# `--start-seq` rotation semantics audit + fix — design note

Pipeline coverage hole from AGENTS.md §Pipeline coverage holes:
`RegDataset.predict_load` (`preframr/train/regdataset.py`) selects
the predict target by `--start-seq` index into a df-map filtered by
`predict_set`. The selection IGNORES `n_rotations`: regardless of
the dump's rotation count, `predict_load` always loads
`.0.blocks.npy` (rotation 0).

For `MAX_PERM > 1` (e.g. prodlike `max_perm=2`, mini `max_perm=2`),
half (or more) of the rotations on disk are unreachable via
predict.

Docs-only this commit. Implementation = `preframr/train/regdataset.py`
edit, blocked on prodlike completion.

## Symptom evidence

`preframr/train/regdataset.py` (predict_load):

```python
df_map = df_map.drop_duplicates("dump_file").reset_index(drop=True)
...
target_row = kind_df.iloc[start_seq]
target_file = target_row["dump_file"]
...
blocks_path = target_file.replace(DUMP_SUFFIX, ".0.blocks.npy")   # always rotation 0
target.add(blocks_path, SeqMeta(irq=irq, df_file=target_file, i=0))
```

`target_row` has an `n_rotations` column populated by the tokenize
stage (e.g. n_rotations=2 for max_perm=2). The predict path
discards it and hard-codes rotation index 0.

Compare with the training-side `load()` path (`preframr/train/regdataset.py`):

```python
for _, row in df_map_df.iterrows():
    n_rot = int(row["n_rotations"]) if pd.notna(row["n_rotations"]) else 0
    ...
    for i in range(n_rot):
        blocks_path = df_file.replace(DUMP_SUFFIX, f".{i}.blocks.npy")
        ...
        target.add(blocks_path, SeqMeta(irq=irq, df_file=df_file, i=i))
```

Training-side iterates all rotations correctly. The bug is
predict-only.

## Impact

- **Training is unaffected.** All rotations are seen during fit.
- **Predict-side audition / WAV generation** can only address the
  first rotation of each SID. For prodlike (max_perm=2), rotation
  1 is unreachable — half the rotation diversity is invisible to
  manual audition.
- **Smoke gates on memorise** use start_seq=0 and max_perm=1
  typically; they don't trip the bug today. The bug latent only
  for predict-side workflows that vary across rotations.
- **`drop_duplicates("dump_file")`** at line 887 is the
  cosmetic-defensive operation that AGENTS.md flagged. It's a no-op
  given the tokenize-side df-map convention (one row per dump_file)
  — but its presence in the predict path suggests the author was
  aware that df-map might carry duplicates, then never wired
  rotation-aware addressing. The comment in AGENTS.md is correct:
  the dedup HIDES that rotations are missing, since the dedup
  collapses to one row even if rotations were spread across rows
  in some legacy df-map format.

## Audit probe

Before implementing the fix, run a quick probe to confirm the
expected scope:

```python
# preframr_experiments/audit/audit_start_seq_rotations.py
"""Count rotations on disk + rotations addressable by --start-seq.

For a given workdir's df-map.csv:
  - n_files_total = len(df_map)
  - n_rotations_total = sum(n_rotations)
  - n_addressable = n_files_total  (rotation 0 of each file)
  - n_unreachable = n_rotations_total - n_addressable

Report: predict-side coverage ratio = n_addressable / n_rotations_total.
For prodlike with max_perm=2 and full rotation occupancy: 0.5.
"""
```

Runtime ~1 minute on a populated workdir (CSV read + arithmetic).
Output committed alongside the fix.

## Fix options

### Option A — flat indexing across (dump, rotation) pairs

Rebuild predict's selectable list as `[(dump_file, rotation_idx),
...]`. `--start-seq N` selects the N-th entry in the flattened
list.

```python
df_map = df_map.drop_duplicates("dump_file").reset_index(drop=True)
flat = []   # [(dump_file, rotation_idx, row), ...]
for _, row in df_map.iterrows():
    n_rot = int(row["n_rotations"]) if pd.notna(row["n_rotations"]) else 1
    for i in range(n_rot):
        flat.append((row["dump_file"], i, row))
...
target_file, rot_idx, target_row = flat[start_seq]
blocks_path = target_file.replace(DUMP_SUFFIX, f".{rot_idx}.blocks.npy")
target.add(blocks_path, SeqMeta(irq=int(target_row["irq"]),
                                df_file=target_file, i=rot_idx))
```

**Pros:** single int CLI flag stays; backward-compatible at
start_seq=0 (still selects rotation 0 of the first file).
**Cons:** index meaning changes — start_seq=1 was "second file
rotation 0" pre-fix, becomes "first file rotation 1" post-fix if
max_perm>=2. Operators with cached `--start-seq` values need to
recompute.

### Option B — separate `--start-rot` flag

Keep `--start-seq` as file index; add `--start-rot` (default 0).

```python
target_row = kind_df.iloc[start_seq]
rot_idx = getattr(self.args, "start_rot", 0)
n_rot = int(target_row["n_rotations"])
if rot_idx >= n_rot:
    raise ValueError(f"--start-rot {rot_idx} >= n_rotations {n_rot}")
blocks_path = target_row["dump_file"].replace(DUMP_SUFFIX, f".{rot_idx}.blocks.npy")
```

**Pros:** preserves `--start-seq` semantics; explicit per-axis.
**Cons:** new CLI flag; downstream consumers (audition scripts,
predict harness) need updating.

### Option C — tuple-form `--start-seq "<file_idx>,<rot_idx>"`

```python
ap.add_argument("--start-seq", type=str, default="0")
...
parts = self.args.start_seq.split(",")
file_idx = int(parts[0])
rot_idx = int(parts[1]) if len(parts) > 1 else 0
```

**Pros:** one flag, both axes.
**Cons:** stringly-typed; harder to grep + integrate; surprising
on the wire.

## Recommendation

**Option A (flat indexing).** It keeps the CLI clean (one int
flag), is backward-compatible at start_seq=0 (the most common
case), and matches the training-side `load()` path's flattening
intuition. The "start_seq semantics changed at max_perm>=2"
caveat is small — predict-side workflows have always treated
start_seq as opaque-pick-one-thing; flattening makes that
intuition true.

## Implementation plan (Option A)

1. **Land the audit probe** (`preframr_experiments/audit/audit_start_seq_rotations.py`)
   as a standalone script with no side effects. Run it on the
   current prodlike workdir to baseline the coverage ratio.
2. **Update `predict_load`** in `preframr/train/regdataset.py`:
   - Replace the `kind_df.iloc[start_seq]` indexing with the
     flattened `(file, rot)` list.
   - Pass the rotation index into `SeqMeta` and `blocks_path`
     construction.
3. **Update the range-check error message** to reference the
   flattened length (so out-of-range errors cite the right
   number).
4. **Reuse pattern in the legacy `_load_via_reparse` fallback**
   IF it has the same bug (audit before edit).

## Validation strategy

**L0 — unit (`tests/test_regdataset.py`):**
- Fixture: tiny df-map with 3 dump_files, each with n_rotations=2.
  Synthesise the 6 `.{i}.blocks.npy` files (empty NumPy arrays).
- Assert: `predict_load` at `start_seq` ∈ [0, 5] picks the
  expected (file, rotation) pair without raising.
- Assert: `start_seq=6` raises out-of-range with a clear message.

**L1 — audit probe re-run:**
- Post-fix on the same workdir, the probe reports
  `n_addressable == n_rotations_total` (coverage 1.0).

**L2 — predict-side smoke:**
- Re-run `run_memorize_int_test.sh` (max_perm=1; semantics
  unchanged). Assert: predict output byte-identical to pre-fix.
- Re-run an audition with `--start-seq 1` on a max_perm=2 workdir.
  Pre-fix: error or unexpected output; post-fix: rotation 1 of
  file 0 loads cleanly.

## Backward compatibility

- **At `start_seq=0`:** semantics unchanged (rotation 0 of file
  0).
- **At `start_seq > 0` with `max_perm == 1`:** semantics
  unchanged (rotations only at index 0; flattening degenerates).
- **At `start_seq > 0` with `max_perm > 1`:** semantics
  CHANGE. Operators picking specific files by index need to
  re-derive their start_seq values.

Document the change in the args.py help text and in
AGENTS.md when the fix lands. Affected workflows (orinnx
audition, predict-side reproduction harness) need a one-pass
audit to confirm no scripts hard-code start_seq values that
need updating.

## Effort

- Audit probe: **~0.5 day** (script + 1 prodlike-workdir run).
- `predict_load` fix + tests: **~0.5 day**.
- Backward-compat audit (grep `start_seq` in scripts /
  audition harnesses): **~0.2 day**.
- AGENTS.md + args.py help-text update: **~0.1 day**.

Total: **~1.5 days**. Lands after `loop_lookahead_prodlike`
completes (touches `preframr/*`).

## Order of operations

1. Land this design (reviewer pass).
2. Land audit probe (`preframr_experiments/audit/audit_start_seq_rotations.py`) —
   no side effects; can land mid-run if useful for diagnostics,
   but realistically waits for prodlike completion to bundle.
3. Run the probe; commit baseline output to
   `data/start_seq_rotation_audit.json`.
4. Land `predict_load` fix + L0 tests.
5. Land help-text + AGENTS.md update.
6. L2 smoke runs.

## Out of scope

- **Train-side rotation re-shuffle.** Training already iterates
  all rotations correctly; no change needed.
- **`--end-seq` / range selection.** Single-file audition is the
  current contract; multi-file ranges are a separate feature.
- **Per-rotation diagnostic dumps.** Could be useful for debugging
  rotation-specific learning failures; out of scope here.

## Risks

- **Operator workflow drift:** anyone using `--start-seq N` on a
  max_perm>1 workdir is implicitly relying on the pre-fix
  behaviour ("pick the N-th file"). After the fix, "pick the
  N-th (file, rotation)" might surprise. Mitigation: AGENTS.md
  callout + args.py help text + a runtime info-log at
  `predict_load` that prints the resolved (file, rotation).
- **Stale predict-set artefacts.** A workdir generated pre-fix
  has `.0.blocks.npy` and `.1.blocks.npy` files but no
  rotation index encoded in df-map's wireformat (just
  `n_rotations`). The fix relies on `n_rotations >= 1` being
  honest; existing artefacts satisfy this.
