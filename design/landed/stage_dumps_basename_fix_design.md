# `_stage_dumps` basename-collision fix — design note

Pipeline coverage hole from AGENTS.md §Pipeline coverage holes:
`_stage_dumps` (`integration_tests/experiments/base.py:307-326`)
copies dumps via `dst_dir / src.name`; HVSC paths that share a
basename (different composer dirs) silently overwrite each other.
**Prodlike `train.list` carries 46 colliding basenames → 50 files
lost → 4437 entries stage as 4387 dumps (1.1% smaller than spec).**

A/B validity holds (deterministic per arm — both arms see the same
4387-dump corpus), so the in-flight `loop_lookahead_prodlike`
decision is not invalidated. But:

- Per-composer eval breakouts under-report participation for
  composers whose SIDs got dropped.
- `eval_per_composer.py` per-composer-skip (deferred bug) is the
  same root cause: composer breadcrumb is lost when paths flatten
  to basename.
- Future re-pins that grow the corpus drift the collision count
  upward — silent corpus shrinkage scales.

**Blocked on `loop_lookahead_prodlike` completion** — the fix
edits `experiments/base.py` which is part of the runner surface
the AGENTS.md mid-run-edit rule forbids. Design only this commit.

## Symptom evidence (2026-05-12 prodlike train.list)

46 basenames collide; 50 dumps lost. Top collisions:

```
5  Axel_F.1.dump.parquet                # 5 distinct composers
3  Trapped.1.dump.parquet
2  Yesterday.1.dump.parquet
2  Wasting_Time.1.dump.parquet
2  Visage.1.dump.parquet
... + 41 more 2-way collisions
```

Per `integration_tests/experiments/base.py:315`:

```python
shutil.copy(src, dst_dir / src.name)   # basename collision overwrites
```

The last-write wins; HVSC alphabetic-walk order determines which
composer survives.

## Downstream consumers of the staged tree

Affected:

1. **`parse.py --reglogs <glob>`** — globs `<train_subdir>/*.dump.parquet`.
   46 paths missing → 46 fewer parsed parquets.
2. **`stftokenize.py`** — same glob; same loss.
3. **`df-map.csv`** — `dump_file` column lists the staged paths
   (`/scratch/preframr/train/Foo.1.dump.parquet`), one row per
   surviving dump. The 46 collided dumps are absent.
4. **`predict_load` / `eval_per_composer`** — read df-map, look up
   per-file metadata. Composer attribution requires the original
   HVSC path (`MUSICIANS/<L>/<Composer>/<sid>`), which is gone
   after the flatten.

## Fix options

### Option A — namespace-preserving subdir layout

Stage as `<dst_dir>/<composer>/<basename>` instead of
`<dst_dir>/<basename>`. Tree mirrors HVSC `MUSICIANS/<L>/<Composer>/`
flattened by composer (drop the letter-bucket level).

```python
def _stage_dumps(rels, src_root, dst_dir, logger):
    ...
    for rel in rels:
        src = src_root / rel
        if not src.exists(): ...
        composer = _composer_from_rel(rel)        # parses MUSICIANS/<L>/<Composer>/
        sub = dst_dir / composer
        sub.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, sub / src.name)
        staged += 1
```

**Globs updated:** `--reglogs /scratch/preframr/train/*/*.dump.parquet`
(one extra `*`). Same applies to `eval_*` subdirs.

**Pros:**
- Eliminates basename collisions (composer dirs are unique).
- Composer breadcrumb survives → unblocks `eval_per_composer`.
- One-level deeper glob is a one-character change.

**Cons:**
- ~25 composer dirs created under train/, eval_a/ each — `ls` is
  slightly noisier.
- Tests / scripts that assume `train/foo.dump.parquet` paths need
  updating (`profile/audit_eval_leak.py`, validate_branches,
  some integration tests).

### Option B — HVSC-rel-path-preserving layout

Stage as `<dst_dir>/<rel>` (preserving the full HVSC tree:
`MUSICIANS/<L>/<Composer>/<sid>`).

```python
shutil.copy(src, dst_dir / rel)   # rel = "MUSICIANS/C/Crisps/Foo.1.dump.parquet"
```

**Globs:** `--reglogs /scratch/preframr/train/MUSICIANS/*/*/*.dump.parquet`.

**Pros:** zero ambiguity; rel-path is the canonical HVSC reference
and matches the .list file format.

**Cons:**
- Heavier glob; 3 extra wildcards.
- More mkdir calls (~25 composers + 25 letter dirs).

### Option C — content-addressed rename on collision

Detect basename collision at stage time; append a hash suffix to
collided files: `Axel_F.1.dump.parquet` →
`Axel_F.1.{8charhash}.dump.parquet`. df-map gets the renamed path.

**Pros:** flat tree preserved; no glob changes.

**Cons:**
- Composer attribution still lost (rename doesn't add composer).
- Downstream code that reads dump filenames for SID identity
  (predict-side composer breakouts) gets opaque hashes.
- Doesn't unblock `eval_per_composer`.

## Recommendation

**Option A (composer-subdir).** It eliminates collisions AND
preserves composer attribution AND requires minimal glob changes.
Option B's full HVSC tree is overkill — the letter-bucket level
adds nothing once the composer dir is in the path. Option C
doesn't address the parent bug (composer-attribution loss).

## Implementation plan (Option A)

1. **Extract `_composer_from_rel`** helper from
   `audit_engine_families.py:69` (already battle-tested regex
   `MUSICIANS/[A-Z0-9]+/([^/]+)/`). Hoist to `base.py` next to
   `stage_dumps`.
2. **Update `stage_dumps`** to create per-composer subdirs +
   shutil.copy into them. Document the new layout in the
   function docstring.
3. **Update glob construction in `run_arm`** (`base.py:548-563`):
   - `train_glob = "/scratch/preframr/train/*/*.dump.parquet"`
   - eval globs: `"/scratch/preframr/<subdir>/*/*.dump.parquet"`
   - parse stage gets the same union.
4. **Update `build_eval_reglogs_arg`** (`base.py:705-721`) to
   emit the extra `*` in the glob.
5. **df-map writes the new paths.** No regdataset.py change
   needed; `dump_file` column just gets composer-prefixed paths.
6. **`predict_load` composer extraction** — once df-map carries
   composer-prefixed paths, `eval_per_composer.py` can recover
   composer from the path. Fold into the same commit or a
   sibling commit (per-composer eval is a separate gated feature).

## Validation strategy

**L0 — unit (`tests/test_experiments.py` or new
`tests/test_stage_dumps.py`):**
- Fixture: a tiny rel list with two paths sharing a basename
  (`MUSICIANS/A/Foo/Bar.1.dump.parquet`,
  `MUSICIANS/B/Baz/Bar.1.dump.parquet`). Synthesise empty parquets
  at the src root.
- Assert: both files stage; counts are 2; resulting layout has
  `train/Foo/Bar.1.dump.parquet` and `train/Baz/Bar.1.dump.parquet`.

**L1 — smoke regen.** Re-run `run_memorize_int_test.sh` end-to-end.
Assert: pre-fix vs post-fix smoke dump count unchanged (smoke list
has no collisions); val_loss / val_acc within seed σ.

**L2 — prodlike staging regression check.** Add an `assertEqual`
in `run_arm`: `len(rels) == count_after_staging` (within the
existing skip-warning tolerance). Without the fix, this asserts
4437 → 4387; with the fix, 4437 → 4437.

**L3 — full prodlike spec re-run.** Post-fix prodlike re-tokenize
should yield 4437 dumps (vs 4387 today). Token-count delta on the
50 newly-included dumps is the regression measurement.

## A/B validity preservation

For specs already-resolved (e.g. mini batch 2026-05-10/11), the
fix changes the corpus shape; their results don't directly
translate. Decision: re-running mini specs is NOT required; the
~1% corpus shrinkage is within seed σ on those decisions. Note
the fix-version in any subsequent prodlike report.

The in-flight `loop_lookahead_prodlike` will land on the pre-fix
corpus (4387 dumps). Its result is comparable across la1/la3
because both arms see the same 4387. Post-fix re-pin should
re-run loop_lookahead_prodlike on the 4437-dump corpus IF the
prodlike Δ is on the threshold of significance — otherwise the
1% shrink doesn't affect the flip decision.

## Effort

- Helper + `stage_dumps` rewrite + glob updates: **~0.5 day.**
- Tests (L0 + L2 assertion): **~0.3 day.**
- Smoke + mini regression (L1): **~0.2 day** + ~30 min compute.
- Documentation (AGENTS.md §Pipeline coverage holes resolved):
  **~0.1 day.**

Total: **~1 day.** Lands after `loop_lookahead_prodlike` completes.

## Out of scope

- **Per-composer eval (`eval_per_composer.py`)** — separate gated
  feature; benefits from this fix's composer-breadcrumb
  preservation but lands separately. The fix DOES NOT enable
  per-composer eval on its own; it removes the blocker.
- **Re-pinning the prodlike train.list to drop the 46 colliding
  basenames.** Doesn't fix the bug — collisions can recur on any
  future re-pin. The structural fix is the staging-layout change.

## Order of operations

1. Land this design (reviewer pass).
2. Implement `stage_dumps` rewrite + helper + glob updates in
   one commit.
3. Land L0 + L2 tests.
4. Smoke regen + mini regression.
5. AGENTS.md update: move §Pipeline coverage hole entry to
   Resolved.
6. Future: post-fix prodlike re-pin can land alongside
   engine_fingerprint_evalb re-pin (composer subdirs + new
   eval-B subset lists in one commit).
