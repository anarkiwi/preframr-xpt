# `GENERALIZE_MIN_VAL_ACC` floor — calibration design

Pipeline coverage hole from AGENTS.md §Pipeline coverage holes:
the `generalize` spec ships in **calibration mode** (`MIN_VAL_ACC=0`)
because no canonical baseline was settled when the spec landed. The
infrastructure exists (`experiments/generalize.py:_generalize_gate`
reads `GENERALIZE_MIN_VAL_ACC` from env); only the value to set is
missing.

This note specifies the calibration procedure that produces that
value once 2-3 canonical baselines have run.

Docs-only this commit; no code change. Implementation = setting an
env-var default or hard-coding the floor in the spec.

## Status today

`experiments/generalize.py:35`:

```python
floor = float(os.environ.get("GENERALIZE_MIN_VAL_ACC", "0"))
```

`_generalize_gate` returns `True` whenever `val_acc >= floor`. With
the default `floor=0`, the gate is effectively a no-op (every
non-NaN val_acc passes); the gate exists only as report-mode text.

When set, the gate would cause `run.py` to exit non-zero on a
spec run whose `val_acc_at_best_loss` falls below the floor —
catching regressions before they propagate downstream.

## Calibration target (AGENTS.md text rule)

> Set to ~2/3 of median `val_acc_at_best_loss` once 2-3 canonical
> runs settle.

The "2/3 of median" choice protects against:

- **Seed σ** at canonical tier (TBD; per DESIGN.md §5 mini σ is
  0.0009; canonical σ is expected larger but not yet measured).
  Floor at 2/3-median is well below the noise floor of the canonical
  distribution.
- **Encoder regressions** that shift val_acc by O(0.005-0.020)
  (mid-size encoder change scale). 2/3-median catches a regression
  that erases ~33% of the model's generalisation capability —
  catastrophic-only.
- **Configuration drift** (wrong tkvocab, wrong body size, wrong
  data tier) which typically drops val_acc by O(0.02+).

It will NOT catch:
- Subtle (~1σ) regressions — by design (those need an A/B, not
  a hard floor).
- val_acc INFLATION (e.g. test leakage); a floor only catches
  drops.

## Procedure

### Inputs

The calibration needs **2-3 canonical baseline runs** of the
`generalize` spec at HEAD (or any stable commit) — N seeds per
run (default 1; bump to 3 for σ estimation per Framework
follow-up "Multi-seed default for canonical").

Each run produces `val_acc_at_best_loss` per (arm, seed). For the
single-arm `generalize` spec, that's N values per run, R*N total
across R runs.

### Compute

1. **Median val_acc** across all (run, seed) values:
   `floor_basis = numpy.median([all val_acc_at_best_loss values])`
2. **Floor:** `floor = floor_basis * 2 / 3`
3. **Document** the basis (number of runs, seeds, commit shas,
   resulting median) in `AGENTS.md §Pipeline coverage holes
   resolved` so future agents can re-calibrate if the canonical
   tier changes.

### Sanity bounds

The chosen floor must satisfy:

- `floor > 0` (floor=0 reverts to calibration mode).
- `floor < median - 3σ` (so legitimate seed variance doesn't trip
  the gate). σ from the canonical-tier seed table once available;
  pre-data, use a placeholder rule: `floor ≤ median - 0.01`.
- `floor ≤ 0.5 * max(observed val_acc)` (extra slack against
  outlier-high runs poisoning the median).

If any bound fails, report the conflict and don't set the floor.
Common reasons: too few seeds, atypical run (data corruption), or
the median is genuinely too low (e.g. corpus is too hard for the
body size).

## Where the floor lives

Three placement options:

### Option A — env-var default at spec level

```python
# experiments/generalize.py
DEFAULT_MIN_VAL_ACC = 0.045   # set by calibration

def _generalize_gate(art):
    floor = float(os.environ.get("GENERALIZE_MIN_VAL_ACC", str(DEFAULT_MIN_VAL_ACC)))
    ...
```

**Pros:** explicit, in-tree, version-controlled. Operator can still
override via env. Future calibrations land as a single-line PR.
**Cons:** one constant per spec; if another spec wants a similar
gate, copy-paste.

### Option B — central calibration registry

```python
# experiments/calibration.py
MIN_VAL_ACC_FLOORS = {
    "generalize": 0.045,
    "memorize": 0.6,         # legacy run_memorize MIN_ACC analogue
    # ...
}
```

`_generalize_gate` reads `MIN_VAL_ACC_FLOORS["generalize"]` as the
default; env override still works.

**Pros:** central place to audit floors; easy to grep.
**Cons:** indirection; spec authors might forget to add an entry.

### Option C — spec-level `ExperimentSpec.min_val_acc_floor` field

```python
@dataclasses.dataclass
class ExperimentSpec:
    ...
    min_val_acc_floor: float = 0.0
```

The base `predict_gate` consumes the field; per-spec gates can
override.

**Pros:** typed, discoverable, makes the gate a first-class
contract.
**Cons:** touches `base.py` (blocked on prodlike completion); the
field would be used by exactly one spec for now.

## Recommendation

**Option A** initially: minimal surface, no `base.py` edit, lands
without runner changes. If a second spec gains a similar floor,
re-evaluate (likely promotes to Option B or C).

## Implementation (Option A, post-calibration)

1. Run `generalize` spec ≥2 times at HEAD (or use any 2 sufficiently
   recent canonical-tier runs that haven't regressed). Capture
   `val_acc_at_best_loss` from each (arm, seed).
2. Compute median + 2/3-median per §Procedure.
3. Land a single-line PR in `experiments/generalize.py`:
   `DEFAULT_MIN_VAL_ACC = <value>`. Update the docstring with the
   basis (commit shas, N seeds, median, σ if known).
4. AGENTS.md update: move §Pipeline coverage hole entry to
   §Resolved, record the floor value + calibration basis.

## Validation

**L0 — calibration sanity:** the floor passes the bounds in §Sanity
bounds. Document the median, the σ estimate, and the floor's
distance from each.

**L1 — current-HEAD regression check:** re-run `generalize` at HEAD
with the new floor; assert it passes (sanity-check that the floor
isn't immediately self-tripping).

**L2 — deliberate-failure check:** run a corrupted-config arm
(e.g. `--learning-rate 1e-1`) and assert the gate trips. Confirms
the floor catches real regressions.

## Open questions

- **Tier coupling.** The floor is calibrated for `canonical` tier.
  Specs at other tiers (`mini`, `prodlike`, `smoke`) would need
  their own floors — calibration is tier-specific because the
  data sizes differ. **Decision:** scope this design to
  `generalize` only (canonical); other tiers calibrate
  independently when motivated.
- **Re-calibration cadence.** When the canonical-tier corpus
  re-pins (HVSC version bump, composer-set change), the floor's
  basis is invalidated. **Convention:** re-pin commit MUST be
  followed by a re-calibration run; the spec's `DEFAULT_MIN_VAL_ACC`
  is updated in the same PR or marked stale.
- **Per-spec vs global floor.** `memorize` has a separate MIN_ACC
  floor (memorize is a memorization gate, not generalization).
  Not consolidating: the semantics differ.

## Effort

- 2-3 canonical baseline runs to settle the median: **~4-8 hr
  GPU time** (single-arm canonical, ~60-120 min/arm × 3 seeds = 3-6
  hr at the existing seeds=1 spec default, or 8-12 hr at seeds=3).
- Calibration math + PR: **~30 min**.
- AGENTS.md update + L1/L2 validation runs: **~1 hr**.

Total: **~half a day of operator time** (mostly waiting on GPU);
trivial design + implementation effort.

## Order of operations

1. Land this design (reviewer pass).
2. **Wait for `loop_lookahead_prodlike` to complete** (frees the
   GPU for the canonical baselines).
3. Run `generalize` spec 2-3 times; capture val_acc.
4. Compute floor; land the PR.
5. Update AGENTS.md.

## Out of scope

- **Per-Eval-B floors** — Eval-B subsets are smaller (~16 SIDs);
  noise floor is too wide to support a hard val_acc gate. Use
  Eval-A floor only.
- **val_loss floor.** val_loss is secondary KPI; per DESIGN.md §5
  it's a tie-breaker, not a winner metric. Floor would conflate
  the two.
- **Per-composer floors.** Premature; per-composer eval is gated
  on the `_stage_dumps` composer-breadcrumb fix
  (`design/landed/stage_dumps_basename_fix_design.md`).
