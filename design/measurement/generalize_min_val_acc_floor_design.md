# `GENERALIZE_MIN_VAL_ACC` floor — calibration design

**Status:** Pending impl — calibration only; the env hook exists
(`preframr_experiments/specs/generalize.py:_generalize_gate` reads `GENERALIZE_MIN_VAL_ACC`,
default 0 = no-op). First event-model data point exists (atoms-only baseline `val_acc` 0.561,
2026-06-12); set the floor once 2–3 canonical event-model runs settle.

## Procedure

1. Collect `val_acc_at_best_loss` per (arm, seed) from 2–3 canonical `generalize`-family baselines
   at a stable commit (the atoms-only baseline counts as one).
2. `floor = (2/3) × median(values)`. The 2/3-median target catches catastrophic-only regressions
   (config drift, encoder breakage ~0.02+), deliberately not ~1σ effects (those need an A/B).
3. **Sanity bounds** (if any fails, report and don't set): `floor > 0`; `floor < median − 3σ`
   (pre-σ-data: `floor ≤ median − 0.01`); `floor ≤ 0.5 × max(observed)`.
4. Land as a spec-level default (`DEFAULT_MIN_VAL_ACC = <value>` in `generalize.py`, env still
   overrides), with the basis (commits, seeds, median) in the docstring; record in AGENTS.md.
5. Validate: L1 — re-run at HEAD, gate passes; L2 — a corrupted-config arm (e.g. `--learning-rate
   1e-1`) trips it.

## Scope notes

- Floors are **tier-specific** (calibrated for canonical only) and **invalidated by a corpus
  re-pin** — a HVSC/tier change must be followed by re-calibration in the same PR or the floor is
  marked stale.
- A val_acc floor catches drops only, not inflation/leakage; eval-B subsets are too small for a hard
  floor (eval-A only). The content-tier verdict and the
  [generation quality gate](../generation/generation_quality_gate.md) are separate, sharper gates —
  this floor is just the cheap catastrophic-regression tripwire.
