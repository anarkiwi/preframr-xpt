# `per_tier_heads_entropy_prodlike` — REFUTED 2026-05-23

Three-run follow-on to `per_tier_heads_mos_prodlike` (v10 refute) testing
whether `--per-tier-mos-entropy-lambda` can recover prompt-conditioning
at prodlike capacity. Mini retest PASSED cleanly; prodlike capacity
attenuation refutes the thread definitively at both lambda values
tested at prodlike.

## Experiment thread

| run | tier | lambda | spec | verdict |
|---|---|---|---|---|
| `per_tier_heads_entropy_retest` | mini, 3 seeds | 0.01 | `specs/per_tier_heads_entropy_retest.py` | **PASS** (val_acc 0.1483±0.0016, div_ratio 1.535, collapse 0/12) |
| `per_tier_heads_entropy_sweep_mini` | mini, 1 seed | 0.02 | `specs/per_tier_heads_entropy_sweep_mini.py` | passes — div_ratio 1.596 (mini peak), val_acc 0.1490 |
| `per_tier_heads_entropy_sweep_mini` | mini, 1 seed | 0.05 | same | over-regularised — div_ratio 1.400, val_acc 0.1499 |
| `per_tier_heads_entropy_prodlike` (v11) | prodlike, 1 seed | 0.01 | `specs/per_tier_heads_entropy_prodlike.py` | borderline FAIL — div_ratio 1.123 (gate 1.2), 3/4 PASS |
| `per_tier_heads_entropy_prodlike_v12` | prodlike, 1 seed | 0.02 | `specs/per_tier_heads_entropy_prodlike_v12.py` | **REFUTE** — div_ratio 1.038, collapse 42%, content acc 1.98× (just under 2.0×) |

## Gate result (v12 = peak prodlike entropy attempt)

| # | Criterion | v10 (λ=0) | v11 (λ=0.01) | **v12 (λ=0.02)** |
|---|---|---|---|---|
| 1 | content acc ≥ 2× baseline AND ≥ 0.12 | 0.1358 (2.20×) PASS | 0.1261 (2.04×) PASS | **0.1222 (1.98×) FAIL** |
| 2 | ≥5/8 eval_b lifts | 8/8 PASS | 8/8 PASS | 8/8 PASS |
| 3 | collapse_rate at T=0.5 ≤ baseline (8%) | 33% FAIL | 8% PASS | **42% FAIL** (5/12, periods [2,2,2,2,4]) |
| 4 | diversity_ratio at T=0.5 > 1.2 | 1.031 FAIL | 1.123 FAIL | **1.038 FAIL** |

v12 audit artefacts: `integration_tests/data/audit/per_tier_heads_entropy_prodlike_v12/`
(audit_checkpoint_per_class + loop_detection + prompt_conditioning JSONs +
12 stream CSVs).

## Failure-mode interpretation

The mini sweep showed a clean inverted-U with peak at lambda=0.02
(diversity_ratio 1.596 vs 0.01's 1.535 and 0.05's 1.400). The
mini → prodlike attenuation analysis from v11 (29% of mini lift
transfers) predicted v12 at lambda=0.02 would hit diversity_ratio
~1.14 at prodlike. Actual: **1.038, BELOW v11's 1.123**.

Two things broke the extrapolation:

1. **Lambda is non-monotonic at prodlike, not just at mini.** At
   prodlike, lambda=0.01 IS the peak. Raising lambda further
   over-flattens the router posterior so the model loses prompt
   discrimination — diversity_ratio degrades AND collapse_rate
   explodes (8% → 42%) because the looser router lets the model
   fall into period-2 attractors. The mini curve's peak at 0.02
   does not transfer because prodlike capacity makes the router
   posterior naturally sharper; the regulariser hits the
   sharpening-vs-discrimination tradeoff at a lower lambda.
2. **Content acc erosion is real.** v10 → v11 → v12: 0.1358 →
   0.1261 → 0.1222. Higher lambda costs content prediction
   capacity even when the curve hasn't peaked. v12's 1.98× ratio
   is the first time we've slipped under the 2.0× criterion on
   this thread.

The honest framing: **v11 (lambda=0.01) was the prodlike ceiling
for entropy regularisation**. The thread is exhausted; tuning the
knob cannot get us past criterion 4.

## What this changes

1. **Refutes entropy regularisation as a sufficient fix for v10's
   prodlike router saturation.** It helps (v11 collapse_rate 33% →
   8%, content acc lift preserved) but cannot recover
   prompt-conditioning to the baseline level (1.40 plain-CE
   reference). The architecture has a ceiling that no single
   hyperparameter can break.
2. **Do NOT escalate `--per-tier-heads --per-tier-mos-entropy-lambda
   0.01` to defaults.** v11's borderline result was the optimistic
   reading; v12 confirms the ceiling. Defaults stay at single-head
   plain CE for now.
3. **Promote queue item 2 (cluster-conditional content head) to
   in-flight.** Design at `integration_tests/design/cluster_conditional_content_head_design.md`.
   Phase 0 starts now (offline cluster index, ~1 hr fogbank).
4. **Mini retest data retained as positive evidence** for the
   router-architecture framing (vs the gradient-dominance framing
   that mask_structural_loss refuted). The entropy mechanism
   demonstrably moves the right needle at mini; the cluster head
   is the next intervention in that framing's direction (commit
   to acoustic region before token).

## Open questions for the cluster-head work

- **Does the cluster head's two-stage sampling pattern transfer
  better mini → prodlike than the entropy-tuned single-stage
  router?** Hypothesis yes: the cluster commit happens at
  inference, not training, so it doesn't get attenuated by
  training-time capacity tradeoffs. Phase 3 verifies.
- **Is the 2.0× ratio criterion right at prodlike?** v10's 2.20×
  baseline ratio was within reach; v11/v12 slipped below or to
  the line. Recalibration to ratio ≥ 1.8× would give the cluster
  head a fairer gate against the same baseline.

## References

- v10 (original Phase 3 refute): `per_tier_heads_mos_prodlike.md`
  (this directory).
- mask_structural separator: `mask_structural_loss.md` (this directory).
- Cluster-head design (in-flight): `../../../../preframr/integration_tests/design/cluster_conditional_content_head_design.md`.
- Specs: `../../specs/per_tier_heads_entropy_{retest,sweep_mini,prodlike,prodlike_v12}.py`.
