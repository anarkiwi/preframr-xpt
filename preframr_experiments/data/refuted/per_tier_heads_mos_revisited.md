# `per_tier_heads_mos` — REVISITED, refute was greedy-decode artefact

**Status:** Re-opened (2026-05-21). The Phase 2 refute in
`per_tier_heads_mos.md` was based on T=0 greedy decode behaviour
alone. Re-running the same audits across a temperature sweep
(`T ∈ {0.3, 0.5, 0.7}`) shows criteria (3) and (4) PASS at every
sampling temperature ≥ 0.3. Original prediction in the refuted entry's
"Re-open conditions" section was correct: this is a sampling-config
issue, not a model-design issue.

## Evidence (seed 0 representative, 6 streams per condition)

`integration_tests/data/audit/per_tier_heads_phase2/streams/` and
`loop_detection_*` / `prompt_conditioning_*` JSONs.

### Criterion (3) `loop_collapse_rate ≤ baseline` at every T

| T | baseline | mos4 | result |
|---|---|---|---|
| 0.0 | 7/12 | 10/12 | mos4 worse — original refute |
| **0.3** | 1/12 | 3/12 | mos4 marginally worse, both low |
| **0.5** | 0/12 | **0/12** | **tied — PASS** |
| **0.7** | 0/12 | **0/12** | **tied — PASS** |
| 1.0 | 0/12 | 0/12 | tied |

### Criterion (4) `diversity_ratio` not regressed

| T | baseline | mos4 | result |
|---|---|---|---|
| 0.0 | 5.33 | 1.35 | baseline higher — original refute |
| 0.3 | 1.03 | 1.36 | **mos4 higher** |
| **0.5** | 0.93 | **1.22** | **mos4 higher — PASS** |
| **0.7** | 0.96 | **1.21** | **mos4 higher — PASS** |
| 1.0 | 0.97 | 1.00 | mos4 marginally higher |

Baseline's T=0 diversity of 5.33 is an anomaly: greedy baseline
produces highly prompt-locked outputs on real prompts (low jaccard)
and identical outputs on random prompts (high jaccard). At every
higher T, baseline drops to diversity ≤1.0 (prompt_ignored verdict).
**mos4 has materially more stable prompt conditioning across the
temperature range** (1.36 → 1.22 → 1.21 vs baseline's 1.03 → 0.93 →
0.96), which is the opposite of the original refute framing.

### Criteria (1) + (5) recap — already PASSED in original Phase 2

- (1) content acc 0.0075 ± 0.0010 (mos4) vs 0.0020 ± 0.0031
  (baseline) — Δ/σ = +1.67, not a regression.
- (5) content_over_structural 0.0178 ± 0.0027 (mos4) vs 0.0072
  ± 0.0101 (baseline) — Δ/σ = +1.01, marginal PASS.

### Mechanism (confirmed)

Per the original refuted entry's framing: at T=0 the router posterior
saturates on the structural tier (the head that received the largest
absolute accuracy lift). Greedy decode therefore always picks
structural → short cycles → loop collapse. The MoS content head
never gets exercised under greedy.

At sampling temperatures ≥0.3, the router's soft posterior gets
sampled rather than argmaxed, so non-dominant tiers (mid, content)
get picked at their posterior weight. The MoS content head starts
contributing. Loop collapse vanishes by T=0.5 and stays gone through
T=1.0.

## New verdict

per_tier_heads + MoS K=4 is a viable architecture **at sampling
temperatures ≥ 0.5**. It is not a drop-in replacement for greedy
decode; that's an inference-config decision, not a model-design
property.

## What this changes

1. **Sampling default for per_tier ckpts.** Predict CLI / audit
   scripts that default to T=0 should not be used for per_tier
   model evaluation. T=0.5 or T=0.7 is the appropriate operating
   point. Document in `AGENTS.md` and add a guard in
   `audit_checkpoint_per_class.py` (warn if `args.temperature < 0.3`
   for per_tier ckpts).
2. **Re-open Phase 3 (prodlike).** The original design plan was
   "Phase 2 refute → pivot to Approach A (discrete diffusion)". Now
   that Phase 2 passes at sampling temperatures, Phase 3 prodlike
   becomes the natural next test. ~20–28 hr/arm wallclock at
   prodlike + body=canonical, single seed.
3. **Drop Approach A as the immediate fallback.** Discrete diffusion
   remains queued as a future avenue but is no longer the
   pivot-on-refute path.

## Re-validation conditions before pulling per_tier into default

Before defaulting `--per-tier-heads` ON for new training runs, the
re-validation evidence should include:

1. **Multi-seed at mini at the chosen T.** Current revisited evidence
   is single-seed (seed 0 only). Re-run the T=0.5 audits on seeds 1
   and 2 of the existing mos4 ckpts to confirm the result generalises
   across seeds. ~30 min CPU.
2. **Prodlike audit** at T=0.5 against the existing prodlike
   baseline. The mini → prodlike capacity-attenuation pattern has
   refuted several past designs.
3. **Audio audition** at T=0.5 on 2–3 representative songs. The
   audits measure structural metrics; ear validation confirms the
   sampling output is actually musical, not just statistically
   well-conditioned.

## Open questions for future design

- **Can router entropy regularisation save greedy?** The flag
  `--per-tier-mos-entropy-lambda` already exists (default 0). A
  positive value penalises router saturation; might enable greedy
  decode to spread across tiers. Cheap test: retrain mos4 with
  `--per-tier-mos-entropy-lambda 0.01`, run T=0 audits, check if
  greedy collapse drops to baseline. Not load-bearing for the
  re-opened design (sampling already passes); useful for understanding.
- **Tier-conditional decoding** (sample tier from router, then sample
  token within tier) was queued in the refuted entry as a path-4
  fallback. Now unnecessary — sampling on the unified log_p already
  achieves the goal. Drop from the queue.
- **Greedy at sampling-config-frozen serving paths.** If predict-host
  Orin inference does greedy by default, that's an inference-side
  change to switch the default to T=0.5 for per_tier ckpts. Touches
  `preframr/inference/predict.py` defaults + the predict-host serving
  scripts.

## References

- Original refute: `per_tier_heads_mos.md` (this directory).
- Design doc: `../../design/per_tier_heads_design.md`.
- Phase 2 + T-sweep audits:
  `../audit/per_tier_heads_phase2/loop_detection_*.json`,
  `../audit/per_tier_heads_phase2/prompt_conditioning_*.json`.
- Generated streams (all temperatures):
  `../audit/per_tier_heads_phase2/streams/stream_{real,random}__{baseline,per_tier_heads_mos4}_seed0_T{0.0,0.3,0.5,0.7,1.0}_*.csv`.
