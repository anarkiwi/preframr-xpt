# `per_tier_heads_mos` Phase 3 (prodlike, body=canonical) — REFUTED 2026-05-22

Phase 3 prodlike A/B for per-tier heads + MoS K=4 (the Approach C
target re-opened after Phase 2's sampling-mitigation pass). Spec
`preframr-xpt:preframr_experiments/specs/per_tier_heads_prodlike.py`,
single seed, target arm only (baseline arm pulled from a separate
prodlike baseline run at the same epoch envelope). v10 ran 50/60
epochs (early-stopped); best ckpt epoch 41 step 2254, val_loss 4.4104.

Phase 2 mini lift (val_acc 1.80× baseline) was real and reproducible.
Phase 3 prodlike confirms the lift on the teacher-forced TB scalars
(eval_a content acc 2.20× baseline) but fails three of four gate
criteria. The mini → prodlike capacity-attenuation pattern observed
on earlier designs (`contrastive_*`, `voice_traj_distributed_*`)
repeats.

## Gate result

Criteria from `integration_tests/design/per_tier_heads_design.md`:

| # | Criterion | Verdict | Numbers |
|---|---|---|---|
| 1 | eval_a content acc ≥ 2× baseline AND ≥ 0.14 | **FAIL** (borderline) | baseline 0.0618, target 0.1358; ratio **2.20× PASS**, absolute **0.14 FAIL by 0.0042** |
| 2 | ≥ 5/8 eval_b families show non-zero content acc lift | **PASS** | 8/8 positive (lifts +0.4 pp to +8.0 pp) |
| 3 | loop_collapse_rate at T=0.5 ≤ baseline | **FAIL** | target 33% (4/12), baseline 8% (1/12); periods [2,2,3,3] |
| 4 | diversity_ratio at T=0.5 > 1.2 (real-vs-random) | **FAIL** | target 1.031 (`prompt_ignored`), baseline 1.401 (`prompt_used`) |
| 5 | no structural regression > 1σ | mixed (informational) | eval_a structural flat (+0.0004); eval_b -3 to -9 pp; marquis -30 pp on n=607 (noisy) |

Audit artefacts (main repo, single-seed):
- `integration_tests/data/audit/per_tier_heads_phase3/audit_checkpoint_per_class_v10_target.json`
- `integration_tests/data/audit/per_tier_heads_phase3/audit_checkpoint_per_class_v10_baseline.json`
- `integration_tests/data/audit/per_tier_heads_phase3/loop_detection_v10_T0.5.json`
- `integration_tests/data/audit/per_tier_heads_phase3/prompt_conditioning_v10_{target,baseline}_T0.5.json`
- `integration_tests/data/audit/per_tier_heads_phase3/streams/` (24 streams: real+random × target+baseline × 6 each)

Best ckpt: `/scratch/tmp/preframr_experiments/results/per_tier_heads_prodlike/per_tier_heads_mos4/seed0/tb_logs/preframr/version_0/checkpoints/best-epoch=40-val_loss=4.4104.ckpt`.
Baseline ckpt: `…/per_tier_heads_prodlike/baseline/seed0/…/best-epoch=59-val_loss=5.4574.ckpt`.

## Failure mode interpretation

The interesting signal is the **direction** of criterion 3 and 4
failure. Both regressions concentrate on random prompts:

| metric | target | baseline | direction |
|---|---|---|---|
| collapse_rate on real prompts | 17% (1/6) | 17% (1/6) | parity |
| collapse_rate on random prompts | **50% (3/6)** | 0% (0/6) | target much worse |
| mean_jaccard on real outputs | 0.114 | 0.119 | parity |
| mean_jaccard on random outputs | **0.140** | 0.371 | target much *less* responsive to prompt |

Read directly: at T=0.5 the target model produces almost-identical
output distributions for real and random prompts (diversity_ratio
1.03 ≈ 1.0). The baseline still differentiates strongly (ratio
1.40). The per-tier model's output statistics depend on the **router
posterior**, not the prompt content — exactly the "router saturation"
failure mode the cheap router-entropy retest in
`per_tier_heads_mos_revisited.md` "Open questions" was queued to
probe.

Criterion 1 failing by 3% on the absolute floor (while passing 2.20×
on the ratio) is a separate, weaker signal. The mini Phase 2 lift
was 1.80×; prodlike preserved and slightly amplified the ratio. The
absolute number being below 0.14 reflects that the **prodlike baseline
content acc is itself low** (0.0618 — only half what mini baseline
content acc was, ~0.13). The 2× target on a low baseline lands at a
low number. This is a calibration miss in the gate, not a model
failure per se: ratio is the right comparison; the absolute floor was
set against mini-tier expectations.

If the cheap router-entropy retest reshapes the router posterior
without sacrificing the content lift, criteria 3+4 may flip without
re-running prodlike. Criterion 1 will likely remain borderline on
absolute and pass on ratio.

## What this changes

1. **Re-state Phase 3 verdict.** AGENTS.md "In flight: Phase 3" should
   move to a verdict line; the next-session work item is the
   router-entropy retest at mini, not "run the audits".
2. **Do NOT flip `--per-tier-heads` default in `args.py`.** The Phase 2
   re-open required prodlike confirmation before defaulting; that
   confirmation didn't land.
3. **Recalibrate the absolute floor in `per_tier_heads_design.md`
   criterion 1.** The 0.14 floor was set when mini baseline content
   acc was 0.13 (so target 0.20 ≈ 1.5× looked like a clean win). The
   prodlike baseline halved that to 0.0618, so the ratio target (2.0×)
   is the load-bearing one and the absolute floor should be
   re-derived against the prodlike-baseline measurement, not re-used
   from the mini design.
4. **Promote queue item 1: router-entropy retest at mini.** Trigger
   condition (per `model_loss_queue.md`) is satisfied: "Phase 3
   prodlike mos4 refutes AND the audit shows router saturation as
   the dominant failure mode." The diversity_ratio collapse to ~1.0
   is the router-saturation signature. Single-arm `mos4 +
   --per-tier-mos-entropy-lambda 0.01` at mini, body=large, 60-ep,
   3 seeds; re-run T=0.5 audits.

## Open questions for the router-entropy retest

- **Does lambda 0.01 close the diversity_ratio gap on real-vs-random
  *at mini*?** If yes, escalate to prodlike. If no, the failure mode
  is deeper than router saturation (mixture collapse or softmax
  bottleneck) — escalate to queue item 2 (cluster-conditional content
  head, `model_loss_queue.md`).
- **Does lambda 0.01 cost any content-acc lift?** Entropy regulariser
  spreads the router posterior; that may dilute the tier-specialisation
  win that gave the +1.80× val_acc at mini. Phase 2 gate should track
  both the diversity_ratio and the val_acc together; pass requires
  ratio > 1.2 AND val_acc within 1σ of the mos4 mean.
- **Is the prodlike capacity gap symptomatic of MoS K being wrong?**
  K=4 was picked to match the mini A/B; prodlike may need K=8 or 16
  to express the broader content distribution. Cheap re-test as a
  follow-on if router-entropy doesn't recover the metrics.

## References

- Original Phase 2 refute: `per_tier_heads_mos.md` (this directory).
- Phase 2 re-open: `per_tier_heads_mos_revisited.md` (this directory).
- Design doc: `../../../../preframr/integration_tests/design/per_tier_heads_design.md`
  (main repo).
- Phase 3 spec: `../../specs/per_tier_heads_prodlike.py`.
- Queue: `../../../../preframr/integration_tests/design/model_loss_queue.md`
  (main repo) item 1.
