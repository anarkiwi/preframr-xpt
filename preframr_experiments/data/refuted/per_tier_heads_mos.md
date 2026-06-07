# `per_tier_heads_mos` (per-tier heads + MoS content head, Approach C) — REFUTED at Phase 2 gate

**Status: RE-OPENED 2026-05-21.** Initial refute (below) was a
greedy-decode artefact. Temperature sweep at T ∈ {0.3, 0.5, 0.7}
shows criteria (3) and (4) PASS at every sampling temperature. See
`per_tier_heads_mos_revisited.md` for the load-bearing evidence and
the new verdict.

**Original status (kept for history):** Refuted at Phase 2 mini A/B
(`per_tier_heads_mini_body_large`, 3 seeds, 2026-05-21). Criteria 1
and 5 pass; criteria 3 and 4 fail at T=0 greedy. Per
`design/refuted/per_tier_heads_design.md`: "Refute if any of (1)–(4) fail."

Revisit gate (now satisfied): re-open if **sample-time mitigations**
close the greedy collapse gap without sacrificing the teacher-forced
lift documented below.

## Hypothesis

Per `design/refuted/per_tier_heads_design.md` (Approach C): replace the
single softmax head with four tier-specific heads (structural / mid /
content / zero) + a router that selects the tier. The content head
uses Mixture-of-Softmaxes (K=4) to express multi-modal continuations.
Marginal-factorisation produces a unified (B,T,V) log-prob via
`router_log_p[t] + tier_log_p[t]` scattered into a disjoint partition.

Predicted win: lifts content-tier accuracy specifically (the
hypothesised bottleneck for cross-composer generalisation), without
regressing structural / mid.

## Implementation landed

- `preframr/train/model.py`: `MoSHead`, `PerTierHeads`,
  `per_tier_unified_log_p`, tier partition build + uncertainty-weighted
  multi-task loss; flags `--per-tier-heads`,
  `--per-tier-content-mos-k 4`, `--per-tier-mos-entropy-lambda`.
  Initial Phase 1 impl had three bugs that surfaced only at full-scale
  training:
  1. Scatter dtype mismatch (bf16 target vs fp32 src under autocast).
  2. `self.log` calls inside `_per_tier_training_step` traced by dynamo
     hit a PL 2.6 × torch 2.12 introspection bug
     (`inspect.getfullargspec(self.training_step)` on a bound method
     under FakeTensor tracing).
  3. The contributions-list + stack approach materialised 4×(B,T,V)
     in fp32 = 32 GiB peak; OOM on 24 GiB at body=large.
  All three fixed in the same commit; final shape is a single in-place
  scatter (the partition is disjoint, so the marginal collapses).
- `tests/train/test_per_tier_heads.py`: 15 unit tests (incl. the
  mixed-dtype regression that would catch bug 1 above).

## Phase 2 result (`per_tier_heads_mini_body_large`, 3 seeds)

Spec: `integration_tests/experiments/per_tier_heads_mini_body_large.py`.
Tier: mini, body=large (8L/512/1536), 60 max-epochs, batch=4 ×
accum=8, 3 seeds per arm.

### Aggregate val_acc (training-loop teacher-forced metric)

| arm | val_acc_at_best_loss | wallclock/seed |
|---|---|---|
| `per_tier_heads_mos4` | **0.1494 ± 0.0028** | 16.6 min |
| `baseline` (single head CE) | 0.0832 ± 0.0074 | 5.5 min |

Δ = +0.0662, **1.80× baseline**, 2.7× the pooled-σ floor. Far above
the criterion-1 3σ threshold. Wallclock cost: 3× slower per seed.

`val_loss` numbers not cross-arm comparable: per_tier passes
pre-log_softmaxed unified log_p to `chunked_cross_entropy`, which
re-applies log_softmax. Argmax (hence val_acc) is invariant; loss
magnitude is not. Instrumentation bug, separately tracked.

### Per-tier audit (`audit_checkpoint_per_class.py`, max-blocks 16, eval set)

| tier | baseline | mos4 | Δ | Δ/σ |
|---|---|---|---|---|
| structural | 0.240 ± 0.058 | 0.421 ± 0.006 | +0.180 | +3.08σ |
| mid | 0.040 ± 0.011 | 0.129 ± 0.016 | +0.089 | +4.66σ |
| **content** | **0.0020 ± 0.0031** | **0.0075 ± 0.0010** | **+0.0054** | **+1.67σ** |
| zero | 0.0 | 0.0 | 0 | — |
| c_over_s | 0.0072 ± 0.0101 | 0.0178 ± 0.0027 | +0.0106 | +1.01σ |

**Key finding:** the +1.80× macro val_acc lift is driven by
**structural** and **mid** tier gains (+3.08σ, +4.66σ), not the
content tier the design targeted. Content lifts +3.75× relative
(0.20% → 0.75%) but is +1.67σ — below the 3σ threshold that would
be a strong signal, and absolute accuracy remains marginal.

### Generation-based audits (seed 0 representative, 6 streams per condition)

`generate_for_audit.py` → `loop_detection_audit.py` /
`prompt_conditioning_audit.py` at T ∈ {0.0, 1.0}.

| metric | T | baseline | mos4 | direction |
|---|---|---|---|---|
| loop_collapse_rate | 0.0 | 7/12 = 0.58 | **10/12 = 0.83** | **WORSE** |
| loop_collapse_rate | 1.0 | 0/12 = 0.00 | 0/12 = 0.00 | tied |
| diversity_ratio | 0.0 | 5.33 | **1.35** | **WORSE** |
| diversity_ratio | 1.0 | 0.97 | 1.00 | tied |
| verdict @ T=0 | — | prompt_used | prompt_used | both pass |
| verdict @ T=1 | — | prompt_ignored | prompt_ignored | both fail |

## Criterion table

| # | criterion | result |
|---|---|---|
| 1 | content acc not regressed within 3σ | **PASS** (+1.67σ improvement) |
| 2 | router_accuracy > 0.7 | not measured by audit |
| 3 | loop_collapse_rate ≤ baseline @ T=0 AND T=1.0 | **FAIL** (T=0: 0.83 > 0.58) |
| 4 | diversity_ratio not regressed | **FAIL** (T=0: 1.35 < 5.33) |
| 5 | c_over_s ≥ baseline + 1σ | **PASS** (0.0178 > 0.0173, marginal) |

Per design's refute clause: any of (1)–(4) failing → refute. (3) and
(4) both fail → **refuted**.

## Mechanism

The unified log_p at temperature 0 is dominated by the **structural**
tier — that tier got the largest absolute accuracy lift (+18%) and
the router strongly prefers it on most positions. Greedy decode
therefore picks structural tokens repeatedly, producing short cycles
(loop collapse 0.83). The MoS content head — the supposed load-bearing
piece — almost never gets selected at decode time under greedy.

This is a **routing-distribution problem at inference, not a
representation problem**. The teacher-forced metric (val_acc) sees the
benefit because it conditions on the true tier; the autoregressive
greedy generator gets stuck on whichever tier the router favours.

## Re-open conditions

Re-open this design if **inference-side mitigations** close the
generation-quality gap without sacrificing the teacher-forced lift.
Concrete things to try, in increasing order of design intrusion:

1. **Temperature sweep on greedy substitute.** Run loop_detection at
   T ∈ {0.3, 0.5, 0.7, 1.0}. If T=0.7 already gives mos4 ≤ baseline
   collapse and diversity_ratio ≥ baseline, the refute may be a
   greedy-decode artefact rather than a real property.
2. **Top-k / nucleus on the unified log_p.** Same idea, sample-time only.
3. **Router entropy regularisation.** `--per-tier-mos-entropy-lambda`
   already exists; default 0. A small positive value (say 0.01) may
   prevent the router from saturating on structural at greedy.
4. **Tier-conditional decoding.** At decode time, force the router to
   spread mass across tiers (e.g. sample tier ~ router probability,
   then sample token within tier). Decouples greedy-on-unified-log_p
   from the marginal-factorisation training objective.
5. **Joint distribution learning.** If (1)–(4) all fail, the marginal
   factorisation assumption (`P(v) = Σ_t π_t · P_t(v)`) may be too
   restrictive vs the discrete-diffusion approach (Approach A).

## Decision

Pivot to **Approach A (discrete diffusion on content tier)** per the
design's escalation path. Re-open this entry only if the sample-time
mitigations above succeed at mini.

## References

- Design: `../design/refuted/per_tier_heads_design.md`.
- Phase 0 audits (framing): `../data/audit/loop_detection_real.json`,
  `../data/audit/prompt_conditioning_sample.json`.
- Phase 2 data: `../data/audit/per_tier_heads_phase2/` (per-class
  JSONs, loop_detection JSONs, prompt_conditioning JSONs, generated
  streams).
- Spec: `../../integration_tests/experiments/per_tier_heads_mini_body_large.py`.
