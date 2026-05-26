# Per-tier output heads + MoS content head (Approach C)

**Status:** **REFUTED at prodlike** (mos + entropy variants — router posterior
saturates, outputs ignore prompt content; see
`../data/refuted/per_tier_heads_mos_prodlike.md` +
`per_tier_heads_entropy_prodlike.md`). The all-tier val_acc lift it produces is
**structural, not content** (re-seen in the 2026-05 re-arc mini triage: +0.057
all-tier acc, content unverified). Superseded `multi_modal_objective_design.md`
Approach C bullet after `contrastive_infonce_auxiliary` refuted.

## Problem (re-anchored)

`multi_modal_objective_design.md` framed the bottleneck as
"per-token CE cannot express the multi-modal target distribution
that musical continuation requires". The follow-on bet (Approach
B, InfoNCE auxiliary) refuted at mini sweep (`L0.1_K32` regressed
below baseline; `L0.05_K64` lift 0.0008 → 0.0043 single-seed,
smaller than the same baseline at 60 vs 30 epochs: 0.0008 → 0.0063)
and at prodlike epoch 8 (eval_a content acc 0.0000). InfoNCE failed
on two counts: (a) it is a softmax-over-(K+1) classifier that pulls
mass onto GT — the failure mode it claimed to fix — and (b) random
distractors over the full 32K vocab are dominated by trivially-
out-of-tier samples, diluting the within-content discrimination
signal.

The framing survives the refutation: the *bottleneck remains
distribution-shape on content-tier positions*. The intervention
must (1) factor the prediction by tier so structural and content
positions are modeled with the objective each requires, and
(2) replace per-content-position CE with a genuinely multi-modal
density model.

## Hypothesis

A shared transformer body feeding four tier-specific output heads,
combined via a marginal-factorization router and trained with
uncertainty-weighted multi-task loss, will lift content-tier
prediction without regressing structural-tier prediction. The
content head uses a Mixture-of-Softmaxes (MoS, Yang et al. 2018)
density model — K=4 mixture components, each a softmax over the
content sub-vocab, combined with state-conditioned mixture weights —
directly modelling the multi-modal target distribution.

## Approach

### Architecture

Shared transformer body (same `llama3_2` backbone, no change). Four
output heads on the final hidden state `h`:

- `head_structural`: `Linear(d → |V_structural|)` + softmax. CE loss
  on structural-tier GT.
- `head_mid`: `Linear(d → |V_mid|)` + softmax. CE loss on mid-tier GT.
- `head_content_mos`: K=4 component softmaxes over `V_content`.
  Component `k` projection `Linear(d → |V_content|)`. Mixture
  weights `π_k(h) = softmax(Linear(d → K))`. Final content
  distribution `P(v|h) = Σ_k π_k(h) · softmax_k(h)[v]`. NLL on
  content-tier GT.
- `head_zero`: `Linear(d → |V_zero|)` + softmax. CE on zero-tier
  GT (small; degenerate cases).
- `head_router`: `Linear(d → 4)` + softmax over tier ids. CE loss
  on GT tier.

`V_structural`, `V_mid`, `V_content`, `V_zero` partition the full
vocab per `_vocab_id_to_class_tier` (existing). The current single
projection `Linear(d → V=32768)` (~24M params with d=768) decomposes
into:

| head | params | notes |
|---|---|---|
| structural | d·\|V_s\| ≈ 0.768·8K → 6M | unchanged shape |
| mid | d·\|V_m\| ≈ 0.768·3K → 2.5M | unchanged shape |
| content (K=4 MoS) | 4·d·\|V_c\| + d·K ≈ 4·0.768·21K → 64M | 4× content sub-vocab projection |
| zero | d·\|V_z\| ≈ 0.768·tiny | negligible |
| router | d·4 ≈ 3K | negligible |
| **total** | **≈73M head params** | vs ~24M single head |

Net delta: +49M params on the head, all on the content path. Body
unchanged. Fits 24 GB envelope at body=large mini and prodlike
canonical body (current prodlike body 125M params + 24M head = ~150M
total → ~200M with new heads; well under 24 GB even with KV +
optimizer state at `batch_size=2 + accumulate=16`).

Predict-host envelope: KV cache (16L × 4 kv_heads × 64 head_dim, GQA,
bf16) is body-only. Heads run once per token, no caching. MoS adds
4× the matmul on `Linear(d=768 → |V_c|=21K)` per token, ~0.07 GFLOP
per token vs original 0.02. Per-token wallclock impact on Orin: <5%
on the linear-projection arm; the body forward dominates.

### Marginal factorization (no discrete router at inference)

`P(v | h) = Σ_t P(tier=t | h) · P(v | tier=t, h)`

At inference, compute all four head distributions + the router
posterior; combine into a single unified probability vector over
the full vocab; sample (or argmax) from the unified vector. No
discrete tier decision is ever made — avoids the "teacher-forced
router vs autoregressive generation" mismatch the original
`multi_modal_objective_design.md` flagged for Approach C.

### Training loss

Uncertainty-weighted (Kendall & Gal) multi-task loss reusing
the existing `learnable_class_loss` machinery, extended to five
task heads (4 tier heads + router):

```
L = Σ_t exp(-2·log_σ_t) · 0.5 · L_t + log_σ_t
  + exp(-2·log_σ_r) · 0.5 · L_router + log_σ_r
```

`L_t` is CE for {structural, mid, zero} and MoS-NLL for {content}.
`log_σ_t` is per-tier learnable scalar (already in HEAD; extend to
include router). `L_router` is the four-way CE on tier ids.

### Inference path

`predict_lib.py` and `constrained_decode.py` extend to call the
multi-head forward, compute the unified posterior, and feed that
into the existing sampler. `StreamState` keeps its current vocab
filter logic — applied to the unified posterior. Hooks already
factored: `_last_token_logits` takes `model_out` and currently
returns `(B, V)`; the multi-head path returns `(B, V)` after
fusion, no signature change.

## Phase plan

| phase | scope | wallclock | gate |
|---|---|---|---|
| 0 | Diagnostic: (a) docker image rebake (post-restructure), (b) `prompt_conditioning_audit.py` + `loop_detection_audit.py` on `content_floor_check` ckpt (mini, 60-ep baseline) and on best `voice_traj_distributed_set_diff_freq_prodlike` ckpt at temperatures 0 / 0.8 / 1.0. (c) Land random-prompt mode in `predict.py` (~30 LoC: skip `get_prompt` dataset read, emit uniform-sampled vocab of length `prompt_seq_len`). | 2–4 hr (rebake ~10 min, ~6 streams × 2 ckpts × 3 temps ≈ 36 generation runs × 1–3 min each on 4090 ≈ 1–2 hr) | Audits emit JSON to `data/audit/per_tier_heads_phase0.json`. **No hard gate** — Phase 0 anchors design decisions empirically before commit. |
| 1 | Implementation behind `--per-tier-heads` (default OFF) + `--per-tier-content-mos-k 4` (default 0 ⇒ disabled). Changes: `model.py` (four heads + router + MoS NLL + uncertainty-weighted aggregation), `predict_lib.py` + `constrained_decode.py` (unified-posterior fusion), `audit_primitives.py` (extend `tier_accuracy` to optionally take router outputs). New unit tests under `tests/`: `test_per_tier_heads_forward.py` (shape + numeric checks), `test_mos_nll.py`, `test_router_marginalisation.py` (compare unified-posterior to `Σ_t π_t · P_t` on toy input). Lint + `run_tests.sh` clean. | 1–2 days impl | Tests pass; smoke training on smoke-tier corpus (7 SIDs) completes one epoch without NaN. |
| 2 | `per_tier_heads_mini_body_large` A/B at mini tier, body=large, 60 max-epochs, **3 seeds** (variance bound). Arms: `per_tier_heads_mos4` (target, first per `spec.arms` convention) and `baseline` (CE, single head — `content_floor_check` config). Audit via `audit_checkpoint_per_class.py` + `prompt_conditioning_audit.py` + `loop_detection_audit.py` on each best ckpt. | ~25–30 min/arm × 3 seeds × 2 arms = **2.5–3 hr** sequential | (1) content acc on `eval` not regressed vs baseline (3σ floor on 3-seed std); (2) `router_accuracy` > 0.7; (3) `loop_collapse_rate` ≤ baseline at temperature 0 AND 1.0; (4) `prompt_conditioning.diversity_ratio` not regressed; (5) `content_over_structural` ratio ≥ baseline + 1σ. Refute if any of (1)–(4) fail. |
| 3 | `per_tier_heads_prodlike` at prodlike tier, body=canonical (16L/768/2048), 60 max-epochs, 1 seed (compute budget). Arms: `per_tier_heads_mos4` target, baseline. Eval suite: `eval_a` + 8 `eval_b_*` families. Audits as Phase 2. | ~10–14 hr/arm × 2 arms = **20–28 hr** sequential | (1) `eval_a` content acc ≥ 2× baseline AND ≥ 0.14 (apush4x baseline ceiling); (2) ≥ 5 of 8 `eval_b_*` families show non-zero content acc lift; (3) `loop_collapse_rate` ≤ baseline; (4) `diversity_ratio` > 1.2 on real-vs-random; (5) no structural regression > 1σ. |
| 4 | If Phase 2 refutes → write refuted entry `data/refuted/per_tier_heads_mos.md`; pivot to Approach A (discrete diffusion on content tier). If Phase 3 refutes → write refuted entry; pivot to Approach A. | — | — |

**Total wallclock from doc approval to prodlike verdict:**
- Phase 0: 2–4 hr (Phase 0c random-prompt mode is the only blocking impl).
- Phase 1: 1–2 days agent work.
- Phase 2: 2.5–3 hr wallclock.
- Phase 3: 20–28 hr wallclock.
- **End-to-end:** 3–4 days assuming Phase 2 passes and goes straight to Phase 3.

## Decisions taken (rationale)

These three choices supersede design-doc-time questions:

1. **Phase 0 first.** The InfoNCE post-mortem framing ("the model
   has no generalizable representation") rests on per-token argmax
   accuracy on content tokens. Per-token argmax conflates "model
   has learned the song style" with "model produces the exact
   continuation in held-out song". The right diagnostic — diversity
   ratio + loop-collapse rate at multiple temperatures — exists
   (`prompt_conditioning_audit.py`, `loop_detection_audit.py`) but
   has never been run on a trained ckpt. Phase 0 establishes the
   empirical anchor before another loss-engineering bet.

2. **MoS K=4 on the content head.** Considered alternatives:
   - *CE with topK soft labels* — cheaper, but defines "plausible
     alternatives" by surface statistics (corpus n-gram counts),
     not by a learned latent. Won't generalise across composers
     where the topK distribution is composer-specific.
   - *Temperature-scheduled CE* — addresses sharpness, not
     multi-modality. Doesn't model the stated bottleneck.
   - *Approach A (discrete diffusion)* — maximalist. Different
     generation algorithm (K-step iterative denoising), different
     predict path, less-mature literature for discrete tokens.
     Reserved as the fallback if MoS refutes.

   K=4 is the standard MoS default (Yang et al.); large enough to
   capture genuine multi-modality (note-vs-rest, melodic-vs-
   chord-tone) without exploding head parameters. Tunable in Phase 2.

3. **3 seeds at Phase 2.** The InfoNCE sweep was single-seed; the
   "5× content acc lift" was within seed variance of nothing. Three
   seeds at mini cost ~3 hr — cheap insurance against the same
   methodological error. Phase 3 reverts to 1 seed (compute budget).

## Risk + non-goals

- **Body bias toward structural.** Structural CE provides the
  dominant gradient by sample count (~25% of positions but high
  per-position acc). Uncertainty weighting (`log_σ_t`) re-balances;
  validated in Phase 2 by checking that body hidden-state norms
  don't shift between baseline and target arms.
- **MoS mode-collapse.** K=4 components can collapse to a single
  mode if mixture weights saturate. Mitigation: add entropy
  regularisation on mixture weights (`λ_entropy · H(π)` term).
  Default `λ_entropy = 0`; tune in Phase 2 if mixture weights
  observed degenerate.
- **Router-marginal disagreement.** Unified posterior assumes
  conditional independence within tier; if true tier boundary is
  fuzzy at some positions (e.g. `mid` tokens that behave content-
  like), router posterior may be miscalibrated. Mitigation: emit
  router entropy in Phase 2 metrics; if router consistently
  high-entropy on fuzzy positions, validates the soft-router design.
- **Not a fix for data scale.** Orthogonal to melody-transfer
  augmentation (`preframr-aug:preframr_aug/augment_melody_transfer.py`). User has
  explicitly deprioritised data-scale interventions for this
  iteration; this design takes that as given.
- **Not a fix for predict-host envelope.** Body forward dominates
  per-token wallclock on Orin; head changes are <5% of inference
  cost. KV cache layout unchanged.
- **Per-token argmax-acc metric staying suspect.** Phase 0 may
  show diversity_ratio > 1.2 on baseline ckpts, undermining the
  framing. In that case: this design is refuted *before
  implementation* and we re-open the metric question. Phase 0 is
  cheap; this is the right place to find out.

## Acceptance criteria for advancing from this design

Reviewer (user) approves:

1. The three decisions above (Phase 0 first, MoS K=4, 3 seeds at
   Phase 2).
2. The total wallclock envelope (~3–4 days end-to-end).
3. The architecture sketch (shared body, 4 tier heads + router,
   marginal-factorization unified posterior, MoS-NLL content head,
   uncertainty-weighted loss).

On approval: Phase 0 launches; once Phase 0 audit lands, Phase 1
implementation begins behind `--per-tier-heads` flag (default OFF).
Phase 1 → 2 gate is unit tests + smoke training. Phase 2 → 3 gate
is the five-criterion mini A/B. Phase 3 verdict determines
escalation (Approach A) or landing (Approach C is the win).

## References

- `multi_modal_objective_design.md` — original four-approach
  framing; this design replaces its Approach C bullet with concrete
  implementation.
- `../data/refuted/contrastive_infonce_auxiliary.md` — InfoNCE
  refutation (to be authored alongside this design landing).
- `audit_primitives.py` — `tier_accuracy`, `detect_tail_cycle`,
  `distinct_n` reused.
- `profile/audit_checkpoint_per_class.py` — Phase 2/3 audit caller.
- `profile/prompt_conditioning_audit.py` — Phase 0 + 2 + 3
  diversity-ratio audit.
- `profile/loop_detection_audit.py` — Phase 0 + 2 + 3 collapse-rate
  audit.
- Yang, Z., et al. (2018). "Breaking the Softmax Bottleneck: A
  High-Rank RNN Language Model." ICLR. — MoS reference.
- Kendall, A., & Gal, Y. (2017). "Multi-Task Learning Using
  Uncertainty to Weigh Losses for Scene Geometry and Semantics."
  CVPR. — uncertainty-weighted loss reference.
