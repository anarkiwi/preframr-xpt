# Multi-modal output objective design

**Status: REFUTED (umbrella, concluded).** The per-token-CE-bottleneck thesis: argmax CE cannot
express the multi-modal next-content distribution, so change the output objective. All three
branches were tried and refuted at prodlike — B (InfoNCE contrastive), C (per-tier MoS + entropy
router), A (discrete diffusion). The leverage proved to be **representation/tokenization** (the
tokenizer-side win that lifted the ~0.13 content ceiling; under the v3 event model the atoms-only
baseline reads eval_a content 0.479), not the objective. Evidence stub:
`preframr_experiments/data/refuted/multi_modal_objective.md` (+ per-branch stubs). The full design
(approaches, phase plan, implementation sketch) is in this file's git history.

## Anti-queue — model-side bets NOT to re-attempt

Salvaged from the retired `model_loss_queue.md`; the model-side arc is **closed**. Detailed
refutations live in `preframr_experiments/data/refuted/`.

- **Plain InfoNCE with random distractors** (Approach B) — refuted. Re-open ONLY with cross-composer
  *targeted* negatives (distractors from similar structural contexts in OTHER composers), and only
  if the dominant signal is the eval_b-vs-eval_a gap.
- **Per-tier heads / MoS + router-entropy** (Approach C) — refuted at prodlike (router saturates);
  [`per_tier_heads_design.md`](per_tier_heads_design.md).
- **Discrete diffusion content head** (Approach A) — refuted (sampling-side, no CE change);
  [`content_diffusion_design.md`](content_diffusion_design.md).
- **Cluster-conditional content head** — refuted (same ceiling, diversity ~1.0–1.2);
  [`cluster_conditional_content_head_design.md`](cluster_conditional_content_head_design.md).
- **Static class-weighted CE** (`weighted_token_loss` / `learnable_class_loss`) — refuted; don't add
  another tier-weight knob.
- **Approach D (DPO / energy sequence ranking)** — refuted in design (weak per-step gradient,
  expensive inference); re-open only with a fresh decoding-time story.
- **Per-voice auxiliary supervision** — same model-side class; never beat the ceiling;
  [`per_voice_aux_supervision_design.md`](per_voice_aux_supervision_design.md).
