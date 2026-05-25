# Multi-modal output objective design

## Problem

Argmax cross-entropy assumes a unimodal next-token distribution. SID
content tokens (FREQ_LO, PWM_PRESET, FC_PRESET, AD, SR) are genuinely
multi-modal: many equally-valid next-notes given musical context.
CE penalises hedging across plausible options, the model collapses
to fixed-point loops under greedy decode, and `content/structural`
accuracy ratio plateaus around 0.10 even at prodlike scale.

Empirically reproduced across `accuracy_push_prodlike_4x` (val_acc
0.177 ceiling, fixed-point collapse), and now
`voice_traj_distributed_set_diff_freq_prodlike` epoch 13 (per-tier
audit: structural 40-79%, content 1-12%, c/s 0.025-0.188 across
eval subsets). Every encoder-level intervention to date has moved
structural accuracy or alphabet width without breaking this ceiling.

## Hypothesis

The objective, not the encoding, is the bottleneck. Per-token CE
cannot express the multi-modal target distribution that musical
continuation requires. A distribution-aware objective should
preserve diversity in content predictions (so plausible alternatives
aren't penalised against each other) while keeping structural
prediction sharp.

## Approaches considered

### A. Discrete diffusion over content positions (maximalist)

Replace CE on content positions with a discrete-diffusion denoising
objective (D3PM / MaskGIT line). Structural positions keep CE.
- Pros: explicit multi-modal modelling; established literature.
- Cons: iterative K-step generation per token; new training loop;
  new predict path; less mature for discrete tokens.

### B. InfoNCE-style contrastive auxiliary loss (minimalist)

Add an auxiliary loss alongside CE: at each content position, sample
K distractor tokens from the vocab, train the model to assign
higher logit to GT than to all K distractors. Run on top of standard
CE. The contrastive term teaches "don't predict obvious wrongs"
without forcing argmax on the GT token. ~50 lines on top of the
existing training loop.
- Pros: minimal infrastructure change, same generation pipeline,
  A/B-testable at micro_mini in days. Standard NLP trick.
- Cons: doesn't directly model the distribution shape; teaches
  "rank GT above distractors" which is weaker than "match the full
  posterior". May not be enough.

### C. Per-tier head separation with distribution objectives

Two output heads sharing the backbone: structural head (CE, sharp),
content head (flow/diffusion/contrastive over content vocab).
Tier-router decides which head fires per position from the token
context. Backbone learns features useful to both.
- Pros: clean decomposition matching the per-tier audit. Each head
  can use the objective best fit to its prediction shape.
- Cons: tier-router itself is a prediction problem at inference
  (you need to know the next token's tier before predicting); may
  require teacher-forcing during training but autoregressive at
  generation, with mismatch risk.

### D. Energy-based / DPO-style sequence ranking

Train an energy function E(prompt, continuation) to score real
continuations lower than corrupted ones. Per-sequence loss, not
per-token. Inference via MCMC or rejection sampling.
- Pros: bypasses per-token argmax entirely.
- Cons: weak per-step gradient signal; expensive inference; doesn't
  fit autoregressive generation cleanly.

## Recommendation: B → C → A

Start with B (cheapest, fastest to validate). If InfoNCE-style
contrastive surfaces content-tier lift at micro_mini scale,
escalate to C for a cleaner architecture. A is the maximalist
fallback only if B+C both refute.

D deferred indefinitely: doesn't fit autoregressive generation
without sampling overhead the deploy host can't carry.

## Phase plan

| stage | scope | wallclock | gate |
|---|---|---|---|
| 0 | Design doc landing + review | --- | user approval |
| 1 | InfoNCE auxiliary loss in `model.py`; flag `--infonce-content-loss`; per-position contrastive on content tier only | impl: 1 day | unit tests + lint |
| 2 | `contrastive_micro_mini` Stage-C A/B (contrastive on/off, both with current baseline pipeline -- no token juggling) | 10 min/arm | gate signals: content/structural lift, distinct-n4 not regressed |
| 3 | Promote to `contrastive_mini` if Stage 2 PASS | 15 min/arm | val_acc Δ ≥ +0.005, cross-engine improving |
| 4 | Promote to `contrastive_prodlike` if Stage 3 PASS | ~6-11 hr/arm | full evaluation + audition |
| 5 | If Stage 2 refutes: design Approach C | --- | new design doc |

## Validation

Algorithmic gates only (no audio review required):

1. `content_over_structural` ratio per eval subset. Pass if combined
   exceeds baseline by ≥50% relative on eval_a AND lifts at least
   one cross-engine subset above 0.10.
2. `distinct_n4` on greedy continuations. Pass if combined does not
   regress (contrastive shouldn't tighten the distribution; if it
   does, that's a different pathology than CE collapse).
3. `loop_collapse_rate`. Pass if combined ≤ baseline.
4. Prompt-conditioning audit (when available). Real-vs-random
   diversity_ratio > 1.2 on the trained checkpoint.

## Risk + non-goals

- **Not a fix for data scale.** Melody-transfer augmentation is
  orthogonal and may need to land alongside.
- **Not a fix for the local-context decoder mandate.** Still
  prompt-only generation; song-global state stays disqualified.
- **May regress structural accuracy.** Contrastive auxiliary loss
  could distract the model from structural CE. Mitigate via loss
  weight λ on the auxiliary term, validate at Stage 2.
- **K sampling cost.** InfoNCE needs K distractors per content
  position. K=32 doubles the per-position effective vocab pass;
  may slow training 10-20%. Tunable.

## Implementation sketch (Stage 1)

```python
def _content_contrastive_loss(logits, targets, tier_mask, k=32):
    """InfoNCE on content positions: rank GT above K random distractors."""
    # logits: (B, T, V), targets: (B, T), tier_mask: (B, T) bool
    B, T, V = logits.shape
    content_pos = tier_mask.nonzero()
    if content_pos.numel() == 0:
        return logits.new_zeros(())
    pos_logits = logits[content_pos[:, 0], content_pos[:, 1]]  # (N, V)
    gt_ids = targets[content_pos[:, 0], content_pos[:, 1]]      # (N,)
    distractors = torch.randint(V, (pos_logits.size(0), k), device=logits.device)
    gt_logit = pos_logits.gather(1, gt_ids.unsqueeze(1))         # (N, 1)
    distractor_logits = pos_logits.gather(1, distractors)        # (N, k)
    combined = torch.cat([gt_logit, distractor_logits], dim=1)   # (N, k+1)
    return F.cross_entropy(combined, combined.new_zeros((), dtype=torch.long).expand(combined.size(0)))
```

Wired into `Model.training_step` as `loss = ce_loss + lambda * contrastive_loss` with `lambda` from a new CLI flag `--infonce-content-loss-weight` (default 0).
`tier_mask` derived from `self.vocab_tier_id` buffer (already built).

## Acceptance criteria for advancing from this design

Reviewer (user) approves Approach B as the first bet. Stage 1 implementation lands behind a CLI flag (default OFF). Stage 2 micro_mini A/B runs unattended in < 30 min with auto-abort gates active. Stage 2 verdict determines escalation.
