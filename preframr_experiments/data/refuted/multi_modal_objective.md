# `multi_modal_objective` (umbrella: the per-token-CE-bottleneck thesis) — REFUTED, concluded

**Hypothesis:** argmax cross-entropy is the content bottleneck — it cannot express the multi-modal
next-content distribution, so a distribution-aware output objective should lift content accuracy and
prompt-conditioned diversity. Design: `design/refuted/multi_modal_objective_design.md` (carries the
model-side anti-queue).

## Why refuted (all three branches, at prodlike)

- **B — InfoNCE contrastive auxiliary**: `contrastive_infonce_auxiliary.md` (this directory).
- **C — per-tier heads / MoS + entropy router**: router posterior saturates, outputs ignore prompt;
  `per_tier_heads_{mos,mos_prodlike,mos_revisited,entropy_prodlike}.md`.
- **A — discrete-diffusion content head**: sampling-side, no CE change; `content_diffusion.md`.
- (D — energy/DPO sequence ranking: refuted in design, never built.)

Every branch concentrated at the same ~0.13 eval_a content ceiling that **tokenizer-side
representation** then lifted (full_macros 0.13→0.32-class; v3 event model atoms-only 0.479). The
class-level conclusion: learnability is won on the representation side; the objective was never the
binding constraint.

## Do not revisit (the class) without

- The current encoding's content-tier CE at or near its data ceiling (so representation is
  exhausted as a lever), AND
- a failure signature that is distribution-shape-specific (samples degenerate while argmax CE is
  healthy), AND
- a mechanism that does not depend on a learned per-position tier router (the shared
  saturation failure).
