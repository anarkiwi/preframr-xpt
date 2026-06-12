# `cluster_conditional_content_head` — REFUTED 2026-05

**Hypothesis:** the content head fails because two acoustically-different content tokens get the
same CE penalty as two acoustically-equivalent ones; conditioning the content head on a learned
cluster posterior (acoustic-region clusters, C=256) lets the model spend uncertainty on "which
region" vs "which token within it", recovering prompt-conditioned diversity without losing content
accuracy. Design: `design/refuted/cluster_conditional_content_head_design.md`.

## Why refuted

- Mini A/B (`cluster_C256`): **diversity_ratio 1.194 vs plain-CE baseline 1.596** — the cluster
  head, like every router-driven head, sharpens at the cost of prompt-conditioning and never
  recovers the ≥1.2 gate at scale-relevant settings.
- Content accuracy stayed at the same ~0.13 eval_a ceiling the whole model-side class hit
  (entropy-thread context: v10 router 1.031/0.1358, v11 1.123/0.1261, vs plain CE 1.401/0.0618 —
  the architectures trade diversity for content acc along one frontier; none move the frontier).
- The parent run it was gated on (`per_tier_heads_entropy_prodlike_v12`) refuted, closing the
  per-tier-router line it depended on.

The motivating ambiguity ("acoustically-equivalent token set") was later addressed on the
representation side — the v3 canonical contract collapses chip-equivalent writes at encode time —
which is the lever this head was trying to emulate in the loss.

## Do not revisit without

- Evidence that, **under the v3 event encoding**, content-tier CE is limited by intra-class
  acoustic equivalence (e.g. per-op audits showing mass split across chip-equivalent values that
  canonicalization provably did not collapse), AND
- a router/conditioning mechanism with a demonstrated fix for posterior saturation (the failure
  mode shared by every per-tier/cluster head at prodlike).
