# Cluster-conditional content head (queue item 2)

**Status: REFUTED.** Condition the content head on a learned cluster posterior (acoustic-region
clusters) so content-position uncertainty splits "which acoustic region" from "which token within
it". Hit the same ~0.13 eval_a content ceiling; diversity_ratio ~1.0–1.2 never recovered to the
plain-CE baseline's 1.4–1.6 (mini `cluster_C256`: 1.194 vs 1.596), and the
`per_tier_heads_entropy_prodlike_v12` run it was gated on also refuted. The model-side class is
closed (see the [`multi_modal_objective_design.md`](multi_modal_objective_design.md) anti-queue).
Evidence stub: `preframr_experiments/data/refuted/cluster_conditional_content_head.md`. Full design
in git history.
