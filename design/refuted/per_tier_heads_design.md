# Per-tier output heads + MoS content head (Approach C)

**Status:** **REFUTED at prodlike** (mos + entropy variants). Shared body + per-tier output heads
with a MoS (K=4) content head and a learned tier router. At prodlike the router posterior saturates
and outputs ignore prompt content (diversity_ratio ~1.0–1.1 vs baseline 1.4; content acc ~0.13 ≈
ceiling); the all-tier val_acc lift it shows is **structural, not content**. Evidence stubs:
`preframr_experiments/data/refuted/per_tier_heads_{mos,mos_prodlike,mos_revisited,entropy_prodlike}.md`.
The model-side class is closed ([`multi_modal_objective_design.md`](multi_modal_objective_design.md)
anti-queue). Full design + phase history in git history. Code note: `heads.py` MoS/per-tier machinery
remains in `preframr/train/model/` for the record.
