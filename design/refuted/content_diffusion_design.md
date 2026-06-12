# Content-tier discrete diffusion head (Approach A)

**Status:** **REFUTED.** The maximalist branch of
[`multi_modal_objective_design.md`](multi_modal_objective_design.md): replace per-position CE on
content positions with a D3PM absorbing-state discrete-diffusion denoising head (reusing the
per-tier router infrastructure). A sampling-side change that did not move the CE outcome; re-seen
flat in the 2026-05 re-arc mini triage (−0.002 val_acc vs the mos4+entropy baseline), and the
approach it was the fallback for (per-tier heads) refuted at prodlike. Evidence stub:
`preframr_experiments/data/refuted/content_diffusion.md`. Full design in git history.
