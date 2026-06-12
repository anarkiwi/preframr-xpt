# `content_diffusion` (D3PM absorbing-state content head, Approach A) — REFUTED 2026-05

**Hypothesis:** per-position parametric softmax under-fits the genuinely multi-modal next-content
distribution; a discrete-diffusion denoising objective over content positions (structural keeps CE)
models the full posterior. The maximalist branch of `multi_modal_objective_design.md`, built on the
per-tier router infrastructure. Design: `design/refuted/content_diffusion_design.md`.

## Why refuted

- It is a **sampling-side** intervention: the CE-measured content bottleneck did not move — re-seen
  flat in the 2026-05 re-arc mini triage (**−0.002 val_acc** vs the mos4+entropy baseline).
- Approach C (per-tier MoS), whose refutation this was the designated fallback for, refuted at
  prodlike on router saturation — the shared infrastructure (tier router) is the broken link, and
  diffusion inherits it (the router must still decide content-vs-structural per position).
- The content ceiling was subsequently lifted tokenizer-side (event model atoms-only eval_a content
  0.479 vs the ~0.13 ceiling) with plain CE — confirming the objective was never the binding
  constraint.

## Do not revisit without

- A demonstration that prediction is good but *sampling* is degenerate under the current encoding
  (content-tier CE healthy, samples fail the generation quality gate in a way temperature/top-k
  cannot fix), AND a per-step inference budget the predict host can carry (K-step denoising
  multiplies decode cost; see `design/performance/orin_inference_optimization_design.md`).
