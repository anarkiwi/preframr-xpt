# Per-voice multi-target auxiliary supervision

**Status: Refuted (by class), never run.** Scoped 2026-05-20, no code: auxiliary per-voice heads
(gate/pitch/waveform/ADSR) forcing musical state into the body's hidden activations. A model-side
content intervention — the whole class refuted at the ~0.13 ceiling that tokenizer-side
representation then lifted (see the [`multi_modal_objective_design.md`](multi_modal_objective_design.md)
anti-queue). Forcing state into activations is not a substitute for exposing it in the token stream.
Its A/B spec also named zoo macros that no longer exist, so it would not run as written. Full design
in git history.
