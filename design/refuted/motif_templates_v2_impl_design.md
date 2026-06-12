# MotifDict v2 — value-slotted motif templates

**Status:** **REFUTED 2026-05-27** — built (tokens 0.21.0–0.23.0) and A/B'd. Shape-keyed motif
templates with content slots (template token + slot values, lossless expand), de-fragmenting the v1
motif vocab and exposing motif-carried content to the content tier. v2 recovered most of v1's
regression but content-tier never beat no-motif (v2 0.036 vs baseline 0.045). Evidence + do-not-
revisit condition: `preframr_experiments/data/refuted/motif_pass.md`. (The corpus-mined motif-pass
v1 design was deleted; this doc is the surviving record of the motif direction.) Full design in git
history.
