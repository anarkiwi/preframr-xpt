# `macro_coarsening` (simple-coarsening alphabet shrink) — REFUTED

**Status:** Experiment E1 in
`integration_tests/design/macro_coarsening_research.md` refuted the
alphabet-row tradeoff. `coarsen_pass` retained as a tracker-export
tool (post-encoder transform for human-readable output), NOT as an
encoder pass.

## Hypothesis

A "coarse" alphabet that buckets nearby register values (e.g.
quantises CTRL and ADSR bytes at coarser granularity = L) would
shrink alphabet size, letting the model see a smaller effective
vocab and potentially generalising better.

## Refutation

Experiment E1: re-tokenise the smoke corpus at coarsening levels
L ∈ {4, 8, 16, 32}; count alphabet size + row count + train a small
model to compare val_acc.

Result at L=16 (the sweet spot from a pure alphabet-shrink
perspective):

- Alphabet shrink: **9%** (vs L=1 baseline).
- Row growth: **+160%** (coarsened bytes can no longer use
  PATTERN_REPLAY; macros that depend on byte-identity break down).
- Val_acc: degraded (more rows to predict, smaller alphabet helps
  but doesn't compensate).

At L=32, alphabet shrinks ~15% but row growth exceeds +250% —
even worse net. At L=4, alphabet barely shrinks (~3%) and row
count is comparable to baseline — no gain either direction.

Net: alphabet shrink at any L is dwarfed by the row growth.
Coarsening as an encoder pass is strictly worse than the standing
encoder.

## Retained use

The `coarsen_pass` itself stays in the codebase as a
**tracker-export tool**: external tools that consume the
post-encoder representation (e.g. visualisation, tracker round-trip)
can ask the encoder to apply coarsening as a downstream
post-processing step. This use does NOT affect training; the
encoder pipeline ingests un-coarsened bytes.

## Evidence

`integration_tests/design/macro_coarsening_research.md` §E1 —
per-L alphabet/row/val_acc table.

## Do not revisit without

- A coarsening scheme that preserves PATTERN_REPLAY compressibility
  (the current break is structural: coarsened bytes no longer
  match across repeats, so PATTERN_REPLAY fires less often).
- OR independent evidence that smaller alphabet at fixed row count
  helps val_acc generalisation in this architecture — the existing
  `cents_sweep` (2026-05-10) already showed cents=25 (finer
  alphabet) is worse on val_acc than cents=50 (default), so the
  generalisation-via-smaller-alphabet hypothesis is independently
  challenged.
