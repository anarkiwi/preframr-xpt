# `head_row_class` — REFUTED (pre-2026-05-10, design phase)

**Status:** Experiment E1 in
`head_row_class_design.md` (removed) refuted the
arms-vs-baseline alphabet/atom tradeoff. No implementation landed.

## Hypothesis

Replace the per-row class token at the head of each (voice,
register, payload) tuple with a finer-grained class-token scheme
that distinguishes within-frame writes from frame-boundary writes
from palette-replay rows. Goal: reduce per-token entropy by making
the class structure more explicit to the model.

## Refutation

Experiment E1: re-tokenise the smoke corpus with the proposed
class-token scheme; count atoms (rows) and alphabet size; compare
to the baseline `--no-instrument-pass` encoder.

Result: **+1.70× atoms** (row count grew 70%) for ~5% alphabet
reduction. The atom growth dominates: longer sequences fragment
the macro structure that the existing class-token scheme captures
in compact tokens. Generalisation (val_acc proxy via held-out
encodability) was uniformly worse across the smoke set.

The redesign moves complexity from the alphabet axis into the
sequence-length axis without any net gain on either.

## Evidence

`head_row_class_design.md` (removed) §E1 — atom
counts, alphabet sizes, per-track breakouts.

## Do not revisit without

- A finer class-token scheme that adds atoms < +20% (vs the
  current +70%), AND
- A clear theoretical argument why finer class tokens help
  generalisation (the current scheme is information-equivalent
  to a coarsened version, so finer-granularity is strictly
  redundant unless the model fails to learn the coarsening —
  which the val_acc data refutes).
