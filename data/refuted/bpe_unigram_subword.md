# REFUTED — vanilla BPE / Unigram subwording on the BACC token stream

**2026-06-20.** Applying a vanilla subword tokenizer (BPE or Unigram) on top of the sparse BACC
token-id stream is a learnability NO-GO. Measured on real recovered streams (Monty/5TT/Grid_Runner/
A Mind Is Born + GoatTracker tunes), with a field-id annotation asserting id-equality to the
serializer:

- **Welding:** vanilla BPE @vocab1024 welds **62.5% of token occurrences / 74.8% of mass** across
  field boundaries (NOTE↔INSTR, INSTR↔DUR↔PORTA↔DT, whole rows); Unigram 65% / 79%. 73% of the
  apparent ~3× compression IS the welding.
- **No melodic win:** only ~21 distinct multi-NOTE subwords (~1.4% of mass) — the inline backward-LZ
  (`REPEAT`) already factored repeated phrases before the subword layer sees them.
- **Learnability:** vanilla BPE collapses induction-copy **0.886 → 0.114** and grows a flat ~5.5-bit
  MI tail across all lags (long-range SGD shortcuts) — the same failure mode as the previously
  refuted old-codec BPE, reached via welding instead of dense-trace density-fitting.

Also refuted: pre-collapsing to a whole-field value vocab before subwording (pushes welding to 86%).

**Not refuted (but help-neutral):** HARD field-boundary-segmented BPE (merges can never cross a
field) — 0% welding by construction, ~1.4× compression (mostly the one-shot header/table prologue),
induction-copy preserved (0.741). It does not hurt, but buys little; the representation is already
near the learnable regime (raw copy 0.886). Full analysis: `design/encoding/bpe_unigram_subword.md`.
