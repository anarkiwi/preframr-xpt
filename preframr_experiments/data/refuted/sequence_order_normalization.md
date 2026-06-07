# `sequence_order_normalization` (intra-frame write-order collapse) — REFUTED 2026-05-27

**Hypothesis:** engines share content vocabulary (full-atom content cosine 0.96)
but diverge on per-frame write ORDER; canonicalizing the inaudible write-order
degree of freedom would make engines look alike in the training token stream,
freeing sequence-modeling capacity for content (a representation/generalization
lever). Design: `design/refuted/sequence_order_normalization_design.md`.

**Decided by CPU audit + render proof, no model A/B** (`audit/audit_seq_order_norm.py`,
8 eval_b engines, post-full_macros, image `anarkiwi/preframr:0.2.3`).

## Why refuted

Decomposing the per-frame reg-tuple divergence (cross-engine cosine):

| signature | cosine | gap |
|---|---|---|
| SET (composition only) | 0.466 | — |
| MULTISET (+ multiplicity) | 0.343 | −0.123 multiplicity |
| TUPLE (+ order) | 0.297 | −0.046 order |
| TUPLE voice-canonicalized (legal, audio-safe reorder) | 0.306 | recovers **+0.009** |

The SET→TUPLE gap (+0.169) is 0.123 **multiplicity** + 0.046 order. Legal,
voice-respecting, audio-safe reordering recovers only **+0.009 (~5%)**. And the
multiplicity is content, not redundancy: **84%** of the 284,857 intra-frame
repeated writes carry DISTINCT values (genuine sub-frame modulation the SID
renders); only 16% are dead same-value rewrites. So the divergence beyond the
shared register SET is mostly real audible modulation content a model must learn
— not normalizable notation.

The reorder itself is genuinely inaudible (canonical `(reg,subreg)`-sort renders
at corr 1.000000, maxabs ≈ 6e-4; reg-state byte-identical) — it is simply not
where the cross-engine divergence lives.

## Do not revisit without

- A measurement showing the order component (MULTISET→TUPLE, currently −0.046)
  is materially larger on the corpus/tokenizer in question (re-run
  `audit_seq_order_norm.py --mode divergence`), AND
- a reason the gain is content-tier-relevant despite ~5% legal-reorder headroom.

**Methodological note:** the first decomposition treated VOICE_REG as a free,
independently-movable marker — WRONG. VOICE_REG sets the active voice; its writes
are voice-relative and travel with it (a write may never cross a VOICE_REG). The
honest decomposition reorders whole voice-block units / sorts within a voice run.

## Salvage

The 16% dead same-value rewrites are a small, clean, inaudible token-count
reduction — folded into the redundant-writes note in
`design/encoding/audio_equivalence_normalization_design.md` (the still-open per-write
sibling), not a standalone direction. The audit stays as a reusable instrument.
