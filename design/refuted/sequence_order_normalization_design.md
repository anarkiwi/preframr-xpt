# Sequence-order normalization (intra-frame write-order collapse)

**Status:** **Refuted as a generalization lever 2026-05-27.** Collapse the per-frame write-order
degrees of freedom that are audio-safe to canonicalise. The audit found only the canonical
voice-respecting reorder is inaudible (≈ the order dumps already use — a near-no-op), recovering
~5% of the cross-engine divergence; **84% of repeated writes are genuine sub-frame modulation
content**, not reorderable notation. Decided by CPU audit + render proof, no model A/B. Evidence +
do-not-revisit: `preframr_experiments/data/refuted/sequence_order_normalization.md`. Historical
note: the v3 canonical contract later licensed a *measured* set of placement liberties — the sound
descendant of this idea, earned per-rule by chip measurement rather than asserted globally. Audit
tool `audit/audit_seq_order_norm.py` kept as instrument. Full design in git history.
