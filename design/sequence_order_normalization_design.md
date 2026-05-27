# Sequence-order normalization (intra-frame write-order collapse)

**Status:** **Refuted as a generalization lever 2026-05-27.** The audio-safe
reorder *works and is inaudible*, but it recovers only ~5% of the cross-engine
divergence it was meant to close — the gap is dominated by genuine sub-frame
**modulation content**, not by reorderable notation. Decided by CPU audit +
render proof (below), no model A/B needed. Do-not-revisit stub:
`preframr_experiments/data/refuted/sequence_order_normalization.md`. The
sequence-level sibling of
[`audio_equivalence_normalization_design.md`](audio_equivalence_normalization_design.md)
(per-write `val` collapse, still open). Generalization axis / representation;
see [[representation-thread]].

## Hypothesis (refuted)

Engines share content vocabulary (full-atom content cosine 0.96) but diverge on
per-frame write ORDER, so canonicalizing the inaudible write-order DoF would make
engines look alike in the training token stream, freeing sequence-modeling
capacity for content. **Tested: the divergence is mostly not order.**

## What the audit measured (`audit/audit_seq_order_norm.py`)

8 eval_b engines, post-`full_macros`, `anarkiwi/preframr:0.2.3`, CPU.

**Decomposition of the per-frame reg-tuple divergence** (`--mode divergence`):

| signature | cross-engine cosine | gap |
|---|---|---|
| SET (composition only) | 0.466 | — |
| MULTISET (+ multiplicity) | 0.343 | **−0.123 multiplicity** |
| TUPLE (+ order) | 0.297 | **−0.046 order** |
| TUPLE voice-canonicalized (legal reorder) | 0.306 | recovers **+0.009** |

The SET→TUPLE gap (+0.169) I first read as "write order" is actually 0.123
**multiplicity** (how many times each reg is written per frame) + 0.046 order.
Legal, voice-respecting, audio-safe reordering recovers only +0.009 (~5%).

**Multiplicity is content, not redundancy:** of 284,857 intra-frame repeated
writes to a register, **84% carry distinct values** (genuine sub-frame modulation
— PWM, vibrato, fast arps the SID renders), only 16% are dead same-value
rewrites. So the −0.123 multiplicity gap is mostly real audible content engines
genuinely differ on; a model must learn it, it cannot be normalized away.

**Voice semantics (the correction that exposed this):** VOICE_REG sets the active
voice; its following writes are voice-relative and MUST travel with it — a write
may never cross a VOICE_REG boundary (that reassigns it to another SID channel).
The only legal reorders are (a) stable-sort writes *within* a voice run and (b)
move whole voice-block units. Treating VOICE_REG as a free, independently-movable
marker (my first decomposition) was wrong and inflated the apparent order
headroom; the voice-aware decomposition above is the honest one.

## The reorder is genuinely inaudible (just low-value)

`--mode fidelity` (raw renderable writes, 6 dumps): per-frame SID reg-state is
byte-identical under any stable reorder (invariant — same-reg order preserved ⇒
last-write-wins unchanged); the canonical `(reg,subreg)`-sort renders at
**corr 1.000000, maxabs ≈ 6e-4** (float rounding) — inaudible, because it matches
the hardware byte-order convention (lo-reg before hi-reg, voices contiguous) the
dumps already follow. Controls confirm the constraints are load-bearing: a random
value-latch shuffle drifts (corr 0.98–0.9994, it splits lo/hi byte-pairs); moving
CTRL is plainly audible (corr 0.72–0.91). So an audio-safe reorder exists — it
just isn't where the cross-engine divergence lives.

## What remains (small)

- **Redundant same-value rewrite dedup** (the 16% dead-overwrite slice): a clean
  inaudible token-count reduction, but small and partly covered by `DedupSetPass`.
  Belongs with the redundant-writes note in
  [`audio_equivalence_normalization_design.md`](audio_equivalence_normalization_design.md),
  not as a standalone direction.
- The audit stays as the **iterable instrument** — re-run on new tokenizer
  versions / corpora to confirm the order/multiplicity split holds before
  reconsidering.

## References

- Audit + render proof: `preframr_experiments/audit/audit_seq_order_norm.py`.
- Per-write sibling (still open):
  [`audio_equivalence_normalization_design.md`](audio_equivalence_normalization_design.md).
- Engine-divergence framing: `tokenization_vs_music_llms.md`, [[representation-thread]].
- Reg constants: `preframr_tokens/stfconstants.py` (VOICE_REG=−126, CTRL 4/11/18).
