# Encoding principles — fidelity × context-efficiency × learnability

**Status:** Reference (evidence re-anchored to the v3 event model 2026-06-12; the P1–P8 results were
earned on the retired substrate — the principles stand, the cited ops are historical and live in
this file's git history). The single rubric for SID stream encoding; designs that trade one axis for
another must say which and why.

**Learnability framing.** The axes are not co-equal
([`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md)): **fidelity is a
hard constraint (the gate), context-efficiency is a bounded constraint, learnability is the
OBJECTIVE** — maximize learnability among fidelity-valid, budget-feasible encodings. The
learnability axis is measurable training-free (`audit/learnability_triage.py`).

## The three axes

1. **Fidelity (the floor) — v3 canonical.** `decode(encode(x))` must reproduce
   **`stream.canonical_writes(x)`** exactly: an intra-frame permutation + derivation of the dump's
   writes with zero drops, where every canonicalization liberty is licensed by a pinned reSID
   measurement ([`sid_render_fidelity_contract.md`](sid_render_fidelity_contract.md)). Checked by
   `stream.encode(verify=True)` on every encode, fail-loud. **No lossy tier, no escape path, no
   WAV-audition exception** — values are byte-exact, only canonical *placement* is licensed, and a
   new placement liberty requires a new chip measurement (write-count-matched, per-write-clocked).
2. **Context efficiency.** Tokens per song, bounded by the deploy envelope (Orin: PROMPT=2048 /
   MAX=8192). BPE merging over the fixed atoms is the lever (the vocab dial).
3. **Learnability.** How well a bounded model can *predict* the next token; has structure (below).

The axes conflict, and efficiency that is fidelity-neutral is **not** learnability-neutral — the
defining result: a Unigram merge that bought context at zero fidelity cost destroyed the pitch-onset
signal (0.66 → 0.009) by welding it into thousands of compounds.

## Learnability sub-principles (each earned by a measured result)

- **P1 — Separability.** Each content decision is its own low-cardinality token, never fused with
  unrelated content. *Earned by:* de-merging lifted pitch-onset 0.009→0.658. *v3 embodiment:* typed
  value nibbles + kind-led events; BPE merges remain the surface to watch (do learned merges re-weld
  content boundaries? — check the merge table before blaming structure).
- **P2 — Locality.** Predictive context should be near the decision — but locality only helps where
  cross-song-predictable structure is being separated; it cannot manufacture predictability for a
  multi-modal target (→ P6). Cross-voice de-multiplexing is the open locality lever
  ([`lane_demux_hypothesis.md`](../encoding/lane_demux_hypothesis.md)).
- **P3 — Don't multiplex the target.** Interleaving independent streams (voices) into the
  next-token position dilutes each stream's signal; the per-voice line is the real prediction
  target. (Still structurally true in v3's frame groups — same hypothesis doc.)
- **P4 — Voice/identity is structural, not content.** Surface a structural variable explicitly and
  locally if cheap, but it is not itself the lever (*earned by:* localizing voice id was
  content-neutral). *v3:* explicit `VOICE_*` tags.
- **P5 — Alphabet size ≠ learnability.** Shrinking a field's cardinality doesn't help if the
  *sequence* structure is the hard part (*earned by:* semitone-binning the onset shrank the alphabet
  28% and left the predictability ceiling flat). Fix entropy at the representational source; don't
  just bin.
- **P6 — Use the right yardstick.** Genuinely multi-modal targets (absolute onset pitch caps ~0.51
  even for a memorizing n-gram) structurally undersell exact-token accuracy — score them
  distributionally and by audition
  ([generation quality gate](../generation/generation_quality_gate.md)).
- **P7 — Provenance invariance.** The same musical gesture must encode to the same tokens however
  the source stream produced it (hand-written per-frame writes vs driver table) — otherwise the
  model learns two unrelated things and can leverage neither. *v3 embodiment:* by construction — the
  event grammar has no literal/passthrough path, so every stream expresses in the one universal
  alphabet. The acid test stands: explicit-write and driver-table versions of one gesture must
  yield identical events.
- **P8 — Interpret freq through ctrl, and prove inaudibility before dropping anything.** The control
  register assigns each frame's role: TEST-bit frames hold the oscillator (freq there is the one
  near-inaudible write — absorbable only to a *nearby* value); noise-frame freq is timbre, not pitch
  (and noise can accent a *pitched* note — never classify pitch by waveform); release-phase and
  combined-waveform freqs are audible. **"Not melodic pitch" ≠ "discardable"** — every claimed
  inaudibility must be emulator-proven (preframr-audio pinning tests), which is exactly how the v3
  canonical liberties were licensed. And the long tail of hard engines is recurring mechanism to
  recognize, not noise to go lossy on — lossy is a last resort after tracing every engine.

## The checklist (apply to any encoding change)

1. **Fidelity:** `decode(encode(x)) == canonical_writes(x)` exactly? New canonicalization liberty ⇒
   new reSID measurement, else invalid (no WAV-audition exception).
2. **Separability:** does any token (atom or learned merge) fuse independent content decisions?
3. **Locality:** how many tokens between the decision and its determining context; reducible without
   breaking fidelity?
4. **Multiplexing:** is the prediction target one coherent stream or several interleaved?
5. **Cardinality vs sequence:** is the difficulty the alphabet (binnable) or the sequence structure
   (don't quantize)?
6. **Yardstick:** is exact-token accuracy meaningful here, or is the target multi-modal
   (distribution + audition)?
7. **Context budget:** net token delta vs the Orin envelope; justify growth against the
   learnability gain.
8. **Provenance invariance:** would hand-written and driver-produced versions of the gesture encode
   identically?
9. **Triage before training:** does `learnability_triage` (seq_len 8192, window mode) rank the
   change ≥ the incumbent on per-frame h_k + induction-copy?
