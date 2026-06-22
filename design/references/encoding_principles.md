# Encoding principles — fidelity × context-efficiency × learnability

**Status:** Reference (evidence re-anchored to the BACC step/tracker codec; the P1–P8 results were
earned on retired substrates — the principles stand, the cited ops are historical and live in
this file's git history). The single rubric for SID stream encoding; designs that trade one axis for
another must say which and why.

**Learnability framing.** The axes are not co-equal
([`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md)): **fidelity is a
hard constraint (the gate), context-efficiency is a bounded constraint, learnability is the
OBJECTIVE** — maximize learnability among fidelity-valid, budget-feasible encodings. The
learnability axis is measurable training-free (`audit/learnability_triage.py`).

## The three axes

1. **Fidelity (the floor) — BACC residual-zero.** The recovered program must render byte-exact to
   the ground-truth dump over all 25 registers: `residual = 0` by construction. `recover_from_sid`
   recovers the program from the bus trace; any SID write not explained by score+generator is a
   non-zero residual to trace, not an escape lane
   ([`sid_render_fidelity_contract.md`](sid_render_fidelity_contract.md)). **No lossy tier, no escape
   path, no WAV-audition exception** — residual-zero is the gate, and a new placement liberty requires
   a new chip measurement (write-count-matched, per-write-clocked).
2. **Context efficiency.** Tokens per song, bounded by the deploy envelope; the target is the whole
   song in context (training context default 4096; stretch goal ≥90% corpus under 4096). **BPE is
   NOT the context lever (refuted)** — the lever is generator recovery / whole-song-in-context, not a
   vocab dial.
3. **Learnability.** How well a bounded model can *predict* the next token; has structure (below).

The axes conflict, and efficiency that is fidelity-neutral is **not** learnability-neutral — the
defining (historical, BPE-refuted) result: a Unigram merge that bought context at zero fidelity cost
destroyed the pitch-onset signal (0.66 → 0.009) by welding it into thousands of compounds.

## Learnability sub-principles (each earned by a measured result)

- **P1 — Separability.** Each content decision is its own low-cardinality token, never fused with
  unrelated content. *Earned by:* de-merging lifted pitch-onset 0.009→0.658. *BACC embodiment:* the
  recovered program emits each decision (note, instrument, BACC param) as its own token in the VOCAB=34
  alphabet; no fused compounds (BPE refuted as the lever).
- **P2 — Locality.** Predictive context should be near the decision — but locality only helps where
  cross-song-predictable structure is being separated; it cannot manufacture predictability for a
  multi-modal target (→ P6). Cross-voice de-multiplexing is the open locality lever
  ([`lane_demux_hypothesis.md`](../landed/lane_demux_hypothesis.md)).
- **P3 — Don't multiplex the target.** Interleaving independent streams (voices) into the
  next-token position dilutes each stream's signal; the per-voice line is the real prediction
  target. (Still structurally true in v3's frame groups — same hypothesis doc.)
- **P4 — Voice/identity is structural, not content.** Surface a structural variable explicitly and
  locally if cheap, but it is not itself the lever (*earned by:* localizing voice id was
  content-neutral). *BACC:* voices are de-muxed by construction (per-voice tracker row streams).
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
  model learns two unrelated things and can leverage neither. *BACC embodiment:* by construction —
  recovery runs from the rendered register writes (the bus trace), so provenance (which driver, or
  hand-written code) collapses at the input; every gesture expresses as the one BACC primitive. The
  acid test stands: explicit-write and driver-table versions of one gesture must yield identical programs.
- **P8 — Interpret freq through ctrl, and prove inaudibility before dropping anything.** The control
  register assigns each frame's role: TEST-bit frames hold the oscillator (freq there is the one
  near-inaudible write — absorbable only to a *nearby* value); noise-frame freq is timbre, not pitch
  (and noise can accent a *pitched* note — never classify pitch by waveform); release-phase and
  combined-waveform freqs are audible. **"Not melodic pitch" ≠ "discardable"** — every claimed
  inaudibility must be emulator-proven (preframr-audio pinning tests), which is exactly how the BACC
  canonical liberties were licensed. And the long tail of hard engines is recurring mechanism to
  recognize, not noise to go lossy on — lossy is a last resort after tracing every engine.

## The checklist (apply to any encoding change)

1. **Fidelity:** does the recovered program render `residual = 0` (byte-exact vs the dump over all 25
   regs)? New canonicalization liberty ⇒ new reSID measurement, else invalid (no WAV-audition exception).
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
9. **Triage before training:** does `learnability_triage` (whole-song-in-context 4096) rank the
   change ≥ the incumbent on per-frame h_k + induction-copy?
