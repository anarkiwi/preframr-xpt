# Encoding principles — fidelity × context-efficiency × learnability

**Status:** Reference. The single rubric for how to encode the SID register stream as
model tokens. Other encoding designs should be checked against this; when they trade one
axis for another, say which and why. Distilled from the 2026-05 melody-onset arc (de-merge
win + voice / op48 / semitone results).

**Learnability framing.** These three axes are not co-equal: subordinate to
[`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md), **fidelity
is a hard constraint (the floor/gate), context-efficiency is a bounded constraint, and
learnability is the OBJECTIVE** — you maximize learnability *among* fidelity-valid,
budget-feasible encodings. The "priority order when they conflict" below is the
constraint-resolution rule (you can never ship a non-faithful or out-of-budget encoding),
**not** a statement that fidelity/efficiency outrank the goal. When a fidelity-neutral,
budget-neutral choice remains, the learnability axis decides — and is now measurable
training-free (`audit/learnability_triage.py`).

## The three axes

A token encoding is judged on three axes — two constraints then the objective; the order
below is how they resolve **when they conflict**:

1. **Fidelity (the floor).** `decode(encode(x))` must reproduce the SID register writes
   that matter for audio. Byte-exact is the default; the *content tier* is allowed to be
   *deliberately* lossy (cent-binned slope/preset/transpose), but only behind the 12-SID
   WAV audition gate. A change that is not byte-exact and not audition-gated is invalid.
2. **Context efficiency.** Tokens per song, bounded by the deploy envelope (Jetson Orin:
   PROMPT=2048 / MAX=8192, KV ~16 KiB/token). Fewer tokens = more musical context per
   window. Unigram merging is the main lever here.
3. **Learnability.** How well a model can *predict* the next token. This is the axis the
   melody arc showed we were silently sacrificing, and it has structure (below).

The axes conflict. The central finding of the arc: **the Unigram merge that buys context
efficiency (axis 2) destroyed melody learnability (axis 3) at zero fidelity cost (axis 1)** —
the pitch onset went from 0.66 to 0.009 purely from merging. Efficiency that is
fidelity-neutral is *not* learnability-neutral.

## Learnability sub-principles (each earned by a result)

- **P1 — Separability.** Each content decision should be its own low-cardinality token,
  not fused with unrelated content into a compound. *Evidence:* disabling Unigram (`--tkvocab
  0`) lifted op45 V0_HI 0.009→0.658 — merging had welded the pitch onset into ~9489 compound
  tokens bundling pitch + shape. A 2-value atom is learnable; a 9489-way compound is not.
- **P2 — Locality.** The tokens needed to predict a decision should be *near* it — but only
  when the decision is predictable at all. *Evidence (refined):* the de-merge win is partly a
  locality/separability fix. BUT locality has a ceiling: for the V0_LO *magnitude*, a
  cross-song adjacent 2-gram tops out at ~0.30 and the model already hits 0.35 — so forcing
  onset adjacency (pairing / voice-lanes) adds nothing the model isn't already extracting.
  Locality helps only where there is cross-song-predictable structure being separated; it can't
  manufacture predictability for a multi-modal target (→ P5/P6). Cross-voice de-multiplexing remains a
  separate lever (now subsumed into the generator-MDL pipeline,
  [`generator_mdl_representation.md`](../encoding/generator_mdl_representation.md)), *not* for the magnitude.
- **P3 — Don't multiplex the target.** Interleaving independent streams (voices) into the
  next-token position dilutes any one stream's signal. *Evidence:* melody is three voices
  multiplexed by the frame header; the per-voice line is the actual prediction target.
- **P4 — Voice/identity is structural, not content.** Surface a needed structural variable
  explicitly and locally if it's cheap, but it is not itself the lever. *Evidence:* moving
  voice id onto the VOICE reg (local) was content-neutral (V0_HI 0.668 ≈ 0.658) — voice
  attribution wasn't the blocker. ([`voice_encoding_reference.md`](voice_encoding_reference.md))
- **P5 — Alphabet size ≠ learnability.** Shrinking a field's cardinality does not help if the
  *sequence* structure is the hard part. *Evidence:* semitone-quantizing the onset shrank the
  alphabet 5.70→4.04 bits but left the 2-gram predictability ceiling flat (~0.51) — the
  magnitude is genuine pitch-range/leap entropy, not removable cent-jitter. Fix entropy at the
  representational source; don't just bin.
- **P6 — Use the right yardstick.** Where a target is genuinely multi-modal, exact-token
  accuracy structurally undersells the model. *Evidence:* exact pitch magnitude caps at ~0.51
  even for an in-sample memorizing n-gram → a generalizing model emits a *plausible*, not the
  *exact*, next pitch. Score such targets distributionally + by audition.
- **P7 — Provenance invariance (one universal primitive, however it was produced).** The same
  musical gesture must encode to the **same** token regardless of how the source SID stream
  produced it — whether hand-written as raw per-frame register writes, or generated by a driver's
  table/command. Our ornament model is a *universal driver*; the encoder's job is to express any
  tune in it **wherever the gesture is recognizable**, not to passthrough an explicitly-written
  gesture as RESID just because no driver was invoked. *Why:* if a hand-written arp and a
  chord-table arp tokenize differently, the model learns two unrelated things and can leverage
  neither universally; folding both into one `ORN_TYPE_ARP` lets it learn + generate the gesture
  once. *Evidence/instance:* the shared fast-melodic-run RESID gap (Trap 98.8% / Baggis 75.6%,
  and Commando/Camerock equally) is exactly this — a fast melodic line, explicit or wavetable, is
  the same thing and must fold into universal SKEL notes, not RESID. This *strengthens*
  RESID-as-completeness: **RESID = any gesture we failed to recognize and fold into a universal
  primitive, regardless of provenance** — a recognizer gap to close, not a tune to tolerate. The
  acid test is a **provenance-invariance** assertion: an explicit-write stream and a driver-table
  stream of the *same* gesture must produce *identical* ORN/SKEL tokens. (Drives backlog #13/#15.)
- **P8 — Read the control register, not freq alone; the long tail is a recurring engine mechanism (a
  codebook), not a stack of new exact primitives — and not a lossy floor (lossy is a last resort, only
  after tracing every engine).** *Evidence (2026-05-30 RESID-archetype program):* the
  per-frame freq cannot be interpreted in isolation — the control register (gate / **test bit** /
  **waveform**) assigns each frame's role: a **TEST-bit** frame's freq is don't-care (oscillator held)
  — but NOT a classic gate-based hard-restart frame, which is in release where freq IS audible (don't
  conflate the two HR mechanisms); a **noise**
  frame's freq is *timbre not pitch* (a note-onset noise-tik accents a *pitched* lead — Facemorph —
  it is NOT a drum), and a note with no pitched frame is percussion. So you may base the melodic note
  on its *pitched* frames (landed, control-aware `_rebased_note`). **BUT "not melodic pitch" ≠
  "discardable".** Emulator-proven (single source of truth: `preframr-audio` 0.5.5
  `test_freq_write_audibility.py` + `test_register_canonicalization.py`, under the renderer's REAL
  per-write timing — see [`sid_render_fidelity_contract.md`](sid_render_fidelity_contract.md)): a
  **TEST-bit frame's freq** is the one freq write that does not reach the output, but only absorbable to
  a NEARBY value (a wild multi-octave triangle jump leaks through the pre-TEST inter-write window, so
  absorb to the adjacent note's freq, not an arbitrary constant). **PW and waveform bits on a test-bit
  frame ARE audible** (the pulse threshold / held DC level take effect in that window) — NOT discardable.
  A **noise**-frame freq is the noise pitch/colour (fully audible), a freq change during
  **release** is audible (release-0 is NOT instant), and **combined-waveform** freqs are audible — so
  those freqs must still be ENCODED (a percussion/effect channel), never absorbed to 0. Discarding any
  non-test write is provably wrong; *prove* a write is inaudible with the emulator before dropping it.
  Measure melodic RESID=0 on the **pitched** content; the noise/percussion is a separate audible
  channel. AND: once the control-explained and segmentation-explained RESID is removed, the **residue
  is genuinely-noisy content that no EXACT parametric primitive reproduces** — widening SLIDE and a
  uniform-freq SWEEP both reproduce 0/9 of the real wide ramps. Driving that residue toward RESID=0 is
  therefore a **deliberate, audition-gated content-tier fidelity relaxation** (a lossy parametric fit
  is more learnable than raw per-frame RESID), NOT another lossless ORN type. Don't ship a lossy
  primitive on an exact round-trip; gate it on the WAV audition (axis 1's "deliberately lossy,
  audition-gated" clause).

## The checklist (apply to any encoding change)

1. **Fidelity:** byte-exact round-trip? If lossy, is it content-tier and audition-gated?
2. **Separability:** does any single token fuse multiple independent content decisions
   (esp. via Unigram merges crossing content boundaries)? If so, split them
   ([`melody_merge_split.md`](../landed/melody_merge_split.md)).
3. **Locality:** for the decision you care about, how many tokens separate it from the context
   that predicts it? Can that distance be reduced without breaking fidelity?
4. **Multiplexing:** is the prediction target one coherent stream, or several interleaved?
5. **Cardinality vs sequence:** is the difficulty the alphabet size (binnable) or the sequence
   structure (not binnable — don't quantize)?
6. **Yardstick:** is exact-token acc meaningful for this target, or is it multi-modal (use
   distribution + audition)?
7. **Context budget:** net token delta vs the Jetson envelope; if it grows, justify against the
   learnability gain.
8. **Provenance invariance:** would a hand-written version and a driver-produced version of this
   same gesture encode to the *same* tokens? If not, the encoder is recognizing only one
   provenance — close the recognizer gap (P7), don't leave the other as RESID.

## How the existing designs map

- `freq_v0_interval` — P1/P5: makes the onset *sign* a separable ~2-value atom (the bankable
  melody win); the magnitude residual is P6 territory.
- `melody_merge_split` — P1/P2: un-welds cross-boundary Unigram merges.
- `generator_mdl_representation` — supersedes the per-pass pitch/ornament stack: one self-verifying
  generator decomposition over all channels (P1 separability + P7 provenance-invariance by construction).
- `voice_encoding_reference` — P4: voice is a structural variable, now single-sourced.
- Refuted by these principles: op48 single-byte interval (wrong factoring, P1), semitone-quantize
  (P5), voice-feature injection (`voice_trajectory`, P4), write-order normalization
  (`sequence_order_normalization`).
