# preframr tokenization vs other music LLMs — a critical comparison

**Status:** Reference/positioning (re-anchored 2026-06-12 to the v3 event model; the original
comparison of the retired macro substrate is in git history). Compares the *scheme itself*;
literature positioning lives in [`related_work.md`](related_work.md).

## What preframr tokenizes (v3)

The token stream is the **SID's register-write program** — not a score, not audio. A fixed
**127-atom event alphabet** (varint value digits, register/voice tags, note-index/freq-residual/PW
step-and-ramp kinds, NOTE_ON with folded envelope lifecycle, ramp shapes, typed value nibbles,
KEYFRAME segment brackets — see the
[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens)), with **BPE over the atoms as
the only dictionary** (the vocab dial; ~98% live at tkvocab 2048). No DEF/REF ids, no literals, no
escape path. Fidelity is the v3 canonical contract: `decode(encode(ow)) == canonical_writes(ow)`,
self-verified on every encode; decode → registers → cycle-exact reSID audio, no vocoder. Measured
collapse vs the 16-bit raw floor: 7.8× (order-0) / 23× (order-1). Scope: single-speed non-digi
(~92% of corpus).

## The paradigm landscape

| Paradigm | Representative | Token = | Reconstruction |
|---|---|---|---|
| **Register/engine-event** (preframr) | this repo | chip control-write event | canonical-exact replay |
| Symbolic event | Music Transformer, REMI(+), CP-Words, MMM, Anticipatory MT | note-on/off, time-shift, bar/pos | synth-dependent; timbre not encoded |
| Neural audio codec | MusicGen, MusicLM (EnCodec/SoundStream RVQ) | quantized acoustic frame, K codebooks | lossy neural decode |
| Continuous/hier. VQ | Jukebox, audio-diffusion likes | VQ code or latent | lossy neural decode |

## Where it's better

1. **Fidelity for free.** The stream *is* the executable; replay is chip-exact under a deterministic
   emulator. MIDI discards the sound; codecs never recover the original. For a chiptune corpus this
   is the correct ground truth — and it makes **audio-verified augmentation** possible (perturbations
   provably inaudible), a move MIDI/codec models can't make cleanly.
2. **Domain structure without learned codebooks.** Events are grounded in the engine (notes as
   intervals over a recovered per-voice table, ramps as shapes, envelope lifecycle on NOTE_ON) —
   the inductive bias CP-Words approximates with score heuristics, here hardware-faithful.
3. **A tiny, fully-used alphabet + a legible dictionary dial.** 127 atoms with BPE on top makes the
   capacity/coverage trade explicit and sweepable, vs a codec's fixed lossy bottleneck.

## Where it's worse (and what bites)

1. **Sequence length.** Frame-locked events are far finer than note events: tunes average ~30k
   tokens and **82% exceed the seq_len-8192 window** — the model trains on KEYFRAME-led windows,
   never whole tunes. This is the binding constraint behind
   [`../generation/long_range_structure.md`](../generation/long_range_structure.md).
2. **Engine specificity.** The vocabulary means nothing off-SID: no cross-instrument transfer, no
   borrowing internet-scale MIDI/audio corpora. (It also means a *phrase* prompt must be compiled
   into SID events — [`../generation/prompt_interface_design.md`](../generation/prompt_interface_design.md).)
3. **Data scale.** HVSC is the ceiling (~tens of K songs). The old long-tail/dead-vocab pathologies
   are resolved by the fixed alphabet (~98% live vocab), but corpus size still bounds cross-composer
   generalization; augmentation (preframr-aug) is the lever.

Resolved since the original comparison: the **self-inflicted content ambiguity** (many
near-equivalent atoms for one sound) — the v3 canonical contract collapses chip-equivalent writes at
encode time, and the content ceiling moved 0.13 → 0.479 (atoms-only eval_a) with it. The model-side
refutations predicted this: the ceiling was a property of the tokenization, not the model.

## References

Huang+ '18 (Music Transformer); Huang+Yang '20 / Hsiao+ '21 (REMI, CP-Words); Ens+Pasquier '20
(MMM); Thickstun+ '24 (Anticipatory MT); Défossez+ '22 (EnCodec); Copet+ '23 (MusicGen);
Agostinelli+ '23 (MusicLM); Dhariwal+ '20 (Jukebox).
