# preframr tokenization vs other music LLMs — a critical comparison

**Status (2026-05-24):** reference/positioning doc. Critically compares
preframr's register-event + macro Unigram tokenization with the
dominant music-LLM token paradigms — what is genuinely different, and
where it is theoretically better or worse. Companion to
`music_llm_landscape_and_fail_fast_plan.md` (which ranks *ideas to
borrow*); this doc argues the *scheme itself*.

**Learnability framing.** The "ceiling is the tokenization, not the model" conclusion is exactly [`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md)'s thesis (the architecture is already exonerated); compare schemes by copy-fraction / per-frame h_k, not gzip-style compressibility.

## What preframr tokenizes

The token stream is the **SID's register-write program**, not a score
and not audio. Atoms are `(op, reg, subreg, val)` events at PAL-frame
(50 Hz) granularity, with structural markers (`FRAME`, `DELAY`,
`VOICE`) and a hand-designed **macro layer** that collapses recurring
register-write patterns into recognised ops, organised by a loss tier:

- **lossless tiers** (`zero`/`structural`/`mid`/`bit_exact`): SET/DIFF
  primitives, `HARD_RESTART`, `CTRL_BIGRAM`/`CTRL_TRIPLE`, `FREQ_RUN`,
  `FREQ_VIBRATO`, `FREQ_NUDGE`, `RELEASE_UPDATE`, `LEGATO`, `LOOP`/
  `BACK_REF`, `VOICE_BLOCK_ORDER`. Reconstruct the register stream
  **byte-for-byte** → bit-exact SID replay (verified by the per-frame
  register oracle + `compare_renders`).
- **content tier** (deliberately lossy): `slope` (ramp fit), `preset`
  (PW/FC table snap), `transpose` (16-cent freq-delta bin). Trade
  bit-exactness for compression; mostly inaudible (measurable).

Atoms (~5.5K after the FREQ_TRAJ rework; was ~10K) train a **Unigram**
sub-token model (HF `tokenizers`, `tkvocab`). Decode is deterministic and cheap: tokens → registers →
`pyresidfp` → exact 6581/8580 audio. No neural vocoder.

## The paradigm landscape

| Paradigm | Representative | Token = | Reconstruction |
|---|---|---|---|
| **Register/engine-event** (preframr) | this repo | chip control-write event | bit-exact replay (lossless tiers) |
| Symbolic event | Music Transformer, REMI(+), CP-Words, MMM, Anticipatory MT | note-on/off, time-shift, bar/pos (CP groups concurrent attrs) | synth/soundfont-dependent; timbre not encoded |
| Neural audio codec | MusicGen, AudioCraft, MusicLM (EnCodec/SoundStream RVQ) | quantized acoustic frame, K codebooks | lossy neural decode |
| Continuous/hier. VQ | Jukebox, audio diffusion (Suno/YuE-likes) | VQ code or latent | lossy neural decode |

## Head-to-head

| Axis | preframr register-event + macros | symbolic/MIDI | audio codec (RVQ) |
|---|---|---|---|
| What's captured | the *synthesis program* (timbre + notes + FX) | abstract score (notes only) | the *acoustic surface* |
| Timbre fidelity | exact (it is the patch) | lost (timbre-agnostic) | approximate (codec artefacts) |
| Decode | deterministic, exact, ~free | deterministic given a synth | heavy neural, lossy |
| Inductive bias | hardware-faithful, event-sparse, macro = hand-coded musical prior | score structure (bars/voices) | none (raw acoustics) |
| Seq length / song | long, frame-locked (~10K Unigram tokens) | short (note events) | long (50 Hz × K books) |
| Vocab | ~5.5K atoms → Unigram (8192 vocab, 100% used; ~11% long-tail post-FREQ_TRAJ) | small, dense | fixed codebook, fully used |
| Cross-instrument transfer | none (SID-specific) | high (MIDI is universal) | high (any audio) |
| Data scale available | HVSC (~tens of K songs) | internet-scale MIDI | internet-scale audio |
| Augmentation | verified-inaudible perturbation + transfer (audio ground truth exists) | transposition (no audio check) | hard (no symbolic handle) |

## Where it's theoretically better

1. **Fidelity for free.** The stream *is* the executable; the lossless
   tiers replay the original chip output bit-for-bit with a deterministic
   cycle-exact emulator — no vocoder, no soundfont, no codec loss. MIDI
   discards the actual sound; codecs never recover the exact original.
   For a chiptune corpus this is the correct ground truth.
2. **Domain structure without learned codebooks.** The macro layer is a
   hand-designed, (largely) lossless grouping of musically-meaningful
   gestures — vibrato (`FREQ_VIBRATO`), slides (`FREQ_RUN`/`SLOPE`),
   hard-restart, legato — analogous to CP-Words' attribute grouping but
   grounded in the engine rather than a heuristic score grammar.
3. **Explicit loss knob.** A tiered zero→content loss spectrum makes the
   fidelity/compression trade *legible and selectable*, unlike a codec's
   single fixed lossy bottleneck or MIDI's all-or-nothing abstraction.
4. **A measurable representation/content split.** Because exact audio is
   recoverable, near-equivalent token streams can be *proven* inaudible
   (`per_frame_rel_rms`), enabling verified-inaudible augmentation and
   acoustic-equivalence vocab collapse — moves MIDI and codec models
   can't make cleanly (no audio ground truth / no exact reconstruction).

## Where it's theoretically worse (and what bites us)

1. **Sequence length.** Frame-locked register events are far finer than
   note events; ~10K tokens/song strains context and predict-host
   throughput. (Mitigation drafted: `compound_token_tokenizer_design.md`
   — REMI+/CP-Words-style grouping, 3-5× target.)
2. **Engine specificity.** The vocabulary only means anything for the
   SID. No cross-instrument/genre transfer; we cannot borrow the
   internet-scale corpora that make audio/MIDI models work.
3. **Data scale.** HVSC is the ceiling (~tens of K songs). The *original*
   symptom (pre-FREQ_TRAJ): 38% of ~7376 atoms occurred <10×, capacity
   wasted on a long tail behind the ~13% content-acc ceiling.
   **Measured update (2026-05-26, prodlike `full_macros`; from the
   `tokens.csv` count column):** the FREQ_TRAJ rework + 8192 vocab trim
   largely closed this — alphabet ~5491 atoms, **only ~11% long-tail
   (<10×)**, and the Unigram vocab is **100% utilised at tkvocab 8192**
   (the "~91% dead" was an artifact of the oversized 32768 cap). The worst
   per-family long-tail is still ~66% but on far smaller families (≤210
   atoms, vs the old 1926). This tracks `full_macros` being the content win
   (eval_a content 0.160→0.287): the long-tail + dead-vocab levers this
   bullet names are the ones that got pulled, tokenizer-side. Codecs still
   have small, fully-utilised codebooks; MIDI/audio models drown the tail
   in data.
4. **Hand-engineered = brittle.** The macro layer is bespoke and
   bug-prone — the multi-frame-drain off-by-one (0.14.1) silently
   corrupted audio while passing value-sequence tests, caught only by a
   purpose-built per-frame oracle. Learned codec tokens have no such
   hand-design surface (at the cost of opacity).
5. **Tokenization-induced ambiguity.** The content tier creates
   near-equivalent atoms (e.g. many `SET freq_lo` values that snap to
   one preset) the model must disambiguate from sparse data — an
   ambiguity a single codec quantization or canonical MIDI avoids. This
   is the suspected root of the content ceiling; it is *self-inflicted by
   the scheme* and is exactly what acoustic-equivalence normalization
   (`audio_equivalence_normalization_design.md`) targets.

## Net take + implied levers

preframr's scheme wins decisively on **fidelity** and **domain
inductive bias**, and uniquely supports **audio-verified augmentation** —
the right choice for bit-exact chiptune generation, where a codec or
MIDI representation would either lose the sound or the timbre. It pays
in **sequence length**, **engine specificity**, **data scale**, and a
**self-inflicted content-token ambiguity** that caps content accuracy.

Critically, the three levers the project keeps re-deriving fall straight
out of this comparison — they are each "buy back a property other
paradigms get for free":
- **compound tokens** → buy back symbolic-event sequence brevity;
- **acoustic-equivalence normalization** → buy back codec-style
  single-quantization (collapse near-equivalent content atoms);
- **augmentation** (this doc's sibling) → buy back data scale, using the
  one advantage the others lack (exact audio ground truth).

Model-side interventions (per-tier heads, MoS, diffusion, contrastive)
were all refuted at the content ceiling; the comparison predicts that —
the ceiling is a property of the *tokenization*, not the model.

## References

External: Huang+ '18 (Music Transformer); Huang+Yang '20 / Hsiao+ '21
(REMI, CP-Words); Ens+Pasquier '20 (MMM); Thickstun+ '24 (Anticipatory
MT); Défossez+ '22 (EnCodec); Copet+ '23 (MusicGen); Agostinelli+ '23
(MusicLM); Dhariwal+ '20 (Jukebox).

Internal: `music_llm_landscape_and_fail_fast_plan.md`,
`compound_token_tokenizer_design.md`,
`audio_equivalence_normalization_design.md`,
`preframr-aug:design/melody_transfer_augmentation_design.md`,
`preframr-tokens:TOKEN_IMPROVEMENTS.md`.
