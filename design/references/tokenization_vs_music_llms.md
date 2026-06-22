# preframr tokenization vs other music LLMs — a critical comparison

**Status:** Reference/positioning (re-anchored 2026-06-22 to the BACC codec; the v3-event-model and macro-era
comparisons are in git history). Compares the *scheme itself*; literature positioning lives in
[`related_work.md`](related_work.md).

## What preframr tokenizes (BACC)

The token stream is **the recovered generative PROGRAM of the SID playroutine** — not a score, not audio, and
no longer the raw per-frame register trace. The codec decompiles a `.sid` (`recover_from_sid`, white-box) into
per-voice tracker rows + **pitch-invariant instrument generators** (vibrato/slide/arp/PWM/ADSR/sweeps all
expressed as one **bounded-accumulator (BACC)** primitive: `value += rate every dwell frames`, with a boundary
∈ {wrap-N, reflect, none}, width ∈ {8,12}-bit, output map ∈ {absolute, base+offset, note-table-scaled}, or a
table-walk). Notes ride a **canonical 12-TET A440 grid** (the same concert pitch is the same token across
drivers); repeated phrases dedup via an **inline backward orderlist**, and a transposed repeat is a single
backward **TRANSPOSE** op. The alphabet is a **tiny VOCAB=34** (LEB digits + REPEAT/TRANSPOSE markers); a
single learnable vocabulary spans Hubbard / GoatTracker / lft (DMC in progress). See the
[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens).

Fidelity is **residual = 0, byte-exact**: the recovered program renders back to the ground-truth dump
byte-for-byte over all 25 registers, every frame (`verify_residual`). A lossy codec trivially hits any token
budget, so the budgets only mean anything under this gate.

The decisive scale result: the recovered program is **sparse**. Whole tunes fit one small context window —
Monty 1,313 tokens (0.075 tok/frame), 5_Title_Tunes 1,394 (0.680), Grid_Runner 2,817 (0.180), A_Mind_Is_Born
496 (0.061) — all residual-zero, ~10× smaller than the retired frame/event codec. Training context default is
4096 (whole-song-in-context); the stretch goal is ≥ 90% of the corpus under 4096 tokens.

## The paradigm landscape

| Paradigm | Representative | Token = | Reconstruction |
|---|---|---|---|
| **Recovered playroutine program** (preframr) | this repo | a BACC op / score row / orderlist ref | byte-exact chip re-render (residual = 0) |
| Symbolic event | Music Transformer, REMI(+), CP-Words, MMM, MuPT/NotaGen | note-on/off, time-shift, bar/pos | synth-dependent; timbre not encoded |
| Learned symbolic codebook | MuseTok (RQ-VAE over bars) | quantized bar code | lossy decode to MIDI |
| Neural audio codec | MusicGen, MusicLM (EnCodec/SoundStream RVQ) | quantized acoustic frame | lossy neural decode |
| Continuous/hier. VQ | Jukebox, audio-diffusion | VQ code or latent | lossy neural decode |

## Where it's better

1. **Fidelity for free, and a sparse program to model.** The stream *is* the executable program; the render is
   chip-exact under a deterministic emulator (residual = 0). MIDI discards the sound; codecs never recover the
   original. Recovering the *generator* (not the dense trace) is also what makes the stream short enough to fit
   a whole tune in one window — the per-frame modulation that made a score-level scheme look unattainably long
   becomes a few instrument parameters. It also makes **audio-verified augmentation** possible (perturbations
   provably inaudible).
2. **Domain structure without learned codebooks.** Ops are grounded in the actual playroutine (notes as an
   absolute A440 grid index, modulation as bounded-accumulator parameters rendered through the note table) —
   the inductive bias CP-Words/MuseTok approximate with score heuristics or a learned VQ, here
   hardware-faithful and lossless.
3. **A tiny, fully-used alphabet.** VOCAB=34 spans every driver; no long-tail/dead-vocab pathology, no lossy
   codec bottleneck. (Subword merging is *not* the context lever — vanilla BPE/Unigram welds across field
   boundaries on this stream and is refuted; `../encoding/bpe_unigram_subword.md`.)

## Where it's worse (and what bites)

1. **Recovery is the hard part.** The win depends on a white-box decompiler reaching residual = 0 for the
   tune's driver. Per-driver coverage is partial (Hubbard / GoatTracker / lft land; the generic bus-trace path
   covers more; DMC + others open) — a tune whose generator isn't yet recovered is out of scope, by design
   (HARD RULE #0: never fall back to storing the dense trace). This replaces the old "sequence length" pain
   with a *coverage* frontier.
2. **Engine specificity.** The vocabulary means nothing off-SID: no cross-instrument transfer, no borrowing
   internet-scale MIDI/audio corpora. A *phrase* prompt must be compiled into the SID program domain
   ([`../generation/prompt_interface_design.md`](../generation/prompt_interface_design.md)).
3. **Data scale.** HVSC is the ceiling (~tens of K songs, stratified by the SIDId tracker catalog). Corpus size
   bounds cross-composer generalization; augmentation (preframr-aug) is the lever.

The earlier **content-ambiguity** pathology (many near-equivalent tokens for one sound) is gone by
construction: the recovered program is canonical, and the model-side content interventions that stalled at a
~0.13 ceiling were diagnosed as the frame/event codec signal-fitting a dense trace — the BACC codec is the
representation-level fix. Re-baseline the content metric on the BACC stream.

## References

Huang+ '18 (Music Transformer); Huang+Yang '20 / Hsiao+ '21 (REMI, CP-Words); Ens+Pasquier '20 (MMM);
Qu+ '24 (MuPT); Wang+ '25 (NotaGen); the MuseTok RQ-VAE tokenizer '25; Défossez+ '22 (EnCodec); Copet+ '23
(MusicGen); Agostinelli+ '23 (MusicLM); Dhariwal+ '20 (Jukebox). Decompilation/trace-synthesis cousins in
[`related_work.md`](related_work.md).
