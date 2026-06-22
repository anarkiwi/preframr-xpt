# Related work — how preframr sits in the literature

**Status:** Reference (positioning; refreshed 2026-06-22). The project's distinctive angle moved with the
codec: it is no longer "an LM over the raw per-frame register stream + an MDL DEF→REF codebook" (that frame/
event-codec era is refuted — `data/refuted/`). It is now **recovering the generative PROGRAM from a
deterministic playroutine trace, verified byte-exact** — `trace = VM(program)` — and then training a model on
that recovered program. This reframes the nearest literature from *symbolic-music tokenization* toward
**neural/trace-driven decompilation and program synthesis from execution traces**. Cross-ref
[`tokenization_vs_music_llms.md`](tokenization_vs_music_llms.md),
[`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md),
[`../encoding/sid_player_decompiler.md`](../encoding/sid_player_decompiler.md).

## The one-paragraph placement
Each facet is precedented; the **integration is not**. Chiptune-ML precedents capture the chip's register/
command stream but **up-normalize it into note-level scores/MIDI before modeling**. The decompilation line
(below) recovers re-executable programs from binaries/traces and is increasingly held to a *re-executability /
byte-equivalence* gate — exactly preframr's residual-zero gate — but targets general code, not a generative
music program, and is not paired with a downstream generative LM. The symbolic-music-as-program / MDL line
gives the "music as the shortest program that outputs the surface" theory; the learnability-theory line gives
the automaton/causal-state basis for ordering the recovered program for an AR transformer. **Genuinely novel/
unaddressed combination:** *byte-exact recovery of a generative music program from a deterministic hardware
playroutine trace (one bounded-accumulator VM, residual = 0), then an autoregressive transformer trained on
that program under a learnability-theory ordering.* No surveyed work unites trace→program recovery, the
byte-exact gate, the music-generation target, and the learnability lens.

## By facet

### 1. Neural / trace-driven decompilation — the NEW nearest neighbor (the angle the codec moved to)
The codec is, structurally, a domain-specific **decompiler**: lift a deterministic execution trace to a
re-executable program and verify by re-execution. The 2024–2025 decompilation literature converges on the
same gate preframr already enforces.
- **Re-executability / byte-equivalence as the gate.** "Evolving Exact Decompilation" (BED, Schulte et al.)
  pursues **byte-equivalent** decompilation (recompile → identical bytes, falling back to function-granularity
  equivalence). <https://eschulte.github.io/data/bed.pdf> "Context-Guided Decompilation: A Step Towards
  Re-executability" (ICL4Decomp, 2025) scores success only when "the compiled binary executes without runtime
  errors and produces output identical to" the ground truth — a behavioral-equivalence gate.
  <https://arxiv.org/html/2511.01763v1> *Contrast:* both target general C/binary and measure equivalence on
  test inputs; preframr's gate is **stronger** (residual = 0 over the WHOLE 25-register, every-frame trace, by
  construction — no escape hatch) and its recovered "program" is the generative artifact the model is trained
  on, not the end product.
- **Program synthesis from execution traces.** "Program Synthesis from Partial Traces" (Ferreira, Nicolet,
  Dodds, Kroening, 2025) synthesizes source verified to be **behaviorally equivalent** to observed partial
  traces. <https://arxiv.org/pdf/2504.14480> Trace-driven semantic lifting (Tracexp / "dual decompiler",
  2026) reconstructs high-level pseudocode from a **single observed deterministic execution**.
  <https://arxiv.org/pdf/2601.16681> *Contrast:* these recover code from a few traces for understanding/
  security; preframr recovers a *generative* program (a VM of bounded-accumulator ops + a score) from a
  complete deterministic trace and then *generates* new traces from it.
- **LLM-based decompilation at scale.** LLM4Decompile / Decompile-Bench, GenNm, SK2Decompile — neural
  translation of assembly→C with symbolic alignment and functional-validation feedback.
  <https://www.cs.purdue.edu/homes/lintan/publications/gennm-ndss25.pdf>,
  <https://arxiv.org/pdf/2509.22114> *Contrast:* single-shot translation without a hard byte-exact invariant;
  preframr is white-box (reads the driver's RAM state-variable updates from the bus), not a learned lift.

### 2. Chiptune / VGM ML — closest music precedents, but they model a *derived* score
- **NES-MDB / LakhNES** (Donahue et al.) — logs NES APU register writes (VGM), then **emulates the APU to
  derive a downsampled note-level score**; LakhNES trains Transformer-XL on a 631-event note-level (delta-time
  + note on/off) representation. <https://arxiv.org/pdf/1806.04278>,
  <https://archives.ismir.net/ismir2019/paper/000083.pdf>
- **YM2413-MDB** — converts the OPLL FM chip's binary command stream **into MIDI** (669 files).
  <https://arxiv.org/abs/2211.07131>
- *Contrast:* all normalize the hardware stream **up** into notes/MIDI and discard the chip-exact realization.
  preframr recovers the *program that generated* the register writes and regenerates them byte-exact — it
  keeps the chip-faithful generative structure (vibrato/PWM/arp as bounded-accumulator parameters) that a
  note-level score throws away.

### 1+2. Nearest neighbor on INPUT — `desidulate` (same author)
- **desidulate** operates directly on SID register-write logs (VICE dumps) — the same ground truth preframr
  recovers from — segmenting by GATE 0→1 transitions ("SID Sound Fragments"). MIDI/instrument dataframes are
  OUTPUTS only. <https://github.com/anarkiwi/desidulate>
- *Gap vs preframr:* transcription/fingerprinting, **no program recovery, no generative LM, no byte-exact
  re-render**. preframr ≈ "desidulate's input + a white-box decompiler + a transformer."

### 3. Symbolic-music tokenization — the compound/atom + subword debates (2024–2025)
- **OctupleMIDI / MusicBERT** — one 8-attribute compound token/note; ~4× shorter than REMI.
  <https://arxiv.org/abs/2106.05630> **MMT** — compact sextuple compound token.
  <https://salu133445.github.io/mmt/>
- **MuseTok** (2025) — RQ-VAE bar-wise codes for symbolic music (a *learned* codebook over MIDI bars).
  <https://arxiv.org/abs/2510.16273> **MuPT / NotaGen** (2024–2025) — ABC-notation transformers, patch+char
  decoders. <https://arxiv.org/abs/2404.06393>, <https://www.ijcai.org/proceedings/2025/1134.pdf>
- **MidiTok-BPE / "From Words to Music"** — subword merging on symbolic music: length↓, vocab↑.
  <https://arxiv.org/abs/2301.11975>, <https://arxiv.org/pdf/2304.08953>
- *Contrast + the verified caveat:* these tokenize a *score*; preframr tokenizes a recovered *program* on a
  cross-driver absolute A440 grid. preframr's own subword study **refuted vanilla BPE/Unigram on the sparse
  BACC stream** (it welds across field boundaries and collapses induction-copy) — consistent with the music
  literature, where the strong "BPE bigger-vocab improves results AND speed" claim is unverified; only the
  length/vocab tradeoff is established. (`../encoding/bpe_unigram_subword.md`.)

### 4. MDL / grammar / "music as program" — the theory preframr's thesis rhymes with
- **Meredith — COSIATEC / SIATEC / MTP** — analysis as **lossless compression under MDL/Kolmogorov** (best
  analysis = shortest program outputting the surface); decomposes a score into `⟨pattern, transformation-set⟩`
  pairs. <http://www.titanmusic.com/papers/public/MeredithCMA2016.pdf> *Resemblance, not equivalence:*
  geometric point-set patterns over a score, not a bounded-accumulator generator recovered from a hardware
  trace. (preframr's "program" is *the actual playroutine's* op-program, not a discovered analysis.)
- **GTTM** (Lerdahl & Jackendoff) + grammatical-induction segmentation (LZ78/RePair/Sequitur) — the
  grammar-of-music tradition; reusable non-terminals ≈ a backward orderlist.
  <https://arxiv.org/pdf/2405.18742>
- *Note:* the project's earlier framing as an "MDL DEF→REF generator-primitive codebook" is **refuted** as a
  forward-declaration codebook; the shipped form is an inline backward orderlist + one BACC primitive. The MDL
  *spirit* (shortest generating program) survives; the DEF→REF *mechanism* does not.

### 5. Transformer learnability / expressivity — the ordering theory the design invokes
- **Liu et al., "Transformers Learn Shortcuts to Automata"** — a bounded-depth transformer can simulate a
  finite-state automaton; the basis for ranking an encoding by its automaton/causal-state structure (+ the TC⁰
  upper bound). <https://arxiv.org/abs/2210.10749> (Also Merrill & Sabharwal TC⁰, Olsson et al. induction
  heads, Crutchfield computational mechanics — see `learnability_token_ordering_theory.md`.)

### 6. Voice de-mux / track ordering as a lever
- **MMM** — concatenates per-track event sequences rather than time-interleaving, framing track layout as a
  deliberate choice. <https://arxiv.org/pdf/2008.06048> Supports the de-mux premise (the BACC codec de-muxes
  voices by construction — per-voice row streams, not interleaved frames).

### 7. Transposition-invariant pitch
- Interval/relative-pitch encodings are inherently transposition-invariant. <https://arxiv.org/pdf/1806.08236>
  preframr instead uses an **absolute** cross-driver A440 grid + a backward **Transpose** op (REPEAT + Δ): the
  same concert pitch is one token across drivers, and transposition is recovered as a phrase-level op rather
  than baked into every note token (`../encoding/cross_driver_note_unification.md`).

## What appears genuinely novel (cross-facet)
No single surveyed work unites preframr's defining choices:
1. **Byte-exact recovery of a generative program from a deterministic hardware playroutine trace** — residual
   = 0 over the whole 25-register every-frame trace, one bounded-accumulator VM, no escape hatch — a
   *stronger* gate than the decompilation literature's test-input re-executability.
2. The recovered program is a **generative artifact** (steps + pitch-invariant instrument generators + inline
   backward orderlist), not an end-product readability target — and it is the thing the model is trained to
   continue/generate.
3. **Learnability-theory-guided ordering** (causal-state/automaton, induction-head copy) of that program for a
   bounded AR transformer.
The decompilation line (facet 1) is the closest structural cousin and now shares the byte-exact gate, but it
targets general code and stops at recovery; the chiptune line (facet 2) shares the input but discards the
generative program. The intersection — trace→generative-program recovery + byte-exact + music-generation
target + learnability ordering — is unaddressed.

## Caveats
Web-sourced (2024–2026); decompilation findings are recent and fast-moving. The byte-exact-decompilation
overlap (facet 1) is the highest-value follow-up: it is where preframr can both *borrow* (re-executability
metrics, exemplar retrieval) and *claim* (a hard residual-zero gate the LLM-decompilers don't reach).
