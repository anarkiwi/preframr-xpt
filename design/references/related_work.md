# Related work — how preframr sits in the literature

**Status:** Reference (positioning). A deep-research survey (2026-06-06, fan-out + adversarial verification,
12 verified findings) of research resembling preframr across its eight facets. **Headline: close cousins exist
in every facet, but no single work matches preframr's core choice — generatively modeling the RAW per-frame
SID register-write stream — and no work unites its defining combination.** Use this to position the project,
cite prior art, and avoid reinventing. Cross-ref [`tokenization_vs_music_llms.md`](tokenization_vs_music_llms.md),
[`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md),
[`generator_mdl_representation.md`](../encoding/generator_mdl_representation.md).

## The one-paragraph placement
Each facet is well-precedented; the **integration is not**. The closest chiptune-ML precedents capture the
chip's register/command stream but **up-normalize it into note-level scores/MIDI before modeling**; the one
body of work that analyzes the SID register log *directly* is the same author's `desidulate`, but for
transcription/fingerprinting, not a generative LM. The MDL/grammar line gives the DEF→REF dictionary + "music
as program" theory; the learnability-theory line gives the automaton/causal-state basis for ranking encodings.
**Genuinely novel/unaddressed combination:** generative transformer over the raw hardware register stream +
MDL/lossless DEF→REF generator-primitive codebook + learnability-theory-guided ordering.

## By facet

### 1. Chiptune / VGM ML — closest precedents, but they model a *derived* score
- **NES-MDB / LakhNES** (Donahue et al.) — logs NES APU register writes (VGM), then **emulates the APU to
  derive a downsampled note-level score**; LakhNES trains Transformer-XL on a 631-event note-level
  (delta-time + note on/off) representation. <https://arxiv.org/pdf/1806.04278>,
  <https://archives.ismir.net/ismir2019/paper/000083.pdf>
- **YM2413-MDB** — converts the OPLL FM chip's binary command stream **into MIDI** (669 files).
  <https://arxiv.org/abs/2211.07131>
- *Contrast:* all three normalize the hardware stream **up** into notes/MIDI; preframr tokenizes the per-frame
  register stream itself. This is the sharpest statement of what's different.

### 1+8. Nearest neighbor on REPRESENTATION — `desidulate` (same author)
- **desidulate** operates directly on SID register-write logs (VICE `-sounddev` dumps) — the same input
  preframr tokenizes — segmenting by GATE 0→1 transitions ("SID Sound Fragments"), paralleling preframr's
  voice/frame factorization; MIDI/instrument dataframes are OUTPUTS only. <https://github.com/anarkiwi/desidulate>
- *Gap vs preframr:* GATE-transition (not strict 50/60 Hz frame) segmentation; transcription/fingerprinting,
  **no generative LM**. preframr ≈ "desidulate's representation + a transformer + an MDL codebook."

### 2. Symbolic-music tokenization — the compound-vs-atom + BPE debates
- **OctupleMIDI / MusicBERT** — one 8-attribute compound token/note; ~4× shorter than REMI, ~2× shorter than
  CP (~3.6k vs ~6.9k vs ~15.7k tokens/song). <https://arxiv.org/abs/2106.05630>
- **MMT (Multitrack Music Transformer)** — compact sextuple compound token. <https://salu133445.github.io/mmt/>
- **MidiTok-BPE** — BPE on symbolic music: length↓, vocab↑ (the inverse tradeoff). <https://arxiv.org/abs/2301.11975>
- *Adversarial caveat (verified):* the strong claim "BPE bigger-vocab improves results AND speed" was
  **REFUTED (0-3)** — so the literature does **not** contradict preframr's hypothesis that BPE/Unigram merging
  can hurt learnability; only the length/vocab tradeoff itself is established.

### 3. Low-level synth/chip control-stream modeling — THE thin facet (a real gap)
- Closest survey: **DDSP review** (Hayes et al.) — differentiable DSP control streams for additive/FM/
  subtractive/physical synths, but **explicitly excludes chiptune, retro audio chips, hardware register
  streams, and discrete/symbolic tokenization.**
  <https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2023.1284100/full>
- *Coverage gap:* no surveyed work models a discrete hardware audio-chip register/parameter-automation stream
  as a tokenized sequence for an LM.

### 4. MDL / grammar / DEF→REF dictionary — the "music as program" theory preframr builds on
- **Meredith — COSIATEC / SIATEC / MTP** — analysis as **lossless compression under MDL/Kolmogorov** (best
  analysis = shortest program outputting the surface); decomposes a score into `⟨pattern, transformation-set⟩`
  pairs = a DEF→REF dictionary. *Resemblance, not equivalence:* geometric point-set discovery, not a
  HOLD/ACCUM/SWEEP/TABLE generator vocabulary. <http://www.titanmusic.com/papers/public/MeredithCMA2016.pdf>,
  <https://arxiv.org/pdf/2201.11085>
- **GTTM** (Lerdahl & Jackendoff) + computational impls (ATTA/FATTA/sGTTM) — the grammar-of-music tradition.
  <https://link.springer.com/chapter/10.1007/978-3-319-25931-4_9>
- **Grammatical-induction segmentation** — infers a CFG from a music token sequence (reusable non-terminals =
  DEF→REF rules), benchmarking LZ78/RePair/LONGEST-FIRST/MOST-COMPRESSIVE/Sequitur (LONGEST FIRST best F1).
  <https://arxiv.org/pdf/2405.18742>

### 5. Transformer learnability / expressivity — the theory the design invokes
- **Liu et al., "Transformers Learn Shortcuts to Automata"** — an o(T)-layer transformer can replicate any
  finite-state automaton on length-T input; O(log T)-depth solutions always exist; O(1)-depth shortcut
  simulators are common (Krohn-Rhodes / circuit complexity). The basis for ranking an encoding by its
  automaton/causal-state structure (+ the TC⁰ upper bound). <https://arxiv.org/abs/2210.10749>
- (preframr also cites Merrill & Sabharwal TC⁰, Olsson et al. induction heads, Hahn, Crutchfield computational
  mechanics, Bialek/Tishby predictive information — see `learnability_token_ordering_theory.md`.)

### 6. Voice de-mux / track ordering as a design lever
- **MMM (Multi-Track Music Machine)** — concatenates a per-track event sequence rather than time-interleaving
  tracks (the Music-Transformer default), explicitly framing track ordering/layout as a deliberate choice —
  supporting preframr's facet-6 premise (de-mux + accompaniment-before-melody ordering affects AR
  learnability). <https://arxiv.org/pdf/2008.06048>

### 7. Relative / interval pitch for transposition-invariant generalization
- Interval/relative-pitch encodings are inherently transposition-invariant (transposing leaves the
  representation unchanged) — supporting preframr's interval/scale-degree melody encoding.
  <https://arxiv.org/pdf/1806.08236>

## What appears genuinely novel (verified cross-facet)
No single surveyed work unites preframr's defining triple:
1. a **generative transformer LM over the RAW per-frame hardware register-write stream** (not a derived
   note/MIDI score),
2. an **MDL/lossless DEF→REF codebook of generator primitives** (HOLD/ACCUM/SWEEP/TABLE-style),
3. **learnability-theory-guided (causal-state/automaton, induction-head) encoding + ordering** choices.
Individual facets are each well-precedented; the **integration is the contribution.** The thinnest external
coverage (facet 3, hardware-chip control-stream tokenization for an LM) is also where preframr is most exposed
to "no prior art to lean on."

## Caveats
Deep-research output (web sources, adversarially verified 2/3-vote); treat confidences as the survey reported
(most findings high-confidence; facets 3 and the cross-facet novelty medium). Not exhaustive — a targeted
follow-up on facet 3 (audio-chip / register-automation ML) and on any 2024–2025 chiptune-LM work would
tighten it.
