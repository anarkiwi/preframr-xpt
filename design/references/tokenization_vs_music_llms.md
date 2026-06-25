# preframr tokenization vs other music LLMs — a critical comparison

**Status:** Reference/positioning (re-anchored 2026-06-25 to the **FLAT v2** codec; the BACC-alphabet,
v3-event-model and macro-era comparisons are in git history). Compares the *scheme itself*; literature
positioning lives in [`related_work.md`](related_work.md), the learnability basis in
[`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md).

## What preframr tokenizes (flat v2)

The token stream is **the recovered generative PROGRAM of the SID playroutine** — not a score, not audio, and
not the raw per-frame register trace. The codec decompiles a `.sid` (white-box) into per-voice tracker rows +
**pitch-invariant instrument generators**, then serializes that program into a **typed-atom, learnability-first
flat alphabet**. The v2 design is a deliberate departure from the v1 BACC alphabet (base-16 LEB digits +
inline-LZ REPEAT/TRANSPOSE markers, VOCAB=34): it trades raw compactness for structure a bounded transformer
can actually learn.

The defining choices:

- **Typed-atom alphabet, VOCAB ≈ 576.** A token's numeric RANGE encodes its KIND: structural `0..63`,
  `NOTE 64..223` (`NOTE_ZERO=144`, grid g → `144+g`), `INSTR_REF 224..287`, `CMD 288..319`, `BYTE 320..575`.
  There is **NO base-16 LEB and NO place-value arithmetic** — every numeric field is a single small,
  quantized, domain-natural atom, not digits to be re-assembled.
- **Repetition = inline content-addressed `REF`, not numeric LZ.** A repeated phrase is replayed by a stable
  pattern id (define-at-first-use, `REF P`); a transposed reuse is a small signed Δ on that `REF` — never a
  per-note transpose and never a numeric back-offset (where it WAS). This is induction-head-friendly and
  prefix-valid: a `REF` always points to a def already on its left.
- **Pitch is three orthogonal things.** (1) **NOTE** = a GLOBAL canonical A440 12-TET grid index — the same
  concert pitch is the same token across every driver and every song, so melody is learnable corpus-wide;
  (2) **TUNING** = one per-tune offset (or a tiny per-degree profile) mapping NOTE → the exact `Fn`; (3)
  **DETUNE** = a bounded (≤ ±50 cents), fat-headed expression parameter — never folded into NOTE, never an
  unbounded range. Most tunes use detune 0.
- **Instruments = `GEN_*` generators** (HOLD / RAMP / QUAD / VIBRATO / ARP / TABLEWALK). PW and filter ARE
  generators (`GEN_RAMP`), not per-frame `BYTE` dumps. Each is O(1) in ticks — a constant param count
  regardless of how long it drives — defined once and referenced by notes.
- **Time = the row/pattern grid, implicit.** No `nframes`, no wide 16-bit duration field, no escapes: a note
  holds until the next note-on/keyoff; a pattern is `PAT_LEN` (≈16/32/64) rows; absolute length is the
  orderlist's sequence of small `REF`s; tempo is one small clustered divider atom. Initial state defaults to
  ZERO (no boot dump); a non-zero window entry is a sparse `SEED`.
- **An OPTIMIZE pass** (lossless, SID-output-preserving) canonicalizes the recovered program to **minimum
  causal-state** (sustain-lift, generator absorption, phrase `REF` factoring) before serialization, so the
  model never sees two spellings of one thing.

A single learnable vocabulary spans the whole tracker zoo: GoatTracker, DMC, JCH, FC, Soundmonitor, Hubbard,
Galway, lft, and code-tunes with no pattern bank (A Mind Is Born). See the
[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens) and the v2 spec
(`preframr-tokens/.../FLAT_VOCAB_MIGRATION.md`, §2a–2f, the C1–C8 gate).

Fidelity is **SID-output equivalence**: the notes+instruments render reproduces the ground-truth dump on all
25 registers, every tick, modulo each backend's declared don't-care mask (`verify_residual`, residual-0,
integer-deterministic — never rendered audio). The gate is **structural** (the C1–C8 constraints below), not a
scalar budget: the v1 `< 1 tok/frame` target was a Goodhart metric the frame/event codec hit *pathologically*
as a per-frame dump, so `tok/tick` is now **REPORTED, not gated** (an alert at `> ~1` flags an unrecovered
generator per HARD RULE #0). Digis are EXCLUDED entirely (`DigiExcluded`), never stored.

The C1–C8 structural gate (each closes one degenerate solution): C1 per-LANE efficiency (not aggregate);
C2 generators O(1) in ticks + bounded exceptions; C3 no-LZ, repetition only content-addressed; C4 the tick
denominator derived from the trace, not chosen; C5 render from TOKENS ALONE (no access to the dump); C6 note
falsification (onsets ≪ ticks, few distinct pitch-classes); C7 `BYTE`-atom fraction cap; C8 no wide values /
no escapes (per-field fat-head check).

**Context target = 8192 tokens.** The stream is longer than v1 (compression deliberately traded for
learnability); tunes that overflow stay usable because the INLINE layout is prefix-valid — any prefix/window
is a decodable, continuable song — not via a hard length cap.

## The paradigm landscape

| Paradigm | Representative | Token = | Reconstruction |
|---|---|---|---|
| **Recovered playroutine program** (preframr v2) | this repo | a typed atom — NOTE grid index / `GEN_*` param / row / `REF` | byte-exact chip re-render (residual = 0) |
| Symbolic event | Music Transformer, REMI(+), CP-Words, MMM, MuPT/NotaGen | note-on/off, time-shift, bar/pos | synth-dependent; timbre not encoded |
| Learned symbolic codebook | MuseTok (RQ-VAE over bars) | quantized bar code | lossy decode to MIDI |
| Neural audio codec | MusicGen, MusicLM (EnCodec/SoundStream RVQ) | quantized acoustic frame | lossy neural decode |
| Continuous/hier. VQ | Jukebox, audio-diffusion | VQ code or latent | lossy neural decode |

## Where it's better

1. **Fidelity for free, and a learnable program to model.** The stream *is* the executable program; the render
   is chip-exact under a deterministic emulator (residual = 0). MIDI discards the sound; neural codecs never
   recover the original. Recovering the *generator* (not the dense trace) is what shrinks per-frame modulation
   into a few O(1) instrument params, and the **typed-atom + global-NOTE + content-addressed-`REF`** form is
   what a bounded transformer can cheaply represent (TC⁰ / induction-head terms — see
   [`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md)). It also makes
   **audio-verified augmentation** possible (perturbations provably inaudible).
2. **Domain structure without learned codebooks.** Atoms are grounded in the actual playroutine (notes as an
   absolute A440 grid index, modulation as `GEN_*` parameters rendered through the note table) — the inductive
   bias CP-Words/MuseTok approximate with score heuristics or a learned VQ is here hardware-faithful and
   lossless. Pitch-proximity is token-proximity by construction (one NOTE token per semitone on the grid).
3. **A typed, self-delimiting, prefix-valid alphabet — no unlearnable arithmetic.** A token's range names its
   kind; repetition is content-addressed (induction-copy), not a numeric offset; grammar framing is
   `BEGIN/END` (no length prefixes / no counting/Dyck obligation). No long-tail/dead-vocab pathology (C8
   fat-head check), no wide values, no lossy bottleneck. (Subword merging is *not* the context lever — vanilla
   BPE/Unigram welds across field boundaries on this stream and is refuted; `../encoding/bpe_unigram_subword.md`.)

## Where it's worse (and what bites)

1. **Recovery is the hard part.** The win depends on a white-box decompiler reaching SID-output equivalence for
   the tune's driver. The 645-tracker RE survey collapses HVSC to a handful of models (top-6 drivers = 59%,
   top-30 = 82%); the existing recovery is residual-0 on the 6 top drivers + 6 non-tracker tunes, but the
   **generic FLAT serializer + render-from-tokens path is in-flight, not yet merged** (the GoatTracker flat
   path is merged, PR #150, residual-0). A tune whose generator isn't yet recovered is out of scope, by design
   (HARD RULE #0: never fall back to storing the dense trace) — a *coverage* frontier, not a length one.
2. **Sequence length grew.** Flat v2 is longer than v1 (Grid_Runner flat = 9,480 tokens, over the 8192 target)
   — the explicit trade for learnability. `REF` + `GEN_*` + the OPTIMIZE pass mitigate it, and prefix-validity
   keeps overflow tunes usable, but it is a genuine cost, not a free win.
3. **Engine specificity + data scale.** The vocabulary means nothing off-SID: no cross-instrument transfer, no
   borrowing internet-scale MIDI/audio corpora; a *phrase* prompt must be compiled into the SID program domain
   ([`../generation/prompt_interface_design.md`](../generation/prompt_interface_design.md)). HVSC is the
   ceiling (~tens of K songs); augmentation (preframr-aug) is the lever.

The earlier **content-ambiguity** pathology (many near-equivalent tokens for one sound) is gone by
construction: the recovered program is canonical (the OPTIMIZE pass enforces one spelling), and the model-side
content interventions that stalled at a ~0.13 ceiling were diagnosed as the frame/event codec signal-fitting a
dense trace — the recovered-program codec is the representation-level fix. Re-baseline the content metric on the
flat v2 stream.

## Concerns with the flat codec (raised, and status)

Honest accounting of the objections to the flat design. RESOLVED = closed by a structural choice; OPEN = a
live risk or a trade we accept.

**RESOLVED by design**

- **(a) Place-value LEB arithmetic was unlearnable** → typed atoms: a token's RANGE encodes its kind, no
  digits to reassemble. RESOLVED — there is no place-value field in the alphabet.
- **(b) Numeric LZ back-offsets were unlearnable** → content-addressed `REF` (replay-by-id, define-at-first-use).
  RESOLVED — C3: no offset/REPEAT token exists in the vocab; repetition is what-it-IS, not where-it-WAS.
- **(c) Counting / Dyck-grammar obligation (length prefixes)** → self-delimiting `BEGIN/END`, decode-to-marker.
  RESOLVED — no count prefixes; BYTE data is in a disjoint id range so END markers are unambiguous inside it.
- **(d) Token-proximity ≠ pitch-proximity** → one global NOTE token per semitone on the A440 grid. RESOLVED —
  adjacency in token space is adjacency in pitch (and identical across drivers).
- **(e) Wide-value long tail (16-bit ticks / freq / deltas)** → the no-wide-value policy (C8) +
  global-NOTE/TUNING/bounded-DETUNE + implicit row-grid time. RESOLVED-by-construction and corpus-validated
  (see below): 0 forced wide values across 645 trackers.
- **(f) Goodhart metric-gaming (`tok/frame` hit pathologically as a per-frame dump)** → the C1–C8 structural
  gate; the scalar is demoted to a reported metric (alert at `> ~1`). RESOLVED — efficiency is now a
  *consequence* of recovered structure (per-lane C1, lossless, from-tokens C5, O(1) generators C2), not a
  target to optimize toward.
- **(g) Note-alphabet fragmentation by tuning** → NOTE is global; TUNING is one per-tune value and DETUNE a
  separate bounded param. RESOLVED — tuning/detune never multiply the note alphabet per song.
- **(h) Prefix-validity for 8192 windows / overflow** → inline define-at-first-use + `REF` (def always to the
  left). RESOLVED — any prefix is a valid, decodable, continuable song; no hard length cap needed.

**OPEN / watch**

- **(i) Sequence-length GROWTH.** Flat v2 is longer than v1 — compression traded for learnability. OPEN —
  mitigated by `REF` + generators + OPTIMIZE, but some tunes still overflow 8192 (Grid_Runner = 9,480) and
  lean on prefix-validity rather than fitting whole-song-in-context.
- **(j) Low-redundancy brittleness.** A near-minimal stream has higher per-token entropy / less graceful
  degradation than natural language. OPEN-because the floor is the program — but it stays a *structured
  grammar*, not an entropy-coded blob, so the model still has typed/positional structure to lean on.
- **(k) Render-from-tokens is research-grade.** The schedule-replay (`render_*_from_ids`, C5) for the generic
  path is the core unfinished work and **the generic flat path is not yet merged**. OPEN — GoatTracker flat
  renders from tokens alone (merged); generic does not yet.
- **(l) Prototype-factoring limits in the validation.** The ~26 EDGE tunes (block-dedup ≥ 1 but
  content-addressed design < 1) need the full content-addressed phrase/transpose `REF` — block-dedup alone is
  insufficient; and the 4 flagged DESIGN-MISS are ALL recoverable generators, none a vocab hole — 3
  (`JammicroV1`, `OxyMod_THCM`, `Steve_Turner`) are per-frame `GEN_ARP/VIBRATO/RAMP` the prototype detector was
  too narrow to catch, and the 4th (`Unidentified` = `Jammer/GubbLIITCH.sid`) was RE'd and is NOT aperiodic: a
  35-byte `play=0` self-IRQ **register-spray sizecoder** — one wrapping-sawtooth accumulator byte broadcast to
  every register (256 distinct values, ~100% periodic), trivially a ~10-token generator (≈0.05 tok/frame). Its
  "0% self-similarity" was a double measurement bug (no 50 Hz play cadence → wrong denominator; raw integrated
  freq counted as 2829 "notes" instead of one sawtooth). Reclassify it EDGE (byte-exact generator special-case),
  and re-pick the `Unidentified` bucket representative (a `play=0` sizecoder is a bad stand-in for "SIDId
  couldn't fingerprint"). OPEN-because these are recovery/detector/classification gaps (HARD RULE #0: unrecovered
  generators), NOT vocab holes — the alphabet covers them.
- **(m) `REF` / `TRANSPOSE` must be learned as id + Δ.** This is induction over the def, and
  counting-consistency in free generation remains the residual risk. OPEN-because the self-delimiting grammar
  mitigates but does not *eliminate* the model emitting a dangling/miscounted ref; it is the residual
  free-generation hazard the design narrows, not removes.

**Corpus validation (645 trackers / 60,572 tunes, `/scratch/tmp/design_validation_645.csv`; 645/645 ran,
0 timeout/error):** **514 PASS, 89 DIGI (excluded), 39 EDGE, 3 DESIGN-MISS** (after RE'ing the 4th flagged miss,
`GubbLIITCH`, which is a byte-exact sawtooth-generator sizecoder → EDGE, not a miss; the remaining 3 are
recoverable per-frame `GEN_*`). So **zero genuine vocab holes** — every flagged tune is a recoverable generator. By non-digi HVSC-tune weight, PASS+EDGE cover **57,660 /
58,996 = 97.7%**. design tok/tick over PASS+EDGE: median **0.102**, p90 0.280, p99 0.628, max 0.883 — every
non-miss under 1, 96% under 0.5. TUNING clusters to a small per-tune constant (193 tunes @0c, 232 @+30/+35c,
43 @−25c). The two load-bearing invariants hold with **zero violations: 0 forced wide values (C8) and 0
out-of-bound detune (≤ ±50c)** — the flat design holds; the remaining misses are recovery/detector work, not
alphabet defects. (Two HARD RULE #0 falsifications during the sweep: 28 apparent "wide-value" misses were
sub-audible parked voices → `NOTE_REST` (`REST_FN=256`); 29 apparent "tok/tick≥1" were 72–97% phrase recurrence
the block-aligned dedup couldn't collapse but content-addressed `REF` does → 0.18–0.85.)

## References

Huang+ '18 (Music Transformer); Huang+Yang '20 / Hsiao+ '21 (REMI, CP-Words); Ens+Pasquier '20 (MMM);
Qu+ '24 (MuPT); Wang+ '25 (NotaGen); the MuseTok RQ-VAE tokenizer '25; Défossez+ '22 (EnCodec); Copet+ '23
(MusicGen); Agostinelli+ '23 (MusicLM); Dhariwal+ '20 (Jukebox). Learnability basis (TC⁰ / induction heads /
causal-state): Liu+ '23, Merrill & Sabharwal '23, Olsson+ '22, Crutchfield — see
[`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md). Decompilation/trace-synthesis
cousins and the full literature placement in [`related_work.md`](related_work.md).
</content>
</invoke>
