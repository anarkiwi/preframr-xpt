# Percussion as define-once stamps + voice-agnostic backrefs (proposal)

**Status:** Proposal (iter-2 of the RESID=0 program; see `resid_archetype_program.md`). Not built.
Supersedes the "control-aware noise-frame / percussion-timbre channel" framing — waveform is the
wrong axis.

## The idea (user, 2026-05-30)

A drum is **not a waveform** and **not a per-note offset trajectory**. It is *the same exact series of
register changes, played on a voice, repeatedly* — a low-entropy temporal pattern. The voice it plays
on may change through the tune, but the **sequence is identical**. So encode it like a dictionary:

- **Define the sequence INLINE as a self-contained packed block** (a "stamp") at the point of use —
  all its writes packed contiguously, so the model learns the whole drum as one local unit (token
  locality), not a pattern smeared across the interleaved per-frame timeline. Definitions are
  **redefinable**: a drum's tuning/implementation drifts through a tune, so a changed write-series is
  re-defined inline (a streaming dictionary, not a static preamble).
- **The stamp is the drum's FULL register footprint** — freq + ctrl + PW + ADSR, plus any filter
  writes that are *drum-scoped* (attributed by recurrence, since the SID filter is global).
- **Every occurrence after is a generic BACKREF pointer** — "play stamp K on voice V (here)". The
  stamp is voice-agnostic; the voice rides on the pointer, so the same drum replays on whichever voice.
- **The pointer carries a generic "drum character" attribute** (kick / hat / snare / tom / …) from a
  GLOBAL vocabulary. The model learns *what kind of drum goes here* abstractly → that knowledge
  **transfers across tunes**, while the exact bytes stay bound to the in-context definition (lossless).
- **The definition is laid out as regular small atoms so the downstream Unigram tokenizer
  sub-tokenizes it** — similar drums share sub-tokens and cluster in token space.

This layers the codebook-scope question cleanly: **per-tune exact definitions (lossless) + a global
character vocabulary (transferable).**

## Measured basis (rung-0 fixtures: Baggis/Camerock = JCH, Gridtrap = Crowther)

Probe `audit/probes/resid_percussion.py` (token stream as a musical scope; signatures abs/rel/shape/
ctrl; recurrence + onset-grid + cross-voice):
- **85–90 % of the 503 remaining RESID notes are recurring stamps** (≥3 identical occurrences);
  **~85 % sit on a rhythmic GRID** (inter-onset intervals are clean multiples of a base pulse:
  80/160/320 f).
- Stamps are clearly typed drums, incl. **pitched ones a noise rule misses**: `(2973,65),(1986,65),
  (1251,65),(1115,65)…` (ctrl `0x41` pulse+gate) = a kick/tom down-sweep ×4 gridded; `(8913,129),
  (37745,128)…` (ctrl `0x80` noise) = a hat every 160 f; pulse⊕noise mixes = snares.
- The stamp **IS the exact write-series** → replay is **byte-exact (lossless)**, emulator-safe by
  construction. No timbre-channel approximation, no fidelity tradeoff.
- Coverage is **85 % of NOTES but only 42 % of FRAMES** — drums are short/many; the frame-mass
  remainder is the long held-gate giants (a separate segmentation mechanism, iter 5).
- **Caveat:** on these 3 tunes stamps are single-voice (multi-voice 0–1). Voice-agnostic backref is
  forward-looking (abs signature is value-voice-agnostic, so it's free + correct); confirm cross-voice
  reuse on expansion.

## Token design

Two new token families on top of the existing `op/subreg/val` row model, alongside SKEL/ORN.

### 1. Stamp definition — INLINE and REDEFINABLE (not a preamble)

Definitions are emitted **inline in the token stream at the point of (re)use, NOT hoisted to a kit
header.** A drum's implementation drifts through a tune — the composer nudges its tuning or rewrites
it entirely, sometimes many times. So a stamp is a **streaming dictionary entry**: the first time a
write-series appears it is DEFINED inline; reuses are backrefs; **when the series changes, a new
definition is emitted inline** (the consistency test below versions it automatically — a changed
footprint is a different signature → a new id/def). This is LZ-style, in musical-token form.

A definition is a contiguous block (locality), bracketed so the decoder knows its writes are the
stamp's internal timeline, not song frames:

```
PERC_DEF  id=K  char=KICK  len=N            # header atom (id, drum-character, frame length)
  PSTEP                                       # one stamp-internal frame
    W subreg=FREQ  val=…                      # the FULL voice footprint (see below), voice-RELATIVE
    W subreg=CTRL  val=…                      #   subregs (freq / ctrl / pw / ad / sr) -> voice-agnostic
    W subreg=PW    val=…
  PSTEP …                                     # N PSTEPs, packed contiguously
  [FILT  W reg=CUTOFF val=… …]                # ONLY if filter is drum-scoped (consistency-attributed)
PERC_END
```

### 1a. The stamp is the FULL register footprint (freq + ctrl + PW + ADSR + drum-scoped filter)

A drum is not freq+ctrl — it is **every register write during its span**. Measured on the footprint
probe (`audit/probes/resid_drum_footprint.py`, raw per-voice regs freq R∈{0,7,14} → pw R+2, ad R+5,
sr R+6; global filter 21–24):
- **PW + ADSR are part of the stamp** — usually identical every hit (consistent-in-stamp 90/148 ≈
  61%; e.g. Camerock's `(0,pw,3424),(2,pw,3168),(3,pw,2912)…` PW sweep every hit). Fold the
  consistent ones into the definition; an inconsistent ADSR/PW means it's drifting → its own (re)def.
- **The GLOBAL filter must be ATTRIBUTED, not assumed.** The SID filter is one global cutoff/res/mode
  shared by 3 voices, so filter writes overlapping a drum may be drum-driven OR independent tune
  automation. **Decide by the same recurrence principle that finds the drum: are the in-span filter
  writes IDENTICAL across every occurrence of the stamp?**
  - **Drum-scoped** (consistent every hit) → fold into the stamp (e.g. Camerock: cutoff `41216,33280,
    25344` identical ×all hits; Black_Sun's per-hit cutoff sweep). Measured **~1/3 of drum-overlapping
    filter (41/128)**.
  - **Tune-global** (varies / present on some hits, absent on others) → leave on the global filter
    channel, NOT in the stamp (e.g. Baggis: a cutoff ramp on ONE occurrence, none on the rest).
    Measured ~2/3.
  This rule generalizes to ANY register (and ADSR/PW): **consistency-across-occurrences = scope.**

The block lives OUTSIDE the per-frame song timeline (bracketed by `PERC_DEF`/`PERC_END`); the decoder
buffers it as stamp K. Because the whole sequence is adjacent tokens, the model sees a drum as a unit.

### 1b. Unigram sub-tokenizes the definition → similar drums cluster

The definition is emitted as a run of **regular, small, canonically-ordered atoms** (one per
register-write, fixed subreg order per PSTEP) — deliberately so the downstream **Unigram** tokenizer
can sub-tokenize it. Common sub-sequences across drums (a noise-hat burst, a down-sweep ramp, a
standard kick ADSR) become shared Unigram pieces, so **similar drum definitions share sub-tokens and
cluster in token space** — the model sees two kicks as related even before the `char` tag, and learns
drum "morphemes". This argues AGAINST one opaque blob token per drum (Unigram-opaque) and FOR the
regular-atom layout above. (Contrast the refuted `motif_pass`, which fought Unigram by pre-merging;
here we feed Unigram clusterable structure instead.)

### 2. Backref — a generic pointer placed on the timeline

Every drum hit in the song body (including the first) is one compact token at its onset frame:

```
PERC_REF  char=KICK  id=K  voice=V  [xpose=t]
```

- `voice=V` — which voice this hit plays on (the stamp is voice-agnostic; V supplies the actual
  registers). Onset timing = the token's position in the frame stream (gridded → low entropy).
- `char=KICK` — the GLOBAL drum-character; this is the **surface symbol the model predicts** and the
  transfer handle. `id=K` is the within-tune binding to the kit (usually 1 stamp per character, so the
  model mostly just predicts `char` + voice + when).
- `xpose=t` — optional semitone transpose for *pitched* drums replayed at different pitches (the `rel`
  signature: a tom/kick re-pitched). Absent for fixed (noise) drums.

### Drum-character classifier (global vocabulary)

Assigned algorithmically from the stamp's own writes — `{KICK, TOM, SNARE, HAT, CYMBAL, NOISE_FX,
PITCH_FX}` — from: dominant waveform (noise vs pulse/tri), freq range (low/mid/high), sweep
direction+magnitude (down-sweep → kick/tom), pitched-vs-noise frame ratio (pulse+noise → snare),
length (short hat vs long cymbal). Deterministic, low-cardinality, tune-independent → the transfer
vocabulary.

## Why this is the right shape

- **Lossless / RESID=0:** the definition is the exact writes → byte-exact replay; drains ~85 % of
  rung-0 RESID NOTES with zero fidelity loss (vs. the lossy "content-tier relaxation" feared earlier).
- **Learnable by locality:** packing the sequence contiguously turns a hard "learn a pattern smeared
  over interleaved frames" problem into "learn one local block + a sparse gridded pointer stream".
- **Transfer:** the character vocabulary is global; the model learns *place a kick on the grid*, not a
  tune-specific opaque id → drum knowledge transfers to unseen tunes (the project's goal). The tune's
  actual kick bytes come from its in-context kit definition.
- **Voice-agnostic:** matches the driver reality that drum instruments are reused across voices
  (driver-ref "reuse/banks"); the pointer carries the voice, the stamp doesn't.
- **Token budget:** define-once + cheap pointers; a drum that fired 10× as a 5-frame note (≈50+ rows)
  becomes 1 definition + 10 one-token refs.

## Decode + exactness

`PERC_REF char,id,voice,xpose` → look up the stamp's writes (the latest inline definition of `id`
seen so far in the stream), map each voice-relative subreg to voice V's absolute registers, add
`xpose` to pitched freq writes, replay any folded drum-scoped filter writes on the global regs, and
splice into the timeline at the onset frame for the stamp's `len` frames. Round-trip must be
**byte-exact** vs the raw register log (gate on the emulator + the deterministic per-frame oracle), or
the note falls back to RESID. Because definitions are inline and redefinable, the decoder keeps a live
id→stamp table updated as it streams (a later `PERC_DEF id=K` rebinds K).

## Codebook economics — 150-tune prototype (`audit/probes/resid_drum_codebook.py`, 2026-05-30)

End-to-end mining + consistency-attribution, non-emitting, across 133 parsed tunes (98 with recurring
stamps):
- **Coverage: 96% of RESID notes / 79% of frames** drained by recurring stamps (per-tune median 97.6%
  of notes). Much higher than the 3-fixture 85%/42% — the long held-gate giants are Baggis-specific,
  not corpus-wide.
- **Codebook: median 11.5 defs/tune (p90 36, max 57), 1505 total** — bounded.
- **Redefinitions: median 0.1/drum, p90 1.0, max 5 (348 total over 995 drums)** → inline-redefinable
  is LOAD-BEARING (≈1/3 of drums redefine), not theoretical.
- **Consistency-attribution at scale: 51% of defs fold consistent PW/ADSR; 24% have drum-scoped
  filter** (rest tune-global) — the attribution split is real and substantial.
- **Character: 65% percussion (SNARE 29 / TOM 12 / HAT 9 / KICK 7 / CYMBAL 4 / NOISE_FX 1), 35%
  repeated TONAL/OTHER** — the recurring-stamp mechanism is more general than drums; the tonal 35% is
  the bridge to the melodic **patch-preamble** twin (`patch_preamble_encoding.md`).

## What it leaves (for later iters)

The non-recurring singletons (~10–15 % of notes) and the long held-gate giants (the 58 % frame-mass)
are NOT percussion — they go to iters 3 (held-ARP irregular duration), 4 (freq-domain SLIDE), 5
(held-gate re-segmentation).

## Build plan

1. **Footprint mining pass** — per tune, scan the RESID notes for recurring exact stamps; for each,
   collect the FULL voice footprint (freq/ctrl/PW/ADSR) and run the **consistency-across-occurrences**
   test to (a) confirm the stamp, (b) fold consistent PW/ADSR in, (c) attribute filter writes
   (drum-scoped vs tune-global). Classify `char`. Emit definitions INLINE at first/changed use, ids
   rebindable. Reuse/repair the refuted `motif_pass` mining machinery (same shape, different goal:
   lossless RESID-drain, not content-acc).
2. **Tokens:** `PERC_DEF/PSTEP/PERC_END/PERC_REF` ops + the `char` enum; emit inline defs + body refs
   as regular Unigram-clusterable atoms; decoder keeps a live id→stamp table. Gate flag; default off.
3. **Gate:** deterministic suite green + emulator byte-exact round-trip + re-trace rung-0 RESID (note
   share should drop ~85 % of the recurring-stamp share) + confirm the Unigram vocab clusters similar
   stamps (inspect merges).
4. Expand the probe to a larger rung to size the global `char` vocab, confirm cross-voice reuse, and
   measure redefinition frequency before flipping any default.

## Open questions

- **Redefinition granularity:** how much drift = a new def vs an `xpose`/param of the same stamp
  (a slightly retuned kick is a transpose, a rewritten kick is a new def). The consistency test draws
  the line; tune the tolerance.
- **Near-match stamps:** exact `abs` covers 85 %; `ctrl`-rhythm 90 % with fewer stamps but lossy on
  freq. Keep definitions exact (lossless); use `char`/`shape` only for the transfer tag + clustering.
- **Pitched-drum xpose vs distinct stamps** — one stamp+xpose vs many stamps (`rel` data favors xpose).
- **Cross-tune character→default-stamp** for zero-context drum generation (vs always needing an
  in-tune definition).
- **Drum-scoped filter vs the global filter channel** — when a drum owns the filter for its span but
  the tune also automates it elsewhere, the decoder must restore the pre-drum global filter state
  after the stamp (save/restore), or the drum's filter leaks into the song. Needs a restore rule.
