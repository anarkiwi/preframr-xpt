# Ornament generator-recovery — why the event stream isn't sparse, and the fix

**Status: LANDED diagnosis (2026-06-20).** This is the diagnosis that pointed the arc at generator
recovery; it landed as the step/tracker codec (residual-zero, < 1 token/frame — `../encoding/
sid_player_decompiler.md` "HOW IT LANDED"). "Byte-exactness and sparsity are compatible via
generator-recovery" — confirmed. Kept as the record of the central finding.

**Original (Diagnosis 2026-06-17, operator-led). The central finding of the representation arc.** The
event codec is byte-exact but **not sparse** — it fires on ~half of all frames. The cause is that it
does *signal-level fitting* of the register trace instead of *recovering the per-instrument generator*
that produced it. The per-frame ornaments (vibrato / arpeggio / PWM / wavetable) are **exactly
periodic** and therefore losslessly representable as compact instrument programs; the codec just isn't
recovering them. This unifies every failed generation result and sets the path.

## How we got here

Three representation interventions (voice-form lane-demux, instrument DEF→REF, melody/timbre
factorization) each cleared the training-free learnability proxies and then **failed the generation
bar** — the v4 melody/timbre A/B was a de-confounded null (`copy_novel` novel-content 0.1575 ≈ v2
0.152 / v3 0.161; `free_running_gap` verdict `exposure_bias`) despite the best val_acc (0.622) and the
strongest proxy signal of the arc (see `melody_timbre_factorization.md`). The operator's reframe: a
prompt/representation change can't fix a model whose failure is in *generation itself*, and — crucially
— **events should be sparse** (a few per note), not one per frame. If they aren't, the abstraction is
missing something.

## Finding 1 — the event stream is NOT sparse

Measured on real single-speed tunes (`/scratch/tmp/m0/event_sparsity.py`, 3 composers):

- **~40–62% of all frames carry an event**; **57–77% of event-to-event gaps are DT=1** (an event on
  (almost) every frame). It is a per-frame stream wearing an event costume.

Density decomposition by class (`_collect` events, classified by kind atom):

| | Hubbard | Daglish | Follin |
|---|---|---|---|
| **NOTE onsets (the sparse floor)** | 11% | 14% | 10% |
| freq per-frame STEP (NI/FD/PW) | 34% | 41% | 29% |
| CAS-walk (wavetable / envelope) | 15% | 21% | 10% |
| gesture-recovered (PERIOD/POLY) | 13% | 17% | 10% |
| **total** | **51%** | **62%** | **43%** |

The **note rate is 10–14%** — where a tracker-style stream should sit. Actual density is 3–4× that,
and the excess is per-frame ornament (freq-step + wavetable-walk).

## Finding 2 — the ornaments are EXACTLY periodic (lossless existence proof)

`/scratch/tmp/m0/ornament_periodicity.py` — for every maximal per-frame STEP run, find the smallest
exact period:

- **FD (vibrato):** Hubbard 407 runs / 16,169 frames, Follin 64 runs / 4,108 frames — **all exactly
  periodic**, period 4–12.
- **NI (arpeggio):** Follin 104 runs / 3,416 frames — all exactly periodic, period 2–3.
- **PW (PWM):** all qualifying runs exactly periodic, period 2–64.

Zero drift. A 60-frame vibrato is a period-8 cell (~10 numbers). This is the operator's argument made
empirical: **the playroutine generated these frames from a few bytes of code + tables, so a compact
lossless representation provably exists** — it is the program that produced the log. The earlier claim
that "real vibrato drifts so it can't be parametric" was wrong; any drift is itself deterministic.

## Root cause — signal-fitting vs generator-recovery

The codec covers each value series with `gestures.cover` → `mdl_core.mdl_parse` (optimal HOLD/POLY/
PERIOD). The PERIOD primitive *exists and works on isolated ornaments* (a clean 38-frame vibrato →
one PERIOD gesture). But on whole tunes it recovers only a minority, because it fits gestures
**per-run** while the generator is **per-instrument**:

- **Ornament runs are note-length.** Vibrato/arp reset (re-phase, re-delay) at every note onset
  (~every 8–14 frames). A period-8 cell in a 10-frame note is barely one cycle — PERIOD can't amortize,
  so it falls to per-frame STEPs even though the *same* table runs on every note.
- **Per-note patterns are composite, not pure.** Inspecting per-note FD segments: each note is a
  leading **vibrato-delay** (e.g. 4 zero frames) + a phase-slice of a free-running **LFO** + an attack
  **pitch-glide** + per-instrument **depth**, superimposed. The exact per-frame pattern therefore
  varies by phase and duration (only 2.8–5× exact-pattern reuse across notes), so per-run shape-fitting
  fragments it.
- **The CAS-walk lane has no gesture path at all.** Wavetable / envelope walks go through
  `_typed_cas` as raw per-frame `FLD_CTRL`/AD/SR — never offered to `cover`.

`mdl_core._PMAX = 32` also caps period detection, so PWM with period 45–64 is missed outright.

The fix is to stop fitting the *trace* and recover the *generator*: the ornament is an **instrument
program** (vibrato `{delay, depth, LFO table}`, arp `{table}`, PWM `{rate, depth}`, wavetable) shared
across all notes of that instrument, replayed per-note for the note's duration by a deterministic
render-time player. A note becomes `{pitch, duration, instrument-ref}`. This is the **tracker model**,
and it is byte-exact (the player reproduces the exact register stream) *and* sparse (per-frame ornament
factored out → note rate ~10–14%). There is no lossless-vs-sparse tension; that was a false dichotomy.

## Vibrato recovery prototype (validates the approach; bounds the easy win)

`/scratch/tmp/m0/vibrato_recover.py` — recover shared vibrato cells, express each modulated note as
`(delay, cell, phase, len)`, verify lossless replay:

- **All collapses lossless-verified.** FD events down 1.6–2.6× (e.g. Hubbard ~31 distinct cells cover
  ~277 notes/tune).
- **Modest, because modulation is composite** — a single-cell model only catches pure vibrato; the
  glide+vibrato+depth superposition leaves a large residual. (Caveat: this prototype's "before" is raw
  value-changes; the real codec already recovers some via POLY+PERIOD, so the *marginal* headroom is
  smaller than the raw ratio.)

**Conclusion:** lossless ornament recovery is real and correct, but **note-rate sparsity requires
recovering the *complete* per-instrument modulation program** (glide + vibrato-delay + LFO + depth +
arp + PWM + wavetable), across lanes — genuine driver-program recovery. This is the MDL/generator arc
(`generator-model-prototype`, `log_to_swm_recompiler_design.md`) aimed at the correct target, and a
substantial multi-lane codec project, not a quick patch.

## M0 — the cheap de-risk of the premise (RAN)

Before committing to lossless generator-recovery, validate that a sparse note-level stream is learnable
/ generatable by the small model at all. `/scratch/tmp/m0/` (lossy, notes-only: lead voice, `(pitch,
duration)`, one token-group/note, vocab 65, ~2.5k tokens/tune vs 30–85k atoms):

- 5.0M-param GPT, 6000 steps → **val perplexity 1.63** — it cleanly learns the structure the chip-level
  stream buried.
- **Sampled** (top-k, not greedy) generation produces **coherent-looking melody, not the drone**: 10–20
  distinct pitches, ~50% rests (rhythmic phrasing), notes that move (consec-repeat 0.10–0.41 on 3 of 4;
  one weak sample at 0.84). 4 WAVs rendered (plain pulse synth), non-silent (RMS 0.11–0.16).

M0 is lossy (drops ornaments) and monophonic; long-range form is unproven. But it tests the *floor* —
locally-coherent melody — which the event model never reached. **Listening verdict 2026-06-17: the audio
is coherent melody ("wavs are fine", operator).** The sparse-stream premise is VALIDATED → **Path A is
the committed direction.**

## The two paths

- **Path A (right, expensive): lossless per-instrument generator-recovery codec.** Recover the full
  ornament programs → note-rate-sparse, byte-exact, keeps SID character + polyphony. Build incrementally
  by lane (vibrato → arp → PWM+`_PMAX` → wavetable), measuring density toward the note floor, byte-exact
  at each step. Extends the front-loaded instrument bank (`front_loaded_instrument_encoding.md`) to carry
  per-frame programs, not just the static onset patch.
- **Path B (cheap, RAN): lossy M0** as the go/no-go on the sparse-stream premise. If the audio is
  coherent → commit to Path A. M0 then seeds a two-stage system (note model → arrangement/ornament
  layer) — which is also how the "phrase → arranged SID tune" goal naturally factors.

**Decision status (2026-06-17): M0 audio confirmed coherent → Path A COMMITTED.** Build the lossless
per-instrument generator-recovery codec, lane by lane, byte-exact, measuring density toward the note
floor. Keep M0 alive as the parallel notes-level generation testbed + the seed of the two-stage system.

## Why this matters / implications

- It explains **all** the nulls: the stream is dominated by ornament automation, not music, so a ~1024
  window holds a sliver of structure and the small model drowns; reordering per-frame ornament steps
  (voice-form, factored) can't make them sparse.
- **Byte-exactness and sparsity are compatible** via generator-recovery — the project's lossless
  north-star does not have to be sacrificed for generation (the earlier "lossless is hostile to
  generation" framing was wrong; *signal-level* losslessness is, generator-level losslessness isn't).
- The learnability north-star is served *correctly* here: the sparsest stream is the one closest to the
  generating program.

## Reproduce

Scripts (promote to `preframr_experiments/audit/` if Path A proceeds): `/scratch/tmp/m0/`
`event_sparsity.py` (density + DT), `ornament_periodicity.py` (exact-period proof), `vibrato_recover.py`
(lossless cell recovery + collapse), and `m0_corpus.py` / `m0_train.py` / `m0_gen.py` (the notes-only
de-risk). All torch-free except the M0 trainer/generator (xpt image + GPU).
