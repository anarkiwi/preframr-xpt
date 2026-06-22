# Generic tracker-structure decompilation — recover the PROGRAM, not the per-register trace

**Status: SHIPPED (2026-06-22).**  Byte-exact lift implemented
(`bacc/generic/tracker.py` + `bacc/tracker_serialize.py`), wired into
`generic_serialize` behind a 1-token format tag (the encoder picks the smaller of
the genfits / tracker forms AFTER verifying the tracker form renders identically).
Validated GENERIC across 12 tunes / 8+ driver families — every tune's lift is
byte-for-byte lossless (`unlift` reconstructs genfits/eventfits) AND the chosen
serialized form renders residual-zero w.r.t. the per-register path; every tune
chose the tracker form and every one is smaller:

| tune | driver family | nframes | per-reg resid | lift byte-exact | tokens before | tokens after | ratio |
|---|---|---:|---:|:---:|---:|---:|---:|
| Grid_Runner | GoatTracker_V2 | 1382 | 0 | ✓ | 161,002 | 12,821 | 12.6x |
| Monty_on_Run | Hubbard/DMC | 1382 | 0 | ✓ | 370,861 | 9,995 | 37.1x |
| Travel | DMC (legato-gap) | 1377 | 0 | ✓ | 54,886 | 13,637 | 4.0x |
| Hibernation | DMC | 1382 | 0 | ✓ | 123,230 | 14,798 | 8.3x |
| PC-89_part_3 | MoN/FutureComposer | 1382 | 0 | ✓ | 145,159 | 12,032 | 12.1x |
| Poppol | JCH_NewPlayer (through-comp) | 1382 | 0 | ✓ | 541,871 | 10,687 | 50.7x |
| Gyruss | D.Whittaker (interleave) | 1583 | 0 | ✓ | 61,325 | 10,780 | 5.7x |
| Party_Quiz | Parker Bros (freq-rest) | 1583 | 0 | ✓ | 91,486 | 6,604 | 13.9x |
| Hammurabi | GoatTracker_V2 | 1380 | 0 | ✓ | 100,919 | 12,642 | 8.0x |
| FamiCommodore | GoatTracker_V2 | 1382 | 0 | ✓ | 319,457 | 8,705 | 36.7x |
| Pengon | Keith_Wood (through-comp) | 1382 | **2762** | ✓ | 20,123 | 3,859 | 5.2x |
| Ghost | DMC (digi) | 1382 | 0 | ✓ | 272,616 | 5,209 | 52.3x |

"tokens before" is the per-register genfits serialization (the prior shipping
form); "after" is the chosen tracker form.  **Pengon** is the honest fallback case:
the per-register *recovery* does not fully cover it (resid 2762 — a true
through-composed melody the generic fitter leaves partly residual), yet the lift is
STILL byte-for-byte faithful to whatever the per-register path produced (it
re-expresses, never re-fits), so it is admitted losslessly and stays compact (8
instruments).  Everything below is the original design.

The driver-agnostic generic recovery
(`preframr_tokens/bacc/generic/`) reaches whole-tune residual-zero on the proven
corpus, but its *serialization* is the per-register EXECUTION
(`genfits`/`eventfits`: thousands of per-frame piecewise fits + per-frame `carry`
arrays), so a generic program for Grid_Runner serializes to **2,918,796 tokens**
vs the GoatTracker hand backend's **2,817** (~1000x bloat, ~8x the raw register
dump).  This doc designs — and the companion prototype validates — a generic
DECOMPILER that lifts the byte-exact per-register fits into a TRACKER-like program
(shared instruments + per-voice note events + patterns/orderlist) of the SAME
shape the hand backends emit, so the EXISTING shared score serializer
(`bacc/serialize.py` REPEAT/TRANSPOSE LZ + inline instrument dedup) compresses it
to hand-backend scale.  We recover STRUCTURE; the optimize/serialize pass does the
final squeezing.

Companion / prior art (all read for this design):
[`sid_player_decompiler.md`](../../preframr-xpt/design/encoding/sid_player_decompiler.md)
(the `trace = VM(program)` thesis + HARD RULE #0 + the STEP/TRACKER reframe that
shipped Monty at 0.901 tok/frame),
[`generic_bacc_recovery.md`](../../preframr-xpt/design/encoding/generic_bacc_recovery.md)
(driver-invariant vs driver-specific decomposition; the generator fitter),
[`generic_recovery_from_bustrace.md`](../../preframr-xpt/design/encoding/generic_recovery_from_bustrace.md)
(the bus→25-reg state foundation + note table + the proven 8/8 residual-zero),
[`sid_opset_inventory.md`](../../preframr-xpt/design/encoding/sid_opset_inventory.md)
(the bounded op-set; instruments = parameter sets), and
[`unified_generic_recovery.md`](unified_generic_recovery.md) (the CITG unification —
one clocked indexed-table generator the archetype zoo collapses onto).

---

## 0. The discipline (HARD RULE #0, inherited unchanged)

Lossless / byte-exact: the recovered tracker program must render to the same
`(nframes, 25)` register state — `residual == 0` on the tunes the per-register path
already recovers.  A recovered "instrument" or "pattern" is a genuine REUSED
program object (a parameter set / a repeated event subsequence), never per-frame
raw data in disguise.  The decompiler is a **lossless re-EXPRESSION** of the
already-residual-zero per-register fits into tracker shape: it never re-fits the
trace, never weakens exactness, and never invents structure.  Where structure does
not exist (a true through-composed melody is a note list — fine and small; raw
digi is not a generator), it is surfaced honestly and the per-register path remains
the fallback.  **Do NOT regress the proven corpus.**

---

## 1. Diagnosis — WHERE the 1000x lives (measured, not assumed)

Grid_Runner generic program (residual-zero), measured token breakdown
(`serialize.measure`):

| block | tokens | what it is |
|---|---:|---|
| `genfits` (freq/pw lanes) | 1,994,426 | per-note-on piecewise fits + 3× per-frame `carry` arrays (14,433 ints each) |
| `eventfits` (ctrl/AD/SR/filter/vol) | 923,842 | per-change-point segments serialized verbatim |
| note_table / boot / nframes | 528 | (already compact) |
| **total** | **2,918,796** | (Monty: 1,678,892) |

Two structural pathologies, both "store the execution":

1. **Per-segment re-emission of the SAME instrument.**  Grid has **33,913 genfit
   segments but only 237 distinct fit-SIGNATURES** (the archetype + its structural
   params, ignoring the per-note seed/pitch).  A **143x** redundancy: the same PW
   sweep `citg{table:[256,1792,-2048]}` and the same arp `citg{table:[-3092,…]}`
   are re-serialized at every note-on.  This is the dead-on symptom the thesis
   predicts: "stores the VM's per-register EXECUTION, not the PROGRAM."  74% (grid)
   / 88% (monty) of segments are bare `hold` — a held note value, i.e. the note's
   PITCH, which belongs in a note EVENT, not a stored per-segment constant.

2. **Per-frame carry arrays.**  Each PW lane carries a length-`nframes` `carry`
   array (the freq→PW no-CLC coupling), serialized as one int per frame: ~14k
   tokens × 3 voices = ~43k tokens of pure per-frame data — a HARD RULE #0
   red flag (it is recomputable from the freq generator, never stored).

The fix is NOT to RLE the carry arrays or dictionary-compress the segment list
(band-aids on the symptom).  The fix is to recover the program the composer wrote:
**a small set of shared instruments + a sparse per-voice note-event stream +
pattern/orderlist repetition**, and serialize THAT through the existing score path.

---

## 2. Decompilation framing — the techniques, mapped to each layer

This is a static-analysis / decompilation problem.  The per-register byte-exact
fits are the "disassembly"; the tracker program is the "decompiled source".  Each
layer maps to a classical decompiler technique:

### 2a. Instruments ← library/idiom recognition + def-use value-provenance

A tracker INSTRUMENT is a reusable parameter set (waveform/ADSR/pulse program +
freq-modulation program: arp/vibrato/PW-sweep) that recurs across many notes.
Recovering it is **library-function / idiom recognition** in a decompiler: instead
of re-fitting every note's per-register body, recognize that note N and note M
execute the *same* instrument program at different pitches, recover it once, and
reference it.

Mechanism (lossless, over the already-fitted segments — no re-fit):
- **Slice into note bodies.**  The fitter already slices the generator lanes at
  `note_boundaries` (gate rise ∪ ctrl change ∪ ADSR change) ∪ `pw_sweep_resets` ∪
  `freq_note_onsets` — these ARE the note-on frames.  The non-generator lanes
  (ctrl/AD/SR) change at the same boundaries.  So a NOTE is the tuple of all-lane
  fits over `[on, next_on)` for a voice.
- **Pitch-invariant canonicalization (the dedup key).**  An instrument is its
  per-note body with the PITCH factored out — exactly the "pitch-invariant
  instruments" that collapsed Monty's 818 freq-bodies → 45 (`sid_player_decompiler.md`).
  The CITG fit already separates structure (`table`/`clock`/`mode`/`width`) from
  the per-note seed (`seed`/`v0`/`base`/`ctr0`); an arp's `table` is *relative
  semitone offsets* or *note-table indices* (pitch-relative), a vibrato's
  `amp_step` is `(notetab[n+1]−notetab[n])>>depth` (a depth, pitch-invariant).  The
  instrument key is the fit signature with seed/pitch removed; the note event
  carries the pitch (the `hold` value resolved through the note table) + the
  per-note seed needed to re-render byte-exact.
- **Value-provenance for the shared data tables.**  The note table is already
  recovered this way (a freq write sourced from a contiguous never-written 2-byte
  RAM region).  The instrument/wavetable/arp/pulse tables are the *recovered fit
  tables* themselves (the `citg.table` arrays) — already bus-derived, just not yet
  shared.  Dedup makes the 237 distinct signatures the instrument table.

Result: ~237 instruments (grid) referenced by ~note-on count events, vs 33,913
inline bodies.

### 2b. Note events ← the segmentation already in hand

A NOTE EVENT is `(dt, pitch, instrument-ref, duration[, seed])`:
- **dt / duration** = the note-boundary frame deltas (the segment lengths).
- **pitch** = the `hold` value of the freq lane resolved through the recovered
  note table → the canonical A440 12-TET grid index (Part B), IDENTICAL to the
  GoatTracker/Hubbard token for the same concert pitch (so the cross-driver grid
  and the TRANSPOSE factoring just work).  A note whose freq is not on the clean
  ET grid (a swept/aliased tail) rides the literal-index escape the hand serializer
  already has — lossless.
- **instrument-ref** = the dedup index from 2a.
- **seed** = the minimal per-note residue (initial accumulator phase, vibrato
  `ctr0`, a per-note PW v0) that, with the shared instrument program + the pitch,
  re-renders the note's lanes byte-exact.  This is the irreducible per-note data;
  it is small (a handful of ints), NOT a per-frame array.

This is the same `NoteOn` shape (`bacc/primitive.py`) the hand backends already
emit, so it flows straight into the shared serializer.

### 2c. Patterns + orderlist ← grammar induction / dictionary compression + control-flow structuring

The composer's repetition (a phrase replayed, a bass loop) is **loop structure** in
decompiler terms.  Recovering it is repeated-subsequence factoring over the
note-event stream:
- The shared serializer's `_lz_emit_t` ALREADY does inline backward REPEAT (exact
  phrase repeat) + TRANSPOSE (a phrase replayed at a constant pitch interval on the
  canonical grid = a tracker orderlist Transpose).  This is byte-pair/Re-Pair-class
  dictionary compression over the row list, with a transpose-aware delta.
- So the decompiler does NOT need a separate pattern/orderlist pass: emit the
  per-voice note-event rows and let the EXISTING `_lz_emit_t` factor the repetition.
  The 5-note-bass-×-434-repeats and the transposed phrases collapse there, exactly
  as they do for the hand backends.  (A future refinement can pre-segment explicit
  patterns + an orderlist for an even tighter form, but the LZ already captures the
  compression; structuring it into named patterns is presentation, not bytes.)

### 2d. Program synthesis (the gate)

The whole thing is validated as program synthesis: render the recovered tracker
program (shared instruments + note events) back through the generic VM and require
the `(nframes,25)` state == the bus-state byte-exact.  This is the same
`render_generic` + `residual` harness, re-pointed at the tracker representation.
Residual-zero is the gate; a non-zero residual means the lift dropped information
(fix the lift), never a patch.

---

## 3. The recovered structure → `BaccProgram` (same shape, existing serializer)

The decompiler produces a `BaccProgram` with the populated `score` + `instruments`
the hand backends use, NOT the `genfits`/`eventfits` tables:

```
BaccProgram(
  driver   = "generic",
  nframes,
  boot     = frame-0 25-reg seed,
  instruments = [ <instr 0 body>, <instr 1 body>, ... ],   # the ~237 deduped fit programs
  score    = [ NoteOn(frame, voice, note, instr, lnth, porta/seed), ... ],   # sparse
  tables   = { "note_table": [...] },                       # bus-recovered, shared
)
```

- `instruments[k]` is the pitch-invariant fit program for the k-th distinct
  signature (its archetype + structural params for every lane it animates).  A
  genuine reused program object.
- `score` is the per-voice note-event list, frame-ordered — the SAME `NoteOn`
  dataclass the Hubbard/GoatTracker paths emit.
- Serialization: a generic `lit_emit`/`lit_read`/`delta_of`/`shift` over the
  note-event rows, fed to the SHARED `_lz_emit_t` (REPEAT/TRANSPOSE) with the
  inline-define-instrument-on-first-use `seen` dedup — byte-for-byte the same
  machinery `gt_serialize._row_lit` + `_emit_rows` already use.  No new token ids.

Decode is the inverse: rebuild instruments + score, then `render_generic` over the
tracker representation → `(nframes,25)`.

---

## 4. Why this collapses the size (the arithmetic)

- **Instruments**: 33,913 inline bodies → ~237 shared definitions (defined once,
  inline on first use).  ~143x on the dominant block.
- **`hold` segments → note pitch**: 74–88% of segments were a held constant; they
  become a single note-token per event (resolved through the shared note table),
  not a stored `('hold',{value})` segment.
- **Carry arrays gone**: the freq→PW carry is RECOMPUTED from the (now shared)
  freq instrument at render, never stored.  ~43k tokens removed outright.
- **Repetition**: the per-voice note-event rows go through REPEAT/TRANSPOSE LZ, so
  a looped/transposed phrase costs one copy op, not N literal events — the same
  factoring that gets the hand backends to ~2.8k.

Target: orders-of-magnitude reduction from 2.9M toward hand-backend scale (low
thousands).  The companion prototype MEASURES the achieved number on
Grid_Runner/Monty (`P.measure`) and asserts residual-zero.

---

## 5. Staged, regression-safe build (the migration)

1. **Prototype the lift** (this PR): `genfits/eventfits → (instruments, score)` for
   the two proven tunes, render byte-exact via the existing `render_generic`,
   MEASURE tokens through a generic score serializer.  Keep the per-register path as
   the fallback — the lift is admitted for a tune ONLY when its tracker render is
   residual-zero AND its token count beats the per-register path.
2. **Prove parity**: residual-zero on Grid_Runner + Monty (ground truth: both
   already render residual-zero via the per-register path).  No corpus regression:
   the per-register path stays for any tune the lift does not byte-exactly cover.
3. **Generalize across drivers**: the lift reads only the fit objects + the note
   table (no driver constants), so Monty (Hubbard) lifts identically to Grid
   (GoatTracker).  Extend to the 8-tune corpus, then HVSC.
4. **Ship incrementally**: a design doc + a byte-exact prototype on
   Grid_Runner/Monty with a big measured token drop is the first PR; the full HVSC
   sweep + explicit pattern/orderlist structuring follow.

---

## 6. Honest limits / open risks

- **Per-note seed residue.**  Some instruments need a per-note seed (free-running
  accumulator phase, vibrato `ctr0`) to render byte-exact.  This is genuine
  irreducible per-note data, but it is a few ints per note, not a per-frame array;
  it rides the note event.  If a lane's per-note residue is itself large
  (approaching one number per frame), that note is NOT instrument-reducible and is
  surfaced — the per-register fallback covers it.
- **`piecewise` composites** (a note whose body is several archetypes back-to-back)
  and **cross-lane coupling** (`additive_pw` carry) are the parts the unified-CITG
  doc flags as not-a-single-op.  The lift keeps them as a composite instrument body
  (still deduped if the composite recurs) or, for the carry, recomputes it from the
  sibling freq instrument — never stores the per-frame carry.
- **Pattern/orderlist explicit structuring** is deferred: the LZ captures the
  compression now; naming patterns is a later presentation pass.
- The gate is unchanged: any tune the lift cannot render residual-zero stays on the
  per-register path, surfaced, never faked.
