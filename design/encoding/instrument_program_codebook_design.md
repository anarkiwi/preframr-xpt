# Instrument-program codebook — collapse the note-associated macro cluster

**Status:** Design (2026-06-04). Supersedes the older instrument-state-codebook design (removed; modelled a
*static* per-voice `(waveform,AD,SR)` state and was withdrawn as "not needed for correctness"). Prompted
by: "revisit the driver doc — there is more complexity here than is actually in the drivers; some
operations/sequences are associated with notes, some are not." Grounds in
[`sid_driver_ornament_reference.md`](../references/sid_driver_ornament_reference.md). Direction chosen: **design first,
sign-off before code.**

## 1. The driver's actual representation (what we're matching)

Per the driver reference, every C64 music driver represents sound as exactly three things, split on the
note-association axis:

1. **The instrument = a per-frame register-write PROGRAM, referenced by id, fired at note-onset.** Hubbard:
   an 8-byte record + tables; JCH/SF2: `wave-table | pulse-table | filter-table` by id; defMON: sidTAB
   rows by index. The wave-table is *walked one row per frame from note-on* (waveform + chord), AD/SR are
   loaded at onset, the HR window primes the attack. "Ornament definitions are a small bank referenced by
   id, reused across notes." This is a **sequence**, not a static state.
2. **Pitch ornament (also note-associated)** — `PLAIN/OCTAVE/ARP/SLIDE/VIB/SWEEP`. Already collapsed in
   our code: `decoders.py` dispatches purely on `ORN_TYPE`, zero per-driver branching. **Leave alone.**
3. **Continuous sweeps (NOT note-aligned)** — "PW and filter sweeps persist across notes … encode them as
   continuous per-voice (PW) / global (filter) trajectories." Portamento spans tied notes. Our
   `Gradient`/`Sweep` channel. **Leave alone** (modulo the sweep-vs-program boundary, §5).

So the only *over-built* region is #1: the note-onset instrument program.

## 2. Diagnosis: one driver concept, ten passes

The instrument program (#1) is currently mined by ~ten overlapping passes, each a different
pattern-matcher for the *same* note-onset register sequence, each with its own escape condition:

| pass | what it mines | escape condition (→ residual gap) |
|---|---|---|
| `StampPass` | recurring exact `(freq,ctrl)` **series** per voice | series must recur exactly; non-recurring → miss |
| `PatchPass` | recurring `(AD,SR)` loads per tune | recurrence ≥2; singleton → miss |
| `PresetPass` | wide-val plain SET snap on reg2/21 | only reg2/21 |
| `CtrlWavetablePass` (+nibble, +`onset_def`) | per-frame ctrl/waveform walk; define-on-first | `MINREP≥2`; `fr_reg_count==1`; onset floor; lane-id space |
| `CtrlOscPass` | per-frame ctrl **oscillation** runs | must match an oscillation period |
| `CtrlTriplePass`/`CtrlBigramPass` | gate-event 2/3-grams | fixed n-gram shape |
| `HardRestartPass` | gate-off + AD/SR reload onset window | HR-shaped only |
| `NoteOffPass` | tag gate fall/rise | 1:1 tag, no coverage |

**The residual tail is the direct symptom.** A register write that belongs to the instrument but lands in
the *gap between these heuristics* — a singleton AD, a one-off ctrl byte, a non-recurring waveform step, a
never-gated voice's setup — escapes all ten and falls to raw-SET RESID. The driver has **no such gaps**:
the program is a deterministic per-frame sequence from note-on, and every onset-associated write is part
of it, by id. Today's point-fixes (never-gated-freq, lane-keying) patch individual gaps *inside* the
fragmented model — they don't remove the gaps.

`StampPass` ("recurring `(freq,ctrl)` series per voice") and `PatchPass` ("`(AD,SR)` loads") are the tell:
they are already two field-sliced instances of one program codebook, each with a separate id-space and
recurrence gate. Unify the slices and remove the gates.

## 3. The unified model

Three channels, on the note-association axis, replacing the cluster:

### 3.1 Note segmentation (shared, control-aware)
The boundary is **not** gate-on (legato/held-gate drivers hold the gate across many notes — driver doc
§"Note segmentation caveat"). Use the **intrinsic level-change ∪ gate** detector that
`TrajectoryAnchorPass` pass-1 already implements, with the control register assigning each frame's role
(test/gate-low → HR transient; noise waveform → percussion-timbre; gated pitched → melody). A *note span*
= onset frame → next onset frame on that voice. Continuation frames (held-gate melody, in-progress
sweeps) are **not** new programs.

### 3.2 `InstrumentProgramPass` — the one note-anchored TIMBRE-program codebook
At each note onset, the per-voice **timbre** writes spanning the note span — ctrl-waveform (ctrl hi
nibble), AD, SR, PW-init, the HR primer — form the instrument program. Intern it:

- **Key:** `(voice, program-signature)`, voice derived from the **voice-block context** (`remove_voice_reg`'s
  `v`: VOICE_REG cumsum + FRAME sval), never the canonical reg → no cross-voice conflation (the bug class
  the withdrawn doc found).
- **Codec:** the existing `CodebookFamily` DEF/STEP/COMMIT/REF machinery. A program is a multi-STEP DEF
  (one STEP per frame's writes in the span); REF replays the whole program by id. One id counter, **one
  pipeline stage** (inline, pre-`SubregPass`, pre-canonical-collapse) → no cross-stage id collision, no
  nibble-lane split to desync.
- **Define-on-first:** a program seen once still emits its DEF (no `MINREP` gate); a recurring program
  REFs. ⇒ **every onset-associated timbre write is at minimum a DEF ⇒ zero raw-SET residual on
  ctrl/AD/SR/PW-init, by construction.** This is what removes the heuristic gaps.
- **Timbre only.** The program carries waveform/AD/SR/PW; **pitch stays with the universal ornament stack**
  (§1.2). Clean separation of the two note-associated channels (matches how we already model pitch; avoids
  re-modelling pitch that ARP/SLIDE/VIB own). The JCH "wave-table drives waveform AND pitch" is split at
  our boundary: waveform → program, pitch-transpose → ARP.

Oscillation (`ctrl_osc`) is a *looping* program; a `STAMP` series is an *exact* program; a 1-frame patch
is a length-1 program; HR is the program's onset primer. All become instances of one codebook, not
separate passes.

### 3.3 Sweep channel (non-note) and pitch ornament (note) — unchanged
`Gradient`/`Sweep` keep the continuous PW/filter trajectories; the skeleton+ornament stack keeps pitch.
The only new contract is the **program↔sweep boundary** (§5).

## 4. What it subsumes / keeps

- **Delete:** `StampPass`, `PatchPass`, `PresetPass`, `CtrlWavetablePass`, `CtrlWavetableNibblePass`,
  `onset_def`, `CtrlOscPass`, `CtrlTriplePass`, `CtrlBigramPass`, and `WavetablePass`'s ctrl role. ~10 →
  1.
- **Fold in as markers:** `HardRestartPass` (onset primer of the program), `NoteOffPass` (the span's
  closing gate event).
- **Keep (orthogonal):** the pitch ornament stack (`Skeleton`/`TrajectoryAnchor`/`FreqTrajectory`/
  `FreqOnset`/`FreqNudge`/`PreGateFreq`/`GateSlopeShift`/`PerRegBurst`), the sweep channel
  (`Gradient`/`Sweep`), and the structural passes (`Transpose`/`Legato`/`Subreg`/`VoiceBlock`/`Dedup`/
  `VoiceTrack`/`Loop`/`Init`).

## 5. Contracts — DECIDED (2026-06-04, user sign-off) + verification gates

1. **Program span = timbre-change / HR boundary** (DECIDED). Span the timbre program on the gate/HR
   boundary, independent of the finer pitch-note segmentation: a held-gate legato run is ONE timbre
   program over many pitch-notes (pitch varies inside via the ornament stack). *Verify (§5a):* AD/SR are
   constant within a gate-held span (so they are onset-anchored, not per-pitch-note).
2. **Program↔sweep boundary = set-vs-delta** (DECIDED). The onset PW/filter *set* is program; subsequent
   per-frame *deltas* are the sweep channel. No PW/filter write is double-claimed. *Verify (§5b):*
   non-onset PW/filter writes are predominantly small deltas (sweep-shaped), onset writes are absolute
   sets.
3. **Program REF match = EXACT** (DECIDED). ctrl-waveform/AD/SR are low-cardinality and recur exactly;
   anything non-exact is sweep or RESID, never a fuzzy program match. Define-on-first guarantees
   residual==0 regardless of recurrence rate. *Verify (§5c):* programs recur exactly often enough that
   exact-REF gives real compression (else it still drains to 0, just bigger vocab).

**§5a/b/c VERIFIED (2026-06-04, 861,098 spans, corpus sample step-90, digi-excluded,
`/scratch/tmp/empirical_checks.py` via `register_state`):**
- **§5a:** AD constant within gate-held span **97.0%**, SR **96.3%**; waveform-walk distinct/span mean
  **1.91** (lengths 1–3 dominate). → AD/SR are onset-anchored; the timbre program is a short per-frame
  walk. Contract 1 holds. (The ~3% AD/SR variation = HR multiload / mid-note envelope, carried as
  multi-STEP within the program.)
- **§5b:** non-onset PW changes that are small deltas (|Δ|≤256) **86.5%**. → continuation PW is
  sweep-shaped; the set-vs-delta split holds. Contract 2 holds. *(Filter cutoff read 0 changes —
  `register_state` likely doesn't surface regs 21/22; filter set-vs-delta deferred to impl, separable
  sweep channel.)*
- **§5c:** program `(waveform-walk, AD, SR)` exact-recurrence within tune **98.0%**. → the instrument is a
  small reused bank; exact-REF compresses ~totally, define-on-first covers the 2% unique. Contract 3
  holds; residual==0 by construction.

Remaining verification gates (not design forks):
4. **Byte-exact drain ORDER** — the per-frame audible drain order (ADSR interleave) must be reproduced by
   program replay exactly; risk is in ordering, not per-field logic. Gate corpus-wide with
   `PREFRAMR_VERIFY` / `cb_div_audit.py`.
5. **Vocab / learnability** — one program-id alphabet vs ten small ones; check `tkvocab` size and the
   learnability triage (DEF→REF block-locality must hold; programs are self-contained per note span).

## 6. Why this reaches residual==0 (the standing gate)

Every note-onset-associated register write is owned by the program codebook with **no** recurrence/floor/
period/lane gate to fall through — define-on-first guarantees at least a DEF. Continuous content is owned
by the sweep channel. Pitch by the ornament stack. The only writes left for raw-SET RESID are genuinely
novel non-note content with no sweep/program structure — and the driver doc's claim (universal primitive
set, no per-driver gaps) predicts that set is empty at the register-log input. RESID stays the lossless
escape, but structurally drained, not heuristically chased.

## 7. Migration / gates / release

- New flag `instrument_program` (default OFF, out of `REGISTERED_MACROS` until the audition gate); it
  changes the encoding ⇒ new vocab ⇒ re-tokenize.
- **Acceptance:** `residual_set_census --step 10 reparse=True` == 0 corpus-wide (digi-excluded) +
  reject-claim audit clean + 12-SID WAV audio-equivalence audition before any default flip + per-op
  accuracy unaffected. Validate on the **corpus**, never a sample (the 57-tune sample hid the id
  collision; the census reparse-cache bug hid the true tail — both this session).
- Ship as **tokens 0.45.0** (retires the 0.44.0 nibble/ctrl_wt ops + the deleted passes' ops). Cross-repo
  per [`release_build_cache.md`](../references/release_build_cache.md): tokens `vX`→PyPI → preframr floors → Docker →
  xpt rebake.

## 8. Phasing (always-green, even though design-first)

1. Land segmentation + `InstrumentProgramPass` behind the flag; prove residual==0 + byte-exact corpus-wide
   alongside the existing passes (both paths present).
2. Audio-equivalence audition; flip default.
3. Delete the ten subsumed passes + their ops/decoders/flags in one release once the unified path is the
   default and green.
