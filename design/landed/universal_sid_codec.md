# Universal SID Generative Codec — our own driver-VM (lossless, sparse, learnable)

**Status: SUPERSEDED (2026-06-20).** Right instinct (own VM, universal op-set, pitch arbitrary), wrong
mechanism: it bolted on a residual *escape lane*. The landed codec removed the escape hatch (residual→0
is the gate) and added the step reframe + pitch-invariant instruments — see `../encoding/
sid_player_decompiler.md` "HOW IT LANDED". Kept for the driver-survey + layered-op record.

**Original (DESIGN, 2026-06-17). SUPERSEDES `virtual_tracker_codec.md` (the GoatTracker-target direction —
rejected: forcing other-driver music into GoatTracker's 96-note grid + tempo quantization + 255-row tables
left freq ~85% off-grid and a ~0.84 residual; one tracker's constraints are the wrong cage).** We design
**our own** generative format — a small universal SID-driver virtual machine — that expresses *all* the
real drivers efficiently and losslessly. The surveyed drivers/trackers are **inspiration only**; none is
the target. Motivation/diagnosis: `ornament_generator_recovery.md` (events aren't sparse because ornaments
are per-frame; they're exactly periodic per-instrument programs → a sparse lossless form exists). Scope:
single-speed, non-digi (~92% of corpus). Fidelity target: the existing `canonical_writes(ow)` oracle.

## 0. Principles
1. **Lossless** — our own decoder (the VM) renders to exactly `canonical_writes(ow)`; an **external
   residual patch** (post-VM corrective writes) guarantees byte-exactness; the science is residual → 0.
   Losslessness never depends on any external driver's expressibility.
2. **Universal / superset** — the VM op set is the *union* of the real drivers' mechanisms; any in-scope
   tune compiles. No per-driver branching (the "universal driver" principle).
3. **Efficient / sparse** — note-rate body; per-instrument programs defined once + referenced; ornaments
   are *parametric programs*, never per-frame writes.
4. **Learnable** — every token is a meaningful musical op; structure fits a ~1024 window. Layered
   expressivity (typed > table > residual) keeps the common case compact and musical.
5. **MDL** — the encoder emits the shortest lossless program; per ornament it picks the most compact
   lossless form.

## 1. Driver survey → the universal abstraction (inspiration, not targets)

| driver | song structure | instrument programs | distinctive primitive |
|---|---|---|---|
| **GoatTracker** (pygoattracker) | orderlist→patterns→rows | ADSR + wave/pulse/filter/**speed** tables | note-independent (free-running) speedtable vibrato/porta |
| **SID-Wizard** (pysidwizard) | per-ch sequences→patterns | ADSR/vib/HR + wf/pw/filter tables + **chord table** | `wf_table` (waveform+arp+detune rows); WRPITCH **detune-with-carry**; big/small-FX |
| **defMON** (pydefmon) | arranger→3 patterns; events (flag,slot,slot,note) | **SidTAB rows** (15B) + **jp/dl** jump/delay | low-level **cascade walk** = per-frame program with jump/delay (most general) |
| **WEMUSIC** (reninja; Daglish/Crowther) | per-voice note streams + restart loops | **4 sweep engines**: vibrato, detuned osc, PWM | hand-coded sweep engines + dual freq tables |
| **Hubbard** | 3 tracks→patterns→notes; notework/soundwork | waveform/env/pulsespeed/vibrato | logarithmic vibrato; **ping-pong PWM**; composite **drum** (wave→noise→wave-fall); 16 SFX |
| **Galway** | 3 tracks→patterns→notes; notework/soundwork | fast arps, PWM, ringmod, porta, vibrato, drum | pseudo-polyphony via rapid arpeggiation |
| **zblex-v6** | bytecode | **every register as a wavetable**, bytecode-driven | confirms the universal-bytecode model |

**The deep structure is identical across all of them:** *a per-frame interpreter of per-instrument
programs over the register lanes, sequenced by a note score with loops.* Surface differences (table
formats, command sets, **quantization**) are exactly what we discard. Our VM keeps the union of the
mechanisms and **none of the constraints** — arbitrary frequency (no note grid), frame-precise timing (no
tempo quantization), unbounded programs.

## 2. The model
```
SONG = {
  header:   per-voice { note_table },                 # recovered per-voice freq table (arbitrary entries)
  instruments: [ INSTRUMENT, ... ],                   # defined once, referenced (the bank)
  globals:  [ GLOBAL_PROGRAM, ... ],                  # filter cutoff/res/route + volume
  score:    per-voice [ NOTE, ... ],                  # the sparse body (track-major; merge by onset)
  wiring:   sync/ring edges (ctrl bits 1/2), voice-keyed,
  residual: [ (frame, reg, value), ... ],             # EXTERNAL post-VM patch; target empty
}
INSTRUMENT = { adsr, hard_restart, programs: { pitch?, wave?, pulse?, filter_owner? } }
NOTE = { note_index, duration, instrument_ref, fx? }  # fx = per-note porta/arp/slide overrides
```
A PROGRAM is a per-lane, per-frame instruction stream (§3). Decode = run the VM per frame → register
writes; overlay the residual patch → `canonical_writes`.

## 3. The universal program VM (the decoder contract)

Each instrument program is a frame-stepped stream over the **superset op set** below. **Layered
expressivity** — the encoder uses the most compact lossless layer per program:

**Layer T — typed parametric ops (compact, learnable; the common case):**
- **pitch**: `VIBRATO{shape∈[tri,saw,sq,table], rate, depth, delay, phase_model∈[note_reset,free_running]}`
  (covers GT speedtable, Hubbard log-vibrato, WEMUSIC sweep, SW vibrato); `PORTA{target, rate}` (slides to
  any freq — fixes off-grid/sub-grid); `ARP{table|chord, rate}` (note-index offsets — GT/SW/Galway/Hubbard);
  `DETUNE{delta, carry?}` (SW WRPITCH).
- **pulse**: `PWM{rate, depth, mode∈[sweep,pingpong,set], base}` (Hubbard/Galway/SW/GT).
- **wave**: `WAVESEQ{rows:[(wave,frames)], loop}` (waveform walk; composite Hubbard drum = a short
  WAVESEQ); a row carries waveform + test/sync/ring bits.
- **filter**: `FSWEEP{target, rate}` / `FSEQ{rows, loop}` (global owner, voice-keyed).
- **envelope**: `ADSR{a,d,s,r}` + `HARD_RESTART{frames, prep}`.

**Layer G — general per-frame TABLE / micro-bytecode (lossless fallback; defMON/zblex-style):** when a
program isn't a clean typed form, emit a stepped table over the lane with ops `{DELAY n, SET v, ADD d,
RAMP(d,n), LOOP to, JUMP to, END}`. Expresses *any* per-frame behavior (this is the universal floor that
makes the VM able to encode every driver). Still far sparser than raw writes (one program shared by many
notes), but less compact than Layer T.

**Layer R — residual patch:** anything neither layer reproduces exactly → external corrective writes
(target empty). Guarantees byte-exactness during development; non-zero residual flags an un-modeled
mechanism (the "residual = unmodeled mechanism" principle).

**Per-frame render** (the decoder), per voice, `k`=frames-since-onset, `phase`=free-running counter:
`freq = note_freq(note_index + arp(k), note_table) + porta(k) + vibrato(phase) + detune(k)`;
`pw = pwm(...)`; `ctrl = waveseq(k) | gate | sync/ring`; `adsr` from instrument; globals from
`GLOBAL_PROGRAM`. Same-value-drop + ordering follow `canonical_writes`. **Phase models** and the
free-running counter are first-class (GT speedtable / WEMUSIC sweeps are free-running; many are
note-reset) — the encoder recovers per program which is lossless + shorter.

## 4. Pitch + timing (the two things every tracker constrains and we do NOT)
- **Pitch: arbitrary.** `note_index` over a **per-voice recovered `note_table`** (the exact freq entries
  the tune uses — keep the existing `pitch_grid` recovery, ~20/voice), NOT a fixed 96-note grid. Vibrato/
  porta/detune are exact freq deltas. Sub-grid bass and microtuning are just table entries / deltas. This
  removes the single biggest GoatTracker failure.
- **Timing: frame-precise.** Notes and program steps land on any frame; **no tempo quantization**. Note
  duration is explicit (frames). This removes the GoatTracker frame-lock drift.

## 5. Encoder (decompiler: trace → SONG) — informed by the survey

Per voice, then global; greedy-then-refine MDL; residual fallback at every step keeps it byte-exact.
1. **Note segmentation** (gate 0→1 onsets; duration) — reuse `_typed_cas`.
2. **Per-note lane trajectories** (freq, pw, ctrl, adsr).
3. **Freq decomposition (ordered):** `note_index` (nearest table entry) → **arp** (periodic note-grid
   hops) → **porta/glide** (attack slide to target) → **vibrato** (steady periodic residual: LFO {shape,
   rate, depth, delay, phase}). Knowing the drivers tells us exactly these four components exist and how.
4. **PW** → base + **PWM** (sweep/pingpong; remove any period cap).
5. **Wave/ctrl** → **WAVESEQ** (per-frame waveform walk + bits), shared per instrument; composite drums =
   short seqs.
6. **Envelope** → ADSR + hard-restart (reuse `_fold_envelope`).
7. **Instrument clustering** → notes sharing all programs = one instrument; MDL define-once-reference.
8. **Free-running phase recovery** → fit the global counter rule per program (GT speedtable semantics are
   the reference); fall back to per-note phase if no global rule, else note-reset.
9. **Globals + wiring + residual.** Anything unrecovered → Layer G table, else Layer R patch. Track
   residual fraction (headline) → 0.

## 6. Token serialization (sparse, learnable, byte-exact round-trip)
`HEADER (note_tables) · BANK (instruments = their typed/table programs) · GLOBALS · BODY (track-major
note score: NOTE = [Δnote_index][duration][instr_ref][fx?]) · RESIDUAL (target empty)`. Typed nibbles +
varints, fixed small vocab, positional ids. A note ≈ 3–5 atoms → note-rate sparse. `deserialize(serialize)
== song` exact. New format → its own version key; constrained-decode mask enforces define-before-reference
+ program/score/residual structure.

## 7. Fidelity + efficiency gates (every phase)
- **HARD byte-exact:** `overlay(VM_render(song), residual) == canonical_writes(ow)` on the 5 reference
  drivers + corpus sample. Residual lane guarantees it; science = residual → 0.
- **Cross-check oracles (use the libs we surveyed):** where a tune *is* expressible in GoatTracker/SID-
  Wizard/defMON, optionally compile our SONG → that lib's model and confirm its bit-exact player agrees —
  an independent check that our VM semantics are right. Also enables **export** (editable module) and
  **import** (seed our corpus with real tracker tunes).
- **Sparsity:** body event-frame density → 10–14% note floor (`event_sparsity.py`-style), per lane.
- Repo lint/tests green.

## 8. Build phases (each byte-exact via residual + density-measured)
- **P0 — VM + contract (residual floor).** The VM decoder (Layer T+G+R), the SONG model, token
  serialization, round-trip harness vs `canonical_writes`, encoder that puts everything in Layer R
  (trivially exact). **Reuse the `gtcodec` harness + residual-patch infra already built** (branch
  `feat/gt-decompiler`) — it generalizes directly (drop the GoatTracker `Song`, keep the harness/patch).
- **P1 — notes + envelope + WAVESEQ.** Sparse ctrl/adsr lanes.
- **P2 — FREQUENCY (the proof): note_table + vibrato LFO + porta + arp.** The lane GoatTracker couldn't
  express; our arbitrary-freq + frame-precise VM should drive freq residual → 0 and density → note rate.
  Empirically nail the composite decomposition + free-running phase first.
- **P3 — PWM + filter programs + globals.**
- **P4 — instrument clustering polish + Layer-T promotion (table→typed where it compresses) + wiring.**
- **P5 — constrained-decode mask + version + full-corpus byte-exact + density/residual report + cross-
  check vs the driver libs.**

## 9. Kept / reused / superseded
- **Kept:** `canonical_writes` oracle; `pitch_grid` note-table recovery; `_typed_cas`/`_fold_envelope`;
  the `gtcodec` round-trip harness + external residual-patch architecture (generalizes); the surveyed
  libs as analysis/cross-check **oracles** (NOT targets); the generator-model primitives
  ({HOLD/ACCUM/SWEEP/TABLE} ≈ Layer G/T) the earlier arc prototyped.
- **Superseded:** `virtual_tracker_codec.md` (GoatTracker as the target); the per-frame STEP/RAMP event
  emission and `cover()` signal-fitting as the representation.

## 10. After the codec
Re-encode the corpus on this note-rate-sparse stream; train atoms-only same config; re-read the deciders
that diagnosed the problem (de-confounded `copy_novel` novel-content, `free_running_gap`) + the generation
quality gate. M0 (notes-only) is the validated floor; this keeps full SID character, losslessly. Export to
a real tracker (via the oracle libs) is the editable endpoint for the "phrase → arranged tune" goal.
```
