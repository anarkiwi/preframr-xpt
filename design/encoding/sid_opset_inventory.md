# SID Player Op-Set Inventory — the universal generator grammar, extracted from four real drivers

**Status: ANALYSIS / DESIGN ARTIFACT (Phase 0a, 2026-06-17).** Companion to
`sid_player_decompiler.md`. This document extracts the per-frame **operations** that four
disassembled C64 playroutines apply to the 25 SID registers, merges them into one **union op-set**,
generalizes that union to a minimal **complete computational primitive set**, and proposes a **VM
instruction set** (the decoder's ISA). It is grounded in the actual source of:

- `pygoattracker` — GoatTracker 2 (`gplay.c` port), `src/pygoattracker/player.py` (662 lines, one
  self-contained playroutine).
- `pysidwizard` — SID-Wizard 1.94 (`player.asm` port), `src/pysidwizard/player.py` (2395 lines).
- `pydefmon` — defMON (`docs/SPEC.md` + `pydefmon/defmon_player.py`, 1578 lines).
- `reninja` — WEMUSIC (Ben Daglish / Tony Crowther, The Last Ninja), `docs/engine_annotated.txt`
  + `src/engine.asm` (1007 lines of 6502).

All four are **bit-exact**: each is pinned against its native player running in VICE/resid and
round-trips its own tunes register-for-register, frame-for-frame. So every operation tabulated below
is load-bearing, not a guess.

The SID register file (the finite machine all four write into), per voice V (base = 7·V):

| off | reg | meaning |
|-----|-----|---------|
| 0/1 | FREQ_LO / FREQ_HI | 16-bit oscillator frequency |
| 2/3 | PW_LO / PW_HI | 12-bit pulse width |
| 4 | CTRL | waveform nibble + gate(b0) / sync(b1) / ring(b2) / test(b3) |
| 5/6 | AD / SR | attack-decay / sustain-release envelope |
| — | $D415/16 | filter cutoff lo(3b)/hi(8b) (global) |
| — | $D417 | resonance(hi nibble) + voice→filter routing(lo nibble) (global) |
| — | $D418 | filter mode(b4-6) + master volume(lo nibble) (global) |

---

## 1. Per-driver op tables

Each row is **the computation that produces a register write or a freq/PW/cutoff accumulator
update on a given frame**, with the lane it targets, the runtime state it reads/writes, and its
parameters. "ptr" = a table index that walks; "ctr" = a frame-countdown.

### 1a. GoatTracker 2 (`pygoattracker/player.py`)

| Op | Lane | State (counter / ptr) | Params | Source |
|----|------|------------------------|--------|--------|
| Note→freq lookup | FREQ_LO/HI | `chan.note`, `chan.lastnote` | `FREQ_TABLE[note&0x7F]` | `_freq` L203; `_wave_exec` L501 |
| Wavetable walk (waveform seq + per-row delay) | CTRL waveform | `wave_table_ptr`, `wavetime` ctr | LTBL byte ($00-$DF wave, $E0-$EF gateoff-wave, delay ≤$0F) | `_wave_exec` L475-505 |
| Wavetable jump | (ptr only) | `wave_table_ptr` | LTBL=`$FF`→ RTBL=target | L493-494 |
| Wavetable arp (rel/abs note) | FREQ | `note`, `lastnote` | RTBL byte: `<$80` rel-up (+note), `$80` NOP, else abs `&$7F` | L497-504 |
| Wavetable embedded command ($F0-$FE) | per-cmd | `wave_table_ptr` | dispatches `_wave_command` | L486-487, L361-396 |
| Portamento up/down | FREQ | `freq` accum | `freq ± speedtable[idx]` | `_porta_up/down` L216-220 |
| Tone portamento | FREQ | `freq` accum, `vibtime` | slide `freq` toward `_freq(note)` by speed, clamp | `_toneporta` L222-238 |
| Vibrato (triangle, speedtable) | FREQ | `vibtime` ctr | LTBL=cmpvalue, RTBL=speed; `±speed` per half-cycle | `_vibrato` L240-256 |
| Speedtable realtime calc | (speed source) | `lastnote` | high-bit value → `(freq(n+1)-freq(n))>>shift` | `_speed_value` L206-214 |
| Pulse set | PW_LO/HI | `pulse_table_ptr`, `pulsetime` | LTBL b7 set: `pulse=((L&0xF)<<8)|R` | `_pulse_exec` L532-538 |
| Pulse sweep | PW | `pulse` accum, `pulsetime` ctr | signed RTBL delta, `pulse=(pulse+delta)&0xFFF` | L539-546 |
| Pulse table jump | (ptr) | `pulse_table_ptr` | LTBL=`$FF`→RTBL | L528-531 |
| Filter set (mode/res/cutoff) | $D416/17/18 | `_filterptr`, `_filtertime` | LTBL b7: mode=`&0x70`,ctrl=R; `$00`:cutoff=R | `_filter_routine` L329-359 |
| Filter sweep | cutoff | `_filtercutoff` accum, `_filtertime` ctr | `cutoff=(cutoff+R)&0xFF` | L349-355 |
| ADSR set | AD/SR | — | instrument `ad`/`sr`, or wavetable `SETAD/SETSR` | `_new_note_init` L419; L378-381 |
| Hard restart | AD/SR/gate | `adparam` | pre-note write `adparam` to AD/SR + gate `$FE` | `_get_new_notes` L566-570 |
| Gate / keyoff / keyon | CTRL b0 | `chan.gate` | `wave & gate` ($FE off, $FF on) | `_write_voice_regs` L582; L558-561 |
| Gateoff timer | CTRL gate | `gatetimer` ctr | gate releases when `tick==gatetimer` | L592, L617-625 |
| Tempo / funktempo | (timing) | `tick`, `tempo`, `funktable` | per-row frame count; funktable swing pair | `_play_channel` L603-607; L459-473 |
| Sequencer (orderlist) | (control) | `songptr`, `pattnum`, `repeat` | transpose/repeat/loopsong markers | `_sequencer` L270-296 |
| Pattern walk | (control) | `pattptr` | note,instr,cmd,data per 4-byte row | `_get_new_notes` L548-555 |
| Master volume | $D418 lo | `_masterfader` | `SETMASTERVOL` cmd | L394-396 |

GoatTracker's freq modulators (porta/vibrato) are **free-running accumulators on `chan.freq`**, reset
on note-on (`vibtime=0`); the wavetable arp **overwrites** freq from the note table. Speedtable is the
shared LFO/slide-rate data structure (`STBL`).

### 1b. SID-Wizard 1.94 (`pysidwizard/player.py`)

| Op | Lane | State | Params | Source |
|----|------|-------|--------|--------|
| Note→freq lookup | FREQ | `note`,`octave_shift`,`transpose` | `NOTE_FREQ_HI/LO[note+oct+trans]` | `_compose_sid_writes` L1974+ |
| WF-table walk (waveform + arp + detune, 3-byte rows) | CTRL/FREQ | `wf_pos`, `arp_speed_counter`, `wf_speed_counter` | row=(waveform, arp, detune) | `_tick_wf_table` L1629-1791 |
| WF arp (rel-up / NOP / abs / rel-down / chord) | FREQ | `wf_arp_pitch`, `wf_arp_absolute` | `$00-7E` rel-up, `$7F` chord, `$80` NOP, `$81-DF` abs, `$E0-FF` rel-down | L1732-1785 |
| WF detune column | FREQ_LO carry | `detune` | signed col-2 byte; ADC-with-carry into FREQ_LO (WRPITCH) | L1786-1789, `wf_pitch_carry` |
| WF arp-speed pacing | (ptr gate) | `arp_speed_counter` | dec-and-skip; reload `arp_speed&$3F` | L1663-1667 |
| WF jump / end | (ptr) | `wf_pos` | `$FE` jump abs, `$FF` end/freeze | L1689-1707 |
| Chord-table walk | FREQ | `current_chord`, `chord_pos` | rel pitches; `$7E` return, `$7F` loop | `_tick_chord` L1575-1625 |
| Vibrato (calc, triangle) | FREQ offset | `vibrato_offset` accum, `vibrato_freq_counter` | period=nibble·2, FREQMOD from EXPTAB lookup | `_tick_vibrato` L1318-1390 |
| Tone portamento | FREQ offset | `vibrato_offset` drift→0, `slide_vib=$83` | FREQMOD step toward target | `_tick_portamento` L1392-1438 |
| Pitch slide up/down | FREQ offset | `vibrato_offset`, `slide_vib=$81/$82` | `±FREQMOD` per frame | L1343-1356 |
| INCVIBR ramp | FREQ offset | FREQMOD grows by delay-byte | depth ramp from 0 | L1357-1366 |
| PW-table walk (set / sweep / jump / end) | PW_LO/HI | `pw_pos`, `pw_sweep_count` ctr | row=(mode, lo/delta, kbtrack) | `_tick_pw_table` L1793-1869 |
| PW set | PW | — | b7 set: `pw_hi=b&$7F`, `pw_lo=next` | L1830-1841 |
| PW sweep (16-bit pair) | PW | `pw_hi/lo` accum, `pw_sweep_count` | signed delta, carry hi↔lo | L1842-1869 |
| PW keyboard-track | PW_HI | `pw_kbd_track` | per-note EXPTAB adjust | L1838,1855 |
| Filter-table walk (set / sweep / jump / end) | $D416/17/18 | `filt_pos`, `filter_sweep_count` (global) | row=(mode+res, cutoff/delta, kbtrack) | `_tick_filt_table` L1871-1970 |
| Filter set | cutoff/res/mode | `filt_hi/lo` | b0-2 cutoff_hi=R, mode=`&0x70`, res=`&0x0F<<4` | L1932-1947 |
| Filter sweep (11-bit) | cutoff | `filt_hi/lo` accum | signed delta into 11-bit cutoff | L1948-1970 |
| Filter routing | $D417 lo | `voice_in_filter` per voice | OR of routing bits | `_emit_writes` L2106+ |
| Filter controller arbitration | (global) | `filter_controller_voice` | one voice owns the sweep | L1224-1230, L1895 |
| ADSR set + small-fx override | AD/SR | `*_override` | instrument AD/SR; `$2x/$3x/$5x/$6x` nibble override | `_apply_small_fx` L1107-1154 |
| Hard restart | AD/SR/CTRL test | `hr_timer` ctr | HR-ADSR + CTRL test bit `$08`/`$09`, hold tables | `_start_note` L1165; `_maybe_emit_pre_hr` L1515 |
| Gate / sync / ring (note-FX) | CTRL b0-2 | `ptn_gate`, `sid_ctrl` | `$7D/$7E` gate, sync-on/off, ring-on/off | `_apply_row` L950-978 |
| Legato (max-speed porta) | FREQ | `slide_vib=$83`, FREQMODH=$7F | instrument col `$3F` | L920-945 |
| Note-FX portamento pre-flag | FREQ | `slide_vib=$FF` | `$78` NPORTAM | L979-988 |
| Vibrato note-FX (amp/freq override) | FREQ | vibrato params | `$60-77` overrides vibrato byte | L989-1024 |
| Tempo / funktempo | (timing) | `tempo_left/right`,`tempo_toggle`,`speed_counter` | per-row frames; b7=straight vs swing | `_tick_voice` L530-551 |
| Multispeed (frame_speed≥2) | (timing) | `_multispeed_phase` | re-walk WF (+PW/filter if arp_speed b6/b7) per sub-frame | `_tick_voice_multispeed` L668-711 |
| Sequence (orderlist) walk | (control) | `seq_pos`, `pattern_row` | PlayPattern / Transpose / TempoOverride / Loop / End | `_advance_sequence` L795-826 |
| Master volume | $D418 lo | `filter_mode_vol` | fx `$Ax` | L1042-1044 |

SID-Wizard's freq is **recomputed each frame** from `base(note+arp+transpose) + vibrato_offset +
detune-carry`; the modulators write a separate `vibrato_offset` accumulator (free-running) while the WF
arp **overwrites** the base (note-reset). The 8-bit ADC-with-carry detune chain (WRPITCH) is a quirk
that only nudges FREQ_LO by ±1 — modelled as a carry flag, not a separate op.

### 1c. defMON (`pydefmon/defmon_player.py` + `docs/SPEC.md`)

defMON is the **lowest-level / most explicitly program-like** of the four: a generic **sidTAB** of 256
rows, each a bitmap-packed set of column writes, walked by a per-voice-per-layer **cascade** with
jump/delay control. Two sidcall layers per voice ⇒ a primitive instrument-VM.

| Op | Lane | State | Params | Source |
|----|------|-------|--------|--------|
| Cascade walk (sidTAB row fetch) | (dispatch) | `(row_idx, step_counter)` per voice×layer | DL byte = hold frames; `$80+` = STop | SPEC §3 cascade; `_apply_sidtab_row` L306 |
| Cascade JP redirect | (ptr) | `$1900,row==0`→`$1800,row` target | jump then linear from target+1 | SPEC §3 |
| WGh: set ctrl | CTRL | `ctrl_main` | waveform/gate/sync/ring byte | col L322 |
| WGl: ctrl EOR | CTRL | `ctrl_eor` | XOR mask applied **every frame** before emit | L326, `_emit_frame_writes` L749 |
| AD / SR set | AD/SR | `ad`,`sr` | direct byte | cols |
| TR: set note | FREQ | `current_note`, transpose buffer | b7=absolute else +transpose-buffer | L358 region |
| AF: set slide mode | FREQ | `slide_mode` | `$00` none / `$01-7F` porta / `$80-FF` active | col |
| Note→freq | FREQ | `current_note` | `NOTE_PITCH[note]` + per-voice detune (V0/1/2 +0/+1/+2) | `_pitch_slide_voice` L871 |
| Portamento | FREQ | — | one-frame adjacent-semitone step toward `note+slide_mode` | L908-940 |
| Active pitch slide | FREQ | 16-bit accum, rate LUT | `Y=slide_mode<<1` into rate table, integrate | L944-976 |
| GATE_N zero-slide | FREQ | accumulator | new note clears slide accum + mode | SPEC §3 |
| PW set | PW | `pulse_hi`, `pulse_lo` | `$YX` 12-bit, lo = `byte&$F0` | col PW |
| PW sweep (PS) | PW_LO | `ps_depth` | bit7 ADD/SUB, carry into hi, clamp+flip at $0F/$00 | `_ps_voice` L978-1052 |
| RE: resonance + routing | $D417 | — | 3-way dispatch | col |
| FV: filter mode + vol | $D418 | — | vol forced `$0F` | col |
| CP: cutoff extra | $D416 | `cutoff_extra` | added each frame | col |
| ACID: cutoff slide cmd | $D416 | 16-bit `acc`, `step`, opcode | b7=slide-vs-abs, b6=SBC-vs-ADC; saturate to floor | `_cutoff_slide_step` L815-869 |
| Arranger walk | (control) | per-step pattern index per voice | `$00` silent, `$01-7F` pattern, `$FF`(V1) jump+count | SPEC §2 |
| Pattern event | (dispatch) | duration nibble, GATE_A/B/N, ALT | arms slot_a/b sidcalls, note | SPEC §2 |
| Sub-frame multispeed | (timing) | `sub_frame_count` (CIA) | all NMIs run band+cascade; main-tick adds arranger | SPEC §3 |

### 1d. WEMUSIC / The Last Ninja (`reninja/engine.asm`, `docs/engine_annotated.txt`)

The most striking structural finding: WEMUSIC is **four near-identical add/subtract sweep engines**
sharing one template, plus a note-stream reader. The whole melodic+timbral surface is "a delay-gated
bidirectional 16-bit accumulator over a register lane."

| Op | Lane | State | Params | Source |
|----|------|-------|--------|--------|
| Note-stream read (16-bit cmd) | (control) | `$F0,X` read ptr | 2-byte word; `$0000`=end→reload loop ptr | `fetch_note` $B937, annotated L263-291 |
| Per-voice pattern loop | (control) | `$F6,X` restart ptr | `$0000` reloads from restart ptr | $B95A-B967 |
| Note decode | FREQ + instr | `$B5E2/E3` | hi 5 bits→instr/arp idx, lo 6 bits→note; `$3F`=tie/hold | `decode_cmd` $B952, L292-304 |
| Instrument select | (all params) | `$FC/FD` ptr, `$B5DC,X` | bit-weighted addr build, $2F-byte stride | `play_note` $B9A6, L305-324 |
| Note→freq lookup | FREQ_LO/HI | `note` | `freqtab_hi[note]`, `freqtab_lo[note]` | L106-107, $BA56 |
| **Sweep engine ×4 (shared template)** | freq osc1, freq osc2, PW1, PW2 | per-sweep: delay ctr, direction flag, up-len, down-len | add delta (instr off $0A/$0B) until up-len, flip, sub delta (off $0C/$0D) until down-len | `freq_sweep_1` $B713, L215-251; engine.asm L274-490 |
| └ vibrato (= freq_sweep_1) | FREQ osc1 | `$B630,X` dir, `$B624/B62A,X` ctrs | bidirectional accum on `$B5EE/B5F4` | $B713-B74A |
| └ detuned-osc (= freq_sweep_2) | FREQ osc2 | parallel state | second oscillator, alternated each frame | $B799 |
| └ PW sweep ×2 (= pw_sweep_1/2) | PW | parallel state | bidirectional accum on PW pair | $B829/$B8AF |
| Two-osc detune toggle | FREQ (which osc emits) | `$B60C,X` phase, `$B60F,X` countdown | alternates osc1/osc2 to SID each frame | `sid_output` $BB6C-BB6F, L342-366 |
| Per-voice finetune | FREQ_LO | `$B654,X` | added to freq-lo at output (ADC) | $BB75, $BB9C |
| Octave offset | FREQ | instr off $05/$18 | `note += 12·n` per osc | L316-317 |
| Waveform/gate set | CTRL | `$B606/B609,X` | from instr off $16/$29; hold-tie skips retrigger | L321, $BB8F |
| ADSR set | AD/SR | `$B5E8/B5EB,X` | instr off $00/$01 | L319, $BB60 |
| Filter / volume (voice 3 only) | $D417/18 | `$B657,X & 1` toggle | res=$F0, vol+mode=$08 | $BBAA-BBBC, L367-375 |
| Tempo tick | (timing) | `$B612,X` ctr (reload 6) | 6 frames/step | $B6DA-B6E1 |
| Arp/effect sub-counter | (effect idx) | `$B615,X` 5-bit (AND $1F) | drives arp/instrument step | $B6E4-B6ED |

WEMUSIC's sweeps are **pure free-running** (no note-reset; they ping-pong on their own up-len/down-len
phases) — the opposite end of the phase-model spectrum from GoatTracker's note-reset vibrato.

---

## 2. The UNION op-set — is it bounded and small?

**Headline: YES. The union is bounded, small, and heavily overlapping.** Collapsing the four drivers'
operations by *what computation they perform on which register lane* (not by their format) yields
**~18 distinct operation classes**. Every driver implements a large, overlapping subset.

Legend: ● has it natively; ○ degenerate/special case; — absent.

| # | Union op (computation → lane) | GT | SW | dM | WE | overlap |
|---|-------------------------------|----|----|----|----|---------|
| 1 | **Note→freq table lookup** (note index → FREQ_LO/HI via dual LUT) | ● | ● | ● | ● | **4/4** |
| 2 | **Set register from data** (waveform/ctrl, AD, SR, PW, cutoff, res, mode/vol) | ● | ● | ● | ● | **4/4** |
| 3 | **Table walk with delay** (ptr++ paced by a per-row frame countdown) | ● | ● | ● | ● | **4/4** |
| 4 | **Table jump / end** (`$FF/$FE` → reload ptr or freeze) | ● | ● | ● | ● | **4/4** |
| 5 | **Arpeggio** (rel-up / NOP / abs / rel-down note offset over FREQ) | ● | ● | ○TR | ●arp | **4/4** |
| 6 | **Vibrato** (triangle LFO add/sub to FREQ, counter-paced) | ● | ● | — | ● | 3/4 |
| 7 | **Portamento / tone-porta** (slide FREQ toward target, clamp/snap) | ● | ● | ● | ●(via sweep) | **4/4** |
| 8 | **Pitch slide up/down** (free accumulate ±rate into FREQ) | ● | ● | ● | ● | **4/4** |
| 9 | **Pulse set** (latch PW_LO/HI from data) | ● | ● | ● | ● | **4/4** |
| 10 | **Pulse sweep / PWM** (signed delta accumulate into PW, carry hi↔lo, clamp+flip) | ● | ● | ● | ● | **4/4** |
| 11 | **Filter cutoff set** ($D416 + mode/res) | ● | ● | ● | ○(voice3) | **4/4** |
| 12 | **Filter cutoff sweep** (signed delta accumulate into cutoff, saturate) | ● | ● | ● | — | 3/4 |
| 13 | **Filter routing / controller** (voice→filter bit, one controller) | ● | ● | ● | ○ | **4/4** |
| 14 | **ADSR set + override** (AD/SR, small-fx nibble override) | ● | ● | ● | ● | **4/4** |
| 15 | **Gate / sync / ring / test** (CTRL bit ops; keyon/keyoff/gateoff-timer) | ● | ● | ● | ● | **4/4** |
| 16 | **Hard restart** (pre-note ADSR + test-bit, hold tables N frames) | ● | ● | ○ | ○ | 2.5/4 |
| 17 | **Sequence / arranger / pattern walk** (orderlist→pattern→row + jump/loop/transpose/repeat) | ● | ● | ● | ● | **4/4** |
| 18 | **Tempo / funktempo / multispeed** (per-row frame count, swing pair, sub-frame ratio) | ● | ● | ● | ● | **4/4** |

**Overlap is extreme.** Of the 18 op classes, **13 are present in all four drivers** and the
remaining 5 are present in 3/4 (the absences are *degenerations*, not new ops: WEMUSIC has no
filter-sweep because Last Ninja barely filters; defMON's arp is the `TR` absolute/relative note write
rather than a cyclic table; defMON/WEMUSIC fold hard-restart into ctrl-EOR / waveform retrigger
rather than a distinct test-bit phase). **No driver contributes an op the others can't express.**

Furthermore, **the per-driver tables collapse hard internally too**: WEMUSIC's entire melody+timbre
surface is *one* sweep template instantiated 4×; SID-Wizard's WF/PW/Filter walks are *the same*
three-byte (mode, value, third-col) table-walk machine over three different lanes; defMON's whole
instrument layer is *one* bitmap-column sidTAB walk. The drivers are not 4 different grammars — they
are 4 parameterizations of the same handful of primitives over the same 25 registers.

**Conclusion: the thesis holds.** The union op-set is bounded (~18 classes), small, and the cross-
driver overlap is ~85-90%. This is strong evidence for a bounded universal grammar.

---

## 3. Generalization to a COMPLETE primitive set

The 18 union ops decompose into a tiny set of **general computational primitives over the register
lanes**. Completeness argument: the SID is a *finite* machine (25 byte registers); any per-frame
playroutine is a finite program that, each frame, reads per-tune data + its own state and writes some
registers. The primitives below are exactly {read/index data, conditional, counter, lane arithmetic,
control flow} — i.e. a Turing-complete-over-finite-state primitive set restricted to register lanes.
Anything a 6502 playroutine computes per frame is a composition of these; there is **no operation a
table-driven SID player can perform that is not one of these** (verified: every op in §1 maps to a
composition below, with no escape hatch).

**P1. SET(lane, value)** — latch a constant/data byte into a register lane (or a ZP shadow of it).
Covers union ops 2, 9, 11, 14, 15(set form). Lanes: FREQ_LO/HI, PW_LO/HI, CTRL, AD, SR, $D416/17/18,
and the bit-field sub-lanes (waveform nibble, gate bit, filter-mode bits).

**P2. INDEX(table, ptr) → value** — read a per-tune table at a pointer. Covers note→freq LUT (1),
speedtable/rate-LUT lookups, instrument-record indexing. The table is *program data*, not grammar.

**P3. COUNTER(state, stride, reload, loop)** — a frame-paced countdown/up-count with a reload value
and a wrap/loop action. This is the universal "pacing" primitive: tempo ticks (18), table-walk delays
(3), arp-speed pacing, vibrato period, sweep up-len/down-len, sub-frame multispeed. Every "ctr" column
in §1 is one COUNTER.

**P4. ACCUM(lane, ±delta, [clamp, wrap, carry])** — add or subtract a (constant or data) delta into a
multi-byte register accumulator, with optional carry between bytes, saturation clamp, and direction
flip. This is the single primitive behind **all** continuous modulation: vibrato (6), portamento (7),
pitch slide (8), PW sweep/PWM (10), filter sweep (12), and WEMUSIC's 4 sweep engines. The cross-driver
PWM/filter quirks (carry-in off-by-one, 11-bit vs 12-bit split, floor saturation) are all *parameters*
of ACCUM, not new ops.

**P5. PTR-WALK(ptr, +stride | jump | end)** — advance a table pointer by a stride, or follow a jump
(`$FE`/`$FF`/`$0000` → reload from data), or halt/freeze. Covers table jump/end (4), cascade JP (dM),
note-stream loop (WE), sequence/pattern/orderlist walk (17). Control flow over per-tune data.

**P6. SELECT/COND(predicate) → branch** — choose an action from a marker byte's range/bits. Covers
every "`if byte ≥ $80` then set-mode else sweep-mode", arp dispatch (`<$80`/`$80`/`$81-DF`/`$E0+`),
gate/sync/ring note-FX dispatch (15), the sidTAB bitmap "is this column present", the cascade
hold/STop/JP decision. This is the *decode* primitive.

**P7. ARITH(lane, op, operand)** — small per-lane arithmetic that isn't pure add: XOR (defMON
`ctrl_eor` flicks CTRL bits each frame), AND-mask (gate masking `wave & gate`; transpose `&$7F`),
shift (speedtable realtime calc `>>shift`; rate-LUT `slide_mode<<1`), nibble pack/unpack (PW `$YX`,
ADSR override, resonance nibble), add-octave (`note += 12n`). General byte arithmetic over a lane.

Two **phase-model attributes** parameterize P3/P4 (not new primitives):
- **note-reset** (re-seed accumulator/counter to 0 on note-on): GoatTracker vibrato, SID-Wizard WF arp,
  defMON GATE_N slide-clear.
- **free-running** (accumulator persists across notes, ping-pongs on its own phase): WEMUSIC sweeps,
  GoatTracker speedtable LFO between resets, defMON PS/ACID.

And one **cross-voice wiring** attribute (P1/P6 applied to a shared lane): sync/ring read the *adjacent*
voice's oscillator (CTRL bits, wired in the chip); the global filter has one *controller voice* and a
routing-bit OR across voices ($D417 low nibble).

**Completeness:** {SET, INDEX, COUNTER, ACCUM, PTR-WALK, SELECT, ARITH} + {note-reset/free-running
phase, cross-voice wiring} expresses every op in §1 (mapping given inline). Because they are general
computation (index + branch + counter + lane arithmetic + control flow) over a *finite* register file,
they express **any** finite per-frame playroutine — the only irreducible remainder is the per-tune
data the primitives read (the tables, note lists, params = the music). **No residual escape hatch is
needed or possible**: a non-zero residual means a missing *parameter* of one of these 7 primitives
(e.g. an unmodelled clamp variant), found and added, never an arbitrary-write patch.

---

## 4. Proposed VM instruction set (the decoder's ISA)

The decoder is a 3-voice (+ global) VM stepped once per frame. State = per-voice register shadows
(FREQ_LO/HI, PW_LO/HI, CTRL, AD, SR), global ($D416/17/18), plus the program counters/pointers/
accumulators the instructions below own. A **per-tune program** is a set of tables + note lists + a
control track; each instruction names a lane, a phase model, and its parameters. ISA (concise):

```
; ---- pacing / control ----
TEMPO       reload, [swing_partner]      ; P3 row-tick counter; swing = funktempo pair
SUBFRAME    ratio                        ; P3 multispeed: N table-runs per emit, 1 emit
SEQ.WALK    voice, orderlist            ; P5 orderlist→pattern; markers: PLAY/LOOP/END/TRANSPOSE/REPEAT
PAT.WALK    voice, pattern              ; P5 row reader; row = (note, instr, fx, data)
TABLE.WALK  ptr, stride, delay_col      ; P3+P5 generic instrument-table step (WF/PW/Filter/sidTAB)
TABLE.JUMP  marker, target              ; P5 $FE/$FF/$0000 reload-or-freeze
COND        operand, ranges -> action   ; P6 marker decode (the dispatch primitive)

; ---- pitch (FREQ_LO/HI lane) ----
NOTE        note_index                  ; P2 freq = LUT[note(+oct+transpose)]; per-voice detune add
ARP         {rel_up|nop|abs|rel_down}    ; P5/P2 table-driven note offset; phase=note-reset
PORTA       target, rate                 ; P4 slide toward target, clamp/snap; rate from speedtable
SLIDE       dir, rate                    ; P4 free ±accumulate into FREQ; phase=free-running
VIBRATO     period, depth, phase         ; P4+P3 triangle add/sub; phase=note-reset|free-running
DETUNE      offset | osc-toggle          ; P1/P7 finetune add to FREQ_LO; or 2-osc alternation (WE)

; ---- pulse (PW lane) ----
PW.SET      hi, lo                       ; P1
PW.SWEEP    delta, [cycles|clamp|flip]   ; P4 PWM accumulate, carry hi<->lo, clamp+flip; phase=free
PW.KBDTRACK adj                          ; P7 per-note PW_HI adjust

; ---- filter (global $D416/17/18) ----
FILT.SET    mode, res, cutoff            ; P1 (+ band bits $D418, res nibble $D417)
FILT.SWEEP  delta, [cycles|floor]        ; P4 accumulate cutoff, saturate
FILT.ROUTE  voice_bit                    ; P1/P7 $D417 low-nibble OR; one CONTROLLER voice owns SWEEP

; ---- envelope / control (AD/SR/CTRL lanes) ----
ADSR.SET    ad, sr, [nibble_override]    ; P1/P7
WAVE.SET    ctrl_byte                    ; P1 waveform nibble (gated by gate mask)
GATE        on|off|timer                 ; P6/P7 CTRL b0; gateoff-timer = P3 counter
CTRL.BIT    sync|ring|test [on|off|eor]  ; P7 CTRL b1/b2/b3; EOR = per-frame XOR (defMON WGl)
HARDRESTART adparam, hold_frames         ; P1+P3 pre-note ADSR + test bit, hold tables N frames
```

Per-lane application notes the next phase must honor:
- **FREQ is computed, not stored**: `freq = NOTE(base) [overwritten by ARP] + Σ(VIBRATO/SLIDE/PORTA
  offsets) + DETUNE(+carry)`. Order matters (arp overwrites base; vibrato adds; detune carries into LO).
- **Phase model is per-instruction**: VIBRATO/SLIDE/PW.SWEEP/FILT.SWEEP carry a `note-reset` vs
  `free-running` bit (GoatTracker speedtable & WEMUSIC sweeps = free-running; SID-Wizard/GT vibrato &
  arp = note-reset). The decoder must NOT assume one.
- **Cross-voice wiring**: SYNC/RING bits couple a voice to its neighbor's oscillator (chip-level, no
  data); the global filter has exactly one CONTROLLER voice plus a routing-bit OR.
- **Sub-frame**: multispeed runs TABLE.WALK/SWEEP `ratio`× per frame but emits one register band; the
  arranger/PAT.WALK runs only on the main tick.

This ISA is the union+general primitives, sized so P1 (decompiler) can target NOTE/ADSR/WAVE/TABLE.WALK,
P2 the FREQ stack, P3 the PW/FILT/global stack — exactly the build phases in `sid_player_decompiler.md`.

---

## 5. Gaps / risks (for zero residual on un-disassembled drivers)

1. **The 8-bit ADC-with-carry quirks don't generalize as ops — they generalize as a flag.** SID-Wizard's
   WRPITCH `adc DETUNER` inherits the 6502 carry from the WF-walk branch and nudges FREQ_LO ±1
   (`wf_pitch_carry`, L1646-1789); defMON's pitch/PW/ACID slides all have a `carry_in=0` / `-1` SBC
   off-by-one (SPEC §3, `_cutoff_slide_step` CLC-then-SBC). These are *modeled* in the bit-exact ports,
   but they are emergent from 6502 flag plumbing, not declared operations. **Risk:** an un-disassembled
   driver's modulation arithmetic may carry a *different* flag idiom; ACCUM must expose carry-in /
   saturation / clamp as explicit parameters or these become 1-LSB residuals (which the
   `residual-zero` gate will not tolerate).

2. **Hard-restart is the least-overlapping op (2.5/4).** GoatTracker (`adparam` pre-write) and
   SID-Wizard (`hr_timer` test-bit phase) do an explicit HR; WEMUSIC/defMON fold retrigger into
   waveform/ctrl-EOR. Hubbard and especially **Galway** are famous for idiosyncratic gate/hard-restart
   and **sample/digi** tricks (Galway's arpeggiated "Galway noise", $D418 volume-register sample
   playback) that are NOT in any of these four drivers. **Risk:** digi/$D418-PCM and unusual HR are a
   genuine coverage gap; they may need a CTRL.BIT/SET-at-high-rate characterization that none of the
   four exercises. (Cross-ref the `digi-detection` memory: $D418/Mahoney, SounDemoN, PWM digis.)

3. **Free-running phase recovery is the hard synthesis problem, not the op-set.** WEMUSIC's sweeps and
   GoatTracker's speedtable LFO have **no note-reset**, so the decompiler must recover the accumulator's
   *initial phase and direction* at the trace's start — an op that's trivially present but whose
   *parameter* (where in the ping-pong it began) is hidden. This is the lane that "broke everything"
   (P2 in the build plan). The op-set is complete; the **synthesis search** over free-running phase is
   the residual risk.

4. **Cross-voice sync/ring and the single filter-controller are global couplings.** SID-Wizard's
   `filter_controller_voice` arbitration (last STRTSND wins, L1224) and the reverse voice-order tick
   (L498) are timing-sensitive; sync/ring read the neighbor oscillator. An un-disassembled driver with
   a *different* controller-arbitration or per-voice filter (rare but possible) would need the
   FILT.ROUTE/CONTROLLER model parameterized, not hard-coded to "one controller."

5. **defMON proves the op-set can be *arbitrarily low-level* (a near-general instrument VM).** Its
   sidTAB + 2-layer sidcall cascade is itself a tiny programmable machine; a tune can express modulation
   the other three drivers bake in. **This is reassuring for completeness** (defMON ≈ exists as a proof
   that "general computation over the chip" is the right altitude) **but a risk for the *encoder***: a
   defMON-class or hand-coded driver's per-tune "program" can encroach on what we'd call grammar. The
   line between op (grammar) and parameter (program) must be drawn at "general primitive vs. specific
   table content," exactly as §3 does — and the gate is residual=0, which forces the line to be correct.

6. **Tempo/timing edge cases** (funktempo swing, multispeed sub-frame ratios, defMON's CIA NMI ratio,
   mid-frame stop-finishing) differ per driver and are easy to get 1-frame-off. These are P3 COUNTER
   parameters, fully expressible, but a frequent source of off-by-one residual on first synthesis.

**Bottom line on the thesis:** the evidence **supports** a bounded, small (~18 op / 7-primitive),
heavily-overlapping (≈85-90%) universal op-set with no escape hatch. The residual risk is **not** a
missing op-class; it is (a) **parameterization fidelity** (carry/clamp/saturation variants of ACCUM)
and (b) **synthesis search** (recovering hidden free-running phase + per-tune programs), plus a real
**digi/sample coverage gap** for Galway-class tricks absent from all four drivers. The op-set is the
right altitude; completeness on the un-disassembled corpus is a synthesis + parameterization problem,
to be driven to residual 0 per the build plan.

---

## Real hand-coded corpus drivers (Hubbard / Follin / Galway / Whittaker / Tel / Gray)

**Status: ANALYSIS / DESIGN ARTIFACT (Phase 0a widening, 2026-06-17).** The §1–§5 inventory above was
built from FOUR *tracker/engine* sources, only one of which (WEMUSIC=reninja, Daglish/Crowther) is a
hand-coded driver a corpus composer actually used. This section pulls in the **bespoke hand-coded
drivers** the corpus composers wrote, extracts their op-sets from real disassemblies, and tests
whether the bounded-op-set thesis survives. **It does not.** Two of the six hand-coded drivers carry a
genuine *arbitrary-6502 escape hatch* (Galway EXEC, Follin SEND) that does not decompose into the 7
primitives. The melodic/timbral op-set DOES stay bounded; the *control layer* does not.

### Sources actually read (and provenance)

| Driver | Source read | Where | Status |
|--------|-------------|-------|--------|
| **Rob Hubbard** | `Hubbard_Rob_Monty_on_the_Run.asm` (orig. disasm Anthony McSweeney, ACME port dmx87) + the 1xn.org commented disassembly `rob_hubbards_music.txt` | github.com/realdmx/c64_6581_sid_players; 1xn.org | **fully read, cross-checked** |
| **Tim Follin** | `DRIVE.SRC` "TIMS MUSIC PROGRAM … REWRITTEN IN SWEET 6502 BY STEPHEN RUDDY, VERSION IV, BLACK TURD 1987" — the real C64 SID driver | github.com/KevEdwards/TimGeoffFollinMusicDevDiskArchive (`_Fully_Recovered/06 MUSICK1/.../6502/DRIVE.SRC`) | **fully read** (this is the *actual* Follin C64 engine; the clarets.org writeup that search surfaces is the Atari-ST YM driver, NOT this) |
| **Martin Galway** | `Galway_Martin_Rambo_Loader.asm` + `Galway_Martin_Arkanoid.asm` (reversed dmx87 / Ice Team) | github.com/realdmx/c64_6581_sid_players | **fully read** |
| **David Whittaker** | `Whittaker_David_Panther.asm` (reversed dmx87) | same repo | **fully read** |
| **Jeroen Tel / Maniacs of Noise** | `Tel_Jeroen_Cybernoid2.asm` — MoN player coded by Charles Deenen, "the most advanced music player ever written for the c64" | same repo | **fully read** |
| **Fred Gray** | `Gray_Fred_Mutants.asm` (work-in-progress reverse) | same repo | **fully read** (raw-byte disasm, labels machine-named) |

These are ACME/SWEET source, NOT bit-exact ports validated in VICE like §1's four. So op claims below
are grounded in *read disassembly* (cited by label), but their byte-exactness is the reversers' claim,
not independently re-verified here. Could not find: a Follin tune driven by a DIFFERENT driver (all his
C64 work used Ruddy's DRIVE); a validated Galway *Arkanoid digi* round-trip (sample timing unverified).

### 6a. Rob Hubbard (`Monty_on_the_Run.asm`)

The ancestor. A 3-voice, song→track→pattern walker with per-instrument fx flags. **Every op fits the
existing primitives.**

| Op | Lane | State | Params | Source | Maps to |
|----|------|-------|--------|--------|---------|
| Note→freq lookup | FREQ_LO/HI | `notenum` | `notefreqsl/h[note·2]` | `getpitch` L261-279 | P2 INDEX |
| Triangle vibrato (bit-synth, no table) | FREQ | `counter`, `oscilatval` | `counter&7`, `cmp#4/eor#7` → 0,1,2,3,3,2,1,0; depth = `(freq(n+1)-freq(n))>>vibrdepth` summed osc× | `vibrato` L367-436 | P3 COUNTER + P7 ARITH(eor/lsr) + P4 ACCUM |
| Pulse sweep / PWM (ping-pong $08xx↔$0exx) | PW | `pulsedir`, `pulsedelay` | `pw ± pulsespeed`, flip at $0e/$08 | `pulsework`/`dumpulse` L446-501 | P4 ACCUM + P6 SELECT(flip) |
| Portamento up/down | FREQ | `savefreqlo/hi` accum | `freq ± (portaval&$7e)`, bit0=dir | `portamento` L507-538 | P4 ACCUM |
| "Drums" (HR transient) | FREQ_HI + CTRL | `savefreqhi` | rapid `dec freqhi`; noise on 1st vbl then ctrl-wave | `drums` L549-582 | P4 ACCUM + P6 SELECT |
| "Skydive" (slow drop) | FREQ_HI | `savefreqhi` | every 2nd frame `dec freqhi` | `skydive` L591-605 | P3 COUNTER + P4 ACCUM |
| Octave arpeggio | FREQ | `counter&1` | note vs note+12, look up freq | `octarp` L613-637 | P5/P2 (note offset) |
| Gate / keyoff / release | CTRL b0 + AD/SR | `voicectrl`, `lengthleft` | `&$fe` release, zero AD/SR | `soundwork` L343-359 | P1 SET + P6 |
| Append/legato (no retrigger) | CTRL gate mask | `appendfl` | bit6 of len byte → `ctrl & appendfl` | L236-282, L303 | P7 AND-mask |
| ADSR/PW/ctrl set from instr | AD/SR/PW/CTRL | instr+0..4 | direct | `getinstrument` L290-320 | P1 SET |
| Tempo (global speed ctr) | timing | `speed`,`resetspd` | per-row frame count | `contplay` L146-151 | P3 COUNTER |
| Song→track→pattern walk + `$ff`/`$fe` | control | `posoffset`,`patoffset` | loop/stop markers | `getnewnote` L181-200 | P5 PTR-WALK |

Hubbard has **no filter** in this tune (only sets $D418 volume), **no sync/ring**, and his
"signature" effects (skydive, octarp, bit-synth vibrato) are *parameterizations of ACCUM+COUNTER+
SELECT*, exactly the inventory's prediction. **Hubbard: fits, contributes no new op.** The one notable
wrinkle is the vibrato depth being computed as a *runtime semitone-width* `(freq(n+1)-freq(n))>>depth`
— same trick GoatTracker's speedtable uses (P7 shift), not new.

### 6b. Tim Follin / Stephen Ruddy (`DRIVE.SRC`, "VERSION IV")

A per-voice **bytecode interpreter**: each voice has a PC (`PC_A/B/C`), a 19-entry jump table
(`JUMPA_LO/HI`), and a GOSUB/RETURN call stack. The modulators (vibrato, PWM, portamento) are
free-running accumulators driven by SETVIB/WOBBLE/PORT-armed params. Most ops fit — **but two do not.**

| Op | Lane | State | Params | Source (label) | Maps to |
|----|------|-------|--------|----------------|---------|
| Note→freq lookup | FREQ | `OLDA`/`NOTES_LO/HI[note]` | per-note table | `OKA`/`NOTES_LO` | P2 INDEX |
| Vibrato (delay/rate/limit/dir, free-run triangle) | FREQ | `AV_DEL/DEL1/DIR/LIM/RATE`, `AV3+1/AV4+1` self-mod | bidirectional accum, ping-pong on LIM1/LIM2 | `AVIBON`–`AV8` | P3+P4 ACCUM (free-running) |
| Portamento (slide toward target) | FREQ | `PORTA`,`OLDA`,`TARGETA` | step toward `TARGETA`, snap | `AV9`/`GO_UPA` | P4 ACCUM |
| PWM "wobble" (ping-pong PW $08↔$0e per RATE) | PW | `PWMA`,`AMOD`(dir self-mod) | `pw ± RATEA`, flip at $64/$9b (#100/#155) | `GOA`/`PRATTA`/`CHAN1A` | P4 ACCUM + P6 flip |
| Filter sweep (cutoff drift, claimed by one voice) | $D416 | `CUTOFF`,`FILTRATE`,`FILTCHAN` | `cutoff ± rate`, one CLAIM voice | `MUSIC`/`A_CLAIM`/`A_FILTER` | P4 ACCUM + P1; controller-arb |
| Effect re-trigger (one-shot freq blip) | FREQ + gate | `AEF_TIME/WAIT/GATE/FREQ` | timed override freq + gate | `AT0A`/`A_EFFECT` | P3 COUNTER + P1 |
| Transpose | FREQ | `TRANA` | add to note index | `A_TRANS` | P7 add |
| Tempo / note duration | timing | `COUNA`,`ENDITA` | per-step frame count | `CHAN1A`/`A_END` | P3 COUNTER |
| Pattern walk + **FOR/NEXT loop** | control | `REPA`,`RESA` (loop-ptr+count) | `A_FOR`/`A_NEXT` bounded repeat | `A_FOR`/`A_NEXT` | P5 PTR-WALK + P3 |
| **GOSUB / RETURN (subroutine call stack)** | control | `GOSA` (saved PC) | call subtrack, return | `A_GOSUB`/`A_RETURN` | P5 PTR-WALK (stack) — *new sub-case, see below* |
| **`A_SEND` raw register-write stream** | **ANY $D4xx** | — | reads (reg-index, value) pairs from the track and writes `value → SID,X` until a `$FF` terminator | `A_SEND`/`ATT4`: `lda(PC),Y → tax; … sta SID,X; cmp #255; bne ATT4` | **ESCAPE HATCH — no clean primitive** |

**Follin breaks the bound — but narrowly and at the control layer, not the modulation layer.** The
`A_SEND` op (`ATT4` loop) is *exactly* the "arbitrary write-this-byte-to-that-register" patch the §3
completeness argument asserts "is not needed or possible." It is a general SID-register poke stream
embedded in the per-tune data: the tune can write *any* value to *any* of the 25 registers, in sequence,
with no semantic structure the decoder can recover as NOTE/VIBRATO/etc. To a decompiler, an `A_SEND`
block is opaque — it is P1 SET over an *arbitrary, data-chosen lane*, which is the one thing the bounded
ISA deliberately forbade (its SET targets are fixed named lanes). The GOSUB/RETURN stack is *also* not
in the §4 ISA (which has SEQ.WALK/PAT.WALK but no recursive call stack), though that one is a benign
P5 extension (bounded-depth pointer stack). **Net for Follin: +1 genuine new op (raw register-poke
stream) + 1 ISA extension (call stack).**

### 6c. Martin Galway (`Rambo_Loader.asm`, `Arkanoid.asm`)

Galway's is the most *program-like* of the melodic drivers: a per-voice opcode interpreter (`>=$c0` =
opcode via `v1ops` jump table, `jmp $0000` self-mod dispatch), with CALL/RET/REPEAT/NEXT, instrument-
record copies (SETFQ/SETPM/INSTR5/ISET/CISET), and a **multi-segment piecewise accumulator** for both
freq and pulse.

| Op | Lane | State | Params | Source | Maps to |
|----|------|-------|--------|--------|---------|
| Note→freq lookup | FREQ | `freqsl/h[note]` | per-note | `.other` L243-248 | P2 |
| **4-stage piecewise freq adder** | FREQ | `f1c..f4c` ctrs, `fq1addl/h..fq4addl/h` | run adder N, each `freq += deltaN` for `fNc` frames, then next | `.freq`/`L_2313`-`L_2357` L587-694 | P3 COUNTER ×4 + P4 ACCUM ×4 (chained) |
| **2-stage piecewise pulse adder** | PW | `p1c/p2c`, `pm1addl/h`,`pm2addl/h` | same shape on PW | `.dopulse` L531-581 | P3+P4 chained |
| Arpeggio (offset table walk) | FREQ | `fq3time`,`fqdelay`,`fq1addl,x` | `note + fq1addl[idx]` cyclic | `.arp` L609-623 | P5/P2 |
| Hard-restart / release (test-bit phase) | CTRL b3 + AD/SR | `hrtime`,`rlsc` | test bit `$08`, hold N, clear regs | `.sound` L475-515 | P1 SET + P3 COUNTER |
| Instrument-record copy ops | AD/SR/PW/CTRL/freq/pulse tables | `cinstr`,`cipm`,`cifq` | SETFQ(14B)/SETPM(10B)/INSTR5(5B)/ARP(10B) bulk copies | `op_c2/c6/c8/d8_v1` L362-405 | P1 SET (bulk) |
| CALL / RET / REPEAT / NEXT / TRANS / CALLT | control | `V1SP` stack, `repstack` | subroutine + counted loop + transpose | `op_c0/ca/cc/ce/da/dc_v1` L312-452 | P5 PTR-WALK (stack) |
| **EXEC (`$D4`/`$D8`): `jmp (PARAM1)`** | **ANY** | — | execute arbitrary 6502 at a tune-supplied address | `op_d4_v1` L414-421; Arkanoid `inst_D8_v3` `jmp ($00EF)` L2024 | **ESCAPE HATCH — arbitrary code** |
| ($D418 sample playback — Arkanoid only) | $D418 | sample stack ptrs | 4-voice digi-drum; OUT OF SCOPE | Arkanoid `PSample3`/`inst_60_sample` L5436+ | digi (out of scope) |

**Galway breaks the bound, harder than Follin.** The `EXEC` op (`op_d4`=`jmp(PARAM1)`, and Arkanoid's
`$D8`=`jmp($00EF)`) executes **arbitrary 6502 machine code at a tune-chosen address**. This is not a SID
op at all — it is a fully general escape into the CPU. In Rambo it is used for a Morse-code effect and
`inctrans`; in Arkanoid `exec01/exec02` poke `$D417`/a temp and zero the filter. A decompiler cannot in
general recover what an EXEC body does without disassembling+simulating *its* code — there is no bounded
op it reduces to. **This is the decisive counterexample to "no escape hatch is needed or possible."**
(The 4-stage freq / 2-stage pulse adders, by contrast, are fine: they are just ACCUM chained with a
COUNTER per segment — a piecewise-linear envelope, a *parameterization*, not a new op.) Galway's famous
non-digi sound (the multi-segment pitch/pulse sweeps + arps) is fully inside the op-set; only EXEC and
the (out-of-scope) Arkanoid $D418 digi break it.

### 6d. David Whittaker (`Panther.asm`)

Command-byte engine (`CommandTable` $80-$93: waveform-set, ADSR, duration, tempo, ringmod, sync, vol,
stop). Modulation is a triangle vibrato + 2-segment freq slide + bidirectional PW sweep + relative-arp
tables. **All ops fit.** Notably *includes sync/ring* (`cmd_RingTri`=$14, `cmd_SyncSquare`=$42), which
Hubbard lacked.

| Op | Lane | State | Source | Maps to |
|----|------|-------|--------|---------|
| Note→freq lookup | FREQ | `NoteFreqsL/H[note·2]` | `_noarp` L885-892 | P2 |
| Triangle vibrato (rate/depth/dir, mid-point ping-pong) | FREQ | `B1A`(rate),`B1B`,`B1C`(accum),`B1D` dir bit6 | `L_9537`-`L_958C` L901-984 | P4 ACCUM + P6 flip |
| 2-segment freq slide (add or sub by bit2) | FREQ | `B07/B08` accum, `B0D/B0E` deltas, `B0F` ctr | `L_95C0`/`L_9608` L986-1054 | P3 COUNTER + P4 ACCUM |
| PW sweep (bidirectional, limits B1E/B1F) | PW | `PWL`,`B20`,`B21` dir | `L_962C`/`L_9654` L1056-1100 | P4 ACCUM + P6 flip |
| Arpeggio (relative-offset table, `$88`=loop, `$54`=reset) | FREQ | `ARP`/`ARP2` ptr | `SoundUpdate` L843-877; `ArpTable` L1137 | P5 PTR-WALK + P2 |
| Waveform / sync / ring set | CTRL | `WAVE` | `csetwave` L451-477 | P1 SET |
| ADSR / duration / tempo / vol set | AD/SR/timing/$D418 | — | `padsr`/`pdur`/`ptempo`/`L_93DF` | P1 SET / P3 COUNTER |
| `$88` set-3-bytes-to-B1E/1F/20 (PW limits) | PW-sweep params | — | `L_9297` L419-449 | P1 SET |
| Sequencer (track→pattern, `$88` wrap) | control | `PAT`,`TRACK`,`B03/B05` | `L_9304` L500-550 | P5 PTR-WALK |

No EXEC, no raw-poke stream. **Whittaker: fits, and *adds sync/ring coverage* the §1 four under-exercised
(only SID-Wizard had note-FX sync/ring).** Sync/ring are P1 CTRL-bit SET + the cross-voice wiring
attribute — already in §3, now confirmed used by a real composer's driver.

### 6e. Jeroen Tel / Maniacs of Noise (`Cybernoid2.asm`, Deenen player)

The richest modulation surface of all six, and a stress-test for ACCUM parameterization. It piles on:
tone-arp, vibrato (with **self-modifying-code** to pick add-vs-load), tone-glide with a **runtime 16-bit
division**, wave-arp, pulse-arp, pulse-sweep (multi-segment table), filter-table walk, a second "strange
filter" LFO, pulserun (free PW ramp with $0f→$08 wrap), double-voice detune, "space" effect, drum table,
noise-tick. **All of it fits the primitives — but one sub-op is a notable parameterization stretch.**

| Op | Lane | State | Source | Maps to |
|----|------|-------|--------|---------|
| Note→freq lookup (+ `lonote2` next-semitone for vib width) | FREQ | `lonote/hinote[note]` | `nolengset` L411-435 | P2 |
| Tone-arpeggio (cyclic offset table `arp0..arp7`) | FREQ | `tonearpcounter`,`arpieoklo/hi` | `javib` L554-586 | P5 + P2 |
| Vibrato (depth/period, **self-mod ldy/adc** at `doitnot`) | FREQ | `vibcounter`,`vibstore1/2/3` | `javib2`-`endav` L589-702 | P4 ACCUM + P3; SMC = impl detail |
| Tone-glide **with runtime division** `step=(target-cur)/dur` | FREQ | `glideslo/hi`,`denom` | `rara`-`glideout` L705-823; `nekstbit` divide L784-805 | P4 ACCUM + **P7 ARITH (DIV)** ← new arith op |
| Pulse-sweep (multi-segment, table `pulsetabel`) | PW | `pulsestolo`,`pulsehisto`,`pulsetest` | `glideout`-`pulsestore` L840-952 | P3+P4 + P5 |
| Wave-arpeggio (cycle `wavearp[counter&3]`) | CTRL | `counter2` | `wavetry` L956-975 | P5 + P1 |
| Pulse-arpeggio (cycle `pulsearp`) | PW_HI | `counter2&7` | `pulsearpplay` L979-994 | P5 + P1 |
| Tonesweep-up (per-frame `dec hinote`) | FREQ_HI | `hinotesto` | `sweep` L997-1009 | P4 ACCUM |
| Filter-table walk (`fb0..fb3`: vol/mode + cutoff envelope) | $D416/18 | `filtercount`,`filter` | `filterklooi` L1013-1085 | P5 PTR-WALK + P4 |
| "Strange filter" LFO (ping-pong $D416) | $D416 | `strafilter`,`strfiltest` | `frutsmaarraak` L1086-1140 | P4 ACCUM + P6 |
| Pulserun (free PW ramp, wrap $0f→$08) | PW | `pulserunlo/hi` | `pulserun` L1143-1176 | P4 ACCUM + P6 |
| Double-voice (constant detune add) | FREQ | — | `jeroenshit` L1184-1197 | P7 add |
| Space / noise-tick / drum (table-driven transients) | FREQ/CTRL | `counter2`,`drumtabel`,`starttabel` | L1200-1325 | P3 COUNTER + P5 + P1 |
| Per-voice `byteand` gate mask | CTRL | `byteand` | `nextvoice` L1327-1346 | P7 AND-mask |
| Step-track interpreter (tone-add / step-set / repeat / `$fe`/`$ff`) | control | `tabcount`,`begcount`,`repeatsto` | `h2`-`nextjmp` L245-487 | P5 PTR-WALK + P6 |

**Tel/Deenen: fits — but escalates the "ACCUM parameterization" risk into a genuinely new ARITH op.**
The tone-glide computes its per-frame step as a **runtime 16-bit division** `(target_freq - current_freq)
/ note_duration` (`nekstbit` long-division loop, L784-805), so the glide lands exactly on target after
exactly `dur` frames. This is not ADD/SUB/SHIFT/XOR/AND — it is *division*, a P7 ARITH variant the §3
primitive list did not enumerate (it listed XOR/AND/shift/pack/add-octave). It still *is* P7 (general
per-lane arithmetic over data), so the primitive count doesn't grow — but it proves P7 must be read as
"general integer arithmetic incl. multiply/divide," not the short list §3 implied. The vibrato's
self-modifying `ldy/adc` patch is an implementation detail (P6 SELECT realized as SMC), not a new op.
**No EXEC, no raw-poke stream in this tune** — Tel fits, at the cost of widening P7.

### 6f. Fred Gray (`Mutants.asm`)

A **score-state-machine**: a single `jmp ($e080)` indexes one of ~30 hardcoded "phrase" routines
(`le0bf`..`le451`), each emitting a fixed sub-sequence; advancement is data-driven (`le0c0` step table,
`le001` next-state). Underneath, a shared modulation core: table-walk LFO (`lea14`→`lea36`), portamento
(`leb7f` slide toward target by rate), PW sweep (`lebdc` accumulate), and a **log→linear frequency
reconstruction** (`leba1`: repeated `sbc #$30` to extract octave, mantissa from `lecc4`, then `lsr/ror`
down by octave).

| Op | Lane | State | Source | Maps to |
|----|------|-------|--------|---------|
| Score state dispatch (`jmp ($e080)`, 30 phrase routines) | control | `le0be`,`le001` | `le06a`-`le07c` L125-169 | **P6 SELECT / P5 (data-indexed routine table, NOT arbitrary code)** |
| Phrase step pacing | timing | `le0c0[state]`,`lecb5/6` | `leb20` L906-925 | P3 COUNTER |
| Log-freq reconstruct (octave via `sbc#$30`, mantissa `lecc4`, shift down) | FREQ | `lecb7/8`,`lecbf` | `leba1`-`lebcf` L982-1015 | P2 INDEX + P7 shift |
| Effect/vibrato table-walk LFO into freq accum | FREQ | `lec10`(ctr),`lec14/15`,`lea36` | `lea14` L837-868 | P3 COUNTER + P5 PTR-WALK + P4 ACCUM |
| Portamento (slide `lec11` toward `lec12` by `lec00`) | FREQ | `lec11`,`lec12` | `leb7f`-`leb9a` L960-980 | P4 ACCUM |
| PW sweep (accumulate `lec13` by `lebfd`, pack to PW lo/hi) | PW | `lec13` | `lebdc` L1017-1034 | P4 ACCUM + P7 pack |
| Instrument set (AD/SR/ctrl/freq-eff bulk from `lec25`/`led89`) | AD/SR/CTRL/FREQ | `lebfb..`,`led47` | `leb43`/`led47` L927-1201 | P1 SET (bulk) |
| Arp / note tables (`le5c0`..`le932` relative-offset blocks) | FREQ | per-phrase | scattered | P5 + P2 |
| Gate retrigger via `inc $d404` (SET then INC) | CTRL | — | `leb6a` L949-951; `lead6` L870-880 | P1 SET + P7 inc |

**Fred Gray: fits.** The `jmp ($e080)` looks scary but is a **fixed table of compiled-in routines indexed
by a state byte** — it is P6 SELECT / data-indexed dispatch into the *driver's own* code, NOT execution
of code from the *tune data* (unlike Galway EXEC / Follin SEND, where the target address/bytes come from
the per-tune stream). So it is in-grammar. Gray's only stretch is the log-frequency reconstruction
(`sbc #$30` octave extraction), which is P2+P7, already covered.

### 6g. Updated union — does the bound stay small?

**The melodic/timbral op-set stays bounded and small.** Folding all six hand-coded drivers into §2's
table adds **zero new melody/timbre op-classes**: every pitch/vibrato/porta/arp/PWM/filter/ADSR/gate op
in Hubbard, Follin, Galway, Whittaker, Tel, and Gray maps onto one of the existing ~18 classes /
7 primitives. Several *strengthen* coverage that §1's four under-exercised:

- **sync/ring** — used by Whittaker (`cmd_RingTri`/`cmd_SyncSquare`) and Galway (`SYNCBIT`/`RINGBIT`,
  "rising ringmod effect"); confirms the §3 cross-voice-wiring attribute is real-corpus, not just
  SID-Wizard's note-FX.
- **piecewise-linear (multi-segment) modulation** — Galway's 4-stage freq / 2-stage pulse adders and
  Whittaker/Follin's 2-segment slides confirm ACCUM-chained-by-COUNTER (a *parameterization*, not a new
  op), but the decoder must model **N segments with per-segment delta+length**, not a single sweep.
- **free-running PWM ping-pong** between fixed bounds ($08↔$0e) is *universal* across the hand-coded
  family (Hubbard, Follin, Tel pulserun) — it is the single most common timbre op and is pure ACCUM+flip.

**CORRECTION (2026-06-17, after reading the actual routine bodies + call sites, not just the opcodes).**
The two ops flagged below as "escape hatches" are NOT escapes — that verdict came from reading the
*source mechanism* (`jmp`, a poke-loop) instead of the *output*. We decompile the OUTPUT TRACE (writes to
the 25 SID registers), whose alphabet is finite, so SET is total over single writes and ACCUM/TABLE/POLY
over sequences — **no escape hatch is possible by construction.** Reading what each actually does:

| # | op | What it ACTUALLY does (cited) | Decomposes? |
|---|----|-------------------------------|-------------|
| ~~19~~ | **Galway `EXEC`** (`jmp (PARAM1)`/`$D8`) | the pattern byte points to **fixed driver routines** (not tune-data code): `exec02`=`sta $D417`+`#0` (zero the filter), `inctrans`=`inc transpose`, `morse1-3`+player=walk `.morsedata` table → gate/freq phrase. The address selects a routine from a **bounded driver table** = **P6 SELECT** — *identical* to Fred Gray's `jmp($e080)` which this doc correctly called SELECT. | **YES** — SELECT-dispatch → {SET, transpose(ARITH), TABLE-walk}. |
| ~~20~~ | **Follin `A_SEND`/`ATT4`** | opcode #5 in a clean 17-instruction per-voice VM (beside `A_GATE`/`A_FREQS`/`A_FILTER`/`A_SETVIB`/`A_TRANS`/`A_FOR`/`A_GOSUB`): reads `(reg,value)` pairs, writes to the 25 SID regs until `$FF`. A **multi-SET instruction** for compact patch/effect setup. | **YES** — `N × SET(reg,value)` = P1 looped by P5; the (reg,value) list is per-tune patch DATA. |
| (21) | Subroutine call stack (GOSUB/RET, FOR/NEXT) | counted/recursive control flow over patterns | YES — bounded P5 extension (pointer stack). |
| (P7') | Tel runtime divide | computes a **portamento slope** `(target−current)/time`; the OUTPUT is a linear freq ramp | YES — output = `POLY-1`/`ACCUM`; the divide is source-level slope derivation, recovered from the trace, not a new output op. |

So the union stays **~18 op-classes / 7 primitives** with **NO escape hatch**. "Data chooses the
register/value" is not arbitrariness — SET already takes (register, value) as parameters; the data is the
per-tune program (the music). The bound holds for the whole driver, control layer included.

### 6h. Verdict

1. **Does the bounded ~7-primitive op-set SURVIVE the real hand-coded drivers? YES — fully.** All six
   drivers' pitch/vibrato/portamento/arp/PWM/filter/ADSR/gate/sync-ring operations decompose into the
   existing {SET, INDEX, COUNTER, ACCUM, PTR-WALK, SELECT, ARITH} primitives — no new musical primitive,
   exactly as the thesis predicted, and coverage even improved (sync/ring, N-segment envelopes). **The §3
   claim "no escape hatch is possible" HOLDS** once you model the output trace, not the source: the two
   ops first flagged as escapes (Galway `EXEC`, Follin `A_SEND`) are, on reading the actual routines,
   SELECT-dispatch into fixed driver routines and a multi-SET instruction respectively — both emit ordinary
   register writes that are SET/TABLE/transpose. See the CORRECTION table above.

2. **Did the union grow? No (in primitives).** ~18 op-classes / 7 primitives; +1 benign call-stack
   extension (bounded pointer stack) and the clarification that P7 = general lane arithmetic. The two
   "non-decomposable escapes" were a category error (source mechanism vs output trace) — retracted.

3. **Follin: FITS.** The Ruddy driver is a clean 17-instruction per-voice bytecode VM; `A_SEND` is its
   compact multi-register-set opcode (patch/effect setup) = looped SET. Idiosyncrasy fear unfounded.

4. **Galway: FITS.** `EXEC` is a function-pointer dispatch into the *driver's own* fixed routines
   (filter-zero / transpose / a `.morsedata`-table phrase) — P6 SELECT, exactly like Fred Gray's. The
   Arkanoid `$D418` sampled drums are separately out-of-scope (digi), as the memory notes.

5. **Biggest risk to residual→0, grounded in real code:** it is **NOT** an escape hatch and **NOT** a
   missing op-class. It is **parameterization + synthesis**: **(a) N-segment piecewise modulation**
   (Galway 4-stage adders) — the decoder must fit a *sequence* of ACCUM segments (per-segment delta,
   length), not one sweep (confirmed by the experiment ladder: single-primitive 33% → compositional 68%);
   **(b) general lane arithmetic** (Tel glide-slope) — recover the ramp from the trace with fixed-point
   exactness; **(c) free-running phase recovery** (unchanged from §5.3, now confirmed universal in
   the hand-coded family's PWM/vibrato). The good news: items (b)–(d) are parameterization/synthesis
   problems the build plan already targets; item (a) is the one that genuinely contradicts the "no escape
   hatch" design axiom and needs an explicit decision.

---

## Exact modulation algorithm (vibrato/porta/arp/PWM)

**Status: EXTRACTED 2026-06-18 (Phase 2.5), cited to the actual 6502 disassembly.** This section
extracts the *precise* per-frame modulation programs of the six hand-coded drivers, from
`github.com/realdmx/c64_6581_sid_players` (Hubbard Commando + Monty on the Run, Galway Arkanoid,
Whittaker Panther, Gray Mutants, Tel Cybernoid2) and `github.com/KevEdwards/...FollinMusicDevDiskArchive`
(Follin `…/6502/HIGH.SRC.TXT`). It supersedes §5.3's "free-running phase, hard to recover" note with the
*exact counter+stall+depth+origin program*. File line numbers cite the cloned disassembly.

### The SHARED FAMILY rule (all six)

Every hand-coded vibrato is the **same three-part program**:

1. **Triangle LFO** over a frame counter — a symmetric up/down ramp, NOT `sin`, NOT `f mod period`.
2. **Pitch-scaled depth = a semitone interval right-shifted by a small integer** (constant cents):
   `depth = (freq_table[note+1] - freq_table[note]) >> vib_param`. The vibrato amplitude is a binary
   fraction of *one semitone at the current note* — so in Hz it grows with pitch but in cents it is
   constant. This is the single most important grounded fact: **the depth is not a free parameter, it is
   the note's own semitone step shifted right by `vib_param ∈ {0..7}`.**
3. **A stall/delay arm**: the LFO is suppressed for the first frames of a note (a hold), then arms; the
   counter holds (does not advance) during the delay and at LFO extremes.

The two free-running vs note-reset phase models BOTH appear in the family (so neither is "the" rule):

- **GLOBAL free-running origin** (Hubbard Commando + Monty): one counter `frc`/`counter`, `inc` once per
  frame for *all three voices*, never reset on note-on. Vibrato phase = `frc & 7`, arp = `frc & 1`.
- **Per-voice note-reset origin** (Tel, Follin, Whittaker, Galway): a per-voice counter/accumulator
  re-seeded (or its ramp-in delay re-seeded) on note-on.

### Per-driver detail (cited)

**Hubbard — Commando (`Hubbard_Rob/Hubbard_Rob_Commando.asm`):**
- Global frame counter `frc`: `inc frc` at L38 (once per `_play`), reset only at init L43. Shared by all
  voices.
- **Vibrato** L216-256: `vosc = frc & 7`; if `>=4`, `eor #7` ⇒ triangle phase sequence `0,1,2,3,3,2,1,0`
  (period 8, L216-221). Depth `vdif = (freqs[note+1]-freqs[note]) >> vib` via the `lsr/ror`+`dec vib`
  loop (L226-235); `vib` = `ins_vib` instrument byte (L212). Output `freq = freqs[note] + vosc*vdif`
  (the L245-255 add-loop). **STALL/DELAY**: `if (notelen & $1f) < 6` skip the add ⇒ flat note for the
  first frames (L240-243); also disabled while `notec != 0` sustain branch (L189-193).
- **Arpeggio** L400-422: every other frame (`frc & 1`, L405-407) play `note` or `note+12` (octave arp),
  re-looked-up in the freqs table. 2-frame octave arp, GLOBAL phase.
- **PWM** L275-321: paced by `pdelay` countdown; rate = `pulse & $e0`; 12-bit ping-pong accumulator
  `ins_pwh:ins_pwl`, carry lo↔hi, **flip direction at hi==$0e (max) and hi==8 (min)** (`pdir`, L297-312).
- **Pitch-bend / slide** L323-352: `pbend & $7e` added/subtracted to a per-voice freq accumulator
  `nfqh:nfql` each frame (`pbend & 1` = direction). Free per-frame accumulate (P4).

**Hubbard — Monty on the Run (`Hubbard_Rob_Monty_on_the_Run.asm`):** byte-identical algorithm.
Global `counter` `inc` at L95. Vibrato L367-436 (the author even annotates "this is clever!!" at L385):
same `counter & 7` + `eor #7` triangle (L385-390), same `(notefreqs[n+1]-notefreqs[n]) >> vibrdepth`
depth (L396-405). **STALL/DELAY threshold is `< 8`** here (L416) vs Commando's `< 6` — the *only*
per-tune variation. Octave arp L613+ on `counter & 1` (L596, L618).

**Tel — Cybernoid2 (`Tel_Jeroen_MON/Tel_Jeroen_Cybernoid2.asm`):**
- Per-voice `vibcounter` (3 bytes, L137), **reset to 0 on note-on** (L302-303) ⇒ note-reset phase.
- **Vibrato** L595-701: depth = `(hinote2:lonote2 - hinote:lonote) >> vibrasto` (the `reducesize`
  `lsr/ror`+`dec vibrasto` loop L629-635), `vibrasto = fx1sto & $0f` (L598-600) — **same semitone-step
  >> n constant-cents depth.** Triangle generated by the `vibstore1/2/3` up/down ping-pong (L639-655):
  `vibstore1` = amplitude (`fx1sto & $70`), `vibstore3` = position, `vibstore2` = direction; `subval`
  subtracts and `addval` adds the depth `vibstore3`/`vibstore1>>1` times (L662-693).
- **STALL/RAMP-IN**: `vibcounter` counts up to `#vibtotzover` ($30 = 48 frames) and then **stalls**
  (`cmp #vibtotzover; bcs frag` SKIPS the `inc vibcounter`, L615-619) — the ramp-in delay counter that
  holds at its cap. Plus `vibwait` (`vibtabwait[wavecount]`, L595-597) gates the LFO start.
- **Tone glide** `b17`/`glideout` L705+: per-voice glide accumulator toward target (P4 + clamp).

**Follin — `…/6502/HIGH.SRC.TXT`:**
- **Vibrato** `AVIBON` L856-890: `AV_DEL`/`AV_DEL1` = delay-before-arm (`BEQ AMOD` if depth 0;
  `DEC AV_DEL1; BNE AMOD` HOLDS during the delay = stall, L856-861). `AV5` self-modifies the
  add-vs-subtract opcode (L862-874): `ADC AV_RATE` (up) / `SBC AV_RATE` (down) on the 16-bit freq word
  held in `AV3+1:AV4+1` (self-mod, **free-running per-voice phase**, NOT note-reset). `AV_LIM1` counts
  down; at 0 reload `AV_LIM2<<1` and **flip direction** (`EOR #255` on `AV5+1`, L881-889) ⇒ triangle of
  half-period `AV_LIM2`. **Depth here is absolute** (`AV_RATE * AV_LIM2`), the family's one non-pitch-
  scaled vibrato.
- **PWM** `GOA`/`PRATTA`/`DOWNITA` L894-922: 16-bit `PWMA` ping-pong, `±RATEA` (direction = self-mod
  `AMOD+1` opcode), **flip at upper bound `PWMA_hi==15 && lo>=155` and lower bound `hi<0 || lo<100`**
  (L901-921). The exact PWM flip-bounds + carry rule.

**Whittaker — Panther (`Whittaker_David/Whittaker_David_Panther.asm`):** command-list/table driver.
- **Vibrato** L901-984: per-voice ping-pong on `VD_B1C` (position): `±VD_B1B` (speed) each pass, flip
  (bit 6 of `VD_B1D`) at bounds `0` and `VD_B1A<<1` (depth, L946-952). The centred position
  `VD_B1C - VD_B1A` (L957-961) is converted to a freq delta by a **pitch-relative log-scale shift loop**
  (`adc #$a0` then `asl/rol`+`adc #$18`, L968-984) — exponential depth, then added to the note freq.
  Runs **every other frame** (flip bit 1, L989). Free-running per-voice phase.
- **Arpeggio** L757-781 + `SoundUpdate` L846-883: a per-instrument `ArpTable` of note offsets, walked by
  a per-voice pointer (`VD_ARP`), reset to `VD_ARP2` when the value `>= $54` (L855-864) — table-driven
  relative-transpose arp (P5 PTR-WALK + P7 add).

**Gray — Mutants (`Gray_Fred/Gray_Fred_Mutants.asm`):** label-stripped wavetable driver. Per-voice frame
counter `lecb6` masked `& 7` (L548-549, L582-583, L609-610) indexes per-instrument 3-byte effect/waveform
tables (`le457`/`le467`/`le487`/`le497`, L631-639). No pitch-vibrato LFO of the Hubbard form; its "freq
modulation" is the wavetable's per-row detune column (P5 PTR-WALK over data). Closest to the SID-Wizard
WF-table model already in §1b — its freq residual is dominated by table content, not a recoverable LFO.

**Galway — Arkanoid (`Galway Martin/Galway_Martin_Arkanoid.asm`):**
- **Freq modulation** = a **4-stage piecewise-linear delta program** (`freqAdd1V1`…`freqAdd4V1`,
  L3009-3062): each stage adds a signed `Δfreq` (instrument bytes `+0..+7`) to the running per-voice
  freq accumulator `$F7:$F8` for `+29..+32` frames (per-stage frame counters), advancing stage when a
  counter hits 0. At stage 4 end `resetFreqV1` (L3085-3094) **reloads and loops** the cycle (bit 7 of
  flag `+13`) ⇒ free-running periodic vibrato as a 4-segment triangle; or holds (bit 0). **DELAY**:
  `+12` initial-delay counter `dec`s and suppresses the effect until 0 (L2997-2999). This is the
  N-segment ACCUM the §5.5(a) note flagged — vibrato expressed as ≤4 (delta, length) pairs.

### Implications for the synthesizer (Step B)

The grounded model the freq synth should fit (not blind-search):
- **Triangle** of recovered `period` (Hubbard family = 8; others per-instrument), phase from a **single
  global free-running origin** for the Hubbard family (`frc`), per-voice note-reset for Tel/Galway.
- **Depth = `(note_freq[note+1] - note_freq[note]) >> k`**, fit only the integer shift `k ∈ {0..7}` and
  test the few candidates — the amplitude is otherwise *determined* by the note. This collapses the
  vibrato search from "find the modal cycle" to "find period, k, delay, origin".
- **Stall**: a leading `delay` of flat frames per note (the `notelen < thresh` arm); for the global-phase
  family the triangle is sampled at `(frc - origin) & (period-1)`, so a note that starts mid-phase is NOT
  a phase reset — exactly why a per-note modal cycle drifts and the global origin fixes it.
- **PWM** = ping-pong accumulator with `±rate`, flip at recovered `[lo,hi]` bounds (P4 `step_pingpong`),
  paced by a delay counter — already the right primitive; fit rate + bounds + pace.

What remains genuinely hard (honest): Galway's **N-segment** delta program (§5.5(a)) and Whittaker's
**log-scaled** depth need a per-segment fit rather than one triangle; Gray's freq is **wavetable data**
(no LFO to recover). These are content/parameterization, measured as residual, never patched.
