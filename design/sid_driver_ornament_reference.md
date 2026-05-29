# SID driver ornament generation — reference

**Status:** Reference (background). How C64 SID music drivers generate per-frame **ornamentation** —
pitch (arpeggio / vibrato / portamento), **pulse-width**, and **filter** — at the register level.
Distilled from a 2026-05-29 read of four drivers (References below). Other designs cite this rather
than re-deriving it; see [`ornament_transfer.md`](ornament_transfer.md),
[`encoding_principles.md`](encoding_principles.md),
[`landed/trajectory_anchoring.md`](landed/trajectory_anchoring.md),
[`landed/freq_v0_interval.md`](landed/freq_v0_interval.md),
[`voice_encoding_reference.md`](voice_encoding_reference.md).

## The register targets

Ornament is per-frame (50 Hz PAL / 60 Hz NTSC) modulation of three kinds of SID register, with
different scope:

| target | SID regs | width | scope |
|---|---|---|---|
| **pitch** | `$D400/1` (+7,+14) | 16-bit, via a note→freq table indexed `note<<1` | per voice |
| **pulse width** | `$D402/3` (+7,+14) | 12-bit (hi nibble + lo byte) | per voice |
| **filter** | `$D415` cutoff-lo (3 bits), `$D416` cutoff-hi (8) = 11-bit cutoff; `$D417` resonance (hi nibble) + routing (lo 3 bits, one per voice); `$D418` mode LP/BP/HP (bits 4–6) + master volume | **GLOBAL — one filter, all 3 voices** |

The note→frequency table is universal (Hubbard `FREQ_TBL`, Galway `LoFrq/HiFrq`, SID Wizard
`NOTE_FREQ_LO/HI`, defMON `NOTE_PITCH_LO/HI`), indexed by `note<<1`. **Pitch is fundamentally a
semitone index** into it — an encoding in semitone/interval space (interval-V0) is in the driver's
native domain.

## Two value-domain mechanisms, used for every target

Across all drivers and all three targets, ornament is produced by one of two mechanisms:

- **(A) Note-index offset cycle** — only for *pitch*: an ordered, looped table of **note-relative
  semitone offsets**, stepped one entry per frame, added to the note *before* the freq-table lookup.
  Transposition-invariant, low-cardinality, reused per instrument → naturally a **codebook of
  semitone-offset cycles**.
- **(B) Parametric / table sweep in the target's own value domain** — for vibrato, portamento, PW,
  and filter cutoff: a value swept per frame, either by a **program table** of set/sweep entries
  (SID Wizard) or a **parametric bounded sweep / gradient envelope** (defMON, Hubbard, Galway). These
  are *not* note-relative and *not* a small codebook; they are continuous swept values with a few
  parameters (depth/rate/direction/bounds, or staged gradients). Galway notably uses **one** gradient
  envelope structure (`G0..G3` values + `D0..D3` durations + delay) for vibrato, PW *and* filter.

## Pitch ornament

- **Arpeggio (mechanism A).**
  - *General (tracker) form* — an arbitrary note-relative offset cycle: Galway "OFFSET LIST"
    (`FOLA/FOLB` addr/base, `FOLCI` index, `FOLDC` rate; `ADC offset → TAY → LDX LoFrq,Y`); SID Wizard
    `wf_table` `arp_byte` (`$00–$7E` up, `$E0–$FF` down, `$81–$DF` absolute, `$80` NOP, `$7F` chord
    jump, `$FE` jump/loop; paced by `arp_speed_counter`); defMON `TR` transpose steps across reused
    sidTAB rows (paced by cascade `DL`).
  - *Hubbard early routine* — **octave arpeggio only**: "for the first 50th of a second the current
    note is played, and for the next, current note+12, then the current note again" — a hard-coded
    `note / note+12` toggle at 50 Hz, set by instrument fx **byte 7 bit 2**. Not a general table.
    Many Hubbard tunes = octave-arp + skydive.
- **Vibrato (mechanism B, frequency domain, parametric).** Galway: gradient stages `FMG0..FMG3` +
  durations `FMD0..FMD3` + delay `FMDLY`, added to the freq word. SID Wizard: amp/freq nibbles drive a
  triangle via a FREQMOD table. defMON: freq-word LUT. Hubbard: a single **depth** byte (instrument
  byte 5) raises/lowers pitch slightly.
- **Portamento / slide (mechanism B, frequency domain).** Per-frame add/subtract toward a target.
  Hubbard: a per-note byte = speed, with bit 0 = direction (up/down), applied when the note's flag bit
  marks it bent. defMON: LUT accumulator (active-slide) or single-frame step; `GATE_N` zeroes it.
  SID Wizard: 16-bit accumulator toward the target freq. Galway: `PMG/PMD` gradient.
- **Drum / "skydive" (Hubbard fx bits 0/1).** A fast (drum) or slow (skydive) frequency-down done by
  decrementing the **MSB of the frequency** (8-bit, not 16-bit) — a coarse pitch-drop effect, often
  combined with the octave arp.

## Pulse-width ornament

PW is a **per-voice, absolute 12-bit value that is swept** (square ↔ rectangular timbre). Two forms:

- **Table (SID Wizard `pw_table`, 3-byte rows).** Byte 0 mode: `$80–$FF` set-mode (7-bit PW-hi, bit 7
  set) → byte 1 is PW-lo (absolute 12-bit); `$00–$7F` sweep-mode (cycle countdown) → byte 1 is a signed
  per-frame delta accumulated into the 12-bit value; `$FE` jump, `$FF` freeze. Optional keyboard-track
  in byte 2. Walk rate from instrument multispeed flags.
- **Parametric bounded sweep.** defMON `_ps_voice` (PS column): state = `ps_depth` (bit 7 direction,
  bits 0–6 magnitude); adds/subtracts magnitude to the 12-bit PW each frame and **auto-reverses at the
  extremities** (clamp PW-lo at `$F8`/`$01` and flip the direction bit when PW-hi hits `$0F`/`$00`).
  Hubbard "pulsework": ramps between square and very-rectangular at instrument **byte 6 (pulse speed)**,
  reversing at each extremity; the live PW value is stored back in the instrument. Galway: `PMG/PMD`
  gradient stages.

**PW sweeps persist across notes** (held PW) — defMON `ps_depth` and SID Wizard `pw_pos` are *not*
reset by a new note; SID Wizard resets PW only on a fresh note unless instrument control bit 6
(`pulse_reset_off`) is set. So PW modulation is a continuous per-voice sweep, **not note-aligned**.

## Filter ornament

The SID filter is **global**: one cutoff/resonance/mode shared by all voices, with a 3-bit routing
nibble selecting which voices feed it. So filter ornament is **one tune-global channel**, not
per-voice. Drivers therefore use a **filter-controller voice**: at most one voice drives the cutoff/
resonance/mode sweep at a time; the others only set their own routing bit.

- **Cutoff sweep.** SID Wizard `filter_table` (3-byte rows): set-mode (`$80–$FD`) byte 0 carries mode
  (bits 4–6) + resonance (bits 0–3), byte 1 is the 11-bit cutoff-hi (absolute); sweep-mode (`$01–$7F`)
  byte 1 is a signed delta accumulated into the 11-bit cutoff; `$00` = "filtered but not controller",
  `$FE` jump, `$FF` not-filtered/relinquish. defMON ACID column: a 16-bit cutoff accumulator, either
  absolute load or slide (control byte bit 7 = slide, bit 6 = direction → ADC/SBC), with a floor clamp;
  CP column adds an extra. Galway: `FMG/FMD` gradient stages.
- **Resonance / mode / routing** are low-cardinality absolute sets written by the controller:
  resonance = `$D417` hi nibble; mode LP/BP/HP = `$D418` bits 4–6; routing = `$D417` lo 3 bits, one
  bit per voice (defMON RE column does per-voice set/clear; SID Wizard ORs in each filtered voice).
- **Controller arbitration.** SID Wizard: a voice claims the controller when its `filter_table[0]` is
  non-`$00`/`$FF`, relinquishes on `$FF`; a global `filter_sweep_count` is shared so the sweep
  continues across controller swaps. defMON: last voice to write the RE/FV/ACID globals wins. Galway:
  a `FilterChannel` zero-page byte names the controlling voice.

**Filter sweeps persist across notes** too (global accumulator, not reset by `GATE_N` / new note
unless an instrument reset bit). The Hubbard *early* routine barely uses the filter (its expressive
load is octave-arp + PW + portamento); filter sweeps are heavy in Galway and the trackers.

## Note segmentation caveat: gate-on is not the note boundary

Hand-written / legato drivers **hold the gate** and move pitch via freq-table lookups + portamento
rather than re-gating per note. Confirmed in Hubbard's routine (C=Hacking #5): a pattern note's flag
**bit 6 = "appended to the last (no attack)"** — a legato/tie that does **not** re-trigger the gate;
and the octave arp / portamento run under one sustained gate. So **a gate-segment can span several
melodic notes plus their connecting slides and arp** — gate-on transitions under-segment the melody
for these drivers, mis-reading held-gate note changes as "ornament."

The robust note detector is an **intrinsic level-change** detector (sustained pitch-level changes),
not the gate — exactly the landed `TrajectoryAnchorPass` **pass-1** origin detector
([`landed/trajectory_anchoring.md`](landed/trajectory_anchoring.md)), with the gate as a complementary
signal for re-struck same-pitch notes. (This is the mechanism behind the "gate-anchor refuted"
melody-ladder finding: the inflating "ornament" was partly held-gate melody.) Note also that PW and
filter sweeps are **not** note-aligned (they persist across notes), so they should not be segmented on
note boundaries at all.

## Reuse / banks

Ornament definitions are a **small bank referenced by id**, reused across notes/patterns:

- **Hubbard** — 8-byte instrument records (PW-lo/hi, control/waveform, AD, SR, **vibrato depth**,
  **pulse speed**, **fx byte**); one routine + instrument bank serves a whole module (title / in-game /
  game-over) and was reused across ~30 tunes incl. *Commando*, *Monty on the Run*, *Crazy Comets*.
- **SID Wizard** — instruments hold contiguous `wf_table | pw_table | filter_table`, shared per id.
- **defMON** — sidTAB rows referenced by index, shared across voices.
- **Galway** — per-voice parameter banks `D0/D1/D2`.

This is the structural basis for encoding ornament by table-id with per-instrument / per-composer bank
conditioning.

## Per-driver summary

| | pitch arp | vibrato | portamento | pulse width | filter |
|---|---|---|---|---|---|
| **Hubbard** (early) | octave-only (note/+12 @50Hz), fx bit 2 | depth byte (instr byte 5) | per-note speed+dir byte | bounded sweep @ pulse-speed (instr byte 6) | minimal |
| **SID Wizard** | `wf_table` arp_byte (general offset cycle) | amp/freq-nibble triangle | 16-bit accumulator → target | `pw_table` set/sweep rows | `filter_table` set/sweep + controller voice |
| **defMON** | `TR` steps in reused sidTAB rows | freq-word LUT | LUT accumulator / 1-frame step | `PS` bounded auto-reverse sweep | `ACID` cutoff accumulator + `RE` routing |
| **Galway** | `FOL*` offset list | `FMG/FMD` gradient | `PMG/PMD` gradient | gradient stages | gradient stages + `FilterChannel` |

## Encoding takeaways

- **Arps → a codebook of note-relative semitone offset cycles** (mechanism A), semitone-domain and
  transposition-invariant; add an **octave-arp** special primitive (Hubbard's dominant form).
- **Vibrato / portamento / PW / filter → parametric sweeps** in their own value domains (mechanism B),
  *not* offset codebooks and *not* raw per-frame deltas. One bounded-sweep / gradient primitive family
  generalizes across them (as Galway's shared envelope shows).
- **Filter is global** — encode it as **one tune-level channel** (cutoff sweep + low-cardinality
  resonance/mode/routing sets), not triplicated per voice; this matches `TRAJ_REGS`' treatment of the
  filter cutoff (reg 21) as a single intrinsic, non-gated trajectory.
- **PW and filter sweeps are not note-aligned** (they persist across notes) — segment/encode them as
  continuous per-voice (PW) / global (filter) trajectories, unlike the note-aligned pitch arp.
- **Use intrinsic level-change, not the gate, for note boundaries** on legato/held-gate drivers.

## References

- **C=Hacking Issue 5** (1993), *"Rob Hubbard's Music: Disassembled, Commented and Explained"* by
  Anthony McSweeney — a commented disassembly of Hubbard's first routine (used in ~30 tunes incl.
  *Commando*, *Monty on the Run*, *Crazy Comets*). Authoritative primary source for the Hubbard family:
  50 Hz pattern format, the 4-byte note spec (length + flag bits incl. **bit 6 legato/appended**,
  optional instrument-or-portamento byte, semitone pitch), the 8-byte instrument (PW, control, ADSR,
  **vibrato depth**, **pulse speed**, **fx byte**: bit 0 drum, bit 1 skydive, bit 2 **octave arp**),
  and the per-frame NoteWork/SoundWork loop (vibrato, pulse sweep, portamento, fx).
  <http://www.ffd2.com/fridge/chacking/c=hacking5.txt> (local copy: `/scratch/tmp/chacking5.txt`).
- **defMON** (tracker; Python reimplementation): `/scratch/anarkiwi/pydefmon` —
  `pydefmon/pydefmon/defmon_player.py` (`_pitch_slide_voice`, `_ps_voice` PW sweep,
  `_cutoff_slide_step` + ACID/RE/FV/CP columns, sidTAB cascade), `defmon.py` (note/freq tables, row
  format), tests under `pydefmon/tests/`.
- **SID Wizard** (tracker; Python reimplementation): `/scratch/anarkiwi/pysidwizard` —
  `src/pysidwizard/player.py` (`_tick_wf_table`, `_tick_vibrato`, `_tick_portamento`, `_tick_pw_table`,
  `_tick_filt_table` + filter-controller logic, `_compose_sid_writes`), `model.py` (instrument layout:
  `wf_table | pw_table | filter_table`), tests incl. `test_player_filter_controller.py`.
- **Rob Hubbard — *Commando*** (hand-written 6502; reverse-engineered disassembly):
  <https://gitlab.com/ricardoquesada/c64-commando-2084/-/blob/orig/src/music.asm> (`FREQ_TBL`, per-frame
  freq writes, portamento). A `f####`-labelled disassembly — treat internal detail as lower-confidence
  than C=Hacking #5 / the trackers.
- **Martin Galway — Ocean drivers**: <https://github.com/MartinGalway/C64_music> (e.g. `wizball.asm`,
  `greenberet.asm`) — `LoFrq/HiFrq`, the `FOL*` offset-list arpeggio, the shared gradient-envelope
  mechanism (`*G0..G3` values + `*D0..D3` durations + delay) reused for vibrato / PW / filter, and the
  `FilterChannel` global-filter controller. (Label-level detail via summary; treat as corroborating
  the structural patterns, not exact byte layouts.)
