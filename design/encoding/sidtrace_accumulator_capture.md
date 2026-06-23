# preframr-sidtrace — capture the player's ACCUMULATORS and structure pointers

Status: scoped 2026-06-22, NOT yet implemented. Source: JCH NewPlayer RE (Goto80 `10.sid`,
`deplayroutine` + `da65`). Companion: AGENTS.md "The other face of the wall".

## Problem

Slide / portamento / table-walk players (JCH NewPlayer, RoMuzak, Laxity-family, ...) compute the per-frame
frequency as an INTEGRATOR:

    freq = note_table[arp(note, transpose)] + vibrato_acc + porta_acc

A single compact source command — "slide to note N at speed s" (`$60`) — makes the player add
`interval(N) >> s` to the freq shadow EVERY frame. JCH `10.sid` voice 0 emits **3000 distinct freq values
over 3000 frames (100% unique)**. The output-only generic recovery reconstructs the integral (the "718
distinct pitches" symptom) instead of the one command that generated it — ~4 token/frame, a 52% instrument
pool of big tables.

The compact source IS in RAM (SNAP captures every table: note table, wavetable, patterns, instruments,
filter/pulse programs, orderlists). The FILTER sweep is already fully traced (STSQ samples the cutoff cell
`$1792` + its program pointer `$1790` + the program reads). **The gap is the FREQ pipeline:** the tracer
never samples the freq accumulators, because STSQ/SDCU only sample addresses that appear as a SIDDF *leaf*,
and the accumulators (`$1778` porta, `$17a7` vibrato, `$1795` wave-ptr, `$100c` freq-shadow) are
pass-through stores the slicer jumps over (the freq write slice is 5 stores deep:
`$100c <- +$1778 <- $172d <- note_table[$166f] <- $1014`).

## Changes (priority order)

Files: `third_party/libsidplayfp/src/c64/membus_trace.h` (`sliceSidWrite`, `sliceCellUpdate`,
`sampleStateCells`, `isSiddfStateCell`, `recordIndexedRead`) and `src/sidtrace.cpp` (SDST sections +
`STATESEQ_CAP`).

1. **[HIGHEST] Sample every TRANSITIVE state cell of a `$D4xx`-write slice into STSQ, not just leaves.**
   Have `sliceSidWrite`/`sliceCellUpdate` record every in-RAM store address they traverse into a bounded
   per-write `touchedCells` set; let STSQ sample any such cell that is a state cell. Surfaces `$100c/$100f`,
   `$1778/$177b` (porta), `$17a7/$17aa` (vibrato), `$1795` (wave-ptr). Raise `STATESEQ_CAP` (96 -> ~3
   voices x 6 pitch cells + filter + pulse). *Recovery:* STSQ already feeds Berlekamp-Massey/Daikon;
   `$1778` is an arithmetic series (constant first difference = `interval >> s`) and `$17a7` a triangle —
   both fit a 2-term linear recurrence, collapsing thousands of freq values to "slide to note N, speed s" +
   "vibrato depth/speed".

4. **[HIGHEST, structural] Capture the orderlist -> pattern zero-page POINTER WALK as first-class.** Detect
   zp pointer pairs used as `(zp),Y` bases (`$00fb/$00fc` here) and record the SEQUENCE of pointer values
   (each = a pattern start address) + the `(zp),Y` index range per call. *Recovery:* the pointer-value
   sequence is the orderlist-resolved pattern stream; with the SNAP'd pattern-pointer table (`$19a5/$19c6`)
   and orderlists (`$1954...`), reconstruct orderlist -> pattern -> rows instead of per-frame note events.
   The single biggest structural lever for the pool/token blow-up.

3. **Fix indexed-read attribution for SCALED indices.** IDXR's `base = addr - index` is wrong when the index
   is pre-scaled (`note*2` -> base garbled to `$150a` instead of `$166d`). Record a few observed
   `(index, effective_addr)` samples per read PC so the host fits `addr = base + scale*idx`; also record the
   address of the index register's defining value. *Recovery:* "note table at `$166d`, stride 2, indexed by
   current note", and links the wavetable walk to its pointer `$1795`.

2. **Per-write ACCUMULATOR summary.** When a `$D4xx`-write's source shadow is written >=2x per play-call by
   distinct PCs (from `lastWriteSeq` deltas within a frame), tag the SDDF entry `accumulated=true` and record
   each contributing `(op, addend-cell)`: `freq = base (+) vib_acc (+) porta_acc`. *Recovery:* model each
   accumulator separately rather than treating the summed output as free pitch.

5. **Link control-byte (waveform/gate) provenance to the wavetable.** Ensure the wave-ptr `$1795` (from #1)
   and the `$1826/$17db` reads (from #3) are linked to the `$d404` ctrl write, so gate/waveform changes
   recover as wavetable steps, not per-frame ctrl tokens.

Highest value: **#1 and #4** (they directly kill the "718 pitches" and the pool cost). #3 makes the tables
addressable; #2 and #5 are refinements.

## Note — the reg22 filter lane is already traced

STSQ/SDCU/SDDF already capture `$1792` + `$1790` + the filter-program reads, so reg22 should be collapsible.
The generic OUTPUT-only path (`.sidwr` -> `cover_lane`) does not consume the distill artifact, so reg22's
cost there is a CODEC issue: `cover_lane` must cover the per-frame cutoff output as the piecewise dwell-accum
the filter program produces. Tracked separately as the reg22 codec free-win.
