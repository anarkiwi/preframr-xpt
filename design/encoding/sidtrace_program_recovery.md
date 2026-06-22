# Recovering the *program*, not its output: a white-box execution-trace design for preframr-sidtrace

**Status: IMPLEMENTED (2026-06-22).** preframr-sidtrace now DISTILLS the execution
in the emulator and emits one compact **SDST** artifact per tune (a few KB); the
Python recovery consumes that artifact, classifies memory SMC-correctly by access
type, and lifts the song data byte-exact. The two hard requirements below reshaped
the original design and are now the spec; the legacy multi-GB bus-trace direction
(and the in-progress `BUS_DT2` PC-augmented per-cycle stream) are RETIRED. §0a/§0b
state the requirements; §2.4/§3a document the artifact and classifier as built; the
rest of the document is the original research grounding, retained.

## 0a. REQUIREMENT 1 — self-modifying code must be classified correctly (by access type)

C64 players routinely self-modify code (SMC) for performance. The original
"RAM written-once-at-init, then only read during play = song data" heuristic
MISCLASSIFIES under SMC: code written during play looks like "not data", and SMC
operands pollute the partition. The emulator has full visibility, so classify
memory by **access TYPE**, accumulated per address during emulation:

- **EXEC** — the address was fetched as an instruction opcode OR operand byte
  (`fetchNextOpcode` + the operand fetches `FetchDataByte`/`FetchLowAddr`/
  `FetchHighAddr` mark it). Per phase: `EXEC_INIT`, `EXEC_PLAY`.
- **READ** — read as data on the bus (`READ_INIT`, `READ_PLAY`).
- **WRITE** — written on the bus (`WRITE_INIT`, `WRITE_PLAY`).

From these the SMC-correct definitions are:

- **SMC** `= EXEC_PLAY & WRITE_PLAY` — a location BOTH executed AND written during
  play is self-modifying code (operand patched in place). SMC writes are code
  modification, not song data; excluded by the `EXEC` term, **not** by write-set
  subtraction.
- **song data** `= READ_PLAY & ¬WRITE_PLAY & ¬EXEC ∧ in the loaded image span` —
  init/load-resident memory that during play is READ as data and is NEVER executed
  and NEVER written. SMC code and plain code are both excluded because they are
  `EXEC`. The post-init RAM snapshot of this region IS the song bytes, verbatim.

Validated: a synthetic SMC player (an `INC` that patches the operand of an
`LDA abs` every frame, beside a read-only data table) classifies the patched
operand as SMC and excludes it from the song data, while the read-only table is
recovered byte-exact (`tests/test_distill.py::test_smc_player_classified_correctly_end_to_end`).
Grid_Runner's real GoatTracker player shows 8 SMC bytes, correctly excluded.

## 0b. REQUIREMENT 2 — no multi-GB traces; distill in the emulator

Dumping the cycle-by-cycle bus stream (GBs/tune) and partitioning it in Python is
hopeless at 60,000 tunes (petabytes of I/O). preframr-sidtrace now does the
analysis **in C++ during emulation** (`src/c64/membus_trace.h`) into bounded
fixed-size accumulators (a handful of 64 KiB arrays + small maps; peak memory does
not grow with run length) and emits ONE compact **SDST** artifact per tune. The
Python side (`bacc/generic/distill.py`) consumes that artifact — there is no raw
trace to read. Measured artifact sizes (400 frames): Grid_Runner 9.3 KB,
Ode_to_Music (DMC) 8.3 KB, Hammurabi 6.4 KB, Monty_on_the_Run 10.4 KB,
FamiCommodore 9.7 KB — vs the retired bus trace's ~GBs/tune. The only timestamped
stream kept is `<prefix>.sidwr.bin` (SID writes only, ~one burst/frame, tens of KB),
which is the render/residual-zero gate.

---

(Original research design follows; the staged plan in §7 has been executed.)

This document originally specified *what additional execution information
`preframr-sidtrace` should log* and *how the generic recovery would use it* to
recover the composer's actual program (instrument tables, wavetables, pattern data,
orderlist, note events) **directly from the player's execution**, rather than
inferring it from the register-write output. It is grounded in (a) an audit of the
live `preframr-sidtrace` source and the bus-trace reader, (b) the RE'd ground-truth
players, and (c) the dynamic-analysis literature.

Companion / prior art (all read for this design):
- `preframr-xpt/design/encoding/sid_player_decompiler.md` — the `trace = VM(program)`
  thesis, **HARD RULE #0**, the STEP/TRACKER reframe.
- `preframr-xpt/design/encoding/generic_recovery_from_bustrace.md` — the bus→25-reg
  state foundation, value-provenance note-table recovery, 7/8 residual-zero.
- `/scratch/tmp/re/generic_tracker_decompile.md` — the SHIPPED tracker lift
  (Grid_Runner 161,002 → **12,821** tokens; hand backend **2,817**).
- `sid_opset_inventory.md`, `generic_bacc_recovery.md`, `unified_generic_recovery.md`.

---

## 0. The discipline (HARD RULE #0, inherited unchanged and made *stronger* here)

Recovered tables are **read from the program's own RAM** — genuine program data the
player itself initialized and then only read — **never fabricated, never re-fit from
the output**. Residual-zero (the recovered program renders to the byte-exact
`(nframes,25)` register state) remains the gate. The entire thrust of this design is
that richer tracing lets us *satisfy* HARD RULE #0 more directly than today: instead
of *guessing* that a contiguous never-written RAM region is the note table and
*matching* output values back to it (the current heuristic in
`generic_recovery_from_bustrace.md §2`), we **observe the player read that table
through an indexed instruction** and record the (base, index) decomposition as it
happens. The table is recovered because we watched the program use it.

---

## 1. Why output-only recovery is fundamentally limited (the inverse problem)

`trace = VM(program)` (the shipped thesis). The current generic recovery is given
only the SID-write stream — `VM`'s **output** — and must invert it to a `program`.
That inversion is **under-constrained**: infinitely many programs emit the same
register trace. Concretely:

- A constant held value could be a note pitch, a sustained drone, or a degenerate
  table walk. Output alone cannot tell which.
- An indexed table read `LDA arp,X` and a hand-unrolled sequence of `LDA #imm`
  produce the *same* output bytes. Output alone cannot distinguish a 4-entry arp
  table iterated 400× from 1600 literal writes.
- Two voices sharing one instrument table vs. two voices with byte-identical private
  copies are output-indistinguishable.

So every output-only method — per-register fits, the genfit/eventfit cover, the
tracker lift — is forced to *choose* a program from the equivalence class, and absent
the real provenance it tends toward the program that most cheaply *reproduces the
output*: a **clever lossless RLE/dictionary of the trace**. This is exactly the
measured symptom: Grid_Runner's genfit serialization had **33,913 inline fit
segments but only 237 distinct signatures** (143× redundancy) and per-frame `carry`
arrays — "stores the VM's execution, not the program"
(`generic_tracker_decompile.md §1`). The tracker lift recovered *structure* from
those fits and got to 12,821 tokens, but it is still **~4.5× the hand backend's
2,817**, because it is reconstructing shared instruments and repetition by
*similarity clustering of the output fits*, not by reading the one table the player
actually shared. Diminishing returns are inherent: we are estimating a hidden program
from a lossy projection of it.

**The reframe.** The hidden `program` is not hidden from the *emulator*.
`preframr-sidtrace` *runs* the ~few-KB 6502 playroutine + packed song data that
ORIGINATES every SID write. The program's tables physically exist in C64 RAM; the
player reads them through observable instructions; the control structure is the
observable call/loop/index behaviour. Making that execution observable converts the
inverse problem into a **direct recovery (an "observe-then-read-off" problem)**:
recover the program by watching the program run. This is the white-box move that
every real SID ripper and decompiler already makes (siddump runs the player;
SIDdecompiler traces execution to regenerate 6502 source; rippers "trace every memory
access to fully understand what each byte is doing" [SIDBlaster]). We are not
inventing a technique — we are giving the recovery the inputs the technique needs.

The formal backing for "read the program off the trace" is the dynamic-analysis
literature (§5): **dynamic taint / data-flow** gives *value provenance* (which RAM
byte produced this SID write) [Newsome & Song 2005; Schwartz et al. 2010]; **dynamic
slicing** gives the *def-use chain and control structure* behind each write [Korel &
Laski 1988; Agrawal & Horgan 1990]; **VSA's strided-interval a-locs** are the formal
model of "a load `base+stride·[lo,hi]` is an array of element-size `stride`, length
`hi−lo`" [Balakrishnan & Reps 2004, 2010]; and **trace-based data-structure
excavation** (Howard, REWARDS) is exactly "accesses sharing a base = a record;
constant-stride sweep over a base = an array" [Slowinska et al. 2011; Lin et al.
2010]. All four are *dynamic*, trace-driven, no static disassembly — the regime we
are in.

---

## 2. AUDIT: what `preframr-sidtrace` logs today, and what it can cheaply log

### 2.1 What is logged today (verified against source)

`preframr-sidtrace` (`src/sidtrace.cpp`) runs a *patched* libsidplayfp
(`patches/0001-membus-trace-instrumentation.patch`). The patch hooks the **CPU data
bus** in `c64cpubus::cpuRead/cpuWrite` (`src/c64/c64cpu.h`) and pushes a record to a
process-global `MemBusTrace` (`src/c64/membus_trace.h`):

```c++
struct MemAccess { int64_t cycle; uint16_t addr; uint8_t val; uint8_t rw; };  // rw: 0=read 1=write
```

written to `<prefix>.bus.bin` as packed 12-byte little-endian records — the `BUS_DT`
in `bacc/generic/bustrace.py`. The SID-write subset (`rw==1`, `$D400..$D7FF`) is
split to `<prefix>.sidwr.bin`. `<prefix>.meta.txt` carries `init`/`play`/`load`
addresses, speed, model, cycles-per-frame.

**Crucial findings from the audit:**

1. **Reads ARE already logged, with addresses.** Every CPU read — including the
   player's reads *into its own song-data tables* — is in `.bus.bin` today
   (`rw==0`). This is the value-provenance substrate, and **the recovery currently
   largely ignores it**: `generic_recovery_from_bustrace.md` uses reads only for the
   note-table heuristic (§2 there: "a freq-lo write whose value was just read from a
   contiguous 2-byte-strided never-written region"). The full read stream's
   provenance content is **captured but unexploited**. This is the single biggest
   piece of low-hanging fruit and it needs **no patch at all**.

2. **The PC is NOT logged.** `MemAccess` has no program-counter field. We cannot
   today group writes by the code site that issued them, nor attribute a read to the
   instruction that performed it.

3. **No instruction-level context** (X/Y index registers, effective-address
   pointer, init-vs-play phase) is logged.

### 2.2 What the emulator can cheaply expose (verified against the CPU core)

The pinned libsidplayfp CPU core (`src/c64/CPU/mos6510.h`) already holds, as live
members of `MOS6510`, **exactly** the fields a richer trace needs:

- `int_least32_t instrStartPC;` and `uint_least16_t instrOperand;` (in `CPUDebug`)
- `uint_least16_t Register_ProgramCounter;`
- `uint_least16_t Cycle_EffectiveAddress;`  ← the resolved address of the current
  memory cycle (the `base+X` of `LDA base,X`)
- `uint_least16_t Cycle_Pointer;`  ← the resolved zero-page pointer for
  indirect-indexed `(zp),Y`
- `Register_X`, `Register_Y`, `Register_Accumulator`, `Register_StackPointer`.

So PC, the index registers, and the effective-address/pointer decomposition are all
*already computed by the core every cycle* — logging them is a matter of plumbing,
not new emulation.

**Architectural caveat the next agent must handle (verified):** the bus hook lives in
`c64cpubus` (the `CPUDataBus` implementation), which does **not** hold a pointer to
the `MOS6510`. The CPU calls `dataBus.cpuRead/cpuWrite`; the bus object cannot reach
`instrStartPC` from where the hook is today. Two clean options:

- **(A) Thread a CPU pointer into `MemBusTrace`** — register `&cpu` (and small getters
  for PC / X / Y / effective address) the same way `c64.cpp` already registers the
  `clockFn`. The `record()` call then samples them. Lowest blast radius; mirrors the
  existing clock-getter pattern in the patch.
- **(B) Move/duplicate the record call into the CPU's memory-access microcode**
  (`mos6510.cpp` `Cycle_EffectiveAddress` computation / fetch), where PC, X, Y, and
  the effective address are all in scope. More invasive but gives the cleanest
  per-instruction semantics (and lets us emit one *instruction* record rather than
  re-deriving instruction boundaries from cycle bunching).

Either is a small patch. (A) is recommended for the first cut.

### 2.3 Precedent: the RBT1 / revice bustrace core already captures more

The other in-tree tracer (`/scratch/anarkiwi/cbm/revice/libs/bustrace/`, the `RBT1`
format the bustrace.py reader explicitly *rejects* as "the wrong source") is a
richer, deterministic CPU bus-trace core. It is worth reading as a precedent for the
record schema and for delta-cycle framing, but the design here keeps
`preframr-sidtrace` as the single native source and *extends its* schema rather than
switching formats.

---

### 2.4 The SDST distilled artifact (AS BUILT)

`preframr-sidtrace` emits `<prefix>.distill.bin`, a versioned little-endian binary
(a few KB), documented in `src/sidtrace.cpp` and read by `bacc/generic/distill.py`:

```
magic   "SDST"  | version u16 | reserved u16
init u16 | play u16 | load u16 | subtune u16 | nframes u16 | reserved u16
cycles_per_frame u32 | t0_cycle i64 | load_len u32
sections (tag char[4] + body), terminated by "END\0":
  ACMP  nbytes u32, then (run u16, bits u8) pairs over addresses 0..65535:
        the per-address ACCESS-TYPE map (OR of EXEC/READ/WRITE x INIT/PLAY).
        RLE'd; mostly-untouched RAM compresses to ~nothing. (~2-3 KB)
  SNAP  nbytes u32, then (addr u16, len u16, bytes[len]) runs: the post-init RAM
        snapshot for the SONG-DATA region ONLY -- the maximal eligible
        (¬WRITE_PLAY ∧ ¬EXEC ∧ in loaded image) runs that contain >=1 byte READ
        as data during play. The verbatim song bytes, captured once at the
        init->play boundary; NOT the whole 64 KiB. (~1-5 KB)
  SIDW  nentries u32, then (pc u16, reg u8, _pad u8, count u32, lastVal u8,
        _pad[3]) -- PC-tagged SID-write summary: voice-lane attribution by code
        site (a few dozen entries, not every cycle).
  IDXR  nentries u32, then (pc u16, base u16, stride i32, idxMin u8, idxMax u8,
        _pad u16, count u32) -- indexed-read VSA summary: per indexed-read PC the
        table base (= effaddr - index), element stride, index span (=> length),
        and traversal count. Surfaces orderlist/pattern structure without the raw
        read stream.
```

The recovery reconstructs the BaccProgram (real instruments + sparse score) from
this artifact ALONE: SNAP gives the song bytes byte-exact (HARD RULE #0), ACMP
gives the SMC-correct classification, SIDW gives voice lanes, IDXR gives table
base/size/length/traversal. `build_distill`/`parse_distill` round-trip the
artifact (tested). Measured: Grid_Runner recovers 14 real GoatTracker instruments
(not ~1000 output clusters) and serializes to **2,786 GT tokens** -- below the hand
backend's 2,817 and 4.6x smaller than the tracker lift's 12,821.

---

## 3. What `preframr-sidtrace` should additionally LOG (prioritized)

Each item: what it is · why it directly exposes program structure · the recovery
technique it enables · logging cost. Ordered by **leverage ÷ cost**.

### P0 — (no patch) EXPLOIT the read stream already in `.bus.bin`

- **What.** The existing `rw==0` records with addresses. No emulator change.
- **Why it exposes structure.** The reads the *play* routine makes into RAM that was
  *written during init and never written again* ARE the song-data tables, by the
  player contract (init populates data once; play only reads it — HVSC SID format
  spec; ripper "memory-map" practice). The read addresses + values reproduce those
  tables **verbatim**.
- **Technique enabled.** The "written-once-at-init, read-only-during-play = song
  data" partition (§4.1): compute, per address, the write-set and read-set over the
  trace; the addresses init wrote and play only reads form the data region, recovered
  byte-for-byte from RAM (HARD RULE #0 satisfied by construction). This already
  generalizes the note-table recovery to *all* the player's tables.
- **Cost.** Zero logging cost (already on disk). Pure recovery-side work.

### P1 — PC of every access (the highest-leverage *patch*)

- **What.** Add `uint16_t pc` (the issuing instruction's `instrStartPC`) to each
  record. (Option A in §2.2.)
- **Why it exposes structure.** The PC **groups writes/reads by code site = the
  player's lanes**. A driver writes voice 0's frequency from one `STA $D400,X`-class
  site, voice 1's from another (or the same site with a different X). The set of PCs
  that touch `$D400..$D406` *is* the voice-0 lane; the set that read a table *is* the
  table-walk loop. This turns "237 fit signatures that happen to look alike" into
  "these N writes all came from the same instruction, so they are literally the same
  generator" — provenance by *code identity*, not output similarity.
- **Technique enabled.** Dynamic slicing's def-use over the trace [Agrawal & Horgan
  1990]: the criterion is a SID write; grouping by PC recovers which subroutine/loop
  body produced it. Instrument dedup becomes exact: two notes share an instrument iff
  they execute the same code sites reading the same table region.
- **Cost.** +2 bytes/record (12→14, or pack to keep 12 by dropping the absolute
  cycle to a delta — see §3 note). One getter call per access. Negligible CPU.

### P2 — Effective-address base/index + index register (recovers tables AND traversal)

- **What.** For indexed loads, log enough to decompose the effective address into
  `(base, index)`: the simplest sufficient capture is `Cycle_EffectiveAddress`
  (already logged as `addr` for the access) **plus the index register value
  (`Register_X`/`Register_Y`) at that access**, and/or the instruction's base operand
  (`instrOperand`). `base = addr − index`.
- **Why it exposes structure.** `LDA table,X` reads reveal **base = table start**,
  the **index sequence** = the pattern/orderlist traversal, and the **index span** =
  the table length — *directly*. This is the orderlist/pattern data and its playback
  order, read off the index stream.
- **Technique enabled.** **VSA strided-interval a-loc recovery** [Balakrishnan & Reps
  2004, 2010] + **memory-access-pattern array detection** [Slowinska et al. 2011]:
  the set of effective addresses from one indexed PC forms `base + stride·[lo,hi]` →
  an array of element-size `stride`, length `hi−lo`. The *temporal* index sequence
  (which entries, in what order, how many times) is the per-voice pattern/order
  traversal = the note-event stream's structure.
- **Cost.** +1 byte (index reg) per access if logged unconditionally, or log it only
  for indexed-addressing-mode accesses (cheaper, needs the mode, which the core
  knows). With PC (P1) the base operand can be recovered once per PC from the static
  bytes at `instrStartPC`, so logging just the index value may suffice.

### P3 — init-vs-play phase marker + RAM write-map at end of init

- **What.** A 1-bit (or small enum) phase tag per record: `init` | `play` | `irq`.
  Derive it from the host: `sidtrace.cpp` already knows `initAddr`/`playAddr` and
  *calls them*; mark the phase around the `engine.play()` boundaries and the init
  call. Additionally, **snapshot the set of RAM addresses written during init** (the
  init write-map) — emit it once at the init→play transition.
- **Why it exposes structure.** The region init writes and play never re-writes is
  the **song data, recoverable verbatim** — this is the formalization of ripper
  practice (the "written-once-then-read-only" partition). The init write-map gives
  the exact byte ranges to lift from the post-init RAM image as the program's tables;
  the phase tag lets the recovery ignore init/boot churn and focus on the steady play
  loop.
- **Technique enabled.** Read/write-set partition (§4.1); also pins the boot-prolog
  alignment that already bit the recovery (`generic_recovery_from_bustrace.md §1`) —
  frame 0 = first `play` phase call, derived not guessed.
- **Cost.** Phase bit: ~free (host already brackets the calls). Init write-map: a
  one-shot `O(RAM)` set dump at the transition (≤64 KB bitmap, a few KB typically).
  **This plus P1 is the recommended first patch (§6).**

### P4 — Zero-page pointer dereferences (`Cycle_Pointer`)

- **What.** For indirect-indexed `(zp),Y` accesses, log `Cycle_Pointer` (the resolved
  16-bit pointer) and the zp pointer location.
- **Why it exposes structure.** Drivers walk pattern/instrument data via **ZP
  pointers** that are re-pointed as the song advances (`LDA (ptr),Y`). The sequence of
  pointer values is the orderlist→pattern descent (which pattern, where in it). The zp
  location identifies the per-voice cursor.
- **Technique enabled.** Howard-style base-pointer clustering [Slowinska et al. 2011]:
  accesses sharing a (re-pointed) base pointer are one logical table walked over time;
  recovers pointer-linked structure that flat strided analysis misses.
- **Cost.** +2 bytes on indirect accesses only. Lower priority than P1–P3 because
  many table-driven players (GoatTracker) use absolute-indexed `,X` more than
  `(zp),Y`; valuable for the pointer-heavy drivers (DMC/JCH walk ZP pointers).

### P5 — Call/return (JSR/RTS) structure + play-call boundary / frame counter

- **What.** Log JSR target / RTS (or reconstruct from the stack writes already in the
  trace) and an explicit per-play-call frame index.
- **Why it exposes structure.** Subroutine boundaries = **lane/effect boundaries**
  (per-voice update routine, vibrato routine, the orderlist-advance routine). The
  frame counter makes the IRQ/play cadence explicit instead of re-derived from cycle
  gaps.
- **Technique enabled.** Control-flow structuring for the orderlist/pattern layer;
  cleaner note-boundary detection.
- **Cost.** JSR/RTS is partly recoverable for free from existing stack-page
  (`$0100..$01FF`) write/read records + PC (P1), so this may need *no new field* — a
  recovery-side reconstruction. The frame counter is one host-side integer.

**Record-schema note.** To avoid bloating the (already tens-of-MB) trace, the cheapest
encoding that carries P1+P3 is: keep 12 bytes by switching `cyc int64` → `dcyc`
(delta-cycle, fits well under 32 bits within a frame) and spending the freed bytes on
`pc uint16` + `flags uint8` (phase + addressing-mode + index-reg-present). The reader
in `bacc/generic/bustrace.py` is a thin `np.dtype` and a one-function loader — a new
`BUS_DT2` dtype + version sniff on the 4-byte head (it already sniffs `RBT1`) is a
small, backward-compatible change. P2/P4 fields, when enabled, extend the record.

---

## 4. The recovery algorithm that USES the richer trace

The output is the SAME `BaccProgram` the shipped tracker lift produces (shared
`instruments` + per-voice `score` of `NoteOn` events + `tables`), fed to the SAME
shared serializer (`_lz_emit_t` REPEAT/TRANSPOSE + inline instrument dedup). The
difference is that every block is now **recovered from execution, not fit from
output**, which makes it byte-exact, smaller, and driver-generic.

### 4.1 Song-data tables — read verbatim from RAM provenance (P0 + P3)

1. Partition addresses over the trace into write-set and read-set per phase (P3
   phase tag; falls back to the host init/play bracketing).
2. The **data region** = addresses written during `init`, then in `play` **only
   read** (subtract addresses play also writes — those are scratch / SID shadow /
   self-mod operands). This is the ripper "written-once-then-read-only" partition,
   now mechanical.
3. Lift those byte ranges **verbatim from the post-init RAM image** (the init
   write-map gives the exact ranges; the read stream confirms which are live). These
   ARE the instrument table, wavetable, pulsetable, filtertable, orderlist, and
   pattern bytes — the program's data, not a fit. (Generalizes the existing
   note-table recovery from one table to all of them.)

### 4.2 Tables typed + bounded — VSA over the indexed reads (P1 + P2)

For each indexed-read PC: collect its effective addresses → strided interval
`base + stride·[lo,hi]` [Balakrishnan & Reps]. This gives each table's **base,
element size (stride), and length** without any driver constant — e.g. a stride-2
sweep is a 16-bit table (the note table); a 4-byte stride is GoatTracker's 4-byte
pattern row (`Note|Instr|Cmd|Data`); a split-LR table shows as two stride-1 sweeps
over adjacent bases (GoatTracker's wave/pulse/filter tables are stored left-column
then right-column — a recognizable fingerprint, but recovered, not assumed).

### 4.3 Per-voice note-event stream — from the indexed traversal (P1 + P2 + P4)

- **Voice attribution by PC** (P1): the writes to `$D400+7v` come from voice `v`'s
  lane code site(s). Group the SID-write stream by issuing PC → three (or N) lanes,
  exactly, with **no gate-heuristic guessing**.
- **Note events from the pattern traversal** (P2/P4): the index sequence into the
  pattern table (or the ZP pointer walk) is *literally* the row stream the player
  reads. Each `NoteOn(dt, pitch, instrument-ref, duration[, seed])` is read off:
  `pitch` from the pattern row's note byte (resolved through the recovered note table
  to the canonical A440 12-TET grid, identical to the hand tokens), `instrument-ref`
  from the row's instrument byte, `duration` from the dwell before the index advances.
- This is **dynamic slicing** [Agrawal & Horgan 1990] with the SID write as the
  criterion: the slice is "row r of pattern p, read via this loop, produced this
  write" — the def-use chain made explicit.

### 4.4 Instruments — parameter sets the player reads (P0 + P1)

An instrument is the parameter set the player reads when a row names instrument `k`:
the instrument-table row at index `k` (recovered verbatim in §4.1) **plus** the
wave/pulse/filter sub-tables it points into (the pointers are bytes in that row;
follow them in the recovered RAM image). Because we read the *one shared table the
player indexed*, two notes naming instrument `k` reference the **same recovered
object by identity** — the 143× redundancy collapses to the table's true entry count
(GoatTracker ≤63 instruments), not 237 output-similarity clusters.

### 4.5 Patterns + orderlist — the traversal IS the structure (P2 + P4)

The orderlist is the index sequence into the pattern-number table per voice; the
patterns are the recovered pattern bytes. Repetition (a phrase replayed, a bass loop)
is **visible as the player re-reading the same pattern bytes / re-using the same
order index** — recovered, not induced by LZ over the output. Emit the per-voice rows
and the existing `_lz_emit_t` factors any residual repetition (REPEAT/TRANSPOSE), but
now most repetition is *already* expressed as orderlist reuse, so the LZ has far less
to do.

### 4.6 The gate (unchanged)

Render the recovered `BaccProgram` through the generic VM → require `(nframes,25)`
== bus-state byte-exact. Residual-zero is the gate; non-zero means the recovery
dropped information (fix it), never a patch. The per-register path stays as the
fallback for anything not yet covered (e.g. a genuine through-composed melody, or
FamiCommodore's voice-2 wavetable-pointer PW).

**Why this approaches driver-native size, generically.** The hand backend is small
because it stores *the driver's own tables + a sparse score*. §4.1–4.5 recover
exactly those tables (verbatim, by reading them off RAM) and exactly that score (by
reading the traversal), with **no per-driver constant** — only SID-chip semantics and
the universal init/play contract. So the recovered program is the driver's program,
re-serialized; its size is the driver's data size, not an RLE of the output.

---

## 5. Literature grounding (cite it)

The pipeline is a four-layer dynamic-analysis stack, each layer a named technique:

- **Value provenance — dynamic taint / data-flow** [Newsome & Song, *Dynamic Taint
  Analysis…*, NDSS 2005; Schwartz, Avgerinos & Brumley, *All You Ever Wanted to Know
  About Dynamic Taint Analysis and Forward Symbolic Execution…*, IEEE S&P 2010].
  Tag each data-region byte; propagate along execution; a SID write's taint names the
  source RAM byte/table. Address-taint handling matters for `LDA table,X` (the source
  address is computed). → §4.1, §4.4.
- **Def-use / control over a trace — dynamic slicing** [Korel & Laski, *Dynamic
  Program Slicing*, IPL 29(3):155–163, 1988; Agrawal & Horgan, *Dynamic Program
  Slicing*, PLDI 1990, pp. 246–256]. Backward slice from each SID write = the
  instruction chain (loop body, index increments, the branch taken) that produced it;
  concrete indices disambiguate which table entry was read. → §4.3, P1.
- **Tables + base/index — value-set analysis** [Balakrishnan & Reps, *Analyzing
  Memory Accesses in x86 Executables*, CC 2004, LNCS 2985:5–23; *WYSINWYX*, TOPLAS
  32(6):23, 2010]. Strided-interval a-locs: a load `base+stride·[lo,hi]` is an array
  of element-size `stride`, length `hi−lo`. The trace supplies ground-truth address
  sets, making boundary/element-size recovery exact. → §4.2, P2.
- **Trace-driven structure recovery** [Lin, Zhang & Xu, *Automatic Reverse
  Engineering of Data Structures from Binary Execution* (REWARDS), NDSS 2010;
  Slowinska, Stancescu & Bos, *Howard: A Dynamic Excavator for Reverse Engineering
  Data Structures*, NDSS 2011]. "Accesses sharing a base pointer = a record;
  constant-stride sweep over a base = an array." REWARDS' type sinks ← the typed SID
  registers (freq = 16-bit LE, ADSR = packed nibbles, waveform = bitfield) propagate
  backward onto the source tables. → §4.2, §4.4, P4.

**RE / SID-ripping practice** (the white-box precedent we are matching):
- **siddump** (Lasse Öörni/Cadaver) runs the player under a 6502 emulator, calls init
  once with the subtune in A, then calls play once per frame and snapshots
  `$D400..$D418`, deriving notes by diffing the register file and detecting the
  gate keyoff→keyon edge. `preframr-sidtrace` is the same idea with a full bus trace.
  (github.com/cadaver/siddump)
- **HVSC SID file format spec** — PSID/RSID header, `initAddr` (called once, subtune
  in A), `playAddr` (called per IRQ, ~50 Hz PAL / 60 Hz NTSC). The init-once /
  play-per-frame contract is what makes the written-once/read-only partition sound.
  (hvsc.c64.org/.../SID_file_format.txt)
- **Ripper "memory-map" practice** — rippers run a memory-access-logging sidplay to
  see which addresses init+play touched, then crop to that span; "trace every memory
  access to fully understand what each byte is doing" [SIDBlaster/Raistlin]. The
  read/write-set partition (§4.1) is the formalization of this. **SIDdecompiler**
  (Galfodo) traces execution to regenerate relocatable 6502 source — the strongest
  precedent for white-box recovery.
- **GoatTracker data structures** (the worked-example ground truth, via the manual
  and ChiptuneSAK's `goat_tracker.py` against goattrk2): per-channel **orderlist**
  (pattern# / repeat / transpose / RST+restart), **patterns** (4-byte rows
  `Note|Instr|Cmd|Data`), **instruments** (25-byte rows: AD, SR, wave/pulse/filter
  pointers, vibrato, gateoff, 1st-frame waveform, name), and **split-LR
  wave/pulse/filter/speed tables**. These are the byte layouts §4.1–4.5 must recover
  verbatim and cross-check against `pygoattracker`.

Full citations + URLs are collected in the references block at the end.

---

## 6. Worked example — Grid_Runner (GoatTracker_V2), with size estimate

Ground truth available: `pygoattracker` (`src/pygoattracker/model.py`) defines the
exact GT structures; `tests/test_fixtures/Grid_Runner.sid` is **4,051 bytes** on disk
(player + packed data + header). Current numbers (`generic_tracker_decompile.md`):
per-register genfits **161,002** → shipped tracker lift **12,821** tokens; **hand
backend 2,817**.

**Trace-through of how the richer log surfaces Grid_Runner's structure:**

1. **PC-grouping (P1).** GoatTracker re-blits the whole 25-byte SID shadow file each
   play-call from a fixed RAM shadow (`prov.py` already found `$13BA+r` on Grid). The
   blit is one `STA $D400,X` loop at one PC — so *all* SID writes share that PC, but
   the *shadow-file writes* (the real per-voice updates) come from the per-voice lane
   code at distinct PCs. Grouping the shadow-region writes by PC yields the three
   voice lanes + the global (filter/vol) lane exactly — no gate heuristic.
2. **Init write-map + read/write partition (P0+P3).** Init writes the depacked song
   tables into RAM; play only reads them. The partition hands back the GT
   instrument table (≤63 × 25 B), the orderlists, the patterns (4-byte rows), and the
   split-LR wave/pulse/filter/speed tables — **verbatim**, cross-checkable against
   `pygoattracker`'s reader on the same `.sid`.
3. **Indexed-read VSA (P2).** The pattern-read PC's effective addresses decompose to
   `pattern_base + 4·row` → stride-4 → confirms the 4-byte GT row and the pattern
   lengths. The orderlist-read PC's index sequence is the song order; its repeats are
   the orderlist `Rn`/loop. The note-table read is stride-2 (already recovered).
4. **Note events (§4.3).** Each voice's row stream is read off the pattern traversal:
   `NoteOn(dt, pitch, instr, dur)` directly, with `pitch` through the note table to
   the canonical grid. Instrument refs are the row instrument bytes → the ≤63 shared
   instruments by identity (not 237 output clusters).

**Size estimate.** The recovered program is GoatTracker's own content:
- instruments: ~tens of entries (≤63), each a small parameter row + table pointers;
- per-voice score: sparse `NoteOn` rows, with repetition expressed as orderlist reuse
  (the 5-note-bass-×-N and transposed phrases collapse to order references, not
  literal events);
- shared tables: note table + the four GT tables, stored once.

This is the same shape and content the **hand backend** serializes to **2,817**
tokens. Because the richer recovery reads the *same shared tables and the same
sparse score* (rather than clustering output fits), the expected serialized size is
**hand-backend-class — low thousands, ~2,800–4,000 tokens — i.e. roughly 3–4×
smaller than the current 12,821 and within a small factor of 2,817**, not a
1000× RLE of the output. The residual after instrument-by-identity + orderlist-reuse
is the genuine per-note data (a few ints/note), which is what the hand backend also
stores. (The exact number is what the §7 prototype must MEASURE; the claim here is
the *mechanism* that closes the 12,821→2,817 gap: dedup-by-identity and
repetition-as-orderlist replace dedup-by-similarity and repetition-as-LZ.)

---

## 7. Validation / staged plan

**Highest-leverage / lowest-cost first cut (the minimal patch):**

- **Step 0 (no patch): exploit the read stream already in `.bus.bin` (P0).**
  Implement the write-set/read-set partition and the "written-once-at-init,
  read-only-during-play = data" lift over the *existing* traces. Validate the lifted
  byte ranges against `pygoattracker`'s reader on `Grid_Runner.sid` (the tables must
  be byte-identical to what GT depacks). **This needs zero emulator change** and
  proves the core hypothesis before any C++ is touched. Do this first.
- **Step 1 (the patch): PC-per-access (P1) + init phase tag + init RAM write-map
  (P3).** Implement via §2.2 Option A (thread a CPU pointer + getters into
  `MemBusTrace`, mirroring the existing `clockFn` registration); add a `BUS_DT2`
  dtype + 4-byte version sniff to `bacc/generic/bustrace.py` (it already sniffs
  `RBT1`). Regenerate the patched libsidplayfp (the Makefile already applies the
  patch idempotently and forces a configure/regen). This is the smallest change that
  unlocks voice attribution by code site + the clean init/play boundary.
- **Step 2 (if needed): index-register / effective-address base (P2).** Add only
  after measuring whether P0+P1 already recover the pattern/orderlist traversal (with
  PC known, the base operand is recoverable from the static instruction bytes, so the
  index value may be the only new field needed).

**What to validate, and how (HARD RULE #0 throughout):**

1. **Tables byte-exact vs ground truth.** The §4.1 lifted instrument/pattern/order/
   wave/pulse/filter tables must be **byte-identical** to `pygoattracker`'s
   depack of the same `.sid` (and analogously `undmc`/`unjch` for DMC/JCH tunes).
   This is the genuine-program-data gate: the bytes come from the player's own RAM,
   never fabricated.
2. **Residual-zero render.** The recovered `BaccProgram` rendered through the generic
   VM must equal the bus-state `(nframes,25)` byte-exact — the existing
   `render_generic` + `residual` harness, re-pointed. Non-zero ⇒ the lift dropped
   information; fix the lift, never patch.
3. **Determinism.** The richer trace must stay run-to-run identical (the
   `powerOnDelay` pinning in `sidtrace.cpp` already guarantees this; PC/X/Y are
   deterministic functions of the same execution).
4. **No corpus regression.** The lift is admitted for a tune only when its render is
   residual-zero AND its token count beats the current path; otherwise the shipped
   tracker/per-register path remains the fallback. Keep the 12-tune / 8-driver
   corpus green.

**What to measure.**
- Recovered token count per tune vs the current tracker form (12,821 on Grid) and the
  hand backend (2,817) — target hand-backend-class.
- Distinct-instrument count: should drop from the 237 output clusters to the table's
  true entry count (≤63 for GT).
- Fraction of SID writes whose source RAM byte is identified by provenance (target
  ~100%, matching the existing 15689/15689 shadow-file provenance finding).
- Table byte-exactness vs `pygoattracker`/`undmc` (pass/fail per table).
- Trace-size overhead of the new fields (keep `.bus.bin` within a small constant of
  today via the 12-byte delta-cycle repack).

**Self-modifying-code & RSID caveats** (call out for the implementer): players patch
absolute operands in place; an address that *play writes* may be a code operand
(self-mod state), not data — the write-set subtraction in §4.1 handles this (those
addresses are excluded from the read-only data region). RSID tunes install their own
IRQ rather than the host calling `playAddr`; the phase tag should key off the actual
IRQ entry (recoverable from the `$0314/$0315` vector writes + stack frames) rather
than assuming the host drives play.

---

## 8. Summary of the proposal

Stop inverting the output; **read the program off the execution**. The emulator
already logs reads with addresses (P0, unused) and the CPU core already computes the
PC, index registers, effective address, and ZP pointer every cycle — so a *small*
patch (thread a CPU pointer into `MemBusTrace`; add PC + a phase tag + an init RAM
write-map) makes the player's own tables and traversal **directly observable**. The
recovery then lifts the song-data tables verbatim from the written-once/read-only RAM
region, attributes writes to voice lanes by PC, recovers table base/stride/length by
VSA over indexed reads, and reads the per-voice note-event stream off the
pattern/orderlist traversal — yielding the driver's actual program (hand-backend-class
size, byte-exact, generically), instead of a clever RLE of its output.

---

## References

1. Newsome, J., Song, D. "Dynamic Taint Analysis for Automatic Detection, Analysis,
   and Signature Generation of Exploits on Commodity Software." NDSS 2005.
   http://bitblaze.cs.berkeley.edu/papers/taintcheck-full.pdf
2. Schwartz, E.J., Avgerinos, T., Brumley, D. "All You Ever Wanted to Know About
   Dynamic Taint Analysis and Forward Symbolic Execution (but Might Have Been Afraid
   to Ask)." IEEE S&P 2010. https://users.ece.cmu.edu/~aavgerin/papers/Oakland10.pdf
3. Korel, B., Laski, J. "Dynamic Program Slicing." Information Processing Letters
   29(3):155–163, 1988.
   https://www.sciencedirect.com/science/article/abs/pii/0020019088900543
4. Agrawal, H., Horgan, J.R. "Dynamic Program Slicing." PLDI 1990, pp. 246–256.
   https://dl.acm.org/doi/10.1145/93542.93576
5. Balakrishnan, G., Reps, T. "Analyzing Memory Accesses in x86 Executables." CC
   2004, LNCS 2985:5–23. https://research.cs.wisc.edu/wpis/papers/tr1486.pdf
6. Balakrishnan, G., Reps, T. "WYSINWYX: What You See Is Not What You eXecute."
   TOPLAS 32(6):23, 2010.
   https://research.cs.wisc.edu/wpis/papers/wysinwyx.final.pdf
7. Lin, Z., Zhang, X., Xu, D. "Automatic Reverse Engineering of Data Structures from
   Binary Execution" (REWARDS). NDSS 2010.
   https://www.ndss-symposium.org/wp-content/uploads/2017/09/lin_0.pdf
8. Slowinska, A., Stancescu, T., Bos, H. "Howard: A Dynamic Excavator for Reverse
   Engineering Data Structures." NDSS 2011.
   https://www.cs.vu.nl/~herbertb/papers/howard_ndss11.pdf
9. siddump (Lasse Öörni / Cadaver). https://github.com/cadaver/siddump
10. HVSC SID file format specification.
    https://www.hvsc.c64.org/download/C64Music/DOCUMENTS/SID_file_format.txt
11. SIDdecompiler (Galfodo). https://github.com/Galfodo/SIDdecompiler
12. GoatTracker 2 manual + ChiptuneSAK GoatTracker layout.
    https://chiptunesak.readthedocs.io/en/latest/_modules/chiptunesak/goat_tracker.html
13. Raistlin / SIDBlaster ("trace every memory access"). https://c64demo.com/welcome-to-sidblaster/
14. codebase64 — playing music / IRQ player cadence.
    https://codebase64.org/doku.php?id=base:playing_music_a000-_ffff
