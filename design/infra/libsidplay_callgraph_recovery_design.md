# libsidplay CPU+bus call-graph recovery — one C app to replace headlessvice AND per-driver hand disassembly

**Status: DESIGN (investigation only — no implementation, no PR). 2026-06-21.**

**Goal.** One C application, built on the libsidplay family, that takes a `.sid` and emits TWO artifacts:
1. the per-frame SID register **dump** (retire `headlessvice`/VICE), and
2. a CPU history / register-write **provenance graph** rich enough to recover BACC programs
   **generically**, with NO hand-curated per-driver disassembly (today: `gt_unpack.py`, `lft.py`,
   `targets/dmc`, `hubbard.py`).

Companion docs: the generic-recovery rationale
[`../encoding/generic_bacc_recovery.md`](../encoding/generic_bacc_recovery.md); the thesis
[`../encoding/sid_player_decompiler.md`](../encoding/sid_player_decompiler.md); the dump-contract source
[`sid_to_dump_emulator_design.md`](sid_to_dump_emulator_design.md) (**superseded for APPROACH** — py65 →
libsidplayfp — but its **dump CONTRACT still holds**, cited throughout §3).

---

## 0. Returned summary (read this first)

- **Feasibility: YES, and most of it already exists.** The capability the brief asks to greenfield is
  largely built. Do NOT start from scratch.
- **THE KEY FINDING — `sidtrace` already IS the foundation.** `/scratch/tmp/sidemu/libsidplayfp/tools/sidtrace.cpp`
  is a working C++ app, built on **libsidplayfp** (the accurate modern fork), that already emits:
  (a) `<prefix>.sidwr.bin` — every SID register write `{cycle, addr, reg, val}`, the raw material for the
  headlessvice-compatible dump; and (b) `<prefix>.bus.bin` — the **full CPU bus trace** `{cycle, addr, val,
  rw}`, i.e. every read and write on the 6510 data bus, cycle-stamped. The "full bus sweep = 1,437,769
  exceptions, 0 unexplained" census in the memory log is exactly this app's bus trace fed to
  `exception_sweep.py`/`exception_explain.py`: every SID write classified as table-read / RMW-state /
  immediate. The register-provenance graph the brief wants is **the bus trace plus an offline provenance
  pass that already has working prototypes.**
- **Library recommendation: libsidplayfp, decisively.** The "proven, byte-exact, corpus-wide" dumps are
  produced by sidtrace, which is libsidplayfp-based; the corpus census (`full_scale.py`) validates
  sidtrace's libsidplayfp dump **against the VICE oracle** per-frame and finds the ≥99% byte-exact set. The
  user's "libsidplay2" should in practice be **libsidplayfp** — it is the accurate fork, it is what the
  proven results already use, and the instrumentation hook is already in its tree.
- **Graph schema core.** A columnar bus log (one row per bus access: `cycle:i64, addr:u16, val:u8,
  rw:u8`) is the substrate. PC / call-graph / data-flow provenance are **derived offline by replaying the
  opcode-fetch stream** that is already present in the bus reads (the opcode fetch IS a `cpuRead` of the
  PC). The only C-side change worth making is to **tag each access with its instruction's start-PC** so the
  offline pass need not re-derive PC; everything else stays in Python.
- **Recommended first build step.** Promote `sidtrace` from `/scratch/tmp` into the repo as the dump
  generator and prove **dump byte-parity vs headlessvice** on the existing fixtures (the census already
  shows ≥99%; close it to byte-exact on the fixture set and wire it into `tests/_dump_fixture.py acquire()`
  behind the existing VICE path). Provenance graph + generic recovery is phase 2.

---

## 1. What already exists (investigate-first findings)

### 1.1 `sidtrace` — the C app already exists, on libsidplayfp

`/scratch/tmp/sidemu/libsidplayfp/tools/sidtrace.cpp` (built binary present, ELF not stripped;
`build_sidtrace.sh` links the static `libsidplayfp.a` + the `sidlite` builder). Header comment states its
purpose verbatim: *"white-box SID recovery wrapper around instrumented libsidplayfp."* Usage:

```
sidtrace <file.sid> <subtune(1-based)> <nframes> <out_prefix> [kernal] [basic] [chargen]
```

Outputs (all already implemented):

| file | record (binary, packed) | meaning |
|---|---|---|
| `<prefix>.sidwr.bin` | `int64 cycle, uint16 addr, uint8 reg, uint8 val` | every write to `$D400-$D7FF` (any SID), reg = `addr & 0x1F`. **This is the dump source.** |
| `<prefix>.bus.bin` | `int64 cycle, uint16 addr, uint8 val, uint8 rw` (rw: 0=read,1=write) | **every CPU bus access** — the full read/write trace. The provenance substrate. |
| `<prefix>.meta.txt` | text | format, subtune, songs, init/play/load addr, speed, model, `cycles_per_frame`, `total_cycles`, `n_sid_writes`, `n_bus_accesses`, kernal used |

Engineering already handled: it **streams to disk and clears** per play-chunk (peak RAM = one chunk, not
the whole capture — the comment notes the old buffer-everything path hit ~928 MB/60 s); `SIDTRACE_NOBUS=1`
skips the large bus file for a cheap sidwr-only fidelity pass; optional real ROMs for RSID/KERNAL tunes;
PAL/6581/6526 forced to match the corpus host, but tune flags override when `force*=false`.

### 1.2 The instrumentation hook (already patched into the core)

`src/c64/membus_trace.h` defines a process-global `MemBusTrace` singleton with an `enabled` flag and a
cycle getter. The 6510 data bus is hooked in `src/c64/c64cpu.h` (`class c64cpubus`):

```cpp
uint8_t cpuRead(uint_least16_t addr) override  { uint8_t v = m_mmu.cpuRead(addr);
                                                  MemBusTrace::instance().record(addr, v, 0); return v; }
void    cpuWrite(uint_least16_t addr, uint8_t data) override { MemBusTrace::instance().record(addr, data, 1);
                                                  m_mmu.cpuWrite(addr, data); }
```

`src/c64/c64.cpp` installs the clock getter (`clockFn`/`clockCtx = &eventScheduler`) so every access is
PHI1-cycle-stamped. **Answer to the brief's hook question: yes, every bus read and write is hookable in the
MOS6510 core, and the patch already exists in this tree** — no upstream fork hunt needed.

**Important nuance the brief asks about (PC / instruction fetch / call graph).** The current hook records
the *data bus*, not an explicit instruction-fetch event, so **PC is not a column today**. But in the
libsidplayfp MOS6510 core (`src/c64/CPU/mos6510.cpp`) the opcode fetch is literally
`cpuRead(Register_ProgramCounter)` in `fetchNextOpcode()`, and the debug path already exposes
`cpu_debug->instrStartPC = Register_ProgramCounter`. So:
- **PC and the JSR/RTS call graph are RECOVERABLE OFFLINE** from `bus.bin` alone, by replaying the access
  stream through a tiny 6510 instruction-length table (the first read after each instruction's operand span
  is the next opcode fetch at the new PC). This needs no C change.
- **OR** (recommended, cheap) add an `instrStartPC` field to `MemAccess` populated from the core's existing
  `fetchNextOpcode()`, so each row carries the PC of the instruction that caused it. This turns the offline
  call-graph derivation from "re-disassemble the stream" into a `groupby(pc)` — a few lines in `c64cpu.h` +
  `membus_trace.h`. See §4.

### 1.3 The provenance sweep already runs and is "0 unexplained"

The memory-log claim — *"full bus sweep = 1,437,769 exceptions, 0 unexplained — every write is a
table-read / computed value"* — is produced by feeding `bus.bin` to existing Python:
- `exception_sweep.py` — sweeps tunes, runs sidtrace WITH bus, and `classify()`s every SID write whose
  value the mechanism model can't predict as **table / rmw / immediate** by walking back ≤K bus reads to
  find the RAM source address, then checking whether that source is itself written (RMW state) or read-only
  (table). Unexplained must be 0.
- `exception_explain.py` — per-tune exhaustive: for each unaccounted SET cell, find the SID write and trace
  its value's origin in the bus (RAM source addr → table or RMW var, immediate, or computed). Derives
  per-frame state from the bus directly (`state_from_bus`).
- `bus_table_demo.py` — grounds the INDEX op: shows pw/filter table values are `LDA table,X → STA $D4xx`
  reads of a contiguous RAM region recovered from the access log (no invention).

This **IS** the register-write provenance graph the brief specifies, already implemented as an offline
pass over `sidtrace`'s bus output.

### 1.4 The generic recovery probes already pass on multiple drivers

Per [`generic_bacc_recovery.md`](../encoding/generic_bacc_recovery.md) and the
`/scratch/tmp/sidemu/generic_*.py` probes (note-on, note-table auto-discovery, read/write state
classifier), the driver-invariant signals — note-on (gate-rise), note table (read-only 2-byte-strided
region matching freq writes), score location, generator state (RMW), free-running phase — are all
**auto-recovered with zero hardcoded addresses** on Hubbard (Monty, 5TT), GoatTracker (Grid Runner), and
Galway (Arkanoid). The only residual driver-specific element is the generator *arithmetic*, a bounded
archetype set, not unbounded per-driver cost. The prototype fitter is `generic_fitter.py` (+ `_lft`).

### 1.5 The current dump oracle is VICE; the new generator is libsidplayfp — reconciled

Today's corpus/fixtures come from VICE: `tests/_dump_fixture.py` shells out to the
`anarkiwi/headlessvice` Docker image running `vsiddump.py` over the `asid-vice` `vsid` binary; the dump
is built from the `dump2()` callback on every SID store (contract in
[`sid_to_dump_emulator_design.md`](sid_to_dump_emulator_design.md) §1). `sidtrace` reproduces that dump
from libsidplayfp and `full_scale.py` validates it **against the VICE dump per-frame** (PAL/NTSC-aware),
identifying the ≥99% byte-exact program-ground-truth set. So: **VICE is the legacy oracle; libsidplayfp
(sidtrace) is the replacement, already benchmarked against it.** The memory file is correctly named
`libsidplayfp-groundtruth`.

---

## 2. Library choice — libsidplay2 vs libsidplayfp

**Recommendation: libsidplayfp.** Rationale:

1. **Fidelity.** libsidplayfp is the actively-maintained accurate fork (residfp/reSID-fp filter model,
   cycle-accurate CIA/VIC, accurate `$D41B` osc3 / `$D41C` env3 readback, BA/badline timing). libsidplay2
   is the legacy line; its reSID is older and several readback/timing paths are less accurate. The whole
   point of using the libsidplay family over py65 is exactly these hardware-readback paths that read-driven
   players depend on (the hard part flagged in the superseded py65 doc §1) — libsidplayfp handles them.
2. **The proven results already use it.** Every "byte-exact, corpus-wide" claim in the memory log traces to
   `sidtrace`, which is libsidplayfp. Switching to libsidplay2 would discard the validation already done.
3. **The hook already exists in libsidplayfp's tree** (`membus_trace.h` + `c64cpu.h` patch). No equivalent
   exists for libsidplay2. The CPU core is fully exposed (the static lib links the MOS6510 directly; the
   `c64cpubus` override is the supported extension point).
4. **The user named "libsidplay2" for emulation quality** — libsidplayfp is the same family's modern,
   higher-quality member, so this honors the intent while picking the accurate fork.

Cycle- vs instruction-accuracy: libsidplayfp is **cycle-accurate**; `MemBusTrace` stamps each access with
the PHI1 cycle, so frame binning by IRQ/CIA cadence (the corpus convention) is exact. The builder used by
sidtrace is `sidlite`; for the dump-only path SID *output* fidelity is irrelevant (we record register
writes, not audio), but for any future audio-assisted check, link `residfp` instead — orthogonal to the
trace.

---

## 3. Dump output — headlessvice-compatible (contract reused)

The dump CONTRACT is unchanged from
[`sid_to_dump_emulator_design.md`](sid_to_dump_emulator_design.md) §1 (superseded for approach, contract
still valid). `sidtrace`'s `sidwr.bin` is the raw cycle-stamped write log — the same shape as VICE's
`dump2()` line stream — so the existing `vsiddump.py::process_dump` transform applies directly:

1. `clock = cycle` (absolute; sidtrace already gives absolute PHI1 cycle, no `cumsum` needed).
2. `irq` = absolute cycle of the most recent IRQ; sidtrace can emit IRQ cycles into `meta`/a sidechannel,
   or the host re-derives frame id from the SID-write cycle clusters (sidtrace's header explicitly notes
   "Frames are delineated by the host from the SID-write cycle clusters, the same way the corpus register
   dump uses the IRQ cycle as the frame id").
3. keep registers **0–24**, drop 25–31 (sidtrace already computes `reg = addr & 0x1F`; filter `reg <= 24`).
4. `reduce_res` don't-care masks: PW-high regs 3/10/17 `& 0x0F`; FC-low reg 21 `& 0x07`; res/route reg 23
   `& 0xF7`.
5. `squeeze_changes`: ffill per chip, drop rows where no kept register changed; `+1` boot alignment.

Net: the Python post-processor that currently consumes VICE output is reused **verbatim** with sidtrace's
`sidwr.bin` substituted for the `dump2` line stream. The dump is byte-identical by construction once
emulation parity (§8) holds.

---

## 4. The graph schema

### 4.1 Substrate — the columnar bus log (exists)

One row per CPU bus access, fixed-width little-endian (sidtrace's current `bus.bin`):

```
struct MemAccess { int64 cycle; uint16 addr; uint8 val; uint8 rw; }   // 12 bytes/row
```

Read as a numpy structured dtype on the Python side (already done):
`BUS_DT = np.dtype([("cyc","<i8"),("addr","<u2"),("val","u1"),("rw","u1")])`. Columnar/packed binary is the
right choice for volume: a 3-minute PAL tune is ~177 M CPU cycles; bus accesses are ~1–2 per cycle →
order 10^8 rows → ~1.2 GB raw. sidtrace already streams+clears so the **emulator's** peak RAM is one
chunk; the offline reader memory-maps the file and processes in windows.

### 4.2 The one recommended C-side addition — per-access start-PC

Add `uint16 instrPC` to `MemAccess`, populated from the core's existing `fetchNextOpcode()`
(`instrStartPC` is already tracked in the debug path). New row = 14 bytes. This makes the call graph and
data-flow provenance a `groupby` instead of a re-disassembly:

```
struct MemAccess { int64 cycle; uint16 addr; uint8 val; uint8 rw; uint16 instrPC; }
```

With `instrPC`, the offline pass gets, for free:
- **Call graph / playroutine structure** — JSR pushes a return addr to the stack (writes to `$01xx` with
  `rw=1` straddling a PC discontinuity) and RTS pulls it; grouping accesses by `instrPC` and detecting
  PC-jump edges reconstructs the JSR/RTS nesting → the per-voice code paths.
- **Per-SID-write provenance** — for each `STA $D4xx` (the write row), the `instrPC` names the storing
  instruction; walking back the same-PC-context reads gives the value's source addressing mode:
  - **immediate** — value came from an opcode-operand read (read at `instrPC+1`),
  - **indexed table-read[base,index]** — value came from a RAM read at `base + index` where `base` is
    constant across writes and `index` varies (the `LDA table,X` pattern `bus_table_demo.py` already finds),
  - **accumulator/RMW-cell** — value sourced from a RAM cell that is itself written (read-modify-write)
    each frame → rate/dwell/boundary candidate,
  - **computed** — value not equal to any recent bus read (arithmetic result).

If the C change is deferred, all of the above is still derivable from `bus.bin` alone by replaying the
opcode stream through a 6510 length table — the prototypes (`exception_*.py`, `bus_table_demo.py`) already
do the value-source walk WITHOUT PC. The PC column is an efficiency/clarity win, not a capability gate.

### 4.3 What to record raw vs summarize

- **Raw (C app):** the bus log (+ optional `instrPC`), the SID-write log, meta. Keep the C app a faithful
  **RECORDER** with at most light structural tagging (the `reg = addr & 0x1F` split is the only tagging it
  does, and that's fine).
- **Summarized (offline Python):** the call graph, the RAM read/write-cadence partition (TABLE / STATE /
  FREE-RUN / SCORE — `generic_classifier.py`), the note-table discovery, the per-instrument generator
  fits. These are derived, not recorded.

---

## 5. C-app vs Python split

**Recommendation: C app = faithful recorder; Python = all provenance/taint + BACC fitting.** This is
already the de-facto architecture (sidtrace.cpp records; `generic_*.py`/`exception_*.py`/`generic_fitter.py`
analyze) and it is the right boundary:

- **Determinism & auditability.** The C app does one thing — run libsidplayfp and dump the bus — so its
  output is a pure function of (`.sid`, ROMs, model). All interpretation lives in versioned Python that can
  be re-run on a frozen trace without re-emulating.
- **Iteration speed.** Provenance heuristics (the K-window value walk, archetype fitting) change often;
  keeping them in Python avoids a C rebuild per idea. The expensive, stable part (cycle-accurate emulation)
  is the C app; the volatile, cheap part (taint over a memmapped array) is Python.
- **Volume handling at the boundary.** The C app's job is to get 10^8 rows to disk cheaply (streaming,
  packed binary — done). Python memmaps and windows. Neither side holds the whole trace.
- **Reuse.** `generic_fitter.py` and the `exception_*` sweep are the existing Python analysis; the split
  lets them consume sidtrace output unchanged.

The only thing that belongs in C is what *requires* the emulator's internal state at capture time —
i.e. the cycle stamp (done) and optionally `instrStartPC` (§4.2). Everything else is offline.

---

## 6. Generic BACC recovery algorithm sketch (from dump + graph)

Input: `dump` (per-frame regs 0–24) + `bus.bin` (+`instrPC`). Output: a BACC program — per-voice sparse
score + pitch-invariant instrument generators on the canonical A440/12-TET grid — with NO per-driver hand
code. The pipeline, each step grounded in an existing probe:

1. **Note-on detection (driver-invariant).** Gate-bit (ctrl bit0) 0→1 transitions in the dump =
   note-ons; sparse on all drivers; 96.7% recall, zero false positives vs Monty's white-box score
   (`generic_noteon*.py`). Legato/tie gap closed by a freq-rewrite-without-gate cue (also bus-observable).
2. **RAM partition by read/write cadence (driver-invariant).** Classify every RAM cell from the bus:
   `TABLE` (read ≥50% frames, never written) · `STATE` (RMW, writes concentrated at note-ons) · `FREE-RUN`
   (RMW, writes NOT at note-ons) · `SCORE` (written at note-on cadence). Zero hardcoded addresses;
   recovers Monty's hand-coded `savelnthcc`/`savefreqlo` as SCORE and the PW-sweep counter as FREE-RUN
   (`generic_classifier.py`). **This removes the per-driver RAM map.**
3. **Note-table auto-discovery (driver-invariant).** The read-only 2-byte-strided region whose read pairs
   equal the freqlo/hi written that frame; byte-identical to the true table, ET ratio 1.05946 confirmed,
   on Monty and 5TT (`generic_notetable*.py`). Snap to the corpus-shared A440 grid.
4. **Score extraction.** Read the SCORE region at each note-on trigger → per-voice sparse note-on events
   (pitch index relative to the discovered table, instrument id, duration from the dwell to the next).
5. **Generator fitting (the only driver-specific, bounded part).** For each voice's per-frame deviation
   from the held note (vibrato/porta/PWM/arp), fit the FREE-RUN/STATE accumulators to a BACC archetype
   `value += rate every dwell; boundary; output map` — the bounded archetype set in
   [`generic_bacc_recovery.md`](../encoding/generic_bacc_recovery.md) §2 row 6, via `generic_fitter.py`.
   The exact-modulation algorithms (vibrato `tri(frame&mask)·step`, ping-pong porta, PWM sweep) are
   catalogued in [`../encoding/exact_modulation_algorithm`](../encoding) (recent design commit).
6. **Provenance closure (residual-0 gate).** Any SID write not explained by score+generator is traced via
   the bus to its source (table / RMW / immediate / computed) — `exception_explain.py`. Unexplained must be
   0 (the "0 unexplained" census discipline).

**Worked examples the graph must surface (and the probes show it does):**
- **Monty (Hubbard)** — white-box backend = the bar (0.238 tok/frame, residual-0). Generic pipeline
  recovers its score trigger (`$84CD`), note table (`$8456`≡`$8400` slice), and free-running PW sweep
  (`$84E5`) with zero hardcoded addresses.
- **Grid Runner (GoatTracker)** — partitions cleanly into TABLE/FREE-RUN/SCORE with the same rules,
  reproducing what `gt_unpack.py` extracts by hand.
- **A Mind Is Born** — `amind_*` probes in `/scratch/tmp/sidemu` already trace its tiny driver's state.
- **DMC** — the digi/percussion path; surfaces as FREE-RUN state + table reads, the generic analogue of
  `targets/dmc/format.py`.

---

## 7. What it replaces & honest end state

| Replaced | By | Confidence |
|---|---|---|
| `headlessvice` Docker / VICE `vsiddump.py` (dump generation) | `sidtrace` (libsidplayfp) + reused `process_dump` | **High** — already ≥99% byte-exact corpus-wide; close to byte-exact on fixtures. |
| `hubbard.py` (RAM-map + generator math) | generic partition + fitter | **High** — full worked example passes. |
| `goattracker.py` / `gt_unpack.py` | generic partition + fitter | **Med-high** — partition proven; generator archetypes to be confirmed corpus-wide. |
| `lft.py` / illegal-opcode RSID | sidtrace runs RSID natively (real ROMs) + generic fit | **Med** — emulation handled; LFT's unusual state may need an extra archetype. |
| `targets/dmc` | generic FREE-RUN/table recovery | **Med** — digis are the residual-risk class (50 writes/frame; pathological volume). |

**Honest edge cases.** Multispeed tunes (frame binning), digi/sample players (volume + non-musical writes),
and any driver whose generator arithmetic falls outside the current archetype set will keep edge handling.
The brief's "generic for all drivers" is true for the *RAM map and score/table partition* (driver-invariant,
proven on 4 drivers) and true for generators *up to a bounded archetype set* — not an unbounded per-driver
cost, but not literally zero driver knowledge either (the honest framing from
[`generic_bacc_recovery.md`](../encoding/generic_bacc_recovery.md): flavor A yes, strong flavor B no).

---

## 8. Validation & phasing

**Phase 1 — dump parity (retire headlessvice).** Promote `sidtrace` into the repo (build recipe +
vendored libsidplayfp patch, or a thin submodule). Prove **byte-exact** dump vs headlessvice on the
existing fixtures: run `sidtrace` → `process_dump` and diff against the VICE `.dump.parquet`. The census
(`full_scale.py`) already gives ≥99% corpus-wide and the clean set; tighten to byte-exact on the fixture
set, characterize the <1% divergences (model/readback/multispeed), then wire `sidtrace` into
`tests/_dump_fixture.py acquire()` as a subprocess **alongside** the VICE path (feature-flag, default VICE
until parity is locked, then flip). Risk: the <1% divergence class (readback-driven players, model
forcing) — characterize before flipping.

**Phase 2 — graph schema + one driver generically.** Land the bus-trace emission (already there) and,
optionally, the `instrPC` column (§4.2). Run the generic pipeline (§6) end-to-end on **Monty** and gate on
**residual-0** matching the white-box backend's score. Then **GoatTracker** (Grid Runner) to prove
cross-driver. Risk: provenance fidelity on dense tunes — the dump-only form is provably insufficient
(`generic_lanegrammar_probe.py`), so this MUST use the bus trace (flavor A), not the dump alone.

**Phase 3 — drop a hand backend.** When the generic pipeline matches `hubbard.py` residual-0 on the
Hubbard fixtures, retire `hubbard.py` (lowest-risk: it has the white-box bar to check against). Then
GoatTracker. Keep DMC/LFT hand backends until their archetypes are confirmed. Corpus-wide validation marches
toward the 63,043-subtune census.

**Cross-cutting risks.** (1) **Trace volume** — handled by stream+clear in C and memmap+window in Python,
but the `instrPC` column grows the file ~17%; keep the NOBUS sidwr-only path for the dump-parity work. (2)
**libsidplayfp hook availability** — the patch already exists in this tree; the risk is keeping it in sync
with upstream (vendor the patched core, don't track HEAD). (3) **Whether the generic fitter truly subsumes
the hand backends** — proven for the partition/score/table on 4 drivers; the generator-archetype coverage
is the open empirical question and the reason for the per-driver phased drop rather than a big-bang switch.

---

## 9. Recommended first build step (one line)

**Promote `/scratch/tmp/sidemu/libsidplayfp/tools/sidtrace` into the repo and prove byte-exact dump parity
vs headlessvice on the existing fixtures, then wire it into `tests/_dump_fixture.py acquire()` behind a
flag.** Everything downstream (bus trace, provenance sweep, generic fitter) already exists in prototype and
builds on that same binary.
