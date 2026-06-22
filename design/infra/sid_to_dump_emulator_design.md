# `.sid` → per-frame SID register dump emulator (retire `headlessvice`)

**Status: DESIGN (investigation only — no implementation, no PR).**

> **Approach superseded; contract durable.** The py65/`MPU65ILL` in-process emulator *approach* below
> is superseded — the shipped dump/recovery tool is the deterministic **`preframr-sidtrace`** binary
> (libsidplayfp-based, emits `.sidwr.bin` + `.bus.bin` in one run; see
> [`libsidplay_callgraph_recovery_design.md`](libsidplay_callgraph_recovery_design.md)). But the **dump
> CONTRACT** specified here (§1: the `dump2` write-stream, `reduce_res` masking, `squeeze_changes`,
> the per-frame (nframes,25) array) is **durable** and cited downstream — read it as the live contract.

**Goal.** Replace the `anarkiwi/headlessvice` VICE container with an in-process,
py65/`MPU65ILL`-based emulator that turns a `.sid` file into the **byte-exact**
register-write trace the codec consumes today. End state: the BACC codec parses
**only the `.sid`** and derives the dump in-process; `tests/_dump_fixture.py
acquire()` no longer shells out to Docker.

This is feasible. Per-player byte-exact CPU emulation is already proven three
ways in this repo/tree: the DMC probe (`/scratch/anarkiwi/cbm/undmc/probe_dmc.py`)
reproduces the siddump oracle 100.000 % on a digi tune; the shipped `lft`
backend runs an IRQ-driven illegal-opcode RSID byte-exact; and the generic
`IRQHarness` (`/scratch/tmp/sidemu/irq_harness.py`) already runs `play==0` RSIDs.
The work here is **not** new CPU emulation — it is (1) nailing the exact dump
contract VICE produces, (2) unifying PSID + RSID into one run loop, and (3) the
one genuinely hard part: a **hardware read-model** faithful enough that
read-driven players don't diverge from VICE.

---

## 1. The contract — what the dump is, byte-for-byte

### 1.1 How the dump is produced today (the oracle)

`tests/_dump_fixture.py::_render_dump` runs, in the `anarkiwi/headlessvice`
container, `vsiddump.py` which invokes the patched VICE binary `vsid`:

```
vsid -console -warp -sound -soundwarpmode 1 -sounddev dump -soundarg <fifo>
     -tune <n> -limitcycles <N> <file.sid>
```

`vsid` is built from `anarkiwi/asid-vice` (a fork of VICE). The `dump` sound
device is `src/arch/shared/sounddrv/sounddump.c`. On **every SID register store**
VICE calls (`src/sound.c:1793`):

```c
dump2(maincpu_clk - snddata.wclk,                       // clock_diff
      maincpu_clk - maincpu_int_status->irq_clk,        // irq_diff
      maincpu_clk - maincpu_int_status->nmi_clk,        // nmi_diff
      chipno, addr, val)                                // SID#, reg(0..31), byte
```

which prints one whitespace line `clock_diff irq_diff nmi_diff chipno reg val`
per write. So **the raw oracle is a cycle-stamped log of every write to the SID
register file** — not a per-frame snapshot. `vsiddump.py::process_dump` then:

1. `df["clock"] = clock_diff.cumsum()` → absolute CPU cycle of each write.
2. `df["irq"] = (clock - irq_diff).clip(lower=0)` → absolute cycle of the most
   recent IRQ at the time of the write.
3. `df = df[df.reg <= 24]` → keeps registers **0–24** only (drops 25–31).
4. `reduce_res` masks documented don't-care bits **in place**:
   - regs 3/10/17 (PW-high) → `& 0x0F` (PW is 12-bit; bits 4–7 ignored),
   - reg 21 (FC-low) → `& 0x07` (only 3 significant bits),
   - reg 23 (res/route) → `& 0xF7` (clear bit 3 = filter-external-input).
5. `squeeze_changes` ffills per chip and **drops rows where no kept register
   changed** vs the previous row (a value-change filter, *not* a per-frame
   collapse).
6. Writes Parquet (zstd) with columns `clock, irq, chipno, reg, val`
   (`UInt32, UInt32, UInt8, UInt8, UInt8`).

**This Parquet is the artifact the codec consumes.** The byte-exact target is
this table, modulo the masks in step 4 (already applied) and `squeeze_changes`
(order-preserving change filter). Our emulator must reproduce the same
`(clock, reg, val)` change-stream after the same masking + squeeze.

### 1.2 How the codec reads the dump (the consumer)

`codec/lane_grammar.py::per_frame_state` (and the parallel
`codec/lsp_validate.py::state_seq`) turn the write-stream into an **(nframes, 25)
forward-held register array**:

- Sort by `clock`, keep `chipno == 0`.
- Pick a framing period `cpf`: `None` → `lsp_validate.detect_play_period(cyc)`,
  else explicit (lft passes `16422`).
- `t0 = first_play_cycle(cyc, cpf)` — the cycle of the first regular play burst.
- For each write, maintain a running 32-reg vector `cur`; bin it to frame
  `fi = round((clock - t0) / cpf)`; the last `cur` seen in a frame wins; empty
  frames forward-hold the previous row.

**Definition of a "frame" for the contract:** a `cpf`-cycle bin of the absolute
cycle axis, anchored at the first play burst. For single-speed PAL tunes
`cpf = 19656`; `detect_play_period` returns the raster CPF unchanged unless the
inter-burst median is < 0.9·CPF (multispeed), in which case framing is at the
**sub-frame play period** so every play-call lands on its own row (lossless;
`test_multispeed_framing.py`).

**Multi-write-per-frame / hold semantics:** within a frame the *last* write to a
register wins for the per-frame array, but the **change-stream itself preserves
every write in order** (only `squeeze_changes`'s no-net-change rows are dropped).
"Hold"/"no change" is implicit: a register simply not written keeps its prior
value (forward-held by the consumer).

**Sub-frame / digi (gap #3):** digi players write a register hundreds of times
per frame (DMC ≈ 697 writes/frame). The per-frame array collapses these; the
raw change-stream does **not** — the emulator sees every bus write with its
cycle stamp, so it *natively* preserves sub-frame granularity. This is a bonus
the emulator unlocks (see §6); the must-have contract is only the per-frame
array equality the codec uses today.

### 1.3 Contract summary (what we must reproduce)

A Parquet table `(clock:UInt32, irq:UInt32, chipno:UInt8, reg:UInt8, val:UInt8)`:

- one row per **bus write to SID reg 0–24** whose masked value changes the kept
  state (post-`squeeze_changes`),
- `clock` = absolute CPU cycle (cumulative), `irq` = absolute cycle of the last
  IRQ, `val` already masked per `reduce_res`,
- `chipno == 0` for single-SID tunes (the only case the codec uses;
  multi-SID is out of scope for v1 — see §8).

The pragmatic acceptance test is **`per_frame_state(ours) == per_frame_state(headlessvice)`**
element-for-element on regs 0–24, at the same `cpf`. (Reproducing the exact
`clock`/`irq` columns is a stronger, optional target — useful for the digi
bonus and for `detect_play_period` to behave identically; see §7 risk note.)

---

## 2. Architecture — the minimal C64 around `MPU65ILL`

Reuse `MPU65ILL` (illegal-opcode 6502) for **all** tunes (not vanilla py65; lft
and others need LAX/SLO/… and it is a strict superset). The emulator is the CPU
plus the smallest C64 that makes player code run and reproduce VICE's writes.

Components (prose component diagram):

```
            .sid file
               │  load_psid()  (parse PSID/RSID header)
               ▼
   ┌──────────────────────────────────────────────────────────┐
   │  C64System                                                 │
   │                                                            │
   │  ┌────────────┐   bus    ┌──────────────────────────────┐ │
   │  │ MPU65ILL   │◄────────►│ Memory (64K) + IO decode      │ │
   │  │ (6502+ill) │  rd/wr   │  • RAM $0000–$FFFF            │ │
   │  └─────┬──────┘          │  • $D400–$D41F SID:           │ │
   │        │ .step()         │      writes → shadow + LOG    │ │
   │        │  (cycle count)  │      reads  → SidReadModel    │ │
   │        │                 │  • $DC00–$DCFF CIA1 (read)    │ │
   │   cycle clock ──────────►│  • $DD00–$DDFF CIA2 (read)    │ │
   │        │                 │  • $D011/$D012 VIC raster     │ │
   │        ▼                 │  • $A000/$D000/$E000 ROM (opt) │ │
   │  Run loop / IRQ harness  │  • $0001 bank register        │ │
   │   • PSID: JSR play       └──────────────────────────────┘ │
   │   • RSID: fake IRQ → vector                                │
   │   • advance clock by cycles consumed                       │
   └──────────────────────────────────────────────────────────┘
               │  write-log: (clock, reg, val) per store
               ▼
        mask (reduce_res) → squeeze_changes → Parquet
               │
               ▼   (or, in-process)  per_frame_state(...) → (nframes,25)
```

Key building blocks already exist and should be lifted, not rewritten:

- **PSID/RSID header parse:** `bacc/sidemu.py::load_psid` (handles version,
  data offset, embedded load address when `load_addr==0`, v2 flags). Extend it
  to expose the *clock* flag bits (PAL/NTSC, bits 2–3 of v2 `flags`) and the
  RSID discriminator (magic `RSID`), which the run loop needs.
- **SID write capture:** `sidemu._Mem` / `irq_harness.IRQMem` already latch
  `$D400–$D41F` writes into a 32-byte shadow and log `(reg, val)`. The new
  Memory adds a **clock stamp** to each logged write and a **read model** on the
  IO ranges.
- **Cycle clock:** py65's `MPU.step()` updates `mpu.processorCycles`; accumulate
  it into an absolute `clock` so each logged write carries `maincpu_clk`.
- **PSID call harness:** `sidemu.SIDEmu._call` (JSR-with-sentinel-return).
- **RSID IRQ harness:** `irq_harness.IRQHarness` (waitloop detection, vector
  detection $0314/$FFFA/$FFFE, Kernal-exit traps). This is the generalized form
  of the hardcoded lft machine and is the right basis for the RSID path.

---

## 3. PSID vs RSID — one run loop, two entry conventions

Detect from the header + `play_addr`:

- **PSID** (magic `PSID`, `play_addr != 0`): C64 environment is *simulated* by
  the player driver, not the tune. Run loop = `init(subtune)` once, then **call
  `play()` at the cadence** (one JSR per play-call). This is `SIDEmu.init` +
  `SIDEmu.play_frame` today. Cadence = single-speed → one play/frame; multispeed
  (PSID `speed` bits, or CIA-timer-set) → N plays/frame.
- **RSID** (magic `RSID`, or `play_addr == 0`): the tune is a real program that
  installs its own IRQ/NMI vectors and `CLI`s into a wait loop; the music is
  produced one interrupt at a time. Run loop = `init(subtune)`, **detect the
  waitloop + installed vector**, then **fake one interrupt per cadence tick**
  (push return frame, jump to vector, run to RTI/Kernal-exit). This is exactly
  `IRQHarness`.

**One unified path:** a `System.run(nframes)` that, after init, dispatches each
cadence tick either to a JSR-`play` (PSID) or a faked-IRQ (RSID), chosen once at
load time. The dump-logging Memory and the cadence machinery are shared; only
the per-tick "advance the player" primitive differs. The lft backend is then a
*special case* of the RSID path (its signature match can stay as a fast-path,
but the generic harness already runs it).

RSID subtleties to honor (from VICE/SIDPLAY conventions, needed for byte-exact):
- RSID **must not** be entered with the Kernal banked the PSID way; RSID runs
  with a real reset vector and the player sets `$01`. The generic harness's
  vector probe ($0314 then $FFFA then $FFFE) already covers "Kernal banked out →
  handler at $FFFE".
- RSID `init` is entered with **A = subtune-1** by VICE convention (SID files
  are 1-based externally; the value passed in A is 0-based). Confirm against the
  dumps — the DMC/lft probes pass the raw subtune; v1 must match VICE's exact
  convention (one of the first calibration checks, §5).

---

## 4. THE CRUX — SID/CIA/VIC readbacks (the #1 fidelity risk)

A pure CPU+RAM emulator is byte-exact **only if the player never reads hardware
state**. Many players do. Where a read feeds a *write decision*, a wrong read
value diverges the entire downstream trace. We do **not** need SID audio
synthesis (the dump is writes, not sound), but we **do** need read-register
fidelity. The current harnesses stub these to constants (`mem[$DC04]=0`,
`mem[$DC0D]=0x81`), which is fine for tunes that only poll-to-terminate but
**wrong** for tunes that read a value into the music.

The reads that matter, and the minimal model for each:

| Read | Who reads it | Model needed |
|------|-------------|--------------|
| **SID `$D41B`** (osc3 / "noise random") | Players using "SID as RNG" or osc3-driven mod (arps, random drums) | A tiny **osc3 oscillator**: accumulate `phase += freq3` each cycle/frame; `$D41B` returns top 8 bits of phase for pulse/saw/tri, an LFSR for noise. Must mirror reSID's phase accumulator timing. |
| **SID `$D41C`** (env3) | Players reading voice-3 envelope as a slow LFO | A tiny **ADSR state machine** for voice 3 driven by its gate/AD/SR writes. |
| **CIA1 `$DC04–$DC07`** timers | RSID players timing their own IRQ; tempo-from-timer | **Down-counting timer counters** clocked at φ2 from their latch, with the control-register start/stop/one-shot behavior. |
| **CIA `$DC0D/$DD0D`** ICR | IRQ-source ACK/test | Return "timer fired" appropriately (current `0x81` stub is the common case; a real model sets/clears per timer underflow). |
| **VIC `$D012`** raster, `$D011` bit 7 | Raster-synced players, raster splits | A **raster counter** advancing 0–311 (PAL) / 0–262 (NTSC) at the line rate, derived from the same absolute clock. |
| **CIA2 `$DD00`** etc. | VIC bank / rare | Usually inert for SID music; model on demand. |

**Why this is the #1 risk:** these are *coupled* models (reSID's osc3 phase and
CIA timing are cycle-accurate in VICE). py65 is **instruction-accurate, not
cycle-accurate within an instruction**, and its illegal-opcode cycle counts may
differ from VICE by a cycle here and there. So even a faithful osc3 model can be
off by a few accumulator steps if our cycle clock drifts from VICE's. A read
that the player *latches into a register write* then diverges the trace.

**Mitigation strategy (de-risk early, §9 phase 0):**

1. **Measure the blast radius first.** Before building any read model, run the
   existing dump corpus through a static scan: for each tune, does its player
   *read* `$D41B/$D41C/$DC0x/$D012` at all (instrument the harness's read log —
   `IRQMem.reads` already captures every read)? The memory log already implies
   most tunes don't ("full bus sweep = 1,437,769 exceptions, 0 unexplained"
   suggests reads are well-characterized). Stratify the corpus into
   **read-clean** (CPU+RAM suffices → byte-exact today) vs **read-coupled**
   (needs a model). Ship the read-clean majority first; the coupled set is a
   bounded, named work item, not an open-ended risk.
2. **Build read models incrementally, validated against VICE.** For each read
   register, calibrate the model on the *named* coupled tunes by diffing our
   trace vs the existing headlessvice Parquet at the first divergence (the DMC
   probe's "first divergence (frame,reg)" methodology). osc3/env3 first (most
   common: arps, random drums), then CIA timers, then VIC raster.
3. **The escape hatch that is NOT cheating:** for a read-coupled tune we cannot
   yet match, the shadow-rollout (§5) keeps it on headlessvice. We retire the
   container only for the strata we've proven byte-exact. Honest accounting:
   "N % of the census is emulator-byte-exact; the remainder is read-coupled and
   pending model X." No tune silently degrades.

**Determinism / RNG (the other crux input):** VICE initializes RAM to a known
power-on pattern (not all-zero — VICE's default RAM init is a repeating
`$FF/$00` block pattern with optional randomization, controlled by
`RAMInitStartValue`/`RAMInitValueInvert`/… resources). A player that reads
uninitialized RAM as a seed will diverge from an all-zero RAM. **Action:** read
the asid-vice `src/ram.c` RAM-init pattern and reproduce it exactly in
`Memory.__init__`; if VICE was run with default resources (it is — `vsiddump.py`
passes no RAM resources), match that default. This is a small, fixed, knowable
pattern — pin it as a calibration item (§5).

---

## 5. Timing & cadence

- **Cycle budget / frame:** PAL φ2 = 985248 Hz, raster CPF = **19656**
  cyc/frame (`lsp_validate.CPF`); NTSC = **17095** (`NTSC_CPF`). Select from the
  PSID v2 clock flags (and `load_psid` must expose them).
- **Frame definition for sampling:** match `per_frame_state` — bin absolute
  cycle by `cpf` anchored at the first play burst. The emulator's natural output
  is the *write-stream*; we feed it through the same `per_frame_state` for the
  per-frame array, so framing is identical by construction.
- **The +1 boot offset (DMC probe).** The probe samples **frame 0 = post-init
  state**, then one snapshot **after** each `play_frame` (`emulate()` builds
  `out=[state()]` then appends after each play). The lft backend does the same
  (`out[0]=boot`, then `out[f]=play()`). The dump's first row corresponds to the
  first play burst; the boot/init writes land before `t0`. **The emulator must
  sample after play N, matching the dump's "first play burst = frame 0" anchor**
  — and the init writes (boot frame) are handled by the same `first_play_cycle`
  anchoring the consumer already does. This +1 alignment is a known calibration
  knob, not a mystery; lock it on Monty first (single-speed, well-understood).
- **Multispeed (N play-calls/frame):** PSID multispeed is signalled by the
  `speed` longword (per-subtune 50 Hz vs CIA) **and/or** the player setting CIA1
  Timer-A to a sub-frame period. For RSID it falls out of the CIA-timer-driven
  IRQ rate. The emulator must fire `play`/IRQ at the **true** period so the bus
  write cadence matches VICE; the consumer's `detect_play_period` then frames it
  identically. Galway's `Times_of_Lore` is the gate fixture
  (`test_multispeed_framing.py`).

---

## 6. Sub-frame / digi (gap #3) — a bonus, kept out of the must-have path

Because the emulator logs **every** bus write with its cycle stamp (the raw
`dump2` stream), it already captures the ~697 writes/frame of a DMC digi tune —
the same granularity VICE emits *before* `squeeze_changes`. The per-frame array
the codec uses today collapses these, so for the **must-have** contract we apply
the same collapse and match. But the emulator natively exposes the finer stream,
so gap #3 (sub-frame digi representation) can later consume the un-collapsed
write-log directly. **Flag only:** do not let digi sub-frame work block the
container retirement; the must-have path is the per-frame array.

---

## 7. ROMs & illegal opcodes

- **Illegal opcodes:** always use `MPU65ILL` (`bacc/illegal_mpu.py`). It is a
  drop-in py65 subclass implementing the NMOS combined-ALU illegals (LAX/ALR/
  AXS/SLO/SRE/ANC/…). No tune should run on vanilla py65.
- **ROMs:** some tunes JSR into KERNAL/BASIC/CHAR ROM (the survey's
  `Great_Giana_Sisters` crashed for lack of a Kernal ROM). Design:
  - Add an **optional ROM image set** (KERNAL `$E000–$FFFF`, BASIC
    `$A000–$BFFF`, CHARGEN `$D000–$DFFF`), loadable from a configured path
    (the standard C64 ROMs, same images VICE ships — not committed to the repo;
    resolved like the HVSC mirror).
  - Model the `$01` bank-select register so ROM/IO/RAM banking matches what the
    player expects (PSID players that bank the Kernal in to use its routines;
    RSID players that bank it out to put a handler at `$FFFE`).
  - ROMs are **optional**: read-clean RAM-only tunes don't need them; the loader
    falls back gracefully and a tune that traps into absent ROM is reported as
    "needs ROM," not silently wrong. This mirrors the honest-accounting rule.
- **Determinism:** §4 RAM-init pattern + fixed ROM images = bit-reproducible.

---

## 8. Open questions & the biggest honest risks

1. **Cycle-exactness gap (biggest risk).** py65 is instruction-accurate; VICE is
   cycle-accurate. For **register writes** this is usually irrelevant (the *value*
   written is determined by program logic, not sub-instruction timing) — which is
   why the DMC and lft probes hit byte-exact. It bites only where (a) a player
   reads a **time-coupled hardware register** ($D41B osc3, CIA timer, $D012
   raster) whose value depends on the *exact* cycle, and (b) latches that read
   into a SID write. There, a 1-cycle clock drift can flip a written byte.
   **Cost if it bites:** that tune (or that frame onward) is not byte-exact under
   py65 and stays on headlessvice, OR we accept it into the sub-frame digi bonus
   only. **Mitigation:** the §4 phase-0 scan bounds exactly which tunes are at
   risk; the shadow rollout never ships a divergent tune. We should *measure*,
   not assume, what fraction this is — the memory log's "≥99 % byte-exact census"
   suggests it is small, but that census was over a different (white-box backend)
   path; re-measure for this CPU+read-model path.
2. **`clock`/`irq` column exactness.** The per-frame array only needs *ordering*
   and *binning* to match, not the literal `clock` values. But `detect_play_period`
   and `first_play_cycle` depend on inter-burst **cycle gaps**, so our absolute
   clock must track VICE closely enough that the burst structure (gaps > 2000
   cyc) is identical. Instruction-accurate cycle counts should preserve burst
   gaps (they are thousands of cycles); verify on the multispeed fixture.
3. **RSID `init` register convention** (A = subtune vs subtune-1), stack/flags
   init state, and the exact "wait loop" the IRQ interrupts — calibrate against
   VICE on A_Mind_Is_Born and one CIA-timer RSID.
4. **Multi-SID (`chipno > 0`).** The codec uses `chipno == 0` only; v1 emulates a
   single SID at `$D400`. 2-/3-SID tunes are out of scope (named, deferred).
5. **NMI-driven players** ($FFFA) — the harness handles the vector but NMI cadence
   (often raster or CIA2) needs the same timer model as §4.

---

## 9. Phased build plan (de-risk the readback question earliest)

**Phase 0 — Read-coupling census (de-risks the #1 risk before building).**
Instrument the existing `IRQHarness`/`SIDEmu` read log over the gate fixtures +
a stratified HVSC sample. Classify each tune **read-clean** vs **read-coupled**
(and by which register). Output: the exact share that CPU+RAM-only can already
nail, and the named coupled set. *This decides scope before any code is written.*

**Phase 1 — Unified single-speed PSID path, byte-exact on read-clean tunes.**
Promote `SIDEmu` to the dump-logging `System`: cycle-stamped write-log, `MPU65ILL`,
`reduce_res` masking + `squeeze_changes`, Parquet output. Lock the +1 boot
alignment and `per_frame_state` equality on **Monty_on_the_Run** (single-speed,
read-clean). Add **5_Title_Tunes, Grid_Runner** (PSID gate fixtures).

**Phase 2 — Unified RSID/IRQ path.** Fold `IRQHarness` into `System` as the
RSID per-tick primitive. Validate **A_Mind_Is_Born** (lft) and one CIA-timer
RSID byte-exact at the CIA cadence.

**Phase 3 — Multispeed + digi cadence.** Fire play/IRQ at the true sub-frame
period. Validate **Times_of_Lore** (Galway, multispeed) and **Ode_to_Music**
(DMC digi) — the latter using the DMC probe's existing 100.000 % result as the
oracle.

**Phase 4 — Read models (osc3/env3 → CIA timers → VIC raster).** Build only the
models the Phase-0 census says are needed, each calibrated by first-divergence
diffing against the headlessvice Parquet on its named coupled tunes.

**Phase 5 — ROM + bank model.** KERNAL/BASIC/CHARGEN images + `$01` banking;
validate Great_Giana_Sisters (the known ROM-crash case).

**Phase 6 — Census scale-out + shadow rollout (see §10).**

---

## 10. Validation & shadow-rollout plan (specify; do NOT implement)

**Equality metric.** For each fixture: render the dump both ways (headlessvice
container *and* the emulator), run **both** through `per_frame_state` at the same
`cpf`, and assert element-wise equality on regs 0–24 across all frames (the DMC
probe's `agreement` + `first divergence (frame,reg)` reporting, generalized).
Stronger optional check: the raw `(clock,reg,val)` change-streams match after
`reduce_res`+`squeeze_changes`.

**Fixture gate** (these must be byte-exact before any container retirement):
Monty_on_the_Run, 5_Title_Tunes, Grid_Runner, Need_More_NOPs, Not_Even_Human,
Twilight, A_Mind_Is_Born, Times_of_Lore (Galway multispeed), Ode_to_Music (DMC).
These already live in `tests/_dump_fixture.py::GATE_FIXTURES` with headlessvice
dumps; reuse them as the oracle.

**Corpus scale-out.** Stratified sample → toward the **63,043-subtune census**.
Run emulator vs cached headlessvice Parquet across HVSC; report per-subtune
byte-exact / first-divergence, bucketed by the Phase-0 strata (read-clean vs
each coupled register vs ROM-needed). Track the byte-exact percentage as each
read model lands.

**Shadow rollout (run-both-then-drop).**
1. `acquire()` gains an `emulator=` mode that produces the dump in-process and a
   `shadow=` mode that produces **both** and diffs them, logging any divergence
   (never failing the codec — diagnostic only).
2. CI runs shadow over the gate set + a rotating corpus slice; divergences are
   triaged into a stratum (read model / ROM / cycle-exact).
3. Retire the container **per stratum**, only after that stratum is 0-divergence
   over its census slice. The read-clean majority can drop headlessvice first;
   read-coupled tunes drop as their model lands.
4. Final: `_render_dump`/the Docker dependency is removed; `acquire()` resolves
   the `.sid` and derives the dump in-process via the emulator.

**The `_dump_fixture` / `acquire` change.** Today `_resolve_dump` → `_render_dump`
shells out to `docker run … vsiddump.py`. Target: `_resolve_dump` calls the
in-process emulator (`System(load_psid(sid)).run_dump(subtune, limit_cycles)`)
and returns the Parquet — *no Docker, no FIFO, no container image*. The
`_songlength_cycles` budget logic is unchanged (still bounds the run). During
rollout, an env flag (`PREFRAMR_DUMP_BACKEND=emulator|headlessvice|shadow`)
selects the path so CI can compare; the headlessvice branch is deleted only at
the end. Because the dump is then derived from the `.sid` alone in-process, the
codec's two-input dependency (`.sid` + `.dump.parquet`) collapses to **`.sid`
only** — the stated end state of this migration.

---

## 11. Summary

- **Feasibility: high.** Byte-exact CPU emulation of SID players on
  `MPU65ILL` is already demonstrated (DMC 100.000 %, lft byte-exact, generic
  IRQ harness running `play==0` RSIDs). The contract is fully pinned: the oracle
  is VICE's `dump2` per-write trace (`sounddump.c`), masked by `reduce_res` and
  change-filtered by `squeeze_changes`, consumed as a per-frame (nframes,25)
  array by `per_frame_state`. No SID *audio* synthesis is required.
- **#1 fidelity risk: hardware READBACKS** ($D41B osc3, $D41C env3, CIA timers,
  $D012 raster) latched into write decisions, where py65's instruction- (not
  cycle-) accuracy can drift the read value. **Mitigation:** a Phase-0
  read-coupling census bounds the at-risk set before any build; read-clean tunes
  ship first byte-exact; read models land incrementally, each calibrated by
  first-divergence diffing against the headlessvice Parquet; the shadow rollout
  retires the container only per proven-exact stratum (no tune silently
  degrades). Determinism pinned by matching VICE's RAM-init pattern (`asid-vice
  src/ram.c`) and fixed ROM images.
- **Recommended first build step:** Phase 0 — instrument the existing read log
  (`IRQMem.reads`) over the gate fixtures + a stratified HVSC sample to produce
  the read-clean vs read-coupled census. It costs little, reuses code already in
  the tree, and tells us exactly how much of the corpus the simplest CPU+RAM
  emulator already nails — turning the open-ended fidelity question into a
  bounded, named worklist before a line of emulator code is committed.
```
