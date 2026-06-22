# A generic per-axis SID decompiler: recovering the *generating process* from execution alone

**Status: DESIGN / RESEARCH PROPOSAL (2026-06-22). No build.** This document specifies
a generic dynamic decompiler that, for each SID control axis, reverse-engineers the
*actual computation* the 6502 playroutine uses to drive that axis — its persistent
state, its per-call update rule, and its state→register output map — **purely from
dynamic analysis of the program's execution**, assuming **zero a-priori knowledge** of
any tracker/player format. Table-driven and generative (accumulator / LFSR / closed-
form / self-modifying) recoveries are two outcomes of the *same* procedure. The sole
correctness gate is the **residual-zero render**: re-execute the recovered per-axis
programs and require byte-exact SID register streams.

## 0. Scope, constraint, and what supersedes what

**Oracle constraint (non-negotiable).** The existing depackers (`pygoattracker`,
`undmc`, `gt_unpack`) and the per-driver / identity recovery code (`identity.py`,
`distill.py`, the hand backends, `recover_program`) are **validation oracles only** —
byte-exact pass/fail checks. Their *interface* (confirmed by reading only signatures):

- `recover.residual(program, bustrace) -> (resid, rendered, state)` where
  `sum(resid.values()) == 0` is whole-tune residual-zero (the gate);
- `recover.render_generic(program) -> ndarray[nframes, 25]`;
- a depacker (e.g. `pygoattracker`) emits the tune's data bytes; a recovered region is
  *byte-equal* to that or it is not (yes/no).

This design uses these **only** to assert byte-exactness. It does **not** read, crib,
or depend on any of their parsing/format logic. Nothing below knows what a "pattern",
"orderlist", or "instrument" is.

**What this supersedes.** Two prior families, both *output*-shaped and therefore
**length-proportional**:

1. **The archetype "zoo" / CITG fit** (`unified_generic_recovery.md`,
   `generic_bacc_recovery.md`). It fits ~21 renderers to the per-register *output
   stream*, segmenting by gate-rise note-ons. It is a lossless RLE/dictionary of the
   OUTPUT: its size grows ~linearly with playback length (Goto80 generic measured
   10.5k→34k tokens as the capture window grew). It recovers *playback*, not the
   program. It also fails *structurally* on the generative case: A Mind Is Born has
   **zero note-ons on voices 0/1**, so its segmenter finds nothing to fit
   (`amind .../RECOVERY.md` §"Generic fitter verdict").

2. **The tracker decompiler + recover-by-identity** (the `#143/#145/#148` line, and the
   SDST artifact's SNAP/SIDW/IDXR sections in `sidtrace_program_recovery.md`). These
   assume the init-once/play-per-frame "song-data table" contract and lift a
   contiguous never-written RAM region as "the tables". That contract is a
   *format assumption*: it presupposes the axis is driven by a table read out of
   resident data. Goto80's drivers (JCH NewPlayer, DefMon) have **no depacker** and his
   catalog stays length-proportional under it; A Mind Is Born has **no tables at all**
   (`amind .../RECOVERY.md`: "no note-table / pattern / orderlist … the score is the
   static seed; everything else is generated arithmetically from one counter"). "Where
   is the note table" is the wrong question — a table is only *one* possible frequency
   generator.

The reframe: **a table-walk and a generative recurrence are the same object** — a
state machine `(state, update, output, constants)` — recovered by the same dynamic
data-flow procedure, differing only in whether the update rule happens to dereference
RAM. We never *assume* either; we *observe* which one the program is.

---

## 1. The per-axis recurrence model

### 1.1 The axes

The 25 SID registers ($D400–$D418) partition into control *axes*. We recover an
independent generating process for each:

| axis | registers (per voice v∈{0,1,2}) | width |
|---|---|---|
| freq lo / hi | $D400+7v, $D401+7v | 16-bit pair |
| pulse width | $D402+7v, $D403+7v | 12-bit |
| control (waveform/gate/sync/ring) | $D404+7v | 8-bit bitfield |
| attack/decay | $D405+7v | 8-bit (2 nibbles) |
| sustain/release | $D406+7v | 8-bit (2 nibbles) |
| filter cutoff lo / hi | $D415, $D416 | 11-bit |
| filter res / routing | $D417 | 8-bit bitfield |
| mode / volume | $D418 | 8-bit |

The axis partition is a *register-naming* fact about the chip (which we are allowed —
it is hardware, not a tracker format), **not** a claim about how any program computes
them. Two axes may share state; we discover sharing, never assume it (§2.6).

### 1.2 The model

For each axis A, recover a tuple

```
G_A = (S, U, O, K)
  S : the persistent STATE the playroutine keeps for A  — a finite tuple of bytes
      (zero-page cells, registers, self-modified operands) that carries between
      play-calls.
  U : the UPDATE rule  S' = U(S, K, X)  applied once per play-call. X is the set of
      EXOGENOUS inputs read this call that are NOT part of A's own state: a shared
      counter, a sibling axis's carry-out, a timer/RNG read. Pure-arithmetic axes
      have X = ∅.
  O : the OUTPUT map   out = O(S, K)  : state -> the byte(s) written to A's register(s).
  K : the CONSTANTS U and O reference — recovered by IDENTITY from the program's own
      RAM/operands via execution (never from a format spec). A "table" is K being an
      array that O indexes; a "rate" is K being a scalar U adds.
```

The recovered program for the whole tune is `{G_A}` over the axes, plus the shared
exogenous sources (the master counter, the RNG) recovered once and referenced by `X`.

### 1.3 How every generator class is one instance of this model

The model is deliberately general (a Mealy/Moore machine over a finite byte-state with
arithmetic + indexed-read transitions); the *classes* below are not special cases in
the code, they are descriptions of what `U`/`O`/`K` came out as:

- **Table-walk (Grid_Runner freq).** `S = (idx)`; `U: idx' = idx + stride` (advance
  clock; may stall — §2.5); `O: out = K_table[idx]`; `K = {K_table, stride}`.
  K_table is recovered by **identity** (§2.4): the bytes O reads are read *verbatim
  from the program's own RAM*, at addresses the execution dereferenced.
- **Accumulator / sweep (A Mind Is Born PW, filter cutoff).** `S = (acc)`;
  `U: acc' = (acc + rate) mod 2^w`; `O: out = acc` (or `out = hi(acc)`, or
  `out = 7 + 8*(acc>>7)` for the filter); `K = {rate, w}`. No RAM read — `X = ∅`,
  K is the immediate operands the execution used.
- **"Skydive" (A Mind Is Born voice-0 freq).** `S = (fhi)`;
  `U: fhi' = fhi − 1`, with a reset `if fhi == floor: fhi' = next_start` where
  `next_start` is selected by a bit of the shared counter; `O: out_hi = fhi`. This is
  an accumulator with a data-dependent reset — recovered as a conditional update whose
  predicate is observed in the trace.
- **Gated re-trigger (A Mind Is Born voice-2).** `S = ∅` for the value; the *control*
  axis toggles `ctrl: 0x60↔0x61` every 32 calls — `U` is keyed on a bit of the shared
  counter (`X = {C}`). Pitch/PW are set-once constants (`K`).
- **LFSR (SID-noise-driven or arithmetic PRNG axis).** `S = (lfsr)`;
  `U: lfsr' = (lfsr << 1) | (parity of tapped bits)`; `O` reads a slice. The feedback
  polynomial (the taps) is `K`, recovered by **Berlekamp–Massey** over the observed
  state sequence (§2.7) — *if* the axis is genuinely an LFSR; otherwise BM reports high
  linear complexity and we fall back (§5).
- **Self-modifying-operand state (SMC).** `S` includes the *operand byte of an
  instruction* that the program writes to itself. A location that is both EXEC and
  WRITE during play (already classified by the access-type map, §3) is **not data and
  not constant** — it is mutable state. The data-flow slice naturally pulls it in: the
  write that patched it is in the slice; its value persists between calls; it is a
  state cell exactly like a zero-page byte (§2.3).

### 1.4 Length-independence (the central property)

`G_A` is a **fixed-size** object: `S` is a few bytes, `U`/`O` are short expression
DAGs, `K` is the constants the program actually references. **It does not grow as the
tune plays longer.** A table-walk stores `K_table` once and re-indexes it; the zoo
stored one fit-segment *per traversal* of that table (Grid_Runner: "33,913 inline fit
segments but only 237 distinct signatures", `sidtrace_program_recovery.md` §1). An
accumulator stores `(acc0, rate, w)`; the zoo stored a fresh `accum` piece per
note-on segment. **The recurrence is the program; the trace is its unrolling.** This
is precisely the Daikon/loop-summary stance: a relation `S' = U(S)` that holds across
*all* observed iterations is one fact, not N facts [Ernst et al. 2001].

The residual-zero gate is what makes length-independence *honest*: a recovered `G_A`
must reproduce the byte-exact stream for **all N frames** by re-execution, so it cannot
be a truncated/lossy summary — if it under-fits, the residual is nonzero and the
recovery is rejected (§2.8, §5).

---

## 2. The recovery + synthesis algorithm

The pipeline is a four-stage dynamic-analysis stack: **slice** (isolate what computed
each write) → **symbolize** (extract the per-write computation DAG) → **generalize**
(close the DAG into a recurrence over frames) → **verify** (residual-zero re-execution).
Each stage is a named technique from the literature.

### 2.1 Stage A — per-write backward dynamic slice

**Criterion.** Each executed `STA $D4xx` instance (one per axis-register write per
frame). **Technique: backward dynamic slicing over the Dynamic Dependence Graph**
[Korel & Laski 1988, *IPL* 29(3):155–163; Agrawal & Horgan 1990, *PLDI*, pp. 246–256].

In the DDG each node is one *execution instance* of an instruction; edges are the
data- and control-dependences *actually observed* on this run (Agrawal & Horgan's RDDG
merges instances with identical dependence structure, bounding size by distinct
patterns, not trace length — which is exactly what lets us store the recurrence, not
the unrolling). The backward slice from a `STA $D4xx` instance follows:

- the data edge from the store to the instruction instance that last defined A
  (`LDA/ADC/AND/TXA/PLA/…`); recurse;
- an `LDA $addr` pulls in the instance that last *wrote* `$addr` (could be an earlier
  frame — that is how persistent state enters the slice);
- `ADC`/`EOR`/etc. pull in both operand sources;
- indexed/indirect modes (`LDA tbl,X`, `LDA (zp),Y`) pull in the instances that set
  X/Y or the pointer bytes;
- control edges pull in the `CMP`/`BNE`/`BEQ` instances that decided this path ran
  (this captures the skydive reset predicate and the every-32-frames gate).

The slice contains **exactly** the executed instructions that flowed into that byte —
no statically-possible-but-untaken code, no code that ran but did not contribute. This
is what isolates *one axis's generator* from the rest of the playroutine without any
format knowledge.

### 2.2 Stage B — symbolic expression per write (the computation DAG)

Along the single concrete frame's path, build a **symbolic store** mapping each
location to an expression over a set of *free symbols* (the axis's prior state cells,
the exogenous inputs, the referenced constants). **Technique: concolic / trace-based
symbolic execution** [Godefroid–Klarlund–Sen, DART, *PLDI* 2005; Sen–Marinov–Agha,
CUTE, *ESEC/FSE* 2005; Schwartz–Avgerinos–Brumley, *IEEE S&P* 2010, for the operational
semantics]. Each 6502 op updates the symbolic store as it updates the machine
(`ADC #2` → `A := A + 2`; `LDA tbl,X` → `A := select(tbl, X)`; branches resolve
concretely and append to the path constraint, unrolling the frame). At each
`STA $D4xx`, snapshot A's expression — that **is** the closed form for the byte
written this frame, e.g.

```
$D409  <=  (acc + 2) & 0xFF                 # A Mind PW, no memory read
$D400  <=  notetable[ base + 2*idx ]        # Grid_Runner freq-lo, a table read
$D401+7v <= fhi - 1                          # A Mind skydive
```

The expression is a **DAG** rooted at the written byte; leaves are free symbols
(prior-state cells / constants / exogenous inputs); shared subexpressions share nodes.
**Value provenance** — *which* RAM bytes / state cells flowed in — is the same
information, obtained either as the slice's leaf set or, equivalently, by **dynamic
taint with label-set tags** [Newsome & Song, NDSS 2005; Schwartz et al. 2010]: tag each
candidate state byte and constant, union tags on every ALU op, and read the label set
at the store. We use slicing+symbolic execution as the primary mechanism and taint as
the cross-check that the leaf set is complete.

The **pure-arithmetic / no-memory-read case is handled for free**: if the slice
contains no data `LDA $addr` (only immediates and register arithmetic), the DAG's leaves
are immediates + prior-state cells. There is no table; `K` is the immediate operands.
This is the case that defeats every "find the table" approach and is native here
(A Mind Is Born is *entirely* this case).

### 2.3 State identification (what persists between calls)

A free symbol is **state** for axis A iff its leaf, at the start of frame *i+1*, holds
the value written to it during frame *i* (i.e. it is read this frame and was defined a
previous frame, with no init-only definer in between). Operationally: the leaf's last
definer is in a *prior frame's* slice. State cells are therefore discovered, not
assumed; they may be:

- **zero-page cells** (the common per-voice cursor / accumulator — A Mind's `zp$13`
  counter, `zp$20` section counter);
- **self-modified operand bytes** (EXEC&WRITE during play — already flagged by the
  access-type map; the write that patches the operand is in the slice, the patched
  byte persists, so it is a state cell);
- **CPU registers** carried across calls only if the player preserves them (rare;
  detected the same way).

Everything else in the leaf set is either an **exogenous input** `X` (a shared counter
read but not owned by A — §2.6) or a **constant** `K` (read-only, never written during
play; verified against the access-type map: READ_PLAY & ¬WRITE_PLAY & ¬EXEC).

### 2.4 Constant / table recovery by identity (no format spec)

`K` is recovered by **identity from the program's own RAM**, via the addresses the
execution actually dereferenced — never from a layout spec:

- **Scalar `K`** (a rate, a floor, a mask): the immediate operand bytes the symbolic
  expression used. Read off the instruction stream in the slice.
- **Array `K` (the table sub-case):** for an indexed read `tbl, X` appearing in O, the
  set of effective addresses over the run is a **strided interval** `base + stride·[lo,hi]`
  — **value-set analysis** [Balakrishnan & Reps, *Analyzing Memory Accesses in x86
  Executables*, CC 2004, LNCS 2985:5–23; *WYSINWYX*, *TOPLAS* 32(6):23, 2010]. This
  yields `base` (lower bound), `stride` (element size), and length `(hi−lo)/stride + 1`
  **exactly**, with no driver constant. The bytes are then lifted **verbatim** from the
  post-init RAM image at `[base, base+stride·(hi−lo)]`. They are the table because we
  watched O index them — *not* because a contiguous never-written region was guessed to
  be "the note table". (This is the one place the existing IDXR section of the SDST
  artifact already supplies the right primitive — §3.)

This satisfies the project's HARD RULE #0 (recovered structure is read from the
program's own bytes, never re-fit from output) *by construction* and *generically*: it
applies whether the array is a note table, an arp table, a wavetable, or a sweep-step
table, and it does nothing at all when there is no array (the generative case).

### 2.5 Stage C — generalize the per-frame DAG into the closed recurrence

We now have, per axis, a per-frame DAG `out_i = g(S_i, K, X_i)` and the state cells.
Generalize across frames into `S' = U(S, K, X)`, `out = O(S, K)`:

1. **Inter-frame state relation (the update rule).** Treat each play-call as one
   iteration of a loop and record `(orig(S), S)` pairs across frames. **Technique:
   trace-based likely-invariant / recurrence inference** [Ernst et al., *Daikon*, *IEEE
   TSE* 27(2):99–123, 2001; conf. *ICSE* 1999]. Test the candidate-relation library over
   the observed `(orig(S), S)` tuples and keep the survivors:
   - `s' == s + c` → constant-stride accumulator/counter (A Mind counter `+2`, PW `+2`,
     filter `+8`-per-128);
   - `s' == s − c` → decay/skydive descent;
   - conditional `s' == reset_value when pred(s) else s ± c` → the skydive wrap and the
     wrapaccum modulo (the predicate `pred` is the observed branch from the slice's
     control edges, Stage A);
   - `idx' == (idx + stride) mod P` → table-walk advance.
   The DAG from Stage B *already gives* the functional form per frame; Daikon-style
   testing confirms it is **invariant across all frames** (with statistical
   justification to reject coincidences), upgrading "this frame did `s+2`" to "the
   update rule is `s += 2`".

2. **Advance clock / dwell (the stall axis).** The update may apply only on some frames
   (a table entry held for `dwell` frames; an accumulator stepping on a sub-rate). The
   per-frame change stream `Δ_i = [S changed at frame i]` is itself a sequence; recover
   its period by folding to the **smallest period that replays it** (a loop-summary
   fold). If `Δ` is all-ones → every-frame; if it folds to a short boolean period →
   periodic dwell; if it tracks an exogenous counter bit → the clock is that counter
   (§2.6). This is the one genuine degree of freedom that the zoo over-fit; here it is a
   recovered property of `U`, length-independent.

3. **Synthesis as the closure step (when Daikon templates don't name it).** When the
   inter-frame relation is not a single library template, **synthesize** `U` over a
   small DSL by programming-by-example [Gulwani, FlashFill, *POPL* 2011; Gulwani–Polozov–
   Singh, *Program Synthesis*, FnT PL 4(1–2), 2017]. The DSL spans exactly the 6502
   ALU the slice used: fixed-width `+ − · << >> mod`, bitwise `& | ^`, array select,
   conditional. Each `(S_i, S_{i+1})` pair is one example; version-space intersection
   prunes; **ranking prefers the program with the fewest ops and the operands the slice
   actually used** (so we recover *the program's* `U`, not an arbitrary consistent
   function). Because the DSL leaves and ops are fixed to what Stage A/B observed, the
   search is tiny and grounded — it is not free-form 6502 synthesis.

4. **LFSR closure.** If a state cell's sequence is suspected linear over GF(2), run
   **Berlekamp–Massey** [Massey, *IEEE Trans. IT* 15(1):122–127, 1969; Berlekamp,
   *Algebraic Coding Theory*, 1968] over it: it returns the shortest feedback
   polynomial (the taps = `K`) and the linear complexity `L`. Low `L` ⇒ a genuine LFSR,
   recovered exactly from ~`2L` samples. High `L` ⇒ not linear ⇒ fall back (§5).

The output is `G_A = (S, U, O, K)` — fixed size, regardless of N.

### 2.6 Cross-axis sharing (exogenous inputs `X`) — discovered, not assumed

A state cell read by axis A but **owned** (last-written) by axis B's update, or by a
single master update, is an *exogenous input* `X`, not A's own state. Discovery is
mechanical: the leaf's definer PC is in B's slice (or in a shared update routine), not
A's. This recovers:

- **the shared master counter** (A Mind's `zp$13`, incremented once per IRQ and read by
  freq, PW, filter, and the gate predicate) — recovered *once* as its own tiny `G` and
  referenced by every axis's `X`;
- **freq→PW carry coupling** (the `additive_pw` case the zoo needed a special rule for):
  PW's `U` takes an addend that is the freq accumulator's per-frame carry-out — observed
  directly as a leaf owned by the freq slice;
- **groove ticks** shared across a voice's lanes.

Sharing is the single biggest honesty lever (the prior design's §3b.5 "consistency
prior" and §4e "tautological per-lane clock" concern): a recovered counter is admitted
as a real shared source iff **multiple axes' slices read the same owned cell**. An
"advance vector" that is read by only one axis and folds to no period is not a clock —
it is the output, and the recovery of that axis is rejected as length-proportional
(§2.8). This kills the failure mode where a per-lane "groove" is really stored data.

### 2.7 Stage D — verification by residual-zero re-execution

Assemble `{G_A}` + the shared sources into an executable interpreter state machine and
**re-execute it for N frames** producing an `[N, 25]` register array. Gate:

```
sum(residual(recovered_program, bustrace).resid.values()) == 0
```

over all 25 registers and all N frames (the project's existing `recover.residual`
oracle, used **only** as the yes/no gate). Don't-cares are the documented logging
conventions only (12-bit PW unused hi bits; $D417 bit 3) — exactly as
`amind/RECOVERY.md` already applies them. **No oracle is needed to *recover*; the oracle
only confirms yes/no.** A depacker oracle (`pygoattracker`) may *additionally* assert
that a recovered array `K_table` is byte-equal to the tune's depacked bytes — a stronger
cross-check where it exists, never required.

### 2.8 The anti-length-proportionality gate (built into recovery, not just at the end)

A recovered `G_A` is admitted only if it is **strictly smaller than storing the
output**: `cost(S, U, O, K) < α · N` for fixed α<1. Concretely:

- a table-walk is admitted only if the table is traversed ≥2 full cycles (a single pass
  over a long table = storing the output);
- an accumulator/recurrence is admitted by construction (`(acc0, rate, w)` is O(1));
- an axis whose only "recovery" is a per-frame value list, or an unshared aperiodic
  advance vector, is **surfaced as unrecovered** (§5), never stored as a generator.

This is the same floor the zoo's HARD RULE #0 guards enforced
(`unified_generic_recovery.md` §3c), but now it is a property of the *recurrence*, so
passing it *means* length-independence rather than approximating it.

### 2.9 Multispeed, IRQ source, and framing

The frame boundary is **the play-call cadence**, recovered from the trace, not assumed
to be the PAL raster. A Mind Is Born is the cautionary case: it is **CIA-timer driven at
~16422 cycles**, not 19656 (PAL raster); framing at the raster mis-bins the counter
(spurious +4 jumps, off-by-one dwell — `amind/RECOVERY.md` §3). Multispeed tunes call
play 2×/4×/… per frame. **Recovery of framing:** the master counter's own update (one
`INC` cluster per play-call) *defines* the call cadence; we segment frames at those
clusters (the existing tracer already anchors on the first play-phase SID-write cycle
and clusters writes per IRQ — §3). The recurrence is expressed per *play-call*, so it is
agnostic to the wall-clock rate; the residual-zero gate is evaluated on the same
per-call grid the program runs on.

---

## 3. Minimal bounded in-emulator capture to add (KB/tune)

The tracer (`preframr-sidtrace`, `src/c64/membus_trace.h` + `src/sidtrace.cpp`) today
emits the **SDST** artifact: an access-type map (ACMP), a post-init RAM snapshot of the
song-data region (SNAP), a PC-tagged SID-write summary (SIDW), an indexed-read VSA
summary (IDXR), plus the timestamped SID-write stream (`.sidwr.bin`, the render gate).
This is a few KB/tune and bounded (fixed 64 KiB arrays + small maps).

**What already suffices** (reuse as-is):

- **ACMP** — the access-type map *is* the state/constant/SMC classifier of §2.3–2.4:
  state-or-SMC = EXEC_PLAY & WRITE_PLAY; constant = READ_PLAY & ¬WRITE_PLAY & ¬EXEC.
  Keep it; it is exactly the SMC-correct classification the model needs.
- **SNAP** — the verbatim post-init RAM image is where array `K` is lifted from (§2.4).
  Keep, but **widen** it (below): the song-data-region heuristic in SNAP is a *format
  assumption* (bounds to the loaded image span and to "≥1 byte READ as data"); the
  generic recovery must lift `K` from wherever O dereferenced, including zero page.
- **IDXR** — the per-PC strided-interval `(base, stride, idxMin, idxMax)` is exactly the
  VSA primitive of §2.4 for the table sub-case. Keep.
- **`.sidwr.bin`** — the residual-zero gate. Keep.
- **firstSidWriteCycle + per-IRQ write clustering** — the framing anchor of §2.9. Keep.

**What is MISSING (the gap that blocks recurrence recovery): the per-SID-write
DATA-FLOW.** SDST records *that* PC wrote register r `count` times with `lastVal` — it
does **not** record *what computed the value*: the operands, the source addresses, the
state cells read, or the arithmetic. Without that, you cannot build the Stage-B DAG, so
you cannot recover `U`/`O` for the arithmetic / no-memory-read case (A Mind Is Born),
and you fall back to fitting the output (the zoo). The fix is a **bounded per-write
data-flow record**, keyed by (PC, reg) so it does **not** grow with N:

### 3.1 New SDST section: SIDDF (per-write data-flow summary)

For each distinct issuing **(PC, reg)** of a `STA $D4xx` (a few dozen sites/tune, the
same key space as SIDW), accumulate a **bounded slice/DAG summary**, not a per-frame
stream. The 6510 core already exposes, per cycle, PC / operands / X / Y /
effective-address (confirmed in `mos6510.h`: `instrStartPC`, `instrOperand`,
`Cycle_EffectiveAddress`, `Cycle_Pointer`, `Register_X/Y/A`). Capture, per (PC,reg):

| field | what | why (which stage) | bound |
|---|---|---|---|
| `slice_pcs[]` | the set of PCs in the backward slice of A at this store, walked in-emulator over the **current play-call's** instruction window (a small ring buffer of the last ~few-hundred retired instructions with their A/X/Y/operand/effaddr) | Stage A: the def-use chain = the generator's code | set of PCs (tens) |
| `leaf_kinds[]` | per slice leaf: {immediate(value), ram_read(addr), state_cell(addr), exogenous(addr)} classified via ACMP | Stage B leaves; §2.3 state ID | tens of entries |
| `src_addrs` strided | base/stride/idx-span of any indexed read feeding A (== IDXR but attributed to *this* write's slice) | §2.4 array K | one strided interval |
| `op_seq[]` | the ALU op sequence on the slice's A-defining chain (ADC/EOR/AND/ASL/…/with immediate operands) | Stage B DAG shape; §2.5 synthesis DSL leaves | short op list (tens) |
| `val_lo, val_hi, val_first` | min/max/first written value | §2.8 admission + §2.5 sanity | 3 bytes |

The key is **(PC,reg)** so the record is **O(code sites), not O(frames)** — it stays a
few KB. The in-emulator slicer is a *bounded backward walk* over a small ring buffer of
retired instructions within the play-call (one play-call is short — A Mind is ~1 IRQ;
GoatTracker a few hundred instructions), so peak memory is one ring buffer + the
per-(PC,reg) summaries. This is the single addition that turns "we know *that* this PC
wrote freq-lo" into "we know *how* it computed freq-lo".

### 3.2 New SDST section: STATESEQ (bounded inter-frame state samples)

To run Stage C (Daikon-style inter-frame relation + Berlekamp–Massey), we need the
*sequence* of each candidate state cell across frames — but only for the **few cells the
SIDDF slices flagged as state**, and we can bound it: keep the first `M` samples (e.g.
M=512) per flagged cell (≥`2L` for any recoverable linear recurrence; enough for Daikon
statistical justification) plus a running check that the recurrence the first M imply
*continues* to hold (a cheap in-emulator residual on the cell itself). Per flagged cell:
`addr (u16)`, `samples[M] (u8 or u16)`, `holds_to_end (bool)`. A handful of cells × M
bytes = low single-digit KB. (Alternatively, since the recurrence is the point, store
only `(s0, fitted_step/poly, holds_to_end)` once the in-emulator fold succeeds —
sub-100 bytes — and the host re-derives.)

**Determinism.** All new fields are deterministic functions of the same execution
(`powerOnDelay` is already pinned in `sidtrace.cpp`); the ring-buffer slice and the
state-sample sequence are reproducible run-to-run.

**Net cost.** SIDDF (tens of sites × tens of entries) + STATESEQ (handful of cells ×
M) ≈ **+2–6 KB/tune**, keeping the artifact in the existing few-KB regime — *not* GB,
*not* growing with N. Nothing is streamed per cycle.

### 3.3 What we deliberately do NOT add

- No raw per-cycle bus trace (retired; GB/tune).
- No per-frame value lists for axes (that is the output; storing it is the zoo's error).
- No format/structure tags (pattern/orderlist/instrument) — the recovery has no such
  concepts.

---

## 4. Two worked examples

### 4.1 Table-driven — Grid_Runner (freq recovers as a table-walk recurrence)

**Tune:** `Grid_Runner.sid`, 4,051 bytes (GoatTracker player + packed data). We assume
**nothing** about GoatTracker; we observe execution.

**Recovery of the voice-0 freq axis:**

1. **Slice (Stage A).** Backward-slice each `STA $D400` (freq-lo) and `STA $D401`
   (freq-hi). The slice for freq-lo terminates at an indexed read `LDA tbl,X`-class
   instruction whose X came from a per-voice cursor cell, plus the `STA $D400` blit.
2. **Symbolize (Stage B).** The DAG is `out = select(tbl, base + 2*idx)` for freq-lo and
   `out = select(tbl+1, base + 2*idx)` for freq-hi — a **16-bit table read indexed by a
   cursor** (the note→freq table). No arithmetic on the value: pure `O = K_table[idx]`.
3. **Constant by identity (§2.4).** VSA over the indexed read's effective addresses
   gives `base`, `stride = 2`, `idxMin..idxMax` → the table is `(idxMax−idxMin+1)`
   16-bit entries; lift them **verbatim** from the SNAP RAM image. (Cross-check, *only*
   as oracle: these bytes are byte-equal to `pygoattracker`'s depacked note table —
   yes/no.) The cursor's update is the second slice: `idx' = idx + step` on the note
   advance, holding `dwell` frames (Stage C fold), with a loop-jump at the pattern end
   (the control-edge branch in the slice).
4. **Generalize (Stage C).** `G_freq0 = (S=idx, U: idx advances by the recovered pattern
   traversal with dwell, O: out = K_table[idx], K = {K_table, stride=2})`. The *order*
   of idx values is itself driven by an exogenous per-voice sequencer cursor (another
   `G`), recovered the same way; repetition (a phrase replayed) shows as the cursor
   **re-reading the same indices** — recovered, not LZ-induced.
5. **Verify.** Re-execute → byte-exact freq stream. `sum(resid)==0` over the freq
   registers.

**Recovered size + length-independence.** `G_freq0` is: the note table (recovered once,
shared across all three voices by identity — §2.6, since all three voices' freq slices
read the *same* `base`), plus a per-voice cursor recurrence (a few bytes of state + the
traversal/dwell), plus the sequencer cursor. This is **O(distinct table entries +
distinct cursor program)**, *independent of N frames*. Contrast the zoo: 33,913
fit-segments (one per traversal) collapsing to 237 signatures — i.e. it stored the
*unrolling*. Here the table is stored **once** and the recurrence re-indexes it for
free, so the freq axis size is fixed as the tune plays longer. (The prior identity-based
SDST recovery already reached ~2,786 GT tokens on Grid_Runner by lifting the shared
tables; this design reaches the same shared-table result *generically* — by observing O
index them — and adds the generative case the SDST path cannot do.)

### 4.2 Generative — A Mind Is Born (freq / PW recover as arithmetic recurrence, NO table)

**Tune:** `A_Mind_Is_Born.sid`, **380 bytes** on disk; 254-byte program image, RSID,
IRQ-driven, **no note-table / pattern / orderlist** — the score is the static seed; all
else is arithmetic from one counter (`amind/RECOVERY.md`). This is the case that breaks
every output-fitting and every "find the table" approach. Three execution facts the
generic capture must surface (and does, via §3): **illegal opcodes** (LAX/ALR/AXS/ANC/…
— the probe MPU must implement them; the *emulator already runs the tune*, so the
in-emulator slice is faithful regardless), **self-relocation into zero page** (the
player ends up at `$0031`; the access-type map and the slice follow it transparently —
no static disasm assumption), and **CIA framing at ~16422 cyc** (recovered from the
call cadence, §2.9).

**Recovery, per axis (no slice ever hits a data `LDA` — the native no-read case):**

- **Master counter (shared source).** A two-`INC` cluster on `zp$13` per IRQ; Stage C
  gives `C' = C + 2`. Its high byte `zp$20` is the carry → `section' = section + carry`.
  This is one tiny `G` (`S={C}`, `U: C+=2`, no O of its own), referenced as `X` by every
  axis (§2.6 — multiple axes' slices read `zp$13`).
- **Voice-1 PW (reg9).** Slice → `STA $D409` with `A := C & 0xFF`. DAG leaf is the shared
  counter; no RAM read. `G_PW1 = (S=∅, O: out = C & 0xFF, X={C})` — a continuous +2 PWM
  sawtooth. Pure ACCUM, rate 2, 8-bit wrap. **0 bytes of table.**
- **Voice-0 freq.** Slice → freq-hi `A := fhi`; the inter-frame relation is
  `fhi' = fhi − 1` with a reset `fhi' = next_start when fhi == floor`, `next_start`
  selected by a counter bit (the control-edge branch). `G_freq0 = (S=fhi,
  U: skydive descent + counter-bit reset, O: out_hi = fhi, K = {floor, start-set})`.
  The observed descent `1538,1282,…,2` is a clean rate −256 (Daikon template
  `s' = s − 256` on the 16-bit value). **No note table** — `K` is the floor + the small
  start-set, recovered as immediates. Crucially, **this axis has zero gate re-triggers,
  so the zoo's note-on segmenter finds nothing** (`amind/RECOVERY.md`); the slice-based
  recovery doesn't segment by note-ons at all — it slices the write.
- **Voice-2 (rhythmic bass).** Control axis toggles `ctrl: 0x60↔0x61` every **exactly 32
  frames** — `U` keyed on counter bit 5 (`X={C}`). Pitch (4107) + PW are set-once `K`.
- **Filter cutoff.** Slice → `fc = 7 + 8*(C >> 7)` — steps up by 8 every 128 frames
  (Stage C: accumulator at the section rate; the `>>7` and `*8` are the slice's ALU ops,
  `7` an immediate). 63 changes over the whole tune from one shared counter.
- **Envelopes / AD / SR / vol / res.** Set-once constants from the seed (`K`), no state.

**Verify.** Re-execute `{G_A}` + the counter for 8,190 frames → all 25 registers
byte-exact (with the documented don't-cares), exactly as `amind/verify.py` confirms via
the residual oracle. `sum(resid)==0`.

**Recovered size + length-independence.** The entire recovered program is: **one master
counter recurrence + ~7 tiny per-axis `(S,U,O,K)` tuples, total well under 100 bytes of
parameters, zero tables, zero score events.** `amind/RECOVERY.md` measures the true
generator as the 254-byte image + one counter (0.031 image-bytes/frame), and notes the
lossless lane-grammar miner needs 1,521 ops (0.186 op/frame) — *that* (the lane miner) is
still output-shaped and grows with the lanes' unrolling; **the recurrence model here is
the ~100-byte generator and does not grow with the 8,190 frames at all.** This is the
sharpest demonstration of length-independence: a tune that plays for thousands of frames
is a handful of recurrences over one counter.

### 4.3 Same procedure, two outcomes

Grid_Runner's freq `O` dereferences RAM (so `K` is a recovered array); A Mind's freq `O`
is pure arithmetic (so `K` is a few immediates). **Nothing in the algorithm branched on
"is there a table".** The slice/symbolize/generalize/verify pipeline produced a table in
one case and a recurrence in the other because that is what the programs *are* — which is
the entire thesis.

---

## 5. Honest limits and fallback

The residual-zero render is the **only** correctness gate, and it stays the gate in
every limit case. Where a generator is not recoverable as a fixed-size recurrence, the
axis is **surfaced as unrecovered** (length-proportional fallback, flagged), never
faked and never hidden in a patch.

**Genuine limits:**

1. **Heavy self-modifying code.** §2.3 handles SMC where the modified operand is a state
   cell with a recoverable update. But a player that rewrites *whole instruction
   sequences* (not just operands) per frame changes the slice's *shape* frame-to-frame;
   the DDG instances no longer share structure, so there is no fixed `U`. Detection: the
   slice's op_seq (SIDDF §3.1) is not invariant across frames. Fallback: store the axis's
   output for the SMC-divergent spans (length-proportional, flagged).
2. **Data-dependent control flow / implicit flows.** Pure data-flow slicing/taint misses
   **implicit (control-dependence) flows** [Schwartz et al. 2010, the canonical caveat]:
   a value selected by a branch on a hidden predicate. We capture *taken* branches in the
   slice's control edges (so the *observed* path is exact), but if the predicate depends
   on un-modeled state, the *generalized* `U` may not predict an unobserved branch. The
   residual-zero gate catches this (it will be nonzero on the mispredicted frames); the
   axis is then surfaced. We do **not** silently extrapolate.
3. **True PRNG / high-linear-complexity state.** Berlekamp–Massey recovers an LFSR only
   if the linear complexity is low. A cryptographic-strength or genuinely
   non-deterministic source (e.g. an axis seeded from an un-modeled hardware read) has no
   compact recurrence. Detection: BM reports `L ≈ n/2`. Fallback: surface the axis
   (length-proportional). Note the SID *noise* generator itself is a 23-bit LFSR (BM
   recovers it cleanly); the limit is only a genuinely chaotic *external* source.
4. **Deep cross-axis coupling / n-ary composition.** §2.6 handles 1-deep exogenous
   inputs and the freq→PW carry. A lane built from ≥3 coupled recurrences, or a per-entry
   jump-table loop grammar (multi-target `$FF`/`$FE` loop jumps), needs the model's
   `U`/`X` to be composed n-ary — bounded but a real extension
   (`unified_generic_recovery.md` §4e). The residual gate flags under-coverage.

**The fallback discipline (unchanged from the project's residual-zero stance):** a
non-zero residual is **never** a stored patch and never hidden. The axis (or the
frame-span) that does not close is reported as **unrecovered, length-proportional**, with
the reason (SMC-divergent / control-divergent / high-complexity). Residual-zero stays the
gate corpus-wide; "completeness" means every axis of every tune is either a fixed-size
recurrence or an explicitly-surfaced fallback.

---

## 6. Staged first-step plan + validation across diverse drivers

**Staged plan (each stage: bounded capture addition + recovery, gated by residual-zero,
no oracle needed to recover):**

- **Stage 0 — generative proof on the no-table case, with NO new capture.** A Mind Is
  Born already recovers byte-exact via the white-box run-the-player path
  (`amind/verify.py`, illegal-opcode MPU). Re-express that recovery as `{G_A}` recurrences
  (master counter + per-axis `(S,U,O,K)`) and confirm residual-zero by re-executing the
  *recurrences* (not the player). This proves the model end-to-end on the hardest case
  using existing artifacts — **do this first**, it needs no C++ and validates the model
  shape before any tracer change.
- **Stage 1 — add SIDDF (per-write data-flow summary, §3.1) to the tracer.** The single
  highest-leverage addition: the bounded backward-slice/DAG summary per (PC,reg) over a
  ring buffer of retired instructions. Validate on Grid_Runner: the freq slice must
  surface the indexed read + cursor; the recovered table must be byte-equal to
  `pygoattracker`'s depack (oracle, yes/no) **and** render residual-zero.
- **Stage 2 — add STATESEQ (§3.2) + Stage-C generalization** (Daikon-style relation +
  fold + Berlekamp–Massey). Close the recurrence (not just the per-frame DAG) on both
  worked examples; confirm length-independence by recovering from a *short* capture window
  and rendering a *long* one residual-zero (the decisive length-independence test: the
  recurrence recovered from K frames must reproduce M≫K frames).
- **Stage 3 — cross-axis sharing (§2.6) + admission gate (§2.8)** + the fallback surfacing
  (§5). Drive residual-zero across the corpus; every non-closing axis is surfaced, not
  faked.

**Validation corpus (residual-zero gate, no oracle required; depacker oracle used as
yes/no cross-check only where it exists):**

- **Generative, no tables:** A Mind Is Born (the §4.2 exemplar). Gate: residual-zero over
  8,190 frames; recovered size ~100 bytes; length-independence test (recover from 512
  frames, render 8,190).
- **Table-driven with a depacker:** Grid_Runner (GoatTracker). Gate: residual-zero +
  table byte-equal to `pygoattracker` (yes/no).
- **Depacker-less drivers — the generality test:** **Goto80's JCH NewPlayer and DefMon
  tunes** (no depacker exists, so the SDST identity-lift path cannot validate by
  byte-equality). Gate: **residual-zero render only** (the oracle here is *just* the
  byte-exact register stream; there is no format oracle). Success criterion that the
  prior approaches failed: the recovered size must be **flat as the capture window grows**
  (the prior generic catalog went 10.5k→34k tokens; the recurrence model must not). This
  is the decisive scaling test the design exists to pass.
- **Diverse hand/driver families** (Hubbard, Galway, Follin, the seven RE'd drivers
  Music_Assembler … Master_Composer) for breadth — residual-zero gate, surfacing any
  axis that hits a §5 limit.

**What to measure:** (a) residual-zero pass/fail per tune; (b) recovered-program size and
its **slope vs capture length** (must be ≈0 — the length-independence proof); (c) fraction
of axes closed as recurrences vs surfaced as fallback; (d) for table cases, byte-equality
of recovered `K` arrays vs the depacker oracle.

---

## 7. Summary

Reverse the chip *as it is programmed*, per axis, **from execution alone**: backward-slice
each `STA $D4xx` to isolate its generator [Korel–Laski; Agrawal–Horgan], symbolize the
slice into a per-write computation DAG [DART/CUTE; Schwartz et al.] with value provenance
[Newsome–Song], generalize the DAG across frames into a closed recurrence
`(state, update, output, constants)` [Daikon; FlashFill/program synthesis], recover array
constants by identity from the program's own RAM via VSA strided intervals
[Balakrishnan–Reps] and linear-state constants by Berlekamp–Massey [Massey]. Table-walk
and arithmetic recurrence fall out of the *same* procedure — a table is just `O` indexing
a recovered array; a generative tune is just `O`/`U` over immediates and a shared counter.
The result is **length-independent** (the recurrence is the program; the trace is its
unrolling) and gated **solely** by **residual-zero re-execution** — oracles confirm
yes/no, they are never consulted to derive structure. The one capture the tracer must add
is a **bounded, per-(PC,reg) data-flow summary** (~+2–6 KB/tune), which turns "we know
*that* this PC wrote the register" into "we know *how* it computed it" — the difference
between fitting playback and recovering the program.

---

## References

1. Korel, B., Laski, J. "Dynamic Program Slicing." *Information Processing Letters*
   29(3):155–163, 1988.
2. Agrawal, H., Horgan, J.R. "Dynamic Program Slicing." *PLDI* 1990, pp. 246–256
   (Dynamic Dependence Graph / Reduced DDG).
3. Newsome, J., Song, D. "Dynamic Taint Analysis for Automatic Detection, Analysis, and
   Signature Generation of Exploits on Commodity Software." *NDSS* 2005 (TaintCheck).
4. Schwartz, E.J., Avgerinos, T., Brumley, D. "All You Ever Wanted to Know About Dynamic
   Taint Analysis and Forward Symbolic Execution (but Might Have Been Afraid to Ask)."
   *IEEE S&P* 2010, pp. 317–331.
5. Godefroid, P., Klarlund, N., Sen, K. "DART: Directed Automated Random Testing."
   *PLDI* 2005, pp. 213–223.
6. Sen, K., Marinov, D., Agha, G. "CUTE: A Concolic Unit Testing Engine for C."
   *ESEC/FSE* 2005, pp. 263–272.
7. Cadar, C., Dunbar, D., Engler, D. "KLEE: Unassisted and Automatic Generation of
   High-Coverage Tests for Complex Systems Programs." *OSDI* 2008, pp. 209–224.
8. Godefroid, P., Levin, M.Y., Molnar, D. "Automated Whitebox Fuzz Testing." *NDSS* 2008
   (SAGE). Overview: *CACM* 55(3):40–44, 2012.
9. Ernst, M.D., Cockrell, J., Griswold, W.G., Notkin, D. "Dynamically Discovering Likely
   Program Invariants to Support Program Evolution." *IEEE TSE* 27(2):99–123, 2001
   (Daikon); conf. *ICSE* 1999, pp. 213–224.
10. Gulwani, S. "Automating String Processing in Spreadsheets Using Input-Output
    Examples." *POPL* 2011, pp. 317–330 (FlashFill / version-space algebra).
11. Gulwani, S., Polozov, O., Singh, R. "Program Synthesis." *Foundations and Trends in
    Programming Languages* 4(1–2):1–119, 2017.
12. Balakrishnan, G., Reps, T. "Analyzing Memory Accesses in x86 Executables." *CC* 2004,
    LNCS 2985:5–23 (Value-Set Analysis / strided intervals).
13. Balakrishnan, G., Reps, T. "WYSINWYX: What You See Is Not What You eXecute."
    *ACM TOPLAS* 32(6):23, 2010.
14. Massey, J.L. "Shift-register synthesis and BCH decoding." *IEEE Trans. Information
    Theory* 15(1):122–127, 1969 (Berlekamp–Massey). Berlekamp, E.R. *Algebraic Coding
    Theory*, McGraw-Hill, 1968.
15. (Supplementary, verify year before citing) Kincaid, Cyphert, Breck, Reps. "Closed
    Forms for Numerical Loops." *POPL* 2019. Godefroid, Luchaup. "Automatic Partial Loop
    Summarization in Dynamic Test Generation." *ISSTA* 2011.

### Project-internal context (prior, superseded; read for what is superseded + oracle
### pass/fail interface only — never for format/parsing knowledge)
- `design/encoding/sid_player_decompiler.md` — the `trace = VM(program)` thesis (the
  output-shaped predecessor).
- `unified_generic_recovery.md`, `generic_bacc_recovery.md` — the CITG archetype "zoo"
  (length-proportional output fit; superseded).
- `sidtrace_program_recovery.md` — the SDST artifact + identity-lift (assumes the
  song-data-table contract; superseded for the generic/generative case).
- `amind-wt/research/a_mind_is_born/RECOVERY.md`, `verify.py` — the generative exemplar
  (white-box recovery; the §4.2 ground truth). `bacc/illegal_mpu.py` (MPU65ILL) for the
  illegal-opcode probe.
- Oracle interface only: `recover.residual(program, bustrace) -> sum(resid)==0` gate;
  `recover.render_generic`; `recover_program`; `pygoattracker`/`undmc`/`gt_unpack`
  byte-exact depack (yes/no). **Used solely to assert byte-exactness.**
- Tracer to extend: `preframr-sidtrace/src/c64/membus_trace.h`, `src/sidtrace.cpp`
  (SDST: ACMP/SNAP/SIDW/IDXR + `.sidwr.bin`); 6510 core fields `instrStartPC`,
  `instrOperand`, `Cycle_EffectiveAddress`, `Cycle_Pointer`, `Register_X/Y/A` in
  `mos6510.h`.
```
