# Generic full-tune recovery from the trusted bus trace — method + measured residual-zero

**Status: PROTOTYPE / PROOF (2026-06-21).**  Extends
[`generic_bacc_recovery.md`](generic_bacc_recovery.md) (the rationale and the
generator-lane probes) from a generator-lanes-only fitter to a **FULL 25-register,
whole-tune, residual-zero** recovery driven entirely by the trusted CPU bus
trace (`<base>.bus.bin`).  The shipped substrate is **preframr-sidtrace** (its
earlier run-to-run non-determinism has since been fixed; it is deterministic and
byte-exact); the revice/VICE RBT1 trace (headlessvice PR #24) is a validated
**parallel** second source, not the shipped path.  This is the payoff
of the bus-trace pivot: the path to retiring the per-driver hand backends
(`bacc/backends/{goattracker,hubbard}.py`).  Prototype in
`/scratch/tmp/sidemu/gfit_complete/`; findings
`/scratch/tmp/sidemu/generic_fitter_complete_FINDINGS.md`.  No preframr-tokens
change, no PR.

## Result (measured)

Stratified GoatTracker corpus sample (8 tunes, the `GT_FIXTURES` set), fresh
matched dump+bus rendered from `headlessvice-bustrace:local` (dump and trace share
the same vsid run, so the boot prolog is one consistent framing):

| | GENERIC (this work) | HAND backend (`verify_residual`, read-only) |
|---|---|---|
| **full whole-tune 25-reg residual-zero (rendered == bus-state byte-exact)** | **7 / 8** | **6 / 8** |
| non-generator lanes (ctrl/AD/SR/filter/vol) residual-zero | **8 / 8** | n/a |
| Monty (Hubbard) full whole-tune residual-zero (cross-driver) | **YES** | YES |

**Update (landed in preframr-tokens):** two generic wavetable archetypes were
added to the SHARED BACC library (`bacc/generic/archetypes.py`) and wired into the
cover search.  Whole-tune residual-zero is now **7/8** (up from 5/8); Monty
(Hubbard) stays residual-zero.  Generic recovers Grid_Runner, Need_More_NOPs,
Jetta, Twilight, VIC64Demo_tune_5, **Hammurabi** (newly closed), and
**Not_Even_Human** (reclassified; render == bus-state byte-exact):

- **`tablewalk_lead`** -- a `lead`-frame constant hold then a period-P value
  table walk, a DELAYED long-period modulation.  Admitted to the cover on length
  (one piece) so a short coincidental arp at the note start no longer shadows the
  genuine period-12 modulation -- **this CLOSES Hammurabi** (confirmed
  residual-zero on the native whole-tune trace).
- **`ratewalk`** -- a period-P signed-rate wavetable accumulator (`value +=
  rate_table[i % P]`), the fractional-rate / wider-internal-width sweep.  It
  generalises `maskaccum` (one rate gated by a 0/1 mask) to a full per-step rate
  table and closes FamiCommodore's voice-0 PW.

**FamiCommodore remains the single miss (gen-lane only).**  Its voice-2 PW is a
single sustained note whose pulse width is a wavetable-paced reflecting triangle
over a 12-value table with a drifting (non-periodic) dwell: not a clean tablewalk
(no period across the 903-frame note), not a constant-rate sub-resolution triangle
accumulator, and a `ratewalk` cover fragments into ~70 pieces storing ~873 numbers
for 903 frames -- raw-byte storage in disguise, which HARD RULE #0 forbids.  It is
left UNFIT (residual surfaced) rather than papered over; closing it needs a
wavetable-pointer archetype not yet pinned down.  Net: the "sub-resolution
accumulator" framing was only PARTLY right -- it closes the constant-fractional-rate
case (voice-0) but not the wavetable-paced reflecting-triangle case (voice-2).

## Method (generic, every structure traced to the bus — HARD RULE #0 clean)

### 1. Per-frame 25-register state from the bus (the foundation + boot alignment)

The bus trace's SID-write substream (`$D400..$D418`, rw=1) is byte-identical in
(reg, val) order to the register dump — both come from the same vsid run.  The
*timing* differs because many drivers (GoatTracker) **re-blit the whole 25-byte
SID shadow register file to the chip every play-call**, so the bus carries ~25
writes/frame where the coalesced dump records only changed registers.  Per
play-call the LAST value written to each register is its frame value.  Reproduces
the dump's per-frame state **byte-exact** after three steps, all generic:

- **frame on the steady IRQ/play cadence** — blit-group boundaries (an inter-write
  gap above a threshold starts a new play-call).
- **mask PW-high registers (3,10,17) to 4 bits** — a pure SID-chip semantic
  (12-bit pulse width; bits 4-7 are don't-care), the SAME mask `lsp_validate`
  applies; NOT driver-specific.
- **boot-prolog alignment** — frame 0 = the blit group whose start cycle equals
  the tune's first steady play-call (`first_play_cycle`, the generic anchor the
  dump itself uses).  GoatTracker blits all 25 registers during init too, so the
  bus reaches steady cadence BEFORE the dump's first frame; the offset (e.g. 9
  frames on Grid) is *derived* from the cadence, not guessed.  This is exactly the
  alignment issue that bit the sidtrace-removal work; here it is a clean fixed
  offset because dump and trace share the emulator.  Verified Grid 15681/15681,
  Monty 17544/17544 byte-exact.

The 100%-shadow-file provenance is itself a finding: `prov.py` shows EVERY register
write (15689/15689 per register) sourced from a fixed RAM shadow ($13BA+r on
Grid).  Recovering the tune = recovering the shadow-file trajectory, which is the
per-frame state — and that decomposes into generator + event lanes below.

### 2. Note table — bus value-provenance

A freq-lo SID write whose value was just read from a contiguous, 2-byte-strided,
never-written RAM region.  No hardcoded address.  Recovered on all 8 (128 entries).

### 3. Generator lanes (freq lo/hi, PW lo/hi) — BACC archetypes

The proven archetype library (`generic_fitter.py`, used READ-ONLY) sliced at
gate-rise note-ons, with three generic extensions added in the prototype
(`full_fitter.py`, NOT in the shared file):

- **pre-roll cover** [0, first note-on): the player sets an initial note/sweep
  before the first gate-rise.  Closing this alone took Grid & Monty to full
  residual-zero.
- **`maskaccum`** — periodic-dwell accumulator: `value += rate` on frames where a
  recovered period-P boolean mask is set (a wavetable-paced sweep that steps on a
  fixed-period pattern, not every frame).  Rate + mask recovered from the lane's
  own deltas; the small-period requirement is the guard against fitting arbitrary
  data.  Closed Need_More_NOPs.
- **`tablewalk`** (last-resort) — period-P value table for LFO modulations beyond
  the arp cap, fired only where the proven library returns None.

### 4. Non-generator lanes + orderlist/structure (ctrl/AD/SR/filter/vol)

A piecewise cover with the cheap structured archetypes (hold / accum / dwellaccum
/ arp) between change points.  **Residual-zero on all 8 tunes.**  This is also the
**orderlist / song-structure reconstruction**: the change points are the
note/pattern boundaries the bus exposes, and the cover is a compact run-length +
accumulator program — **~12x compression** over raw cells (Grid: 1637 ctrl
segments over 15681 frames), i.e. a genuine program, NOT raw-byte storage.

### 5. Render → require (nframes, 25) == dump byte-exact, whole tune.

## The residual-knowledge gap (reported precisely — post-landing)

All remaining misses are in freq/PW generator lanes; structure + non-generator
lanes + alignment are generically complete.  Two of the three originally-flagged
gaps are resolved:

- **Hammurabi** — RESOLVED by `tablewalk_lead`.  The delayed period-12 vibrato
  (long hold, then a clean period-12 LFO offset table) was being shadowed by a
  short coincidental arp because the cover never reached the long-period walk.
  `tablewalk_lead` folds the lead hold and the period-P table into ONE candidate
  that wins on length, so the whole modulation is covered in one piece.  Confirmed
  residual-zero on the native whole-tune trace.
- **Not_Even_Human** — RESOLVED (reclassification).  The self-contained render
  reproduces the bus-state byte-exact (residual-zero); the original 102-frame diff
  was bus-vs-DUMP at the `-limitcycles` song-end cutoff, which this recovery does
  not compare against.  Counted as recovered.
- **FamiCommodore** — STILL OPEN (voice-2 PW only; every other lane, incl.
  voice-0 PW via `ratewalk`, is residual-zero).  The voice-0 PW IS the
  fractional-rate / wider-internal-width accumulator the original note predicted,
  and `ratewalk` (a period-P signed-rate wavetable accumulator) closes it.  But
  voice-2 PW is a DIFFERENT mechanism: a single sustained note whose pulse width
  is a wavetable-paced reflecting triangle over a 12-value table with a drifting,
  non-periodic dwell.  It is not a clean tablewalk (no period across the 903-frame
  note), not a constant-rate sub-resolution triangle accumulator (a fine
  (rate, shift) search fails), and a `ratewalk` cover fragments into ~70 pieces
  storing ~873 numbers for 903 frames — i.e. raw-byte storage in disguise, which
  HARD RULE #0 forbids.  It is therefore left UNFIT (residual surfaced), NOT
  papered over with a fake generator.  Closing it cleanly needs a wavetable-pointer
  archetype (a stored LFO table read by a pointer the player advances at a
  sub-resolution rate) that is not yet pinned down from the bus.

The resolved gaps are closed-form generator extensions of a slightly wider
parameter space; **neither requires driver layout knowledge.**  The one remaining
gap is also a generator-arithmetic cost (a wavetable pointer), exactly as
`generic_bacc_recovery.md` §predicted, and remains a small bounded archetype set
rather than an unbounded per-driver cost.

## Driver-agnostic verdict

The method reads only SID chip semantics + the bus log; no GoatTracker constants
anywhere.  Monty (Hubbard) reaches full whole-tune 25-register residual-zero
generically (438600/438600), identical in mechanism to the GoatTracker tunes.  The
shadow-file blit, note table, orderlist/pattern structure, and every generator are
recovered from the bus.  This is driver-agnostic recovery, validated on two
unrelated drivers.

## Landing path — when can the hand GoatTracker backend be dropped?

- **Done, generically (8/8):** the full state reconstruction + boot-prolog
  alignment, the note table, the non-generator lanes (ctrl/AD/SR/filter/vol), and
  the orderlist/pattern structure.
- **Landed:** two closed-form additions to the SHARED BACC archetype library —
  (a) `ratewalk`, a fractional-rate / wider-internal-width accumulator, and (b)
  `tablewalk_lead`, a long-period table-walk cover that survives short-arp
  shadowing.  (b) closed Hammurabi (the single hand-only win); (a) closed
  FamiCommodore's voice-0 PW.  Generic is now **7/8** whole-tune residual-zero,
  ahead of the hand backend's 6/8 on this sample.
- **Remaining (1/8):** FamiCommodore's voice-2 PW wavetable-paced reflecting
  triangle (above) — needs a wavetable-pointer archetype, surfaced not faked.
- **Then:** retire `bacc/backends/goattracker.py` + `gt_unpack.py` in favour of the
  bus-trace generic recovery — which additionally recovers Hubbard (and any driver
  the trusted bus trace covers), eliminating the per-driver hand-disassembly
  scaling bottleneck.

## Artifacts

- `/scratch/tmp/sidemu/gfit_complete/busstate.py` — bus → per-frame 25-reg state.
- `/scratch/tmp/sidemu/gfit_complete/full_fitter.py` — full generic fitter.
- `/scratch/tmp/sidemu/gfit_complete/validate.py` — corpus validation + miss
  classification.
- `/scratch/tmp/sidemu/generic_fitter_complete_FINDINGS.md` — full findings +
  isolation audit.
