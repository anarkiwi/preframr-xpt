# SID Player Decompiler — the universal generator grammar (op-set from real drivers, zero residual)

**Status: LANDED — the live arc reached its goal.** The thesis below (`trace =
VM(program)`, op-set = grammar, per-tune program = music, residual→0 the gate, no escape hatch) is the
spine of the codec that **shipped as the BACC step / tracker representation**: Monty_on_the_Run encodes
**byte-exact (residual-zero) at 0.075 token/frame (1,313 tokens), ~10× reduction, VOCAB=34**. The decisive
reframe on top of the thesis: the frames are *playback*; the composer wrote *steps*. So we encode the
PROGRAM (tracker rows + pitch-invariant instruments on a step grid), not the trace. DECODE =
render steps→frames through the recovered generators (the "audio layer"). The living description of the
landed codec is `landed/README.md` + `AGENTS.md` LIVE ARC; this doc is kept as the durable record of the
thesis and the enduring lessons (HARD RULE #0, recover-the-program-not-the-trace, pitch-invariant
instruments, the transposition trap). The op-set grounding stays live in `sid_opset_inventory.md`.

It is the product of an experiment ladder (Re-Pair grammar induction → parametric canonicalization →
single-primitive fitting → compositional parse → the step reframe) that converged on one conclusion:

## The thesis (what the experiments proved)

The register trace is the output of a **finite playroutine reading per-tune data**: `trace = VM(program)`.

- **Grammar = the playroutine's op-set** — table-walk / counter / loop / conditional / arithmetic over the
  25 SID registers. Tiny, fixed, **general computation over the finite chip → complete**. This is the
  universal grammar, and it is bounded because the machine and the technique set are finite.
- **Program = per-tune data** — instrument tables + note lists + control + params. This is the content:
  per-tune, unbounded, learnable — *the music*.
- **Everything I kept measuring as a "tail" was the program, not the grammar.** 50% cross-tune coverage,
  11,570 "templates", 405 "LFO shapes", 32% "arbitrary" — all *per-tune data* (instrument tables, note
  lists) mistaken for grammar. The residual fell 66%→32% the instant primitives were allowed to *compose*;
  it goes to 0 as the op-set becomes complete.
- **No escape hatch is possible**: the op-set is general computation over the finite chip, so it expresses
  *any* playroutine; the only irreducible part is the per-tune program (the music). **Residual → 0 is the
  GATE, not a lane** — a non-zero residual means a missing op (fix the op-set), never an arbitrary-write
  patch. This is the `residual-zero-non-negotiable` discipline, now structurally enforced.

## The op-set is EXTRACTED from real drivers, not invented

The four libraries are **disassembled playroutines**: `pygoattracker`, `pysidwizard`, `pydefmon`,
`reninja` (WEMUSIC / Daglish-Crowther). The VM op-set is the **UNION of their operations**, generalized to
the minimal general-computation set that makes them complete — grounded in real code, validated by the
fact that each driver's own songs must round-trip exactly. (This is *not* the GoatTracker mistake: we take
the union of their *operations*, never any one's *format* as a cage.) The drivers are also the **cross-
check oracles** — their bit-exact players independently validate the VM.

The op-set is constrained (table-driven players, not arbitrary 6502), which is what makes synthesis
tractable: a per-tune program is `instruments (tables + loop/stride + sweep/LFO/arp/porta params) +
note lists (pitch, duration, instr-ref) + global automation`. Synthesis fits these to the trace.

## Architecture

- **Decoder = the VM** (run a program → per-frame register writes → `canonical_writes`). Grounded in /
  cross-checked against the four real players. We implement the union op-set once.
- **Encoder = the decompiler = program synthesis.** Per tune, synthesize the program whose VM execution
  equals the trace **exactly** (residual measured; target 0). Constrained, parallel across tunes (CPUs).
- **Token stream = the per-tune program** — the learnable, sparse, musical content. The op-set (grammar)
  is the fixed decoder, not in the token stream.

## Why no escape hatch, restated for the gate

There is **no residual lane in the format.** During development, residual is a *diagnostic*: the fraction
of frames the current op-set + synthesizer can't reproduce. The gate is to drive it to 0. Where it sticks,
we diagnose: missing op (extend the op-set — general, grounded), synthesis-search limit (better search,
more CPU), or a genuine chip/timing quirk (characterize it — but the finite-machine argument says it's a
program, so it's expressible). We do **not** ship a tune with residual hidden in a patch; an unfinished
tune is reported as unfinished. Completeness (residual = 0 corpus-wide) is the proof the op-set is the
universal grammar.

## HARD COMPACTNESS GATE (operator, 2026-06-18) — co-equal with residual→0

**The decompiled `Program` must be ≤ 2× the original `.sid` file size. If it exceeds 2×, the design is
WRONG** — it failed to recover the generative structure and is storing expanded/redundant data (we don't
even store the player code the `.sid` includes, so the ideal is *< 1×*). Measured 2026-06-18: the flat
note-list decompiler VIOLATES this on ~7%+ of surveyed tunes (a conservative floor — Hubbard Delta 4.05×,
Tel Rubicon 3.14×, IK+ 3.13×, Lightforce 2.91×; concentrated in note-dense Hubbard/Tel), because it stores
a **fully-expanded per-voice note-list** instead of the **orderlist→pattern→loop reuse** the original uses.
**Residual→0 ≠ compact; both are gates.** Expectation: **most every tune fits a model's context window with
room to spare** (a pattern-compressed tune is hundreds–low-thousands of tokens; a flat 10k-note list is
not). Fix = the missing **sequencing layer**: lossless repeated-subsequence / grammar-induction recovery of
patterns + orderlist + **transpose** on the note-list (the `Program` already has `PatternProgram`/
`OrderEntry`/`ChannelProgram` slots — populate them). This both compresses the melody AND recovers song
form (the long-range structure generation needs).

## HOW IT LANDED (2026-06-20) — the step reframe + the durable lessons

The compactness gate above is what forced the win. A flat per-voice note-list is residual-zero but NOT
compact (it stores playback, not the program). The fix was the **STEP / TRACKER reframe**:

- **Representation:** dump → tracker ROWS per voice `(pitch_interval, duration_in_steps,
  freq_instrument_ref, timbre_instrument_ref)` on a **4-frame step grid**. The composer wrote steps; the
  playroutine renders them to frames. Encode the steps.
- **Pitch-invariant instruments:** instruments are PARAMETERS, not realized per-frame data — vibrato = a
  vibrdepth shift, arp = semitone offsets, PWM = a sweep rate. Rendered byte-exact through the note table.
  This collapsed Monty's 818 distinct "freq bodies" → 45.
- **Repetition:** repeated phrases dedup via an **inline backward orderlist** (backward-reference only —
  no forward declaration, consistent with the model-facing inline-streaming rule).
- **Cross-lane factoring + micro derivation** shared sweeps across lanes;
  `micro` derived from the note table + generator onset, not stored.
- **BACC primitive collapse — the final landing.** After the STEP reframe, the 7 op-set primitives
  collapsed into **one bounded-accumulator (BACC) primitive + table-walk**: a single BACC subsumes
  VIB / SLIDE / ARP / PWM / ADSR / sweeps. VOCAB=34. This is what drove Monty to **1,313 tokens
  (0.075 token/frame), ~10× reduction, residual-zero**.

The journey, for the record: frame event codec → generator recovery (VIB/SLIDE/ARP/HOLD) → frame-level
repetition (REPEAT/LREPLAY, tapped out ~37k) → the STEP reframe → pitch-invariant instruments → cross-lane
sweep factoring + micro derivation → the **BACC primitive collapse (one bounded accumulator + table-walk,
VOCAB=34)** → **0.075 token/frame, residual-zero.** (The earlier STEP-codec figures — 0.901 token/frame,
15,816 tokens — are superseded by the BACC result.)

**The enduring lesson — HARD RULE #0 (the transposition trap, recurred 4×):** nothing is irreducible.
Every "distinct body" was a few instruments rendered at many pitches: `micro` = the note table; arp =
semitones; the period-32 Monty bass = a 5-note pattern × 434 repeats; "pw table data" = sweeps. Judging a
run in isolation by one model class — blind to notes + repetition — is the error, every time. Keep HARD
RULE #0 prominent (memory `no-irreducible-runs`).

## Build phases (each: residual measured + driven down; op-set stays bounded)

- **P0 — extract op-set + build the VM + validate on the drivers' OWN songs.** Read the four disassembled
  models; tabulate their operations; implement the union as the VM (decoder) over the 25 registers. Build
  the round-trip harness vs `canonical_writes` (reuse the `gtcodec` harness on `feat/gt-decompiler`).
  **Validate: each driver's native songs (built via its lib) must round-trip through OUR VM exactly** —
  this proves the VM covers each driver's ops (residual 0 on their own tunes by construction). Then measure
  residual on a sample of *other-driver* corpus tunes (Hubbard/Galway/Follin) = the decompiler challenge
  ahead. Deliverable: the op-set inventory (the candidate grammar) + the VM + the validation + the baseline
  residual on un-disassembled drivers.
- **P1 — decompiler: notes + envelope + waveform tables.** Synthesize the easy lanes; residual on them → 0.
- **P2 — freq: note + sweep/LFO + portamento + arp** (the lane that broke everything). Residual → 0.
- **P3 — PW + filter sweeps + globals.** Residual → 0.
- **P4 — full corpus.** Drive residual → 0 everywhere; **report the op-set size (the grammar, must be
  bounded) and the per-tune program size (the data).** The op-set is complete iff residual = 0 corpus-wide.
- **Then** — the per-tune program is the learnable token stream: train atoms-only, re-run the deciders
  (`copy_novel` novel-content, `free_running_gap`, generation quality gate). Export = the per-tune program
  → a real tracker module (via the libs) is the editable endpoint.

## Tractability + compute

Synthesis is over the constrained table-driven-player space, parallel across tunes on fogbank/defroster.
Bootstrapping from the four real drivers seeds both the op-set and (for their own tunes) ground-truth
programs, which de-risks the synthesizer. The CPUs do the per-tune synthesis search + the corpus-wide
op-set completeness sweep.

## Kept / superseded
- **Kept:** `canonical_writes` oracle; the `gtcodec` round-trip harness; the four driver libs as op-set
  source + cross-check oracles + export; `pitch_grid` note recovery; the `residual-zero-non-negotiable`
  discipline (now structural).
- **Superseded:** `virtual_tracker_codec.md` (GoatTracker as a cage), `universal_sid_codec.md` (invented
  op-set + residual lane). The residual *lane* is removed entirely; residual is a build diagnostic → 0.
