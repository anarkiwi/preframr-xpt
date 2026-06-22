# Automated generator recovery — MDL inference over the playroutine grammar

**SUPERSEDED (2026-06-20):** the generators (VIB/SLIDE/ARP/PWM) were recovered and landed in the
step/tracker codec (`../encoding/sid_player_decompiler.md`). Kept for the MDL-inference design record.

**Status: DESIGN + PROTOTYPE (2026-06-20).** Replaces the hand-coded per-generator recognizers
(`_as_triangle`, `_as_arp`, SLIDE/HOLD detection, non-freq sparse-hold) with one principled,
automated engine. Context: the white-box decompiler LIVE ARC (`AGENTS.md`); current driving objective
is getting Monty under 8192 tokens (`design/infra/...`, memory `libsidplayfp-groundtruth`).

## The problem with hand-coded recognizers

Each generator family currently has a bespoke matcher. They are greedy, brittle (the triangle matcher
first missed all non-canonical rotations), incomplete (26 Monty runs left as raw `MOD`; non-freq lanes
needed a separate sparse-hold pass), and they do not generalize to un-disassembled drivers. Every new
modulation shape needs new code.

## Reframe: minimal-generator inference (MDL), made tractable

The correct objective is **MDL**: find the shortest program, in the playroutine op-set grammar, that
reproduces a lane's per-frame sequence EXACTLY. Normally Kolmogorov-intractable — but here the grammar
is **bounded and proven-complete** (the corpus bus sweep: 1.4M exceptions, 0 unexplained), so this is
*bounded* program synthesis, solvable by polynomial-time algorithms, not open-ended search. The user's
framing is the key: the trace is a tiny deterministic program of **accumulators, counters, and
table-walks** — each maps to a classic CS object.

## The engine: three composable algorithms

**1. Berlekamp–Massey / finite differences — accumulator & polynomial curves.**
An order-k integer accumulator emits a degree-k polynomial, whose (k+1)-th finite difference is zero;
generally a constant-coefficient linear recurrence. **Berlekamp–Massey returns the SHORTEST such
recurrence for any sequence, online (one symbol at a time).** It subsumes, with no per-shape code:
hold = order 0 (`Δ=0`), linear ramp / porta / PW-sweep = order 1 (`Δ=const`), quadratic = order 2,
coupled accumulators = higher order. Output = (order, coeffs, initial state): O(order) numbers vs N
deltas.

**2. Exact periodicity detection — LFO / table-walk loops.**
Smallest period p with `x[t]==x[t-p]` exactly (autocorrelation / suffix structure). Encode (p, one
cycle). A triangle vibrato is a periodic sequence whose cycle is itself piecewise-linear, so it
*composes* with engine 1 (periodic-of-ramps) — capturing ALL rotations for free.

**3. Dynamic-programming optimal piecewise segmentation — the composition.**
Real lanes are piecewise (a note onset resets the generator: ramp → vibrato → hold).
`DP[t] = min_{j<t} DP[j] + tokencost(best EXACT model on [j,t))`, O(N·W) with a bounded max-segment
window W. Globally optimal segmentation under MDL — no greedy boundaries. The cost is the **real
serialized token count**, so we minimize actual tokens.

## Why it is strictly better

- **Optimal for the grammar**, not heuristic — provably the fewest tokens with these primitives.
- **Self-correcting on data vs curve**: a genuinely arbitrary arp-offset table (the *music*) has
  minimal recurrence of order = its length → no win → MDL correctly STORES it as data instead of
  over-compressing. Curves get recovered; the composer's choices stay. The "recover the cause, store
  the data" line is drawn automatically.
- **Generalizes across drivers** — no driver-specific code; the op-set IS the grammar.
- **Residual-zero by construction** — the DP admits only models that reconstruct exactly.
- **Composes with backward-reference reuse** — a recovered generator/cycle that recurs is emitted once
  and referenced (ties into the repetition lever).

## Honest risks / open questions

1. **Saturation is the real wrinkle.** SID accumulators clamp (16-bit Fn, 12-bit PW) and truncate —
   nonlinear, so a pure linear recurrence won't reconstruct exactly across a clamp. The model must be
   "linear recurrence **+ explicit clamp/truncation bounds**", reconstructed with the clamp applied
   (AGENTS.md flags saturation as a parameter). BM gives the linear intent; exactness needs the clamp.
   **This is the main thing the prototype must validate.**
2. **Additive decomposition** (base ramp + vibrato offset, as SID-Wizard/Hubbard literally compute):
   BM on the summed signal may not factor it — detect the periodic component (detrend), BM the trend,
   handle the periodic residual; or fit the additive model jointly.
3. **Streaming**: BM is online; the DP segmentation needs an online change-point variant for strict
   any-prefix-valid encoding (doable, slightly more complex). Per-run encoding is already bounded, so
   this only matters for whole-lane.

## Prototype plan (measure before integrating)

Standalone, off the live codec: on Monty's freq + PW lanes, run the BM+periodicity+DP engine, count
tokens under the SAME serialization, and compare to the current hand-coded recovery (VIB/ARP/SLIDE/
MOD/HOLD). Questions: does it match or beat the hand-coded token count? Does it capture the 26 leftover
SFX runs exactly (clamp-aware)? Where does it win/lose, and is the saturation handling sound? Decide
integration only on the measured result.

## Prototype result (2026-06-20, `/scratch/tmp/sidemu_proto/`)

Built and measured against the live recognizers on Monty, all residual-zero (verified across 1,999
generator spans, 0 failures). **Verdict: correct + strictly more general, but NOT a token win today.**
- **Parity on freq lanes** (within a few %, slightly winning v2) with ONE engine and zero per-shape
  code — confirms the MDL substrate auto-derives VIB/ARP/SLIDE/HOLD. Total payload +9.4% (25,177 →
  27,539), and the gap is **encoding-format overhead, not recovery quality**: the live codec's PW
  `RUN` uses a bespoke tuned sparse-period layout the generic MOD framing doesn't replicate (+18.5%
  on pw v0), plus occasional DP boundary splits.
- **"Leftovers are irreducible" — RETRACTED (disproven 2026-06-20).** The engine called the period-32
  8127-frame voice-1 bass irreducible ("minimal recurrence order = length"). That is wrong, the same
  isolation error as the micro episode: BM ran on ONE 32-delta cycle and found no short LINEAR
  recurrence — but the cycle is a TABLE-WALK OF NOTES that REPEATS, not a recurrence. Dissection: one
  cycle = 8 distinct freqs = ~5 pitches `[0,+12,~+19.5(×4 vibrato),+32,+44]` (a bassline arp), and the
  **exact cycle recurs 434× across the tune**. So the run = `(~5-note pattern) × 434` → tens of tokens,
  not thousands — the MOST compressible part of the song. Voice 2's "period-64" run is the same (4
  pitches, 15×). Lesson: judging a run in isolation by one model class (linear recurrence) hides both
  the note-decomposition and the cross-run repetition. The MDL engine needs a note-table/loop primitive
  and must see repetition; the reduction is NOTE-onset decomposition + the phrase-LZ backward-reference,
  NOT "store the data". These "leftovers" are the phrase-LZ jackpot.
- **Clamp/saturation**: Monty never saturates (`clamp_used=0`), so the wrinkle isn't exercised here;
  the "recurrence + explicit clamp bounds" model is validated synthetically as sound + necessary
  (pure recurrences fail across a clamp; the model wins on a saturating order-2 sweep, cost 10 vs 43).

**Decision**: do NOT integrate for the Monty-<8192 push — it would slightly increase tokens, and the
critical path is repetition (phrase-LZ) + micro, not curve-recovery quality (already at parity). The
engine's value is **generalization to un-disassembled drivers** + retiring per-shape code; revisit at
corpus scale, and only after teaching it the codec's tuned serialized layouts (an encoding-format
task, not an algorithm one).
