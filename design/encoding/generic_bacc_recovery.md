# Generic, driver-agnostic BACC recovery — design analysis

**Status: ANALYSIS (2026-06-20).** The current decompiler reaches residual-zero per driver via a
hand-written backend that hardcodes that driver's RAM map and re-implements its generator arithmetic
(`preframr-tokens/preframr_tokens/bacc/backends/{hubbard,goattracker}.py`). That hand-disassembly is the
scaling bottleneck. This doc asks whether recovery can be made GENERIC, grounds the answer in probes on
four drivers, and proposes the pipeline. The **generator-fitter experiment** §6 recommends is in flight
(prototype `/scratch/tmp/sidemu/generic_fitter.py`); this doc is the durable rationale. Companion docs:
the thesis [`sid_player_decompiler.md`](sid_player_decompiler.md), the op-set
[`sid_opset_inventory.md`](sid_opset_inventory.md).

**Question.** Is there a GENERIC, driver-agnostic way to recover a BACC program from a SID tune, without
per-driver hand disassembly? The framing: "run the entire song and watch how the emulator state generates
a dump."

**Verdict, up front.** Yes for **flavor A** (an *offline*, emulator-assisted recovery pipeline that
auto-builds a per-tune program — like today's backends, but with the RAM map and the score/table/state
partition *inferred*, not hand-written). The driver-invariant signals are strong enough to auto-locate the
score trigger, the note table, and the generator state on three different drivers with zero hardcoded
addresses (probes below). **No** for the strong form of **flavor B** (pure dump-only inference at tokenize
time): the dump alone is provably insufficient to recover the *sparse* program on dense/free-running tunes,
and the probe quantifies exactly why. The residual driver-specific part is **not** the RAM map (inferable)
— it is the **generator arithmetic** (the exact per-frame update rule of each accumulator), and that
reduces to a **small, bounded set of archetypes**, not an unbounded per-driver cost.

All probes are in `/scratch/tmp/sidemu/generic_*.py`, run on real tunes: **Monty_on_the_Run** (Hubbard,
has a residual-zero white-box backend = the bar), **5_Title_Tunes sub-2** (Hubbard 5TT, the catalog's
worst dense tune), **Grid_Runner** (GoatTracker), **Arkanoid** (Galway — a 4th driver with no backend).

---

## 1. The goal, restated precisely

"Watching the emulator generate the dump" gives residual-zero **trivially** — running the playroutine *is*
the byte-exact decoder (that is literally `sidemu.py`). That is not the prize. The prize is **compression
to the SPARSE program**: a per-voice score (sparse note-ons) + pitch-invariant instrument generators +
backward repetition, at **< 1 token/frame**. The white-box Monty backend achieves this (0.238 tok/frame,
residual-zero — re-verified True in this analysis). So the real question is narrower:

> Can the **sparse program** (composer steps + instrument generators) be recovered **generically** from
> the running player, without hand-coding each driver's RAM map *and* re-implementing its generator math?

Decompose into (1) what is driver-INVARIANT and observable generically, and (2) what currently needs
driver-specific knowledge and whether (2) can be replaced by generic inference.

---

## 2. Driver-invariant vs driver-specific decomposition

| Element | Status | Why |
|---|---|---|
| **Note-on / trigger** | **INVARIANT** (chip semantics) | A note trigger is a gate-bit (ctrl bit0) 0→1 transition. Observable from the SID bus on every driver. |
| **Note table** | **INVARIANT mechanism, per-tune data** | Every ET driver has a freqlo/hi table read 2-bytes-at-a-time; it is read-only during play. Auto-discoverable. |
| **Score location in RAM** | **INVARIANT mechanism** | The score is the RAM written at note-on cadence and read at the trigger. Auto-discoverable by read/write timing. |
| **Generator STATE** | **INVARIANT mechanism** | Vibrato phase / porta / PW sweep / length counters are read-modify-written every frame. Auto-discoverable by RMW pattern. |
| **Free-running phase** | **INVARIANT mechanism** | A free-running oscillator = an RMW accumulator whose writes are NOT concentrated at note-ons. Auto-discoverable. |
| **Generator ARITHMETIC** | **DRIVER-SPECIFIC, but BOUNDED** | The exact update rule (`tri(frame&7)·((notetab[n+1]−notetab[n])>>depth)`, ping-pong reflect at 0x08/0x0e, etc.) differs per driver. This is the BACC parameterization. Finite archetype set. |

The current backends hand-code **all six rows**. The probes below show rows 1–5 are *inferable*
generically. Only row 6 needs driver knowledge — and it is a bounded set of archetypes, the BACC
primitive's parameters.

---

## 3. The generic techniques, each with a probe verdict

### 3a. Note-on detection from the SID bus — `generic_noteon.py`, `generic_noteon_vs_whitebox.py`  ✅ STRONG

Detect note-ons purely as ctrl-bit0 0→1 transitions in the per-frame register dump (no RAM, no emulator).
Density is correctly sparse on all drivers:

| Tune | gate-rise note-ons | density/frame |
|---|---|---|
| Monty (Hubbard) | 5328 | 0.30 |
| 5TT sub-2 (Hubbard) | 516 | 0.25 |
| Grid_Runner (GoatTracker) | 4174 | 0.27 |
| Arkanoid (Galway) | 632 | 0.09 |

Against Monty's residual-zero white-box score (5511 events), gate-rise detection scores **5328/5511 =
96.7% recall with ZERO false positives** (every gate rise is a real note-on). The 183 missed events are
**legato/append** notes that re-trigger without a gate cycle (66 flagged legato via lnth bit6; the rest
are same-pitch retriggers / very short notes). **Verdict:** note-on detection is generic and essentially
free; the bounded gap is legato/tie notes, which need either a secondary cue (a freq-rewrite without a
gate cycle, also bus-observable) or the emulator. This is flavor-B-capable for the gated majority.

### 3b. Note-table auto-discovery via RAM-read tapping — `generic_notetable{,2}.py`, `..._verify.py`  ✅ STRONG (flavor A)

Extended the emulator to log RAM **reads** (`_RWMem.__getitem__`). The note table is the read-only,
2-byte-strided region whose read pairs equal the freqlo/hi written that frame. With NO hardcoded address:

- Monty: discovers `$8456`, **contents byte-identical** to the true `$8400` table slice; successive ratio
  **1.05952 ≈ ET 1.05946**. (`$8456 = $8400 + 2·43`: it starts at the lowest *used* note — a non-issue
  since note indices are relative.)
- 5TT: discovers `$1C89` (= true `$1C07` + 2·65), again byte-identical, ratio 1.05952.

**Verdict:** the note table is auto-discoverable per tune, byte-exact, equal-tempered confirmed — exactly
replacing the hand-coded `NOTETAB=$8400`. This is a clean flavor-A build step (needs RAM-read tapping →
offline only). The "starts at lowest used note" detail means the corpus-shared ET table is the right
canonical form (the deferred defcost optimization in the memory log).

### 3c. Read/write state classifier — `generic_classifier.py`  ✅ STRONG (flavor A)

Partition RAM with no hardcoded addresses by per-frame read/write behavior: `TABLE` (read ≥50% frames,
never written) · `STATE` (RMW most frames, writes concentrated at note-ons) · `FREE-RUN` (RMW most frames,
writes NOT at note-ons) · `SCORE` (written at note-on cadence).

Monty hand-map check (the partition recovers the hand-coded addresses):
- `$84CD` savelnthcc (the score trigger the backend *watches*) → **SCORE** (279/292 writes at note-ons).
- `$84F2` savefreqlo → **SCORE** (pitch latch).
- `$84E5` pulse-delay counter → **FREE-RUN** (3198 writes, RMW 2132, only 87 at note-on) — correctly flags
  the free-running PW sweep as generator state, not score.

Generalizes: 5TT yields the score region at `$1CCB+` (= Monty's `$84CB` shifted −0x67F9, matching the hand
backend's TT map) and a TABLE region; **GoatTracker** (no backend) cleanly partitions into TABLE / FREE-RUN
/ SCORE with the same rules. **Verdict:** the score-trigger address, the table region, and the
free-running-vs-resetting state distinction are all auto-recoverable. This is the single most important
result: it removes the per-driver RAM-map hand work **and** resolves free-running generically (the
dense-tune wall) — both flavor A.

### 3d. Dump-only lane-grammar fit (flavor B reach) — `generic_lanegrammar_probe.py`  ⚠️ LOSSLESS BUT INSUFFICIENT ON DENSE TUNES

`lane_grammar.parse_lane` losslessly decomposes every lane into {HOLD, ACCUM, PERIOD, SET} from the dump
ALONE. **Lossless on all four drivers** (driver-agnostic). But the **SET share** (the un-generatorized
residual = note onsets + unmodelled writes) reveals the wall:

| Tune | SET share | reading |
|---|---|---|
| Monty | **7.3%** | mostly HOLD+PERIOD; near the sparse program already |
| Galway Arkanoid | 47.8% | |
| GoatTracker Grid_Runner | 38.7% | |
| 5TT sub-2 (dense) | **62.9%** | the free-running wall: most writes look like bare SETs |

**Verdict:** dump-only lane mining is generic and lossless, but on dense/free-running tunes a majority of
the program degrades to SET ops, because the dump cannot tell that a run of "SET"s on a PW lane is ONE
free-running accumulator sliced at note boundaries — it has no read/write signal to know the accumulator
does not reset. This is exactly the 5TT slicing pathology. **The emulator's read/write classifier (3c) is
what supplies the missing bit** — confirming flavor A is needed for dense tunes and flavor B is not.

---

## 4. Proposed GENERIC offline recovery pipeline (flavor A)

A driver-agnostic build step that replaces the per-driver backend's hand work:

```
.sid + .dump
  │
  ├─ 1. Run play in py65, logging RAM reads, RAM writes, SID writes per frame  (sidemu + _RWMem)
  ├─ 2. NOTE-ON frames     ← gate-bit 0→1 on the SID bus            (3a, invariant)
  ├─ 3. RAM partition      ← read/write classifier                  (3c, invariant)
  │       TABLE  → note table (verified ET) + instrument/wavetables (3b)
  │       SCORE  → the note-on row fields (note, instr, dur, porta…) = composer steps
  │       STATE  → generator accumulators that reset at note-on
  │       FREE-RUN → accumulators that persist across note-ons (seed once, free-run)
  ├─ 4. GENERATOR FIT      ← fit each STATE/FREE-RUN lane's per-frame update to a BACC archetype
  │       (the ONLY step needing the bounded archetype library — §5)
  └─ 5. Verify residual-zero by rendering the BACC program; emit the inline token log.
```

Steps 1–3 are **fully generic and proven** by the probes. Step 5 is the existing render+serialize. Step 4
is where the bounded driver knowledge lives.

**How close can flavor B (dump-only, no emulator at parse time) get?** For *sparse* tunes (Monty-like, SET
share <10%), dump-only mining + gate-rise note-ons + pitch-invariant clustering already recovers most of
the program — flavor B is viable there. For *dense/free-running* tunes (5TT-like, SET share >60%), flavor B
hits a hard wall: the read/write bit that distinguishes a free-running oscillator from a per-note
articulation is **not present in the register dump**. So the honest split is: **flavor A is the general
answer; flavor B works only on the sparse subset and degrades to dense playback exactly on the tunes that
most need the generator recovery.**

---

## 5. Where it still needs per-driver help — and is it bounded?

The only irreducibly driver-specific element is **step 4, the generator arithmetic** — the exact update
rule of each accumulator. The arc has already enumerated this set against real driver source, and it is
**small and bounded**, not unbounded:

- **value-BACC**: `value += rate every dwell frames; boundary ∈ {wrap-8, reflect-12, none}; width ∈ {8,12}`
  — subsumes vibrato-amplitude, porta/slide, simple-PW (wrap-8), ping-pong PW (reflect-12), filter sweeps.
- **table-walk**: octave-arp (`note` vs `note+12`), multi-level arp/wavetable.
- **note-table-scaled output map**: vibrato amplitude `= (notetab[n+1]−notetab[n]) >> depth`.
- **flag coupling**: the no-CLC carry that couples freq→PW (derivable from the freq program).
- **gate/release rule**: `gate-off iff (cmd & 0x20)==0 AND dur==0` + per-instrument `(ctrl,ad,sr)`.

That is **one primitive with a handful of parameters + 2–3 table-walk variants + a gate rule**. Validated
byte-exact on two Hubbard drivers. A *generic fitter* that, given a STATE/FREE-RUN lane's trajectory from
step 3, searches this finite parameter space for the rule that regenerates it residual-zero would replace
the hand-written `_generators()` per driver. The cost is **bounded by the archetype count, not by the
driver count** — exactly the scaling property the question asks for. New drivers reuse the archetypes; only
a genuinely novel generator (rare, and reported as a residual, never patched) would extend the set.

The remaining caveat is *fit ambiguity*: multiple BACC parameterizations can match a short trajectory, so
the fitter needs a cross-tune / cross-note consistency prior (a generator is shared across all its notes —
the pitch-invariance the arc already exploits) to disambiguate. This is the real open research risk, not
the RAM map.

---

## 6. Concrete next-experiment recommendation

**Build the generic generator FITTER (step 4) and measure it against the two white-box backends.** The RAM
map (steps 1–3) is now demonstrably auto-inferable; the unproven link is whether a search over the bounded
BACC archetype space can recover the *same* residual-zero generators the hand `_generators()` encode.

Concretely, on Monty and 5TT sub-2:
1. Run steps 1–3 generically (probes 3a–3c) to get note-ons, note table, score addresses, and the
   STATE/FREE-RUN lane partition — no hand map.
2. For each STATE/FREE-RUN lane, fit the BACC archetype library (value-BACC wrap-8 / reflect-12 / none +
   table-walk + note-table-scaled amplitude) to the per-frame trajectory between note-ons, using
   pitch-invariant cross-note consistency to disambiguate.
3. Render and check residual-zero against the dump. **Success metric:** byte-exact reproduction with the
   generators *fitted, not hand-written*, on both Hubbard drivers, then on Grid_Runner (GoatTracker) where
   the score/table are already auto-found here.

If that lands, the per-driver backend collapses to: a `matches()` fingerprint + the shared archetype
fitter — the hand-disassembly scaling bottleneck is removed. If a specific generator resists the fitter,
that is the precise, bounded place where one archetype must be added (reported as a residual, per the
residual-zero discipline), and the analysis above predicts that set is small.

**Do NOT** invest further in dump-only (flavor B) generator recovery for dense tunes: probe 3d proves the
distinguishing bit is absent from the dump. Flavor B should be scoped to the sparse subset only.

---

## 7. Result — the fitter, built and validated (2026-06-20)

The §6 experiment landed (`/scratch/tmp/sidemu/generic_fitter.py`, timing `generic_fitter_timing.md`).
A greedy sliced cover of each freq+pw lane with a *searched* bounded-accumulator archetype (byte-exact
params only; no per-driver generator math) recovers residual-zero: **Monty 99.0% of segments, 5TT 92.3%,
Grid_Runner 96.4%**, all with the SAME seven archetype families the hand `_generators()` encode. **Proof
it is the hand math, not a curve-fit:** the fitted vibrato `amp_step` equals `(notetab[n+1]−notetab[n]) >>
depth` on 40/40 sampled Monty pieces, using the *generically discovered* note table ($8456). The library
transfers to GoatTracker unchanged.

**Timing (the scaling answer): seconds per tune** — Monty 1.6 s / 18.4k frames (0.30 ms/note, 86 µs/frame),
Grid_Runner 3.8 s, 5TT 3.0 s; py65 dump-gen adds ~0.5 s / 3000-frame subtune. **Full corpus (61,830
subtunes) ≈ 4–17 core-hours = a few MINUTES wall-time on the 72-core box** — embarrassingly parallel across
lanes/tunes. Generic fitting is therefore a practical replacement for the hand backends on the
per-note-resetting majority; each backend collapses to a `matches()` fingerprint + the shared fitter.

**The one gap is the predicted free-running wall.** The classifier flags a free-running pw lane (RMW every
frame, writes not at note-ons); fitting it as ONE continuous generator instead of slicing at note-ons lifts
5TT v0 pw 51%→70% and GT v0 pw 73%→96% — the classifier's bit is exactly what resolves it. But continuous
fitting HURTS lanes that do reset per-note (GT v1/v2 pw → ~49%), so slice-vs-continuous must be **gated on
the classifier** (rule proven, wiring is the remaining work). The residual 5TT pw is the **carry-coupled
additive PW** (the pw step depends on the freq accumulator's carry-out) — a cross-lane coupling, one
archetype to add, not a missing accumulator. Honest dependency: the fitter needs the offline py65
read/write classifier (flavor A) to resolve free-running lanes, exactly as predicted; flavor B cannot.
