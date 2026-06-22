# Unified generic SID recovery — one clocked indexed-table generator subsuming the archetype zoo

**Status: DESIGN / PROPOSAL (2026-06-21).** Read-only analysis of the generic BACC
recovery (`preframr_tokens/bacc/generic/`) after the HVSC sweep grew the archetype
library to ~28 matcher/renderer methods plus several detectors and framing
fallbacks. This doc inventories the accreted zoo, argues it is a special-casing of
ONE parametric generator — a **clocked indexed-table generator** — proposes the
unified op + one fitting procedure that preserves HARD RULE #0, and assesses
migration. No code change, no PR.

> Note on source: the live `main` `archetypes.py` (1115 lines) has the original
> ~14 archetypes; the full ~28-method campaign version is in the worktree
> `.claude/worktrees/sweep-mixed-06/` (1928 lines) — it adds `pingfold`,
> `ratewalk`, `dwellratewalk`, `tablewalk_lead`, `wavetable_ptr`, `maskaccum_stall`,
> the `note_boundaries`/`pw_sweep_resets`/`freq_note_onsets` detectors, and the
> `_walked_table`/`_fold_loop`/`_fold_loop_prefix` machinery. This inventory and
> proposal cover that fullest version (the campaign endpoint).

---

## 0. The thesis this must stay consistent with

`sid_player_decompiler.md`: **`trace = VM(program)`**, the grammar is the
playroutine op-set, the per-tune program is the music, and **residual → 0 is the
gate, never a lane**. HARD RULE #0: every structure recovered from the bus; a
generator is a **closed-form program, never raw data stored byte-for-byte**; a
genuinely irreducible lane is **surfaced, never faked**.

The accreted archetypes ARE the op-set, discovered empirically. The claim of this
doc is that the op-set has collapsed onto essentially one op with a few
parameters, exactly as `generic_bacc_recovery.md` §5 predicted ("one primitive
with a handful of parameters + 2–3 table-walk variants + a gate rule"), and that
the FINDINGS confirm this from the driver side.

---

## 1. Inventory of the accreted zoo

### 1a. Generators (renderer + its matcher) on the freq/PW (16-bit) lanes

| # | name | closed-form value[i] | parameters | bus-recovery of parameters |
|---|---|---|---|---|
| 1 | `hold` | `value` | value | first sample; run = constant prefix length |
| 2 | `accum` | `v0 + rate*i` (16-bit) | v0, rate | v0=seg[0]; rate=seg[1]-seg[0]; run while delta constant |
| 3 | `dwellaccum` | `v0 + rate*floor((i-lead)/dwell)` | v0,rate,dwell,lead | rate=unique nonzero diff; lead=constant prefix; dwell=unique gap between change points |
| 4 | `wrapaccum` | `v0+rate*i` modulo-wrapped in `[lo,hi)` | v0,rate,lo,hi | rate=modal nonzero diff; lo=min; **hi data-determined** from the against-sign step at each wrap (`hi=lo+rate-step`) |
| 5 | `arp` | `freqs[(i//dwell) % P]`, P≤6 | freqs,P,dwell | sample at dwell stride; try dwell∈{1..4}, P∈{2..6} |
| 6 | `glide` | `note_table[(n0+step*floor((i-lead)/dwell)) & 0xFF]` | n0,step,dwell,lead | n0=index of seg[0] in note table; lead=constant prefix; search step∈{±1,±2}, dwell∈{1,2,3,4,6,8} |
| 7 | `vibrato` | `base + tri_phase(ctr0+i)*amp_step` | base,amp_step,ctr0 | base∈{seg[0],min}; amp candidates from `dev/tri_phase`; ctr0∈{0..7} |
| 8 | `vibrato_exact` | byte-wise repeated 16-bit add of `amp`, `tri_phase` times | base,amp,ctr0 | same as vibrato; also yields the freq→PW carry sequence |
| 9 | `vibskydive` | `vibrato_exact` + descending hi-byte counter on parity `par` | base,amp,ctr0,sfh0,par | vibrato params + hi-byte countdown seed/parity from `_has_hi_countdown` |
| 10 | `arp_decay` | `arp` + descending hi-byte counter on parity `par` | freqs,P,dwell,sfh0,par | arp on the non-overlay parity; hi-byte countdown on the other parity |
| 11 | `decay` | `v0 - rate*floor((i+ctr0+1)/every)` after lead hold | v0,rate,every,ctr0,lead | rate=first negative diff; lead=constant prefix; search every∈{1..4}, phase |
| 12 | `pingpong` | reflect accumulator: ±rate every dwell, bounce at lo/hi | v0,rate,lo,hi,dwell,d0,dir0 | rate=modal |diff|; lo/hi=min/max (two reflect conventions); search dwell,d0,dir0 |
| 13 | `pingfold` | mirror-fold fixed-point triangle: internal acc at `frac` extra bits, `acc=2*bound-acc` at bound, emit `acc>>frac` | step,frac,lo,hi,acc0,dir0 | step=`round(mean|diff|*2^frac)`; frac∈{0..3}; bounds=visible extremes shifted; search seed/dir |
| 14 | `maskaccum` | `value += rate` where period-P bool mask set | v0,rate,mask,width | rate=unique nonzero diff; mask=`diff!=0` folded to period P (≤12) |
| 15 | `maskaccum_stall` | longest-prefix `maskaccum`: advance = frames stepping by the *dominant* rate, stall elsewhere | v0,rate,mask,width | dominant rate; advance = `diff==rate`; smallest period ≥`mincycles` cycles |
| 16 | `ratewalk` | `value += rate_table[(ctr0+i) % P]` (width-masked) | v0,rate_table,width | rate_table = per-frame signed deltas folded to smallest period P (≤48), ≥2 cycles |
| 17 | `dwellratewalk` | `value += rate_table[(i//dwell) % P]` | v0,rate_table,dwell,width | dwell=dominant run length of signed deltas; table=deltas at dwell stride; P≤24 |
| 18 | `tablewalk` | `table[(ctr0+i) % P]`, P≤48 | table | smallest P (≥2 distinct, ≥2 cycles) whose first P samples replay seg |
| 19 | `tablewalk_lead` | `value0` for `i<lead`, else `table[(i-lead) % P]` | lead,value0,table | absorb 0..lead of the constant prefix, then smallest replaying period P |
| 20 | `wavetable_ptr` | `table[ptr]`, `ptr += advance[i-1]` (mod P) | table,phase,advance | `_walked_table` (distinct-value sequence + `diff!=0` advance clock); table = `_fold_loop_prefix` of the walk (P 4..32, ≥3 distinct, ≥2 cycles) |
| 21 | `additive_pw` | `pwlo += pulsevalue + carry[i]`, carry from the freq generator | p0,pulsevalue,carry_table | hi byte constant; pulsevalue=modal lo diff; carry∈{0,1} where lo-diff=pulsevalue+1, folded to period ≤8 |

### 1b. Generators admitted on the non-generator (8-bit ctrl/AD/SR/filter/vol) lanes

`fit_event_lane`/`_longest_event_archetype` uses the cheap set — `hold`, `accum`,
`dwellaccum`, `arp` — **plus `wavetable_ptr`** (the dwell-paced page-walk, e.g.
Master Composer, where the whole register file is `page[ptr]` on one shared
pointer). 5 distinct ops here, all already in 1a.

### 1c. Detectors / segmentation / framing (NOT generators — they decide *where* to cut)

| name | role | bus signal |
|---|---|---|
| `gate_noteons` | per-voice note-on slice points | ctrl bit0 (gate) 0→1 rise |
| `note_boundaries` | generalised note-ons: gate rise ∪ ctrl change ∪ AD/SR change | per-voice register changes (legato/hard-restart retriggers) |
| `pw_sweep_resets` | extra freq-lane slice points | PW downward step ≫ median PW step (sweep re-seed) |
| `freq_note_onsets` | extra freq-lane slice points | freq step ≫ median freq step (pure-freq melody advance) |
| `per_frame_state_from_bus` | the 25-reg frame grid itself | blit-group cadence binning, PW-hi 4-bit mask, boot-prolog offset to `first_play_cycle` |
| `discover_note_table_from_bus` | the shared pitch table | freq-lo write whose value was just read from a contiguous 2-byte-strided RAM region |

### 1d. Cover / search / guard machinery

- `_match_prefix` — longest byte-exact prefix of a render vs the segment.
- `_detect_period`, `_fold_loop`, `_fold_loop_prefix`, `_walked_table` — period
  folding (turn a walked value sequence into the smallest looping table).
- `_longest_archetype` / `_longest_archetype_aug` — the greedy longest-run cover
  with a ranked candidate set and length/substantial-length/last-resort early-outs.
- `fit_segment` / `fit_lane` / `fit_event_lane` — slice → greedily tile → render.
- HARD RULE #0 guards (scattered): `_MINRUN`, `_MAXPIECES`, "≥2 cycles" /
  `mincycles`, "≥2 distinct/3 distinct values", "win only if covers
  substantially more (≥2× and ≥36)", "tablewalk is last resort (base is None)".

That is ~21 generator renderers + ~21 matchers + 6 detectors + the cover engine
= the ~28+ "methods" the campaign accreted.

---

## 2. The common structure — one clocked indexed-table generator

### 2a. The evidence: every driver in the FINDINGS *is* this one machine

The reverse-engineering of seven real drivers (Music_Assembler, FutureComposer,
JCH_NewPlayer, Soundmonitor, HardTrack, Master_Composer, FamiCommodore) converges
on a single mechanism with three independent axes:

1. **A pointer/accumulator advanced by a recovered advance-clock.** Every driver
   walks per-tune data with a pointer (or accumulates into a register) stepped by
   a tick/dwell counter:
   - Master_Composer — ONE chip-wide step pointer `$7941` advanced by a per-step
     **dwell table** `$7950[ptr]` (non-uniform groove); every lane is `page[ptr]`.
   - FutureComposer / JCH / FamiCommodore — a **per-voice tick** (FC `$2142,X`
     reset at note-on; JCH `$1795,X` + dwell `$1798,X`; FamiCommodore `$1439,X`
     advanced once per **gated** wavetable tick from groove pair (8,5)) shared
     across that voice's animated lanes.
   - HardTrack — per-wavetable pointers whose **dwell column lives in the table**.
   - Music_Assembler / Soundmonitor — degenerate "advance every frame", with the
     dwell living in per-instrument counters or absent.

2. **Value = TABLE read OR fixed-width accumulate (often both, per entry).**
   - Pure table read: Master_Composer pages, JCH/HardTrack ctrl columns, the
     shared 16-bit **pitch table** (note→freq) every driver has.
   - Fixed-width 16-bit accumulate (carry/borrow wrap): all PW / vibrato /
     portamento lanes.
   - **Mixed within one lane**: FamiCommodore and HardTrack PW tables carry a
     per-entry ctrl byte — `ctrl≥$80` = absolute set (table read), `ctrl<$80` =
     hold `ctrl` ticks accumulating `±val` (accumulate).

3. **Loop grammar.** Universally an `$FF`/`$FE`/`$80` marker giving an explicit
   **loop/jump/hold index**; note-column entries reinterpreted relative-note (→
   shared pitch table) vs absolute via a high-bit flag.

This is precisely the SHARED FAMILY rule the op-set inventory already extracted
from the six classic hand drivers (Hubbard/Tel/Follin/Whittaker/Gray/Galway):
triangle LFO over a frame counter, pitch-scaled depth, a stall/delay arm — i.e. a
*table or accumulator indexed by a clock that can stall*.

### 2b. The unified model

Define the **Clocked Indexed-Table Generator (CITG)**:

```
state:   ptr (table index),  acc (fixed-width accumulator)
params:  TABLE[0..P-1]        # the period-P loop body (values OR signed steps)
         CLOCK                # the advance schedule of ptr (see below)
         MODE ∈ {READ, ACCUM} # ptr selects a value, or a step added to acc
         WIDTH, WRAP          # register width + wrap rule {none, modulo, reflect}
         v0 / acc0, phase     # seed
         LEAD                 # frames of stall before the clock arms (optional)
         loop point           # where ptr wraps (default 0; HARD RULE #0 demands a loop)

per frame i (after the LEAD stall):
   if CLOCK steps at i:  ptr = next(ptr)        # +1 mod P, or loop-jump
   if MODE == READ:      value = TABLE[ptr]
   else:                 acc = wrap(acc + TABLE[ptr], WIDTH, WRAP);  value = acc
```

The **CLOCK** is the unifying degree of freedom. It is one of:
- **every-frame** (advance each frame) — the trackers without a groove tick;
- **periodic** (advance on a period-Q boolean mask / dwell Q) — the tempo-paced
  players;
- **external advance vector** (the recovered per-voice/chip groove tick), which is
  *shared* across the lanes it paces — the separable, non-periodic groove of
  Master_Composer / FamiCommodore.

### 2c. How each archetype falls out as a CITG parameterization

| archetype | CITG parameterization |
|---|---|
| `hold` | TABLE=[v], MODE=READ, CLOCK=never (P=1) |
| `accum` | TABLE=[rate], MODE=ACCUM, CLOCK=every-frame, WRAP=none |
| `dwellaccum` | TABLE=[rate], MODE=ACCUM, CLOCK=periodic(dwell), LEAD=lead |
| `wrapaccum` | TABLE=[rate], MODE=ACCUM, CLOCK=every-frame, WRAP=modulo[lo,hi) |
| `decay` | TABLE=[-rate], MODE=ACCUM, CLOCK=periodic(every), LEAD=lead |
| `maskaccum` / `maskaccum_stall` | TABLE=[rate], MODE=ACCUM, **CLOCK=periodic-mask** (degenerate 1-entry table, the mask IS the clock) |
| `ratewalk` | TABLE=signed steps, MODE=ACCUM, CLOCK=every-frame (per-step rate table) |
| `dwellratewalk` | TABLE=signed steps, MODE=ACCUM, CLOCK=periodic(dwell) (dwell factored out of the rate table) |
| `arp` | TABLE=freqs (P≤6), MODE=READ, CLOCK=periodic(dwell) |
| `tablewalk` | TABLE=values (P≤48), MODE=READ, CLOCK=every-frame |
| `tablewalk_lead` | as tablewalk + LEAD>0 |
| `glide` | TABLE = note-table window walked by `step` entries, MODE=READ, CLOCK=periodic(dwell), LEAD=lead (an indexed table read with stride `step` into the shared pitch table) |
| `wavetable_ptr` | TABLE=values, MODE=READ, **CLOCK=external advance vector** (the general case — every other table generator is this with a structured clock) |
| `vibrato` / `vibrato_exact` | TABLE = 4-entry `base+osc*amp` (or the byte-wise add outcomes), MODE=READ, CLOCK=every-frame; the triangle phase is a fixed degenerate period-8 walk `0,1,2,3,3,2,1,0` over a 4-value table |
| `pingpong` | TABLE=[+rate,−rate] selected by a reflect rule, MODE=ACCUM, WRAP=reflect[lo,hi], CLOCK=periodic(dwell) |
| `pingfold` | same as pingpong with WIDTH=internal `frac` extra bits and WRAP=mirror-fold (`acc=2*bound-acc`) |
| `additive_pw` | TABLE=[pulsevalue], MODE=ACCUM (8-bit lo), plus an **exogenous carry** input from the coupled freq generator — a CITG whose step is `pulsevalue + carry[i]` |

**Reading the table:** `vibrato`, `pingpong`, `pingfold`, `decay` are the cases
where the "table" is implicit (a triangle/reflect/ramp rule) rather than a stored
array, and `arp`/`tablewalk`/`wavetable_ptr` are where it is explicit. Both are
the same op; the only difference is whether the period-P loop body is a recovered
*array* or a recovered *closed-form rule* (triangle, mirror-fold). The cleanest
unification keeps BOTH as table-shapes the matcher can emit: an explicit array, or
a parametric shape (triangle(amp), reflect(rate,bounds)) that expands to the array.

### 2d. What does NOT fit cleanly (be honest)

- **`vibskydive`, `arp_decay`** are genuine *composites*: a base CITG (vibrato or
  arp) with a **second overlaid CITG** — a descending hi-byte counter gated on
  frame parity. They are `base_CITG ⊕ overlay_CITG` where ⊕ writes the overlay into
  the hi byte. This is not one CITG; it is two on one lane (a "drum/skydive"
  layered on a pitched note). It fits the *grammar* (composition of the op) but not
  a single op instance — so a unified op needs a lane-composition rule (or these
  stay as the one explicit composite shape).
- **`additive_pw`** fits as a CITG only if the cross-lane carry input is allowed
  (the step depends on another lane's per-frame carry-out). This is the
  freq→PW coupling — a real driver feature (the no-CLC carry). It is a CITG with an
  *exogenous per-frame addend*, which is a small, principled extension, not raw
  data — but it breaks the "self-contained lane" property.
- **The detectors (`note_boundaries`, `pw_sweep_resets`, `freq_note_onsets`) are
  NOT generators and do not reduce** — they are the *segmentation* layer. They stay
  as-is in any unification; they decide where one CITG ends and the next begins.
- **`pingfold`'s sub-register `frac`** is a genuine extra axis (internal precision
  > register width). It is still a CITG (WIDTH includes fractional bits), but it
  shows the WIDTH/WRAP axis is richer than a single mask.

Verdict: **~19 of the 21 generators are clean CITG parameterizations**; the two
composites (`vibskydive`, `arp_decay`) and the cross-lane `additive_pw` need the
grammar's *composition*/*coupling* rules, not just the single op. That is exactly
consistent with the thesis: the grammar is "the op + composition", and the per-tune
program is the parameters.

---

## 3. The proposed unified algorithm

### 3a. One generator op

Ship **`citg`** (§2b) as the single generator renderer, with the table allowed to
be either an explicit period-P array or a named parametric shape
(`triangle{amp}`, `reflect{rate}`, `ramp{rate}`) that expands to one, and the
clock one of `{every, periodic(Q)/dwell, mask[P], external(advance[])}`. Keep two
auxiliary rules outside the single op: **`compose`** (overlay a second CITG on a
byte range / parity — covers vibskydive, arp_decay) and **`carry-couple`** (a
CITG whose per-frame step takes an addend from a named sibling lane — covers
additive_pw). hold/accum/etc. become *presets* of `citg`, not separate ops.

### 3b. One synthesis/fitting procedure (per note-on segment)

The segmentation and greedy-cover skeleton already in `fit_lane`/`fit_segment` is
kept; only the per-run matcher is unified. For a segment `seg`:

1. **Recover the advance-clock first.** Compute the per-frame change/step stream.
   - If MODE candidate is READ: `advance[i] = (seg[i] != seg[i-1])` (the
     `_walked_table` clock).
   - If MODE candidate is ACCUM: `step[i] = signed_diff(seg)`; `advance = step!=0`.
   Classify the clock: all-true → `every`; folds to a short boolean period → 
   `periodic/mask`; otherwise keep the raw `external` advance vector (this is the
   groove tick, candidate for sharing — see step 5).
2. **Fold the table.** Walk the value/step sequence at the recovered clock; fold
   the longest prefix into the **smallest period-P loop** (`_fold_loop_prefix`
   already does this) with the HARD RULE #0 minima below. If the residual is a
   triangle/reflect/ramp, emit the parametric shape instead of the array (this is
   the vibrato/pingpong/pingfold path, recovered by the existing shape detectors —
   triangle amp from `dev/tri_phase`, reflect/fold from `mean|diff|` + bounds).
3. **Recover WIDTH/WRAP** from the lane width (12-bit PW → 0xFFF, 16-bit freq →
   0xFFFF) and the observed wrap/reflect behaviour at extremes (the data-determined
   `hi` of wrapaccum; the reflect convention of pingpong; the mirror-fold + frac of
   pingfold). LEAD = the constant prefix length absorbed.
4. **Render and take the longest byte-exact prefix** (`_match_prefix`). The greedy
   cover advances by the matched length and tiles the rest.
5. **Cross-lane / cross-note disambiguation (the consistency prior).** A recovered
   `external` clock should be **shared** across the voice's animated lanes (and, in
   the Master_Composer page-walk case, across all lanes chip-wide). After fitting a
   lane, register its clock; prefer a fit that *reuses* an already-recovered clock
   (this is the single biggest honesty lever — it forbids inventing a fresh
   per-lane "groove" that is really stored data, and it matches the driver reality
   that one tick paces many lanes). This also resolves the `additive_pw` carry: the
   carry input is the *already-fitted freq lane's* carry sequence.

### 3c. HARD RULE #0 preservation (the anti-raw-data guards, unified)

The single op MUST NOT be able to degenerate into "store the lane". The existing
scattered guards become three explicit, op-level criteria the `citg` matcher
enforces:

1. **Minimum loop cycles.** A table of period P is admitted only if the matched run
   covers **≥ `mincycles` full cycles** (existing: ≥2 for tablewalk/ratewalk, ≥2–3
   for wavetable_ptr/maskaccum_stall, ≥3 distinct values for the fold). A
   single pass over a long table = storing the deltas raw → rejected. This is the
   load-bearing guard and it generalises cleanly: the loop body is reused, or it is
   not a generator.
2. **"Covers strictly more than raw storage" criterion.** The piece's stored
   parameters must be **smaller than the frames it covers** by a margin: P (table)
   + clock description < run length. The campaign already encodes this as the
   "win only if ≥2× the base run and ≥36 frames" gate for stall/tablewalk and the
   "≥2 cycles" gate. Unified form: a CITG piece is accepted only if
   `cost(params) < α · run_length` for a fixed α<1 (e.g. the FamiCommodore voice-2
   PW that "fragments into ~70 pieces storing ~873 numbers for 903 frames" is
   rejected by this exact criterion — it is the worked example the design notes
   already cite).
3. **Piece cap.** `_MAXPIECES` (64) per note-on segment: a cover so fragmented it
   would be raw-byte storage in disguise returns None → the lane is **surfaced as
   unfit**, never faked. The external-clock fit must additionally not let the
   `advance[]` vector itself become the stored data — it is only admitted when it is
   *shared* (step 5) or *folds to a period*; an unshared, aperiodic, full-length
   advance vector on a single lane is storage and is rejected (the lane is
   surfaced).

With (1)–(3), the unified op has *strictly the same* expressive ceiling as the zoo
and the same floor: it cannot store data any more than the 28 methods could,
because the same minima gate it. The residual-zero gate is unchanged: `sum(resid)
== 0` over regs 0..24.

### 3d. Cover/greedy/early-out structure (kept)

`_longest_archetype_aug`'s structure is preserved but simplified: instead of
calling ~12 matchers and ranking, the unified matcher enumerates a small candidate
set over `(MODE, CLOCK-class, table-shape)` and returns the longest byte-exact
prefix, with the same early-outs (full-cover return; cheap presets first;
external-clock and explicit-array tables behind the substantial-length / last-resort
guards so a coincidental short period never shadows a genuine accumulator/arp).

---

## 4. Migration assessment

### 4a. Would it reduce code / maintenance? — Yes, substantially.

~21 renderers + ~21 matchers collapse to: one `citg` renderer, one `citg` matcher
(with a handful of table-shape sub-detectors reused as-is: triangle, reflect/fold,
the period folder), plus `compose` and `carry-couple`. The detector/segmentation
layer and the cover engine are unchanged. Estimated: the 1900-line zoo → a few
hundred lines, with the *knowledge* (which clocks/widths/shapes exist) preserved as
a small enum rather than a method per case. The accidental complexity that grew in
the sweep — duplicated prefix-scan boilerplate, per-archetype length/tie/last-resort
rules in `_longest_archetype_aug` — is exactly what consolidates.

### 4b. Would it keep the corpus residual-zero / recovered? — Provable, not assumed.

The op is a **strict generalization**: each archetype is a CITG preset (§2c), so
the unified renderer can reproduce every archetype's output *by construction*. The
risk is not the renderer but the **matcher** (does the unified search still *find*
the same parameterization the specialized matcher found?). This is testable without
guessing: the existing synthetic-bus tests (`test_generic_recovery.py`) already
round-trip each archetype (`render_X → fit_segment → render_fit == lane`). The
migration is validated when the unified matcher passes every one of those
round-trips AND the env-gated whole-tune `GENERIC_BUSTRACE` proof stays 7/8
(generator lanes) with FamiCommodore voice-2 PW still surfaced (not faked).

### 4c. Regression risks

- **Matcher coverage loss.** A unified search may pick a *different* (also
  byte-exact) parameterization than a specialized matcher, changing the archetype
  tally / token stream even at residual-zero. Mitigated by the consistency prior
  (§3b.5) and by ranking presets first (cheap, canonical forms win ties).
- **Search cost.** The zoo's per-matcher early-outs are tuned; a naive unified
  enumeration could be slower. Mitigated by keeping the same early-out ladder and
  the `len(unique)>maxp` pre-filters already present.
- **Composites / coupling.** `vibskydive`, `arp_decay`, `additive_pw` are the parts
  that do NOT reduce to a single op (§2d). If they are folded in clumsily the
  recovery could regress on the drum/skydive and 5TT carry-coupled tunes. Safest to
  keep them as the two explicit auxiliary rules (`compose`, `carry-couple`) rather
  than force them into `citg`.
- **Sub-register precision (`pingfold` frac).** The WIDTH axis must carry
  fractional bits or the MusicShop-style fractional triangle regresses.

### 4d. Safe migration / validation path

1. **Implement `citg` alongside** the zoo (new renderer + matcher), do not delete
   anything.
2. **Prove output-equivalence of the renderer**: for every archetype, assert
   `render_X(params) == render_citg(preset(params))` over random params (a pure
   unit test, no bus needed).
3. **Prove matcher parity on the synthetic round-trips**: run the existing
   `test_fit_*_round_trip` suite against the unified matcher; require residual-zero
   on each.
4. **Prove whole-tune parity**: run the `GENERIC_BUSTRACE` gated test on the
   cached corpus traces (Grid_Runner, Monty, the 8-tune `GT_FIXTURES`, plus the
   campaign tunes Hammurabi/FamiCommodore/etc.); require the same residual-zero set
   and the same surfaced gaps. Cross-check against the FINDINGS drivers
   (Music_Assembler … Master_Composer) where traces exist.
5. **Switch the cover to call `citg` first**, fall back to the zoo for any lane
   where `citg` regresses, and log every fallback. Drive the fallback count to 0
   (each is a precise, bounded matcher gap to close — the residual-zero discipline).
6. **Retire the zoo** only once the fallback count is 0 across the corpus.

### 4e. Open questions / where it might lose coverage

- **The external advance clock is tautological per-lane** (`_walked_table`'s own
  docstring says so): honesty lives *entirely* in the sharing/folding guard. The
  open research risk is the disambiguation prior — can the recovery reliably decide
  a recovered groove is *shared* (a real clock) vs *invented* (stored data)? The
  Master_Composer "all lanes change on the same frames" invariant is the strongest
  test; encoding it as a hard requirement for the `external` clock is the key
  design decision to validate.
- **FamiCommodore voice-2 PW** (the one open corpus gap) is a wavetable-paced
  reflecting triangle with a drifting non-periodic dwell. Under CITG it is exactly
  `wavetable_ptr` with an external clock — it closes ONLY IF that clock is
  recovered as a *shared* per-voice groove (it is, per the FINDINGS: the same groove
  pair (8,5) paces note advance, arp, and the wavetable). So the unified model with
  the sharing prior is the principled path to closing the last gap — but it depends
  on recovering the per-voice tick as a first-class object, which the current
  per-lane fitter does not yet do.
- **Composite depth.** If real drivers layer more than two CITGs on a lane, the
  `compose` rule needs to be n-ary; only 2-deep is observed so far.
- **Loop-jump grammar.** The FINDINGS show explicit `$FF` loop-to-index jumps;
  CITG's "loop point" parameter models a single loop, not arbitrary jump tables.
  Multi-target jumps would need the loop point to become a per-entry next-pointer
  (the full wavetable-jump grammar) — bounded, but a real extension.

---

## 5. Bottom line

- **The zoo is one op.** ~19/21 generators are clean parameterizations of a single
  **clocked indexed-table generator**: `value = TABLE[ptr]` (or `acc += TABLE[ptr]`
  width-wrapped), `ptr` advanced by a recovered advance-clock (every-frame /
  periodic-mask / external groove tick), with a loop point, width/wrap, lead-stall,
  and seed. The seven driver FINDINGS independently confirm this is the actual
  hardware mechanism, differing only on clock scope, read-vs-accumulate, and loop
  grammar.
- **The two real residuals** are composition (`vibskydive`/`arp_decay`) and
  cross-lane coupling (`additive_pw`) — handled by two small auxiliary grammar
  rules, not by enlarging the op. The detectors don't reduce; they are the
  orthogonal segmentation layer and stay.
- **HARD RULE #0 survives** because the same three guards (min loop cycles,
  cost<run-length, piece cap + clock-must-be-shared-or-periodic) gate the unified op
  exactly as they gated the 28 methods.
- **Migration is favorable and low-risk if staged**: implement alongside, prove
  renderer output-equivalence, prove matcher parity on the existing synthetic
  round-trip tests, prove whole-tune parity on the cached traces, fall back +
  log + drive to zero, then retire. The payoff is ~1900 lines → a few hundred and a
  principled path to the last open gap (FamiCommodore voice-2) via the shared
  groove-clock prior.
