# Why the generic SID recovery grows a zoo, and the one performant generic model that ends it

**Status: DESIGN ANALYSIS / THINK-PIECE (2026-06-22). No build.** Read-only study of the
question: our generic SID generator recovery keeps accreting archetype variations
(vibrato, arp, pingpong, pingfold, vibskydive, glide, dwellaccum, ratewalk, maskaccum,
wavetable_ptr, the CITG modes, the §2 composites, the literal-table floor). The thing that
*generates* all of it is always a tiny 6502 program with only limited 8-bit arithmetic
(ADC/SBC, the shifts/rotates, AND/ORA/EOR, CMP, INC/DEC, indexed table loads, a few bytes
of zero-page state per voice). **Why can we not have a single performant, generic solution
instead of a growing zoo? Performant AND generic — both.**

The verdict, stated up front and then substantiated: **yes, a single performant generic
solution is achievable, and it is not a better closed-form library.** It is recovering the
**per-voice 8-bit state machine** — `(state cells, per-frame 8-bit transition, output
map)` — which is the literal shape of the actual 6502 code. The zoo grows without bound for
one structural reason: it fits **closed forms to the register OUTPUT**, and the set of
distinct closed forms produced by composing a handful of 8-bit ops is unbounded, while the
set of *state machines* that produce them is tiny and fixed. Output-only recovery is
non-generic (under-constrained: many machines match a finite window) **and** slow (a
combinatorial closed-form search per segment). The performant generic path is per-cell
recurrence/invariant mining over **captured state cells** — `O(n)` per cell, no archetypes —
which becomes tractable *precisely because* the ALU is 8-bit (small modulus, low recurrence
order, table indexing). This both refutes the zoo and obeys the project's HARD RULES: it
recovers the ACTUAL 6502 machine, not an invented op-set, and it has no residual escape lane.

This note grounds every claim in the live code (`preframr_tokens/bacc/generic/archetypes.py`,
`fitter.py`, `recover.py`, `cover.py`) and in a real tiny player — the JCH NewPlayer RE of
Goto80's `10.sid` (`/scratch/anarkiwi/cbm/deplayroutine/out/goto80_10/decompile.c`) — used as
the anchor for "what the state machine actually is."

It is a companion to, and agrees with, three sibling documents:
`generic_sid_decompiler.md` (the per-axis `(S,U,O,K)` recurrence model + the four-stage
slice/symbolize/generalize/verify pipeline + the literature), `sidtrace_accumulator_capture.md`
(the tracer gap that hides the freq accumulators), and `unified_generic_recovery.md` (the
CITG-subsumes-the-zoo migration). What this note adds is the **specific argument from the
archetype code that the zoo is structurally unbounded**, the **JCH state machine read off the
actual 6502**, and the **performance argument** that per-cell recurrence mining is `O(n)`
where the closed-form search is combinatorial.

---

## 1. The zoo is a library of CLOSED-FORMS fit to OUTPUT — and that is why it grows

### 1.1 What all the archetypes have in common: each is a tiny 8-bit state machine + output map

Read the renderers in `archetypes.py`. Strip the surface names and every single one is the
*unrolling* of a fixed per-frame 8-bit update over a few cells of state, plus a map from state
to the emitted register value:

| archetype (renderer) | state cells | per-frame transition `U` | output map `O` |
|---|---|---|---|
| `render_accum` (L371) | `acc` | `acc += rate` (`& width`) | `out = acc` |
| `render_wrapaccum` (L376) | `acc` | `acc += rate`; if `acc≥hi: acc−=span`; if `acc<lo: acc+=span` | `out = acc` |
| `render_arp` (L384) | `ptr`, dwell ctr | `ptr = (ptr+1) mod P` every `dwell` | `out = table[ptr]` |
| `render_glide` (L393) | `idx`, dwell ctr | `idx += step` every `dwell` after `lead` | `out = note_table[idx]` |
| `render_vibrato` (L359) | frame ctr | `ctr += 1` | `out = base + tri(ctr)·amp` |
| `render_vibrato_exact` (L428) | frame ctr, 16-bit `acc` | byte-wise `acc += amp` over `tri(ctr)` | `out = acc`, emit carry |
| `render_pingpong` (L473) | `acc`, `dir`, dwell ctr | `acc += dir·rate` every `dwell+1`; flip `dir` at `lo/hi` | `out = acc` |
| `render_pingfold` (L488) | `acc` (frac bits), `dir` | `acc += dir·step`; mirror-fold `acc = 2·bound − acc` at bound | `out = acc >> frac` |
| `render_vibreflect` (L520) | `vibtime`, 16-bit `freq` | `vibtime ^= 0xFF` past `cmpvalue`, `vibtime += 2`; `freq ± speed` by parity | `out = freq` |
| `render_decay` (L558) | `acc`, every-ctr | `acc −= rate` every `every` | `out = acc` (pre-decrement) |
| `render_dwell_accum` (L564) | `acc`, dwell ctr | `acc += rate` every `dwell` after `lead` | `out = acc` |
| `render_maskaccum` (L572) | `acc`, period-Q phase | `acc += rate` where `mask[phase]` | `out = acc` |
| `render_ratewalk` (L590) | `acc`, `ptr` | `acc += rate_table[ptr]; ptr++` | `out = acc` |
| `render_dwellratewalk` (L604) | `acc`, `ptr`, dwell ctr | `acc += rate_table[ptr]`; `ptr++` every `dwell` | `out = acc` |
| `render_tablewalk` (L582) | `ptr` | `ptr = (ptr+1) mod P` | `out = table[ptr]` |
| `render_wavetable_ptr` (L644) | `ptr` | `ptr = (ptr+1) mod P` where `advance[i]` | `out = table[ptr]` |
| `render_vibskydive` (L450) | vib ctr + hi-byte ctr | vibrato on full value, `hi −= 1` on parity | `out = vib with hi overlay` |
| `render_arp_decay` (L457) | arp ptr + hi-byte ctr | arp walk, `hi −= 1` on parity | `out = arp with hi overlay` |
| `render_additive_pw` (L464) | `pwlo` | `pwlo += pulsevalue + carry_i` (exogenous carry) | `out = pwlo` |

Every row is the same object: **a finite state `S` (a few bytes), a fixed per-frame
transition `S' = U(S)` built from 8-bit arithmetic, and an output map `out = O(S)`.** This is
not a coincidence the code stumbled into — it is forced by the hardware. A SID voice is driven
by a 6502 playroutine that, once per play-call, reads a few zero-page cells, applies a short
straight-line sequence of 8-bit ALU ops, and stores to the `$D4xx` registers. The renderers
are hand-written transcriptions of exactly that shape (the `_njit` kernels in `_njit.py` are the
per-frame loops). The author even acknowledges the unification in the module docstring
(`archetypes.py` L38–44): *"The clean generators above are all special cases of ONE op, the
Clocked Indexed-Table Generator."*

So the *generic object* is already visible inside the zoo: it is the state machine, abstracted.
The CITG (`render_citg`, L746) is the attempt to write that one machine down — and it is the
right instinct (§3 below). But notice what CITG actually is in the code: a **read/accum
state machine over a period-P table, clocked by one of {every, dwell, mask, advance}**, with a
seed/lead/phase/loop and a width mask. That is a *parameterized closed form*, recovered by
fitting the output — and that is exactly where it stops being generic.

### 1.2 Why fitting OUTPUT to closed-forms forces an ever-growing zoo

The composition of even a handful of 8-bit ops produces an **unbounded number of distinct
output closed-forms**, while the underlying machines remain few. The zoo grows because each
genuinely-new *composition* or *boundary rule* a driver uses is a new output shape that no
existing renderer reproduces byte-exact — and the gate is byte-exact-or-nothing, so a new
renderer must be added. The code is a fossil record of this:

- **Boundary/wrap variants multiply.** A bounded accumulator can wrap at the register width
  (`render_accum` `& width`), wrap by an explicit `[lo,hi)` span (`render_wrapaccum`), reflect
  at the visible extreme (`render_pingpong`), reflect *one past* the extreme (the
  `bound_pairs = ((lo−1,hi+1),(lo,hi))` dual convention, `_prefix_pingpong` L1518), or
  mirror-fold at sub-register precision (`render_pingfold`, an internal `frac`-bit accumulator).
  These are *the same accumulator* with four different boundary rules, and each needed its own
  renderer + its own matcher because each is a different output sequence.

- **Clock/dwell variants multiply.** "Add on every frame" (`accum`), "add every `dwell` frames"
  (`dwell_accum`), "add on a periodic boolean mask" (`maskaccum`), "add every frame but step the
  table on a sub-clock" (`dwellratewalk`), "step a pointer on an external groove tick"
  (`wavetable_ptr`). The `_citg_gates` function (L691) enumerates exactly these clock kinds —
  `every / dwell / dwell_ptr / mask / advance` — because each produces a different output even
  with the same state.

- **Composition variants multiply combinatorially.** `vibskydive` = vibrato *plus* a descending
  hi-byte counter overlaid on frame parity (`_hi_overlay`, L444). `arp_decay` = arp *plus* the
  same hi-byte countdown. `additive_pw` = a pulse accumulator whose addend is the *sibling freq
  lane's carry-out* (`render_additive_pw`). Each composite is two state machines whose outputs
  combine on one register; the module docstring (L959) is explicit that these *"are NOT a single
  clean CITG"* and stay as named modes. There are `O(machines²)` pairwise overlays and the code
  has implemented the three that showed up in the corpus. The next driver that overlays a
  *different* pair is a new composite.

- **Table-length and shape variants multiply.** `arp` caps period ≤ 6 (`_prefix_arp` L1412);
  `tablewalk` caps ≤ 48 (`_prefix_tablewalk` L1890); `ratewalk` caps ≤ 48; longer macro-loops
  need `_periodic_candidates` in `cover.py` (L524) to detect the period *directly* because the
  matcher's cap is exceeded. The cap is a knob, and a tune with a period-256 loop falls off the
  end of it (the comment at `cover.py` L524: *"an algorithmic tune's macro-loop (the prompt's
  tune is period-256 in places) exceeds that"*).

The mechanism is general: **a closed form is `O` composed with `U` unrolled. Composition of
8-bit ALU ops is closed under composition but the resulting `(O∘U^n)` family is combinatorially
large** — every new wrap rule × every new clock × every new overlay × every new table shape is a
new output sequence, hence a new renderer + matcher. You are enumerating the *image* of the
machines under unrolling, which is unbounded, instead of the *machines*, which are tiny. That is
the closed-form-of-output trap, and it is why the count only ever goes up.

The `unified_generic_recovery.md` inventory (§1a–§1d) makes the accretion concrete: ~21
renderers, each with a bespoke matcher, plus the detector/segmentation/framing/guard machinery.

### 1.3 Is CITG the intended unification, and exactly where does it stop being generic?

CITG is the right *direction* — it correctly identifies that the family is one read/accum
machine over a clocked table — and the migration was done honestly: `citg_preset` (L954) maps
each zoo archetype to CITG params, a parity test proves byte-identical render, and the cover was
switched to call `_prefix_citg` first with the fallback tally (`CITG_FALLBACK_COUNTS`) driven to
zero before the parallel zoo cover functions were deleted (`archetypes.py` L46–54). The single
cover op is now `_prefix_citg` via `_longest_archetype_aug` (L2405).

But CITG does **not** end the zoo; it *renames* it. Read where genericity breaks:

1. **The §2 composites are still special modes, not the unified op.** `render_citg` (L766–881) is
   a dispatch table: `mode == "vibreflect"` calls `render_vibreflect`; `"reflect"` calls
   `render_pingpong`; `"pingfold"` calls `render_pingfold`; `"vibrato"`, `"vibrato_exact"`,
   `"vibskydive"`, `"arp_decay"`, `"glide"`, `"additive_pw"` each call their original zoo
   renderer. Only *after* that 10-way dispatch (L882) does the genuine "clocked indexed-table"
   core run. CITG is a **tagged union of the zoo**, and the docstring (L67) admits the former
   `render_*`/`_prefix_*` functions *"are RETAINED … as CITG's internal recovery / render
   primitives."* The reflecting triangles (`pingpong`/`pingfold`) and the mirror-fold vibrato
   (`vibreflect`) are flagged in `unified_generic_recovery.md` §2c–§2d as the WRAP=reflect /
   WRAP=mirror cases that *"stay in the zoo as fallbacks"* — they are not expressible as the
   modulo-wrap accumulator the CITG core implements.

2. **The matcher is still a parallel battery of per-shape synthesizers.** `_prefix_citg`
   (L2180–2399) is not one recovery — it is the old zoo's matchers run in sequence and each
   winner translated to CITG params: it calls `_prefix_ratewalk`, `_prefix_wrapaccum`,
   `_prefix_tablewalk`, `_prefix_tablewalk_lead`, `_prefix_arp`, `_longest_dwell_accum`,
   `_prefix_dwellratewalk`, `_prefix_decay`, `_prefix_maskaccum`, `_prefix_maskaccum_stall`,
   `_prefix_wavetable_ptr`, `_prefix_vibreflect`, `_prefix_pingpong`, `_prefix_pingfold`,
   `_prefix_vibrato`, `_prefix_vibskydive`, `_prefix_arp_decay`, `_prefix_additive_pw`,
   `_prefix_glide`. The zoo's *code* is fully intact inside the "unified" matcher; only the output
   *token vocabulary* was unified. Adding a 22nd shape still means adding a 22nd `_prefix_*`.

3. **The literal-table floor is the escape lane wearing the CITG uniform.** When no compact CITG
   mode covers a span byte-exact, `_floor_unfit` (`fitter.py` L65) replaces it with
   `literal_table_citg` (`archetypes.py` L933): *a CITG READ over the ENTIRE observed sequence,
   walked once per frame, LOOP=0* — i.e. **the raw output stored as a length-N table**. It is
   "byte-exact by construction" because the table *is* the output. The doc is candid that this is
   the no-escape floor and instruments it separately (`record_literal_table_floor`) so the
   fraction of floored frames stays measurable and *"~0 for structured tunes."* This is the honest
   admission that output-fitting **cannot** be complete: where the closed-form library misses, the
   only residual-zero fallback is to store the output. It does not violate HARD RULE #0's letter
   (it is a genuine looping table read, the same vocabulary) but it is length-proportional, and a
   tune that leans on it has not had its generator recovered.

So: CITG is the intended unification and it is real progress on *vocabulary* and *code size*. But
it is still **output-shaped** — its candidate set is the zoo's matchers, its modes are the zoo's
renderers, and its completeness depends on a literal-table floor. The genericity ends exactly at
the §2 composites (tagged modes), the per-shape matcher battery (each a `_prefix_*`), and the
literal floor (stored output). The closed-form-of-output trap is intact; CITG narrowed the token
alphabet, not the structural cause of growth.

---

## 2. The generic object is the STATE MACHINE, not the closed-form

### 2.1 The model

A SID voice's modulation is produced by a 6502 playroutine that, per play-call, keeps a few bytes
of zero-page state, applies a short 8-bit transition, and writes the result to the chip. The
generic recovery target is therefore that machine, per voice, per axis:

```
G = (S, U, O, K)
  S : the persistent STATE cells — a finite tuple of bytes (zero-page cells,
      self-modified operands) that carry between play-calls.
  U : the per-frame TRANSITION  S' = U(S, K, X)  — a short DAG of 8-bit ALU ops
      (ADC/SBC with inter-byte carry, ASL/LSR/ROL/ROR, AND/ORA/EOR, CMP-driven
      branch, INC/DEC) plus indexed table reads. X = exogenous inputs read but not
      owned by this axis (a shared counter, a sibling lane's carry-out).
  O : the OUTPUT map  out = O(S, K)  — state → the byte(s) written to the register.
  K : the CONSTANTS U and O reference — scalars (a rate, a shift, a bound) and arrays
      (a note table, a wavetable), recovered by identity from the program's own RAM.
```

This is the same `(S,U,O,K)` model `generic_sid_decompiler.md` §1.2 specifies; this note's
contribution is to show, from the archetype table in §1.1 and the JCH read in §2.3, that **every
zoo archetype is one instance of this model and the model has a tiny fixed representation** while
the zoo does not.

### 2.2 Why this subsumes the entire zoo — exactly

Map the §1.1 table onto `(S,U,O,K)` and the zoo collapses to *parameter settings of one machine*:

- **`S` is at most ~3 cells per axis.** Across the whole zoo the state is: an accumulator (1–2
  bytes), a pointer/index (1 byte), a direction bit, and a dwell/period counter. The JCH player
  (§2.3) uses 5–6 cells for the *entire* freq pipeline of a voice. There are ~256 values per byte
  and the machine has O(1) cells, so the reachable state space is bounded and small.

- **`U` is one of a tiny set of 8-bit recurrences.** Every transition in §1.1 is: `acc += c`
  (affine, mod 2^k); `acc += table[ptr]` (table-driven affine); `ptr = (ptr+stride) mod P` (modular
  counter); `dir ^= 1` / mirror-fold at a CMP-decided bound (piecewise-affine); or an indexed read
  `O = K[idx]`. These are *not 21 different functions* — they are affine-mod-2^k, modular-counter,
  and table-index, possibly gated by a counter and possibly switched by one branch.

- **`O` is `acc`, `hi(acc)`, `table[idx]`, or a bitfield slice.** That is the whole output-map
  vocabulary the zoo uses.

- **`K` is the scalars + the arrays.** The note table, arp table, wavetable, rate table are all
  `K` arrays the output map indexes; the rate, shift, bound, dwell are `K` scalars.

The composites (`vibskydive`, `arp_decay`, `additive_pw`) are **two `G`s whose outputs combine on
one register** — a base machine plus an overlay machine (the hi-byte countdown) or a cross-axis
addend (the freq carry). They are not new primitives; they are *composition of the model with
itself* (one extra machine + a combine rule), which the model expresses natively and the zoo had
to special-case. The reflect/mirror boundaries are **`U` with one CMP-decided branch** (a
piecewise-affine transition), not a new op.

**Length-independence is the property that makes this the generic object.** `G` is fixed-size: `S`
is a few bytes, `U`/`O` are short DAGs, `K` is the constants the program references. It does not
grow as the tune plays longer. The zoo stores **one fit segment per traversal** (this is visible
in the cover: `fit_lane` slices per note-on and `fit_segment` emits a piece per byte-exact run, so
a 3000-frame slide becomes a chain of pieces); the state machine stores the *recurrence* and the
trace is its unrolling. `generic_sid_decompiler.md` §4.1 records the empirical version of this:
Grid_Runner's zoo cover was "33,913 inline fit segments but only 237 distinct signatures" — the
237 are (roughly) the machines; the 33,913 are the unrolling.

### 2.3 The actual machine — JCH NewPlayer, read off the real 6502

The decisive evidence that "the state machine is the generic object" is to read a real tiny player
and confirm the zoo's whole freq family is *one* hand-written `(S,U,O,K)`. From the
da65/Ghidra decompile of Goto80's `10.sid` (`decompile.c`, `thunk_FUN_10c1` = the play-call;
`bVar5` ∈ {0,1,2} is the per-voice offset; addresses are zero-page state cells):

**State cells `S` for the freq axis of one voice:**
- `$1795` — the **wavetable pointer** (the arp/waveform program cursor).
- `$100c/$100f` — the 16-bit **freq shadow** that is blitted to `$D400/$D401`.
- `$1778/$177b` — the 16-bit **portamento/slide accumulator**.
- `$17a7/$17aa` — the 16-bit **vibrato accumulator**.
- `$1775` (vibrato direction bit), `$1769` (vibrato half-cycle countdown), `$1772` (slide ramp).

**Output map `O` + base note (`decompile.c` L229–232):**
```
bVar2  = note_table[$166d + bVar6];               // freq-lo from the note table
$100c  = bVar2 + finetune[$1743];                 // + per-voice finetune (ADC)
$100f  = note_table[$166e + bVar6] + CARRY;        // freq-hi + inter-byte carry
```
This is literally `freq = note_table[index] + finetune`, with the index `bVar6` computed from the
wavetable control byte (next).

**The wavetable pointer transition `U` — and the BRANCHY part (`decompile.c` L206–229):**
```
cVar7 = *(char *)($1795 + 0x17db);                // read the wavetable control byte
if (cVar7 < 0)            -> bVar6 = cVar7 << 1;   // high bit set: RELATIVE arp step
else if (cVar7 == 0x7e)   -> $1795 -= 1; reread;   // '~' : step the pointer back
else if (cVar7 == 0x7f)   -> $1795 = jumptab[$1826]; reread;  // jump (loop point)
else  bVar6 = (cVar7 + transpose[$1014])*2 + octave[$1017];   // ABSOLUTE note index
```
The control bytes `$7e` (return/decrement), `$7f` (jump), and the **sign bit** (negative ⇒
relative-arp-step) are exactly the conditional control flow the prompt names: a small
**finite-state controller** over the table that selects which affine update the accumulators get.

**The accumulators `U` (`decompile.c` L278–301) — the integrator the tracer doesn't capture:**
```
// portamento ($1778/$177b), gated by control flags, direction by $1775:
DAT_172d = (note_table[next] - note_table[cur]) >> shift;   // the slide INTERVAL >> s
if ($1775 == 0)  $1778 += DAT_172d (16-bit, carry to $177b);
else             $1778 -= DAT_172d;
$100c += $1778; $100f += $177b + carry;             // slide folded into the freq shadow

// vibrato ($17a7/$17aa), direction by $17a4:
if ($17a4 == 0)  $17a7 += $179e (carry to $17aa);   // ramp up
else             $17a7 -= $179e;                     // ramp down
$100c += $17a7; $100f += $17aa + carry;             // vibrato folded into the freq shadow
```

Read what this *is*. The entire freq surface of a JCH voice — every one of the "3000 distinct freq
values over 3000 frames" the output-only recovery sees — is generated by:

> `freq = note_table[arp(wave_ptr)] + porta_acc + vibrato_acc`

where `arp` is a tiny finite-state controller over a wavetable (`$7e/$7f`/sign-bit branches),
`porta_acc` is an affine accumulator whose step is `interval(N) >> s` (a *single* slide command's
parameter), and `vibrato_acc` is a reflecting affine accumulator (direction toggled by `$17a4`,
half-cycle by `$1769`). That is **one `(S,U,O,K)` with 5–6 state cells, three affine accumulators,
one branchy table walk, and a shared note table** — and it produces what the zoo would have to
cover with `glide` + `vibrato`/`vibreflect` + `arp`/`tablewalk` + per-traversal pieces + (where
the slide is long) the literal-table floor.

This is the same machine on both ends: the `sidtrace_accumulator_capture.md` summary
(`freq = note_table[arp] + vibrato_acc + porta_acc`, the 3000 pitches being "the integral of ONE
slide command") is *confirmed line-by-line* by this decompile. The compact source is `~6 cells +
3 accumulators + the wavetable controller`; the dense output is just its unrolling. **The generic
object is this machine; the closed forms are its shadow.**

### 2.4 The representation

Concretely, the recovered per-axis program is:

```
G_axis = {
  state:   [ (addr, width) ... ],            # the cells, e.g. ($1778,16),($17a7,16),($1795,8)
  trans:   U,                                # a short typed DAG over {affine_mod_2^k,
                                             #   table_index, modular_counter, branch(cmp)}
  output:  O,                                # acc | hi(acc) | K_table[idx] | bitslice
  consts:  { scalars: {...}, arrays: {note_table@$166d stride 2, wavetable@$17db, ...} },
  clock:   advance-predicate for any stepped pointer (every / dwell / mask / shared-tick),
  exo:     [ shared_counter, sibling_carry ... ]   # X: cross-axis inputs, discovered
}
```

This is `(S,U,O,K)` made executable. The transition `U` is a **piecewise-affine recurrence over
8-bit cells with table indexing** — the minimal generalization of "affine-mod-2^k" needed to cover
the one branch the wavetable controller introduces (§4). It is fixed-size; it is the program, not
the playback.

---

## 3. Why output-only recovery is non-generic AND slow — and the performant fix

### 3.1 Non-generic: output alone is under-constrained

The trace we fit is the per-frame register OUTPUT (`.sidwr` → `per_frame_state_from_bus` →
`cover_lane`). It hides the internal state. Recovering a state machine from output alone is
**under-determined**: many machines reproduce a finite window. The code is full of guards against
exactly this ambiguity, which is the tell that the problem is ill-posed:

- `_prefix_pingpong` (L1518) must try *two* reflection conventions because the output cannot
  distinguish "reflect at the extreme" from "reflect one past it" — both fit, the matcher guesses.
- `_prefix_maskaccum_stall` (L1832) warns that an *"over-eager 'win on any length' form fragments
  the rest of the cover"* — a longer coincidental period can shadow the true generator, so the
  caller only adopts it when it covers *"SUBSTANTIALLY more"* (a heuristic threshold to break a tie
  the output can't break).
- `_walked_table` (L2061) is explicit: the pointer-walk decomposition is *"tautological, so the
  recovery's honesty lives entirely in whether `walked` folds into a SMALL looping table … vs.
  arbitrary data."* Any lane can be "explained" as `table[ptr]` with `ptr` stepping every change —
  the recovery is only meaningful if the table is small, which the output alone does not guarantee.
- The note-on segmentation is a stack of *bus-derived guesses* about where the player retriggers,
  because the output doesn't say: `note_boundaries`, `pw_sweep_resets`, `freq_note_onsets`,
  `freq_rest_boundaries`, `all_voice_boundaries`, then an interleave-period detector
  (`_interleave_period` L2484), each adopted *only if it strictly reduces the un-fit span*
  (`fit_generator_lanes` L136–219). That ladder exists because the true frame boundary is the
  play-call cadence and the true slice is the player's note-on, neither of which is in the output.

The JCH case is the sharp version: the freq shadow `$100c/$100f` is the *sum* of three accumulators
plus a table read. From the output you see only the sum. Decomposing the sum into "note + slide +
vibrato" from output alone is genuinely ambiguous (infinitely many `(a,b,c)` sum to the same
sequence over a finite window), so the output-only path reconstructs the *integral* — "718 distinct
pitches", a 52% instrument pool of big tables — instead of the one slide command that generated it.
That is not a codec inefficiency; it is the under-constraint of inverting a sum without seeing the
addends.

### 3.2 Slow: a combinatorial closed-form search per segment

Because output-fitting must *guess* which closed form (and which parameters) produced each span, it
runs a battery of synthesizers at many positions. The cost is structural:

- The per-position matcher `_prefix_citg` runs ~19 `_prefix_*` synthesizers, several of which sweep
  parameter grids and re-render to check byte-exactness: `_prefix_vibrato` nests `base × amp ×
  phase` (L1264) with two renderers each; `_prefix_vibskydive` nests `amp_lo × amp_hi(0..0x40) ×
  phase × parity` (L1314); `_prefix_pingfold` nests `frac × bound × sub × dir` (L1575);
  `_prefix_vibreflect` nests `center × cmpvalue × vibtime0` (L1683). Each candidate is rendered and
  `_match_prefix`-compared.
- `cover.py` then runs this at the DP's breakpoints to minimize *serialized* token cost
  (`cover_lane`), memoizing per start index (`_citg_cached` L434) and capping the window
  (`_MATCH_WINDOW = 512`) precisely because *"its vibrato / glide / composite sub-searches dominate
  the cover runtime"* (L434) and a *"trivial 3000-frame lane is not charged a 3000-frame vibrato
  scan at every breakpoint"* (L426). The njit kernels (`_njit.py`), the window caps, the start-index
  memoization, and the data-determined boundary candidates in `_prefix_wrapaccum` (L1202, replacing
  an `O(|rate|)` scan that *"had blown up to millions of renders on wide multispeed freq sweeps"*)
  are all there to fight a fundamentally combinatorial search. The whole-tune cover is the
  *seconds-to-minutes per tune* regime the prompt cites — and it is spent **guessing the generator
  from its shadow**.

The cost is `O(positions × candidates × parameter-grid × render-length)`. Every term is there
because the state was not observed.

### 3.3 The performant generic fix: per-cell recurrence/invariant mining over CAPTURED state

If the internal state cells are **observable** (sampled per frame), the inversion-of-a-sum and
the guess-the-shape problems both vanish, and recovery becomes per-cell and linear. The claim
(asserted in `sidtrace_accumulator_capture.md` and `generic_sid_decompiler.md` §2.5/§2.7, and
which this note endorses on the evidence of §2.3): **with the state cells captured, each cell's
update is a SHORT recurrence recoverable per-cell in `O(n)`, generic and fast, with NO archetypes.**

Mechanism, per captured cell sequence `s_0, s_1, …`:

1. **Affine-mod-2^k by first differences.** Compute `Δ_i = s_{i+1} − s_i (mod 2^width)`. If `Δ` is
   constant → `s' = s + c` (an accumulator; the JCH `$1778` slide is a *constant first difference*
   = `interval(N) >> s`). If `Δ` folds to a short period → a dwell/mask/rate-table clock. This is
   the Daikon "likely invariant over all iterations" stance [Ernst et al. 2001]: a relation that
   holds across every observed `(s_i, s_{i+1})` is **one fact**, not `n` facts — so a 3000-frame
   slide collapses to `(s_0, c, width)`, three numbers, in one linear pass.

2. **Modular counter / table index by the same difference test on a pointer cell.** A captured
   pointer cell (JCH `$1795`) exposes the `ptr += stride; wrap at P; jump at loop` directly; the
   table `K` is then lifted by **identity** from the addresses the output map dereferenced (VSA
   strided interval [Balakrishnan–Reps 2004]) — `note_table@$166d` stride 2, `wavetable@$17db`. No
   period search, no fold-or-store ambiguity: you watched `O` index it.

3. **Linear-over-GF(2) state by Berlekamp–Massey.** If a cell is suspected linear over GF(2) (an
   LFSR / bit-counter melody — A Mind Is Born's class), run Berlekamp–Massey over its captured
   sequence: it returns the **shortest** feedback polynomial (the taps = `K`) and the linear
   complexity `L` from `~2L` samples [Massey 1969]. Low `L` ⇒ recovered exactly; high `L` ⇒ report
   "not linear" and fall back (§5). This is `O(n²)` in the worst case but `n` here is the small
   sample budget (`M ≈ 512`), not the tune length.

4. **The sum decomposes because the addends are separate cells.** The JCH `freq = note_table[arp] +
   porta_acc + vibrato_acc` is *trivial* once `$1778` and `$17a7` are captured: each is its own
   affine recurrence (constant Δ for the slide; reflecting Δ for the vibrato), and the output map is
   the observed `$100c = note_table[idx] + $1778_lo + $17a7_lo` (provenance the slice gives
   directly). The "718 pitches" problem is *gone* — not compressed, dissolved — because you no
   longer invert a sum; you read the three addend recurrences off three cells.

The complexity is `O(n)` per cell (a difference pass + a constancy/period check) for the affine and
counter cases, and `O(n)` capture + `O(M²)` BM (with `M ≪ n`) for the linear case. There is **no
parameter grid, no per-position re-render, no candidate battery** — the recurrence is *read*, not
*searched*. This is both the generic solution (it is the machine, not a closed-form library) and
the performant one (linear, per-cell, no combinatorics).

### 3.4 What the tracer must capture to enable it

The recovery is only as generic as the capture. Today `preframr-sidtrace`'s STSQ samples SIDDF
*leaves*, so the freq accumulators — `$1778` (porta), `$17a7` (vibrato), `$1795` (wave-ptr),
`$100c` (freq shadow) — are **pass-through stores the slicer jumps over** (the freq write is 5
stores deep: `$100c ← +$1778 ← $172d ← note_table[$166f] ← $1014`), so recovery is blind to the
slide command and falls back to fitting the sum. This is the *exact* gap `sidtrace_accumulator_
capture.md` scopes. The needed additions (bounded, `O(code sites)` not `O(frames)`, a few KB/tune):

1. **Sample every transitive state cell of a `$D4xx`-write slice** into STSQ, not just leaves
   (`sidtrace_accumulator_capture.md` change #1; `generic_sid_decompiler.md` §3.2 STATESEQ). Raise
   `STATESEQ_CAP` to ~3 voices × 6 pitch cells + filter + pulse. This surfaces `$100c/$100f`,
   `$1778/$177b`, `$17a7/$17aa`, `$1795` as per-frame sequences — the input to the `O(n)` difference
   test and BM.
2. **A per-write data-flow summary (SIDDF)** keyed by `(PC, reg)`: the backward-slice PC set, the
   leaf kinds (immediate / ram-read / state-cell / exogenous via the ACMP access map), the indexed
   read's strided interval, and the ALU op sequence (`generic_sid_decompiler.md` §3.1). This turns
   "PC wrote freq-lo" into "freq-lo = `note_table[idx] + $1778`" — the output map `O` and the cell
   identities, for free, from one play-call's instruction window.
3. **The orderlist→pattern pointer-walk** (`(zp),Y` base sequence, here `$00fb/$00fc`) as
   first-class (`sidtrace_accumulator_capture.md` change #4) — the score/structure, so the note
   *sequence* feeding `arp` is recovered, not per-frame note events.
4. **Indexed-read attribution for scaled indices** (`base = addr − scale·idx`, the `note*2` case),
   so the note table at `$166d` stride 2 is correctly identified (change #3).

These are upstream, bounded, and generalize to every slide/portamento/table-walk player — not a
codec patch. With them, the §3.3 per-cell recovery is enabled; without them, the output-only path
is forced and the zoo (or the literal floor) is the only residual-zero option.

---

## 4. The residual hard parts + the minimal generalization

Capturing state and mining per-cell recurrences is generic and fast for the affine/counter/linear
core, but it is not the whole problem. The genuinely hard parts are real and bounded:

1. **Composition across cells (the output map is n-ary).** The freq shadow is `O = note_table[idx] +
   porta_acc + vibrato_acc` — a sum of one table read and two accumulators, gated by control flags
   (`$174e`, `$1766`, `$174b`, `$17a4`, `DAT_100b` in the JCH play-call). Each *addend* is a clean
   per-cell recurrence, but the *combine* is n-ary and conditionally present. The minimal
   generalization is to recover `O` as an **expression DAG over the captured cells** (the SIDDF
   op-sequence gives its shape directly) rather than assuming a single-cell output — i.e. let `O` be
   `Σ cells` / `select(cells)`, not just `acc`. This is bounded (the DAG is the few ops the slice
   used) but it is the step beyond "one cell → one register." `generic_sid_decompiler.md` §5.4 flags
   deep (≥3-way) coupling as a real extension; the JCH 3-addend sum is exactly the 3-way case and is
   the common one, so n-ary `O` is not optional.

2. **Branch-dependent transitions (the wavetable controller).** The wavetable control bytes
   `$7e`/`$7f` and the sign-bit (`decompile.c` L206–229) make the pointer/arp update **piecewise**:
   step-back vs jump vs relative-step vs absolute-note. The same shows up across drivers
   (`sid_opset_inventory.md` §6: GoatTracker `$FF`/`$FE` wavetable jumps, SID-Wizard `$FE`/`$FF`,
   defMON cascade JP, Galway/Follin `CALL/RET/FOR/NEXT`). A single affine recurrence cannot express
   this, and Berlekamp–Massey will report high linear complexity on the *combined* sequence. The
   minimal generalization is a **small finite-state controller over the linear cells**:
   piecewise-affine, where the *predicate* is the observed branch from the slice's control edges
   (the captured CMP/BNE on the control byte), and each branch arm is a clean affine/table update.
   This is the model `generic_sid_decompiler.md` §2.5(1) names ("conditional `s' = reset when
   pred(s) else s ± c`") and §5.2 caveats (data-dependent control flow): the *observed* path is
   exact, but a predicate on un-modeled state may not generalize to an unobserved branch — caught by
   the residual-zero gate, then surfaced. The control bytes are themselves `K` (a captured table),
   so the controller is recovered as "walk this byte table, dispatch on its ranges" — bounded, but a
   genuine FSM layer above the affine cells.

3. **Free-running phase / direction seed.** A reflecting accumulator (`vibrato_acc` with its
   `$1775`/`$17a4` direction and `$1769` countdown; WEMUSIC's free-running sweeps;
   `sid_opset_inventory.md` §5.3) carries an *initial phase and direction* that is not note-reset.
   With the cell captured this is trivial — the seed is `s_0`, observed. Without capture it is the
   "lane that broke everything" (the free-running phase search). This is an argument *for* capture:
   the hard synthesis problem of recovering hidden phase is eliminated by sampling the cell.

4. **Self-modifying code that rewrites whole sequences.** Where a player rewrites *instruction
   sequences* (not just operand bytes) per frame, the slice shape changes frame-to-frame and there
   is no fixed `U` (`generic_sid_decompiler.md` §5.1). Detection: the captured op-sequence is not
   invariant across frames. Fallback: surface the span (length-proportional, flagged) — never a
   hidden patch. This is rare (operand-level SMC, which *is* handleable as a state cell, is the
   common case).

5. **The honest floor stays.** Where a cell's sequence is genuinely high-complexity (a true external
   PRNG, an un-modeled hardware read), no compact recurrence exists; BM reports `L ≈ n/2`. The axis
   is surfaced as unrecovered, length-proportional, with the reason — the same discipline as the
   literal-table floor, but now an *explicit per-cell verdict* rather than a silent output store.

The shape of the residual is therefore: **n-ary output composition + a small piecewise-affine
controller over the linear cells**. Both are bounded extensions of "affine-mod-2^k per cell"; both
are recoverable from captured state + the slice's control edges; both are gated by residual-zero so
under-coverage is surfaced, not faked. Neither reintroduces a closed-form *library* — there is one
model (cells + piecewise-affine transition + n-ary output), parameterized.

---

## 5. The honest verdict — is performant + generic achievable, and the path

**Yes. A single performant generic solution is achievable, and it is the per-voice 8-bit state
machine, recovered by per-cell recurrence/invariant mining over captured state.** It is performant
(`O(n)` per cell for the affine/counter core, `O(M²)` with `M ≪ n` for the linear core; no parameter
grid, no per-position re-render battery) and generic (it is the model the JCH player *is*, not a
library fit to the JCH player's *output*). It obeys the project's HARD RULES by construction:

- **"Recover the GENERATOR, don't curve-fit the OUTPUT."** Per-cell recurrence mining reads the
  RAM state-variable's update directly from the captured cell + the bus slice — the exact thing the
  rule demands. It does not RLE/compress the delta stream; it recovers `s' = s + c` as three numbers.
- **The generic solution is the ACTUAL 6502 machine, not a new invented op-set.** The model's
  transitions are the literal 8-bit ALU ops the slice executed (ADC/SBC with inter-byte carry, the
  shifts, EOR for the reflect toggle, indexed reads), and its constants are lifted by identity from
  the program's own RAM (the note table at `$166d`, the wavetable at `$17db`). The refuted shape was
  *"any invented op-set + residual escape lane"*; this is the opposite — the op-set is the chip's
  programmer's, and there is no escape lane (high-complexity cells are *surfaced*, not patched).
- **Residual = 0 stays the gate; < 1 token/frame is met by length-independence.** The recurrence is
  fixed-size, so the recovered program does not grow with playback; the dense output is the
  unrolling, recovered for free at render time.

**The zoo is not fundamental.** It is the symptom of one wrong move: fitting closed forms to the
output. Because composing 8-bit ops yields an unbounded family of output closed-forms (every wrap ×
clock × overlay × table-shape is a new sequence) over a tiny fixed family of machines, output-fitting
must enumerate the image (unbounded → the zoo) instead of the machines (tiny → one model). CITG
narrowed the *token alphabet* (real progress on size/maintenance) but kept the *output-shaped*
substance — its modes are the zoo's renderers (`render_citg` L766 dispatch), its matcher is the
zoo's `_prefix_*` battery (`_prefix_citg` L2180), and its completeness rests on the literal-table
floor (`literal_table_citg` L933). The growth pressure is therefore intact under CITG, and the only
thing that removes it is changing what is recovered: the machine, not its shadow.

**The path (staged, gated by residual-zero, each stage bounded and validated):**

1. **Capture the state cells (upstream, in `preframr-sidtrace`).** Implement
   `sidtrace_accumulator_capture.md` changes #1 (transitive state-cell sampling into STSQ; raise
   `STATESEQ_CAP`) and #4 (orderlist pointer-walk), plus the SIDDF per-`(PC,reg)` data-flow summary
   (`generic_sid_decompiler.md` §3.1) and the scaled-index attribution (#3). This is the single
   enabling change; everything downstream is blocked on it. Validate on JCH `10.sid`: STSQ must
   surface `$1778/$17a7/$1795/$100c` as per-frame sequences.

2. **Per-cell recurrence mining (host, no archetypes).** For each captured cell: first-difference
   constancy/period (affine-mod-2^k, dwell, mask, rate-table); pointer-cell modular-counter +
   identity table lift (VSA strided interval); Berlekamp–Massey for suspected-linear cells. This
   *replaces* `_prefix_citg`'s candidate battery with a per-cell `O(n)` read. Validate the
   length-independence property: recover from a short window (e.g. 512 frames), render the full tune
   residual-zero.

3. **n-ary output map + piecewise-affine controller (the residual hard parts, §4).** Recover `O` as
   the captured expression DAG over cells (the 3-addend JCH sum); recover the wavetable controller as
   a small FSM over the captured control-byte table whose predicates are the slice's control edges.
   Gate by residual-zero; surface any cell whose recurrence does not hold or whose linear complexity
   is high (the honest floor, now a per-cell verdict).

4. **Retire the zoo (and the literal-table floor as the structural fallback).** Once per-cell
   recovery + n-ary `O` + the controller reach residual-zero across the corpus with the floored-frame
   fraction at ~0, the `_prefix_*` battery and `render_citg`'s mode dispatch are dead code: the
   machine subsumes them by construction (each archetype is one parameter setting of cells + transition
   + output, proven by the §1.1 table). The literal-table floor remains only as the explicit
   surfaced-unrecovered marker for genuinely high-complexity cells, never as the silent residual sink
   for a generator the search merely failed to guess.

The one-line answer to the question: **we cannot have a single performant generic solution as long as
we fit closed-forms to the OUTPUT, because that image is unbounded; we *can* have one the moment we
recover the per-voice 8-bit STATE MACHINE from captured state cells, because the 8-bit ALU makes that
machine tiny (small modulus, low recurrence order, table indexing) and per-cell recurrence mining
reads it in `O(n)` — generic and performant, the actual 6502, no escape lane.**

---

## Evidence index (files cited)

- `preframr_tokens/bacc/generic/archetypes.py` — the renderers (§1.1 table; L359–662), `render_citg`
  mode-dispatch (L746–930), `literal_table_citg` (L933) the floor, `citg_preset` (L954), the
  `_prefix_*` matcher battery and `_prefix_citg` (L2180–2399), `_walked_table` tautology guard
  (L2061), `_longest_archetype_aug` (L2405), `fit_segment`/`fit_lane` per-traversal pieces (L2442),
  the note-on segmentation ladder (`note_boundaries`/`pw_sweep_resets`/`freq_note_onsets`/
  `freq_rest_boundaries`/`all_voice_boundaries`, L158–317), the interleave detector (L2484).
- `preframr_tokens/bacc/generic/fitter.py` — the re-slice retry ladder (`fit_generator_lanes`
  L88–227), `_floor_unfit` (L65), the note-table provenance recovery (`discover_note_table_from_bus`
  L287).
- `preframr_tokens/bacc/generic/recover.py` — the output-only public path
  (`recover_generic`/`render_generic`, the `.sidwr` → state → fit → render loop).
- `preframr_tokens/bacc/generic/cover.py` — the token-cost DP and its combinatorial-search mitigations
  (`_citg_cached` L434, `_MATCH_WINDOW` L431, `_candidates` L470, `_periodic_candidates` cap-escape
  L524).
- `/scratch/anarkiwi/cbm/deplayroutine/out/goto80_10/decompile.c` — the JCH NewPlayer play-call
  (`thunk_FUN_10c1`): the freq output map (L229–232), the branchy wavetable controller (L206–229),
  the porta/vibrato accumulators (L278–301) — the real `(S,U,O,K)` of §2.3.
- `design/encoding/sidtrace_accumulator_capture.md` — the capture gap (the accumulators are
  pass-through stores STSQ skips) and the bounded fix (§3.4).
- `design/encoding/generic_sid_decompiler.md` — the peer `(S,U,O,K)` recovery model + the four-stage
  pipeline + SIDDF/STATESEQ capture + the literature (Daikon, Berlekamp–Massey, VSA, slicing).
- `design/encoding/unified_generic_recovery.md` — the CITG-subsumes-the-zoo inventory (§1–§2c) and the
  honest §2d composites / §3.6 floor / §4e open coverage.
- `design/encoding/sid_opset_inventory.md` — the four-driver op-set (the wavetable `$FF`/`$FE` jumps,
  the ACCUM/COUNTER/PTR-WALK/SELECT primitives) confirming the branchy-controller + n-ary-output
  residual is cross-driver, and the Galway/Follin escape-hatch caveat (the real bound on "any
  playroutine").
- `AGENTS.md` — HARD RULE #0 (nothing irreducible), "Recover the GENERATOR not the OUTPUT", the refuted
  "invented op-set + residual escape lane", the JCH "other face of the wall" note.
