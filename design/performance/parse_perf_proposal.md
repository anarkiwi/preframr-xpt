# Parse-perf proposal — prioritized, correctness-gated

**Status:** Proposal (2026-06-03). Synthesizes the earlier parse-perf scoping
(the cProfile profiling + the finding that the bottleneck is the arbiter's
per-claim `validate=True` fallback decode, not the `register_state` memo) into a
ranked plan. Companion
deliverable: the preframr-tokens test gate now runs under pytest-xdist (`run_tests.sh -n auto
--dist worksteal`), full pytest 69.6s -> 13.4s; that work is separate from and orthogonal to the
parse-perf changes below.

## The one number that decides priority
Clean 27-tune cProfile (digi-excluded), post the two uncommitted wins (B+partition, ~543s):
**`run_block_refire_passes` = 440s = 86%.** It runs the *entire* 20-pass encoder
(`FREQ_BLOCK_PASSES` + `PASSES`, including the arbiter's `validate=True` decode walks) **once per
block**, after `expand_to_literal_form` decompiles the song back to literal SETs. With training
config (`seq_len=1024`, `frames_per_block=512`, `block_stride=256`) blocks **overlap 2x**, so every
frame is re-encoded ~2x — and all of it re-derives atoms the parse stage *already encoded once* and
wrote to its sidecar.

So the dominant cost is **redundant re-encoding (frequency), not slow decoding (unit cost).** Attack
frequency first. The arbiter-fallback work that earlier scoping chased is real but secondary — B +
partition already took the cheap wins there.

## Priority ladder

### Tier 0 — bank the done work (today, zero new risk)
Commit the two uncommitted, byte-exact-verified changes already in the preframr-tokens tree:
- **B (`state._build_last_diff` groupby)** — 10.8x on that fn, ~12% end-to-end, **zero risk** (0
  mismatches / 60 dfs). Ship unconditionally.
- **Partition (`arbitrate_independent_groups`)** — ~5% net, uneven, proven equal to flat greedy
  under `PREFRAMR_VERIFY_PARTITION`. Ship with the guard.

Captures ~18% that is currently just sitting uncommitted. Nothing else here is worth doing until
this is banked, because the profile that ranks everything else is the *post-B* profile.

### Tier 1 — kill the re-encode multiplier (THE dramatic lever)
**Slice the already-encoded full-song atom stream into block windows instead of
`expand_to_literal` + re-running 20 passes per block.** The parse stage encodes the whole song once;
`iter_self_contained_row_blocks` throws that away and re-encodes. Replace per-block re-encode with:
1. slice the encoded atom stream at the block's `[lo_frame, hi_frame)` markers;
2. fix only the **cross-boundary** state: localize DEF/REF pairs that straddle the cut, expand
   out-of-block `PATTERN_REPLAY/GATE_REPLAY/PLAY_INSTRUMENT/DO_LOOP` references to literal, and
   replay the per-register tick-drain at the boundary.

No pass re-runs; no per-block arbiter decode. Removes the bulk of the 440s **and** the 2x overlap
multiplier at once. This is "route C" in the scoping doc — biggest lever, biggest risk, deliberately
deferred there, but it is where the dramatic gain lives.

**Correctness gate (non-negotiable, and the reason this is safe to attempt):** behind an env flag
(same pattern as `PREFRAMR_VERIFY_PARTITION`), for every block assert
`decode(sliced_atoms) == decode(re_encoded_from_literal)` byte-exact across the corpus, plus the
existing `test_block_refire_contract` (every reference-op producer must survive into the window).
Trust the fast path only after a clean corpus sweep; keep the re-encode path behind the flag for A/B.

### Tier 1b — de-overlap the alphabet build (cheap, do alongside 1)
`make_tokens` needs only the *set* of block atoms; overlapping windows add no new atoms but re-encode
~2x. Build the alphabet from non-overlapping windows (`stride = frames_per_block`) or directly from
the full-song atoms. Low risk; verify the alphabet is identical to the training atom set. Immediate
~2x on the alphabet pass independent of Tier 1's redesign.

### Tier 2 — windowed per-claim arbiter validation (for the tunes Tier 1 doesn't fully cover)
Use the already-built, byte-exact `resume_decode.py` to validate a claim/group over only its frame
window `[claim_frame, reconvergence]` instead of the whole df. This is the only *exact* speedup left
for dense-ctrl tunes (e.g. Bass_Guitar) where the partition degenerated to ~3 big reg-groups.
Moderate risk (touches the byte-exact decode core); gate on `cb_div_audit`. Lower priority than Tier
1 because Tier 1 removes most arbiter *invocations* outright — do it only if dense-ctrl tunes still
dominate after Tier 1.

### Tier 3 — compile the walker floor (last, after frequency is fixed)
Once Tier 1 removes the redundant invocations, the residual is the intrinsic per-row Python decode
(`_walk_frame` 292s + `_dispatch_row` 185s). numba/Cython the SET/DIFF fast path. Biggest effort;
do it **last** because Tier 1 changes how often it runs and therefore its ROI. Don't compile a hot
loop you're about to call 80% less often.

## Why this order
Tier 1 attacks **how many times** the walker runs; Tier 3 attacks **how fast each run is**. Frequency
first: it's higher leverage, and it makes the corpus tractable enough to iterate on the harder
compiled-walker work. Every tier ships behind a byte-exact corpus equivalence assertion vs the
current path (the established `PREFRAMR_VERIFY_PARTITION` / `cb_div_audit` discipline) — that is what
lets "dramatic" and "without sacrificing correctness" coexist.

## UPDATE 2026-06-03 — Tier 1 (structural slice) is a dead end; the real lever is pandas/numpy hygiene

Built the equivalence gate (`/scratch/tmp/block_equiv.py`) and tested "slice the song's
already-encoded atoms instead of re-encoding per block." **Result: 0/33 blocks decode-equal**, and
decisively `blk0` (frame-0 start, no pre-block state) diverges in 442/513 frames. The divergence is
*global*, not at the cut: the reference path's full re-encode + `_norm_pr_order` +
`_consolidate_frames` + `_cap_delay` canonicalise the block in ways a slice doesn't reproduce
(DELAY unrolling, frame consolidation, block-local LoopPass LZ77). Making it byte-exact is a
multi-day decode-core rewrite that *also* changes training tokenisation. **Shelved.**

cProfile of the block path (51 blocks, full_macros) instead pinned the cost on **pandas/numpy
anti-patterns**, which are byte-exact-safe to fix and hit the spread-out per-pass cost:
- **`trajectory_anchor._smooth`** ran a rolling median as a Python comprehension calling `np.median`
  559,630× = **7.4s / 34%** of the block path. Vectorised the full-width interior windows with
  `sliding_window_view` + one `np.median(axis=1)`, edges kept per-index. **Bit-identical (2000/2000
  random incl. NaN/edges), full suite 885 pass. 21.6s → 14.3s (−34%).**
- **`LoopPass` `df.to_dict("records")`** boxed nullable `Int64` cells one at a time (3.6M
  `maybe_box_native` + 3.6M `masked.__iter__`). Built records from `to_numpy()` column arrays
  instead. **Suite green. 14.3s → 13.5s (−5%); call count 38.6M → 34.1M.**

**Combined: 21.6s → 13.5s = ~37% on the block path, both byte-exact (suite + per-frame fidelity +
`test_sid_frame_diff` all green).** Changes are in the preframr-tokens working tree
(`trajectory_anchor.py`, `loop_pass.py`), uncommitted.

- **Nullable `Int*` → plain `int64` in the block hot path (DONE, contained).** NA audit: across the
  whole pass pipeline (parse-time fresh instances + block re-fire + voiced post-passes), **3748/3748
  dfs are NA-free** in all 7 columns (reg/val/diff/irq/op/subreg/description). The freq passes use
  NaN only on extracted numpy arrays, never in the df columns. So nullable Int* is pure boxing in the
  pass loop. `iter_self_contained_row_blocks` now casts the working frame to `int64` and restores the
  canonical dtypes at the block boundary (byte-identical). **Suite green; 13.5s → 12.9s (−4% here).**
  Modest because the cast-back is per-block and the residual boxing is in DataFrame *construction*
  (`pd.DataFrame(rows)`), not iteration.

  **Why NOT global:** the canonical nullable constants can't simply switch to numpy — the early-parse
  stages (`_read_df`/`_combine_regs`/pivot+ffill, *before* the passes) use NA intentionally and were
  out of audit scope. The contained block-path conversion is the safe, hot scope.

**COMBINED RESULT: block path 21.6s → 12.9s ≈ 40%, all byte-exact (full suite 885 pass incl. per-frame
fidelity + `test_sid_frame_diff`).** Changes uncommitted in preframr-tokens:
`trajectory_anchor.py`, `loop_pass.py`, `macros/blocks.py`.

### UPDATE 2026-06-03c — levers 1 (walker) & 2 (columnar construction) done; cheap pandas wins now exhausted

- **Lever 2 — columnar DataFrame construction.** Added `passes_base._rows_to_df` (numpy int64 fast
  path + fallback to object inference on any irregular value) and applied it to `_splice_rows`
  (arbiter/collapse) and `VoiceBlockOrderPass`. `construction.convert` calls 2090 → 934. Byte-exact
  (suite 885 pass). **Wall-time effect within run-to-run noise (~±0.3s)** at this scale — the per-row
  dict→DataFrame inference was a smaller slice than its tottime implied (cumtime overlapped the walk).
- **Lever 1 — decode walker floor.** A true numba *compile* is infeasible without rewriting the whole
  decode model: the decoders (`FreqTrajectoryDecoder`, etc.) and `DecodeState.tick_frame` are
  Python-dict / dynamic-state based (`pending_ft` dict, `pending_set_writes[reg]` lists, link dicts) —
  a multi-day, very-high-risk rewrite of the byte-exact fidelity oracle. **Not attempted blind.** The
  safe slice taken instead: the hot loop indexed numpy arrays element-wise; microbench showed
  `int(ndarray[i])` is **3× slower** than `int(list[i])`. Hoisted the per-row columns as `.tolist()`
  lists (`self.arrs` stays numpy for the vectorised fastpath check). `_dispatch_row` tottime 1.60 →
  1.31s; byte-exact (suite 885 pass). **End-to-end ~1.3%** — small, because the 43% the walker
  represents is dominated by *decoder/state compute*, not dispatch/indexing overhead.

**Honest bottom line:** the byte-exact pandas/numpy hygiene wins are now **exhausted** — block path
21.6s → ~12.9s ≈ 40%, almost entirely from `_smooth`. The remaining ~43% is the decoder/state compute
inside the walk; cutting it materially requires **either** a numba rewrite of the decode core
(multi-day, high-risk, needs explicit scoping/sign-off) **or** reducing the *number* of walks
(windowed per-claim validation via `resume_decode.py` for dense-ctrl tunes; the structural per-block
re-encode removal is the proven dead end). All session changes uncommitted in preframr-tokens:
`trajectory_anchor.py`, `loop_pass.py`, `macros/blocks.py`, `macros/walker.py`, `macros/passes_base.py`,
`macros/passes.py`.

**Remaining ranked levers (next, by leverage):**
1. **numba/Cython the decode core** (`_dispatch_row` + decoders + `tick_frame`) — the intrinsic floor;
   high-risk byte-exact-core rewrite, needs sign-off.
2. **Windowed per-claim validation** via `resume_decode.py` — reduces walk *count* on dense-ctrl tunes
   the partition couldn't help.
3. **Parse-stage pass loop** (`reglogparser.parse` 981-995) — same `int64` fast-dtype treatment as the
   block path; once-per-song, lower leverage.

## SHIPPED 2026-06-03 (PR #49, merged)
The byte-exact pandas/numpy hygiene + xdist test gate landed: `_smooth` vectorize, LoopPass record
build, block-path `int64` (dtype audit), walker `.tolist()`, `_rows_to_df` columnar construction.
Block path 21.6s → ~12.9s (~40%, dominated by `_smooth`); full pytest 69.6s → ~9s under xdist.

## Follow-up analyses (2026-06-03, post-merge)
- **Arbiter static drain-span pruning (best remaining ROI; no decode-core rewrite).** The arbiter
  validates by decoding; it guards two things — (A) a claim is individually lossy (pass bug) and
  (B) claims interact via the per-frame tick-drain. Both are largely STATIC: (A) is a per-pass
  byte-exact contract (the `PREFRAMR_ARBITER_STRICT` philosophy); (B) reduces to interval overlap if
  each claim declares a `drain_span` — and the drain length IS computable from the claim's tokens (a
  FreqTrajectory queues exactly `runtime`/`count`/`period` pending values). Give each claim an
  influence interval `[first_frame, last_write+drain_span]`; non-overlapping claims accept with ZERO
  decode, overlapping clusters decode over their own window (via `resume_decode.py`). Makes the
  partition's empirical `_GROUP_GAP=8` principled. Sound only if drain spans are conservative
  upper-bounds; gate with the `PREFRAMR_VERIFY_PARTITION`/`cb_div_audit` equivalence check. Scope:
  ctrl/freq register-exact claims only (codebook/loop refs are long-range, excluded).
- **prange: not applicable.** All njit is the LoopPass LZ77 kernels. The corpus parse already
  saturates cores at the process level (`ProcessPoolExecutor(cpu_count())`), so intra-kernel `prange`
  would oversubscribe and slow the corpus run; it'd only help single-song latency. Also the candidate
  loop is an argmax reduction (not a clean prange reduction) and the position loop is sequential-greedy
  with loop-carried seed state. The real numba target is compiling the (currently un-jitted) decode
  walk — serial `@njit`, NOT prange.
- **Generators in `iter_self_contained_row_blocks`: keep them.** The generator costs ~0 (yield is
  microseconds); it buys laziness (peak memory ~1 block). The per-block cost is the decode walk inside
  `run_block_refire_passes`, which is row-sequential stateful — not array-vectorizable, and not
  batchable across blocks (block-local LZ77/codebook scope + self-containment). "Build list then yield"
  is strictly worse. `expand_to_literal`/markers/consolidator are already hoisted out of the loop.

## Macro-abstraction consolidation (separate analysis)
The macro *mechanics* are essential complexity; the *registration/wiring* was fragmented across ~5
surfaces glued by a checker. The generator-MDL pipeline
([`generator_mdl_representation.md`](../encoding/generator_mdl_representation.md)) collapses most of that zoo into one
pass, which is the real consolidation; this is performance-NEUTRAL directly but de-risks the decode-core.

## Dead ends (recorded so they're not re-attempted)
- **Structural block slice (reuse song atoms instead of per-block re-encode)** — diverges 0/33 even
  with frame-accurate slicing and clean frame-0 start; reproducing the canonicalisation byte-exact is
  a multi-day decode-core rewrite. Pandas hygiene gives most of the win at a fraction of the risk.
- **Diff-attribution** of the arbiter batch — unsound (19/35 mismatch); the tick-drain spills
  unboundedly, so "claims near a divergence" has no small bound.
- **Pure suffix-resume** from a single stable snapshot — defeated by the accept-heavy fallback (317
  accept / 10 reject): `accepted` grows almost every claim, so there's no reusable prefix.
- **register_state memo (#5)** — 1% hit in block materialization (distinct content per pass). Scope
  it to the parse.py path or retire it.
