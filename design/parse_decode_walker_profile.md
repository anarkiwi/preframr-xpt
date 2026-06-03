# Parse profiling: the decode walker dominates

**Status:** Scoping — profiled 2026-06-03 (tokens 0.42.0, image `anarkiwi/preframr:0.2.17`);
recommendations below are unbuilt. Companion to the AGENTS.md forward-looking item
"Profile + optimize preframr-tokens parsing".

## Method

cProfile around the real per-song unit `parse_runner.write_df` →
`RegLogParser.parse(name)` on one representative song
(`MUSICIANS/D/DRAX/Advanced.1`, 316 KB), `--macro-config full_macros` and the
codebook flag set. Single worker (the corpus run is `ProcessPoolExecutor`,
`cpu_count()` workers — wallclock scales with cores, so the per-song cost below is
what multiplies).

## Finding — it's the pure-Python decode/validation walker, run ~24×/song

| cfg | wall/song | `walker.walk` calls | `walk` cumtime | `register_state` (full decode) calls | decode cumtime |
|---|---|---|---|---|---|
| full_macros | 8.66 s | 24 | 5.08 s (59%) | 12 | 2.85 s (33%) |
| codebook | 7.69 s | 25 | 3.65 s | 13 | 2.19 s |

`FrameWalker` re-decodes the whole token stream row-by-row in pure Python; one
song sums to ~930k `_dispatch_row` calls. Per-row leaves dominate `tottime`:
`state._fastrow_from_arrs` (0.92 s, 929k calls — a row object built per row),
`walker._dispatch_row` (0.88 s), `state.tick_frame`, `decoders._delta_run` /
`expand`. Each full walk costs ~0.21–0.24 s; there are ~24 of them per song.

Two driver categories produce the 24 walks:
1. **Passes that decode-with-state** to do their work — `passes._apply_with_state
   → walker.walk()` and `_SnapshotWalker(df, state).walk()` (passes.py:458/512/601).
   Each freq / loop / snapshot pass walks the whole stream.
2. **The arbiter's byte-exact validation** — `arbitrate(validate=True) →
   _decoded_state → register_state` decodes BOTH the source `df` and the candidate
   `out` for every register-exact pass (arbiter.py:90,93). In the per-claim
   fallback (only when the batch isn't fully lossless) it re-applies + re-decodes
   O(claims) times.

The walker already consumes rows via numpy arrays (`self.arrs[...]`), so the
`to_dict` / `maybe_box_native` cost (~0.8 s, 9%) is NOT the walk — it's the macro
passes building their OUTPUT row-by-row (`transform._row_to_dict` →
`pd.DataFrame(out_rows)`), a separable cost.

## Recommendations (low-risk, ranked by leverage; all guarded by the byte-exact + per-frame fidelity tests)

1. **Thread decoded state through the pipeline — don't re-decode the same bytes.**
   Each validated pass's lossless `out` IS the next pass's input `df`, which the
   next `arbitrate` then re-decodes as its `src_state`. Have `arbitrate` return
   `(out, out_state)` and let the caller feed `out_state` in as the next
   `src_state`. Pure plumbing; the oracle math is unchanged. Targets a large slice
   of the 12 `register_state` calls (~2.85 s → est. ~1.5 s/song). Highest leverage,
   low-moderate risk.

2. **Trim per-row allocation in the walk inner loop.** `_fastrow_from_arrs` +
   `State.__init__` run ~929k×/song (a fresh row/state object per row). Reuse a
   single mutable row buffer and hoist the `self.arrs[...]` / `self.state` lookups
   into locals across the per-frame loop. Mechanical, behaviour-preserving; ~0.3 s
   /song (~3%) at the single largest leaf.

3. **Build pass outputs column-wise, not via per-row dict boxing.**
   `transform._row_to_dict` → `pd.DataFrame(out_rows).astype(...)` round-trips
   through Python dicts + `maybe_box_native` + dtype re-inference (~0.8 s, ~9%).
   Construct the output frame from numpy columns. Contained to the transform
   helpers.

4. **(Measure first) memoize `register_state` on a content fingerprint within one
   song-parse.** 12 decodes/song; if any land on identical df content an exact
   fingerprint (len + hash of the op/reg/val/diff array bytes) returns instantly.
   Lower-confidence than #1 — needs a hit-rate measurement before building, and #1
   removes most of the redundancy more directly.

Not recommended as "low-risk": rewriting the walker (it is the fidelity oracle) or
extending the existing llvmlite/numba jit into `_dispatch_row`/`_walk_frame`.

## Note for the per_reg_burst fix release

The `per_reg_burst` empty-cand+barrier crash fix (tokens `9b2d10f`) is unrelated to
this profile (a cold-path indexing bug). If the parse-perf work is taken before the
tokens release, both can ship in one bump; otherwise release the fix first — it
unblocks the codebook pipeline on the real corpus.
