# Parse-perf — shipped hygiene wins + dead-end registry (historical)

**Status:** LANDED 2026-06-03 (PR #49) for the wins; the doc's remaining "ranked levers" targeted the
arbiter/decode-walk path of the **retired parse-domain pipeline** and are OBE under the v3 event
model. The live perf fact (AGENTS.md wallclock anchors): under v3 the pipeline bottleneck is
**`encode(verify=True)` ~33 min over 856 dumps** (self-verify doubles work by design) — no design
exists for it; open one only if it actually gates iteration.

## What shipped (PR #49, all byte-exact, full suite green)

pandas/numpy hygiene on the block path, 21.6s → ~12.9s (~40%): `_smooth` rolling-median vectorized
via `sliding_window_view` (−34% alone); `LoopPass` records built from `to_numpy()` columns; block-path
nullable `Int64` → plain `int64` (NA audit: 3748/3748 dfs NA-free); walker `.tolist()` hoisting;
`_rows_to_df` columnar construction. Plus the test gate under pytest-xdist: full pytest 69.6s → ~9s.

## Dead ends (do not re-attempt)

- **Structural block slice** (reuse song atoms instead of per-block re-encode) — 0/33 blocks
  decode-equal even with frame-accurate slicing; reproducing the canonicalisation byte-exact is a
  multi-day decode-core rewrite.
- **Diff-attribution** of the arbiter batch — unsound (19/35 mismatch); tick-drain spills unboundedly.
- **Pure suffix-resume** from a stable snapshot — defeated by the accept-heavy fallback.
- **`register_state` memo** in block materialization — 1% hit.
- **prange/numba on the LZ77 kernels** — corpus parse already saturates cores at process level;
  the candidate loops are sequential-greedy.
