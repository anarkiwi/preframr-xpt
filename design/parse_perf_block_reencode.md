# Parse/tokenize perf — the bottleneck is the arbiter's per-claim fallback decode

**Status:** Scoping complete (2026-06-03), implementation pending. Supersedes the
register_state-memo guess in [`parse_decode_walker_profile.md`](parse_decode_walker_profile.md):
that memo is **dead in the dominant path** (1% hit). Profiled + instrumented on fogbank,
codebook config, song = DRAX `Advanced.1` (1 rotation), **with the parse sidecar present**.

## Definitive measurement
`parser_worker` (the tokenize per-song unit), sidecar present so **no re-parse**:
- **355 FrameWalker.walk passes for one song**; `walk` cumtime = 22.6s of 40.6s (55%).
- Caller breakdown (instrumented): **333/355 = `arbiter.py:_decoded_state`** — the arbiter's
  `validate=True` decode.
- Of those: **9 `arbitrate` calls, 327 `_lossless`, ~318 of them the per-claim fallback** (below).
- register_state memo hit rate in this path: **3 hits / 342 misses (≈1%)** — even keyed on
  content-only (no index). The decodes are genuinely distinct content, so the memo can't help.

## Root cause — the arbiter's per-claim fallback
Instrumented on the same song: **9 `arbitrate` calls, 336 `_decoded_state`, 327 `_lossless`** — of
which only ~9 are the main lossless check and **~318 are the per-claim fallback**. The arbiter:
```python
out = _apply(selected); src = _decoded_state(df)
if _lossless(src, out): return out          # the fast path — 1 decode
for claim in selected:                       # FALLBACK when the batch isn't lossless:
    if _lossless(src, _apply(accepted + [claim])):   # re-apply + FULL re-decode PER CLAIM
        accepted.append(claim)
```
So when a batch of ~35 claims isn't lossless together (claims interact via the **ctrl/patch
tick-drain** the validation exists to catch), it degrades to **O(claims) full-df decodes** — 318 of
them here. That's `O(Σ claims × tune_length)`, the term that makes **prodlike → hours**. (Block
materialization re-running passes per overlapping block multiplies the *number* of such episodes, but
the cost *inside* each is the fallback.)

Two earlier wrong turns (recorded so they're not repeated): (1) the *re-parse* hypothesis — the
runner reuses parse.py's `.[0-9]*.parquet` sidecars (verified), so the parse decode is NOT redone;
(2) the *memo* (#5) — premised on pass N `out` = pass N+1 `src`, which holds in the parse.py path
(~6.6% there) but **not** in block materialization (transforms between passes → distinct content →
1% hit).

## Ranked fixes
1. **Incremental per-claim fallback validation (highest leverage, attacks the 318 directly).** The
   fallback's invariant: `src` is fixed; we greedily grow `accepted` and re-decode `_apply(accepted +
   [claim])` from scratch each iteration. Two compounding wins, both byte-exact:
   - **Suffix decode from the first changed frame.** Claims are non-overlapping writes at known frames.
     `accepted + [claim]` differs from the prior iteration only at/after `claim`'s frame; snapshot the
     walker's decoder state at each accept and resume the decode from `claim`'s first frame instead of
     frame 0. Turns each O(tune) decode into O(tune − claim_offset).
   - **Window comparison.** A claim only needs to verify lossless *within its own frames plus the
     tick-drain spill* (the bounded interaction the fallback exists to catch); compare `src` vs decoded
     only over that window, not the whole song. Turns the comparison from O(tune) to O(claim_span).
   Combined: the fallback drops from `O(Σ claims × tune)` to ~`O(Σ claim_span)`. Risk: validation core,
   silent-acceptance class — gate hard on the corpus `cb_div_audit` (a wrong-accept surfaces there as a
   register_state divergence) and keep a full-decode fallback path behind a flag for A/B.
2. **Eliminate per-block codebook re-encoding (orthogonal multiplier).** Block materialization
   `expand_to_literal`s and re-runs the codebook pipeline per overlapping block, multiplying the
   *number* of fallback episodes. Re-localizing the sidecar's already-validated atoms into blocks
   (slice + fix cross-boundary DEF/REF + drains) instead of re-encoding removes that multiplier.
   Bigger redesign; do after (1) if still needed.
3. **De-overlap the alphabet build.** `make_tokens` needs only the *set* of block atoms; overlapping
   windows add no new atoms but re-encode ~2×. Verify the alphabet still matches training atoms.
4. **Compile the walker** (numba/Cython SET/DIFF fast-path). The floor cost; biggest effort.

## Recommendation
Implement **(1)** — it targets the measured 318 fallback decodes directly, is the smallest change with
the largest measured payoff, and is corpus-gatable. Start with the suffix-decode half (lower risk than
window-comparison, already a large win), measure on `cb_div_audit`, then add window comparison.
The shipped memo (#5) should be **retired or scoped to the parse.py path** — it's pure overhead here.
