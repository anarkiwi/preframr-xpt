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

## Feasibility (checked against the walker, 2026-06-03)
Both halves of (1) are **core-walker changes, not self-contained arbiter edits**:
- *Suffix-decode* needs the walker to snapshot/restore its **full** decode state at an arbitrary frame.
  `register_state` only keeps the 25-reg per-frame `snaps`; the resumable state also holds codebook
  tables, loop counters, and the **per-frame tick-drain queue**. Preserves exact output, but touches the
  byte-exact decode core.
- *Diff-attribution* (decode the batch once, re-test only claims near a divergence) is **unsound here**:
  the tick-drain spills an **unbounded** number of frames, so "near a divergence" has no small bound —
  which is the very reason the arbiter validates cumulatively. A clever neighborhood shortcut is exactly
  the silent-corruption class to avoid in this core.

So (1) is the right target but is a deliberate, gated change to the decode core (add resumable
decode-state snapshot/restore to the walker; suffix-decode each fallback candidate from the claim's
first frame). Risk is real; gate on `cb_div_audit` byte-exactness + a slow-path equivalence assertion
under an env flag across the corpus before trusting the fast path.

## Empirical results (2026-06-03, 30 HVSC tunes on fogbank)
Implemented + measured before committing to the core change. Two findings reshape the fix:

1. **Resume primitive works, byte-exact.** `preframr_tokens/macros/resume_decode.py` snapshots the full
   `DecodeState` at a frame boundary and resumes the walk there; validated **240/240** identical to the
   full walk across 40 real (loop-expanded, 2048-frame) dfs at 6 split points each. Frame mapping is
   trivial at arbitrate time (no loop ops present; raw frame count == expanded). This is sound infra.

2. **But the fallback is accept-heavy and concentrated in a FEW huge calls — which defeats both the
   suffix-resume *and* the diff-attribution forms of (1):**
   - Of the per-claim fallback tests: **317 accept / 10 reject**. Greedy *grows* `accepted` on nearly
     every test, so `cur_df` changes constantly — pure suffix-resume from a stable snapshot can't reuse a
     prefix; it would have to re-snapshot after almost every claim.
   - 35 fallbacks over 30 tunes did **8315 greedy decodes** — dominated by a handful of calls with
     `nsel` = 322/557/586/610. One pass proposes **hundreds** of claims that conflict catastrophically as
     a batch (the all-applied decode diverges across ~1800/2048 frames) yet are mostly individually
     compatible (greedy keeps 75/355/395/445). The cost is `O(nsel)` full decodes per big call.
   - **Diff-attribution is unsound here, confirmed: 19/35 mismatch.** When the batch diverges almost
     everywhere, "claims touching a divergence" flags nearly all of them (`nbad`≈`nsel`, fast keeps 0–179
     where greedy keeps 75–445). Batch attribution cannot predict the large lossless subset greedy finds
     incrementally. (`fast_decodes` was 103 vs 8315 — 99% cheaper — but **wrong**, so moot.)

   The only *exact* speedup left is localized suffix-resume that maintains per-frame `DecodeState`
   snapshots and updates only the changed window [claim_frame, reconvergence] per accept. That needs a
   deep copy of `DecodeState` per frame (or per accept), whose overhead is **unproven** and may eat the
   gain on the big `nsel` calls. High complexity + risk in the byte-exact core.

## Culprit pinned: `ctrl_bigram` + `ctrl_triple` (`collapse_runs`)
Per-pass-label breakdown of the 8315 greedy decodes (30 tunes): **`ctrl_triple` 4038 + `ctrl_bigram`
4234 = 99.5%**; `sweep` = 8; everything else 0. `nsel` distribution: 610, 593, 586, 571, 557, 545, …

`collapse_runs` (`run_collapse.py`) builds **one Claim per collapsible CTRL run across all three voice
registers** and hands the whole pile (hundreds) to a single `arbitrate(validate=True)`. The claims
interact only through the **per-register tick-drain**, which is *local*: a collapse's effect is confined
to its run's frames + a short drain; runs far apart in frames, or on different voice ctrl registers,
provably cannot interact (different `pending_set_writes[reg]`; the splice preserves FRAME markers). So
the one giant `O(nsel)` greedy is unnecessary — the claim set is **separable into independent groups**.

## HANDOFF (2026-06-03)
Delivered ~18–20% (B + partition), both **byte-exact verified** (full tokens suite 876 pass / 1 fail,
the 1 being the pre-existing `test_sid_frame_diff` release-gate that fails on clean HEAD too — unrelated,
it's the unmodelled mechanism-B vibrato residual vs an over-eagerly emptied `_KNOWN_FREQ_LOSSY`). The
suite ran with `PREFRAMR_VERIFY_PARTITION=1` throughout and never raised. **Changes are UNCOMMITTED** in
the preframr-tokens working tree: `state.py` (B), `arbiter.py` + `run_collapse.py` (partition),
`resume_decode.py` (new, currently UNUSED infra). Ready to commit; not committed (no request).

**Honest self-assessment:** this stopped at the two cheaper levers and did NOT tackle the real
structural cost. The dominant time is the per-row Python decode walk re-run per overlapping block by
`run_block_refire_passes` (440s / 86%). The high-impact work still open: (1) **windowed per-claim
validation** via `resume_decode.py` (helps the dense-ctrl tunes the partition can't), (2) **per-block
refire redesign** — re-localize parse-stage atoms instead of re-encoding from literal per block — and/or
(3) a **compiled/vectorized walker** for the `_dispatch_row`/`_walk_frame` floor. These are where the
next real gains are; they were deferred as higher-risk byte-exact-core changes, not because they're
lower-value. A lot of session time also went to mis-instrumented probes (digi contamination, pipe
block-buffering, thread oversubscription, `tail -3` truncation) before the numbers were trustworthy —
see [[sweep-tool-hygiene]].

## IMPLEMENTED + MEASURED (2026-06-03, clean cProfile, 27 digi-excluded tunes)
Reprofiled from scratch (digis excluded — the earlier "99.5% ctrl / 8315 decodes" was digi-inflated;
the honest digi-free figure is ~80% arbitrate, 1068 register_state decodes, 666s cumulative).

**B — vectorize `_build_last_diff` (`state.py`):** replaced the per-register full-frame mask scan with
one groupby. **10.8× on that function, byte-exact (0 mismatches / 60 dfs), ~12% of total, zero risk.**
It had been ~40% of the per-decode *setup* (89s `_build_last_diff` + 84s `remove_voice_reg`); B removes
the `_build_last_diff` share. SHIP.

**Partition — `collapse_runs` independent groups (`arbiter.arbitrate_independent_groups`):** split the
single hundreds-of-claims `arbitrate` into per-register, frame-gap(>8)-separated groups validated
against one shared `src` decode. **Byte-exact-equivalent to flat greedy (proven: `PREFRAMR_VERIFY_PARTITION`
guard, no mismatch corpus-wide).** Decode count 1068→900 (**16% fewer**), but UNEVEN: Coop 269→102 (big
win, clustered ctrl) vs Bass_Guitar 470→499 (slight regression — dense ctrl with no frame gaps → ~3 big
reg-groups + per-group batch-check overhead). Net end-to-end ~5%.

**Combined B+partition: 666s → 543s (~18% end-to-end).** The remaining floor is intrinsic: the per-row
Python decode walk (`_walk_frame` 292s + `_dispatch_row` 185s) driven by `run_block_refire_passes`
(440s — the per-block re-encode wrapper) re-running the pipeline per overlapping block.

**Not yet done (the real remaining levers):**
- *Windowed per-claim validation* (use `resume_decode.py` to decode only a claim/group's frame window
  instead of the whole df) — would help the DENSE-ctrl tunes the partition can't (Bass_Guitar), since
  groups bound the window. Complex byte-exact-core change.
- *Per-block refire redesign* (C-proper, the 86%/440s wrapper) — re-localize parse-stage atoms instead
  of re-encoding from literal per overlapping block. Biggest lever, biggest risk.

## Earlier recommendation: partition `collapse_runs` into independent groups
Split the single arbitrate into independent groups (per ctrl register, then by frame-gap > a drain
margin) and validate each group **against the one `src` decode** — independence means accepting one
group never invalidates another, so there is no re-snapshot problem (the thing that defeated localized
validation in the general case). Group validation can use `resume_decode.py` to decode only the group's
window. Net: `O(nsel)` full decodes → `O(#groups)` small decodes, no arbiter/walker core change.
This is **corpus-testable for exact equivalence** to the current single-arbitrate accept set (the guard
that makes it safe). Margin sweep + equivalence being measured now.

`resume_decode.py` is committed infra for the group-window decode. Retire or scope the register_state
memo (#5) to the parse.py path — it's pure overhead in block materialization. Route 2 (kill per-block
re-encoding) remains the orthogonal multiplier if more is needed after this.
