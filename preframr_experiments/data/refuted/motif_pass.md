# `motif_pass` (corpus-mined motif vocab, v1 exact + v2 templated) — REFUTED 2026-05-27

**Hypothesis:** an explicit cross-composer-constrained motif vocabulary (collapse
recurring atom runs into dedicated tokens) gives the model a better content/melodic
representation than Unigram's likelihood-greedy chunking — a learnability win, not a
compression win. Two variants: **v1** exact-match BPE over `(op,reg,subreg,val,diff)`
atoms (motif tokens are loss-tier zero); **v2** value-slotted shape templates
(`MOTIF_OP` shape + `MOTIF_ARG` content slot, the slot tiered as content). Design:
`design/refuted/motif_templates_v2_impl_design.md` (v1 design since removed).

**Decided by the content-tier per_class audit** (the decisive gate; all-tier is
confounded because each arm tokenizes differently). v2 audited in xpt `:0.2.4`
(MOTIF_ARG classified as content). Mini, body=large, 60 ep, 3 seeds, full_macros base.

## Why refuted

Content-tier val_acc (subset `eval`/`eval_a`), seed-stable:

| arm | content acc | content n | enc/song |
|---|---|---|---|
| **no-motif baseline** (full_macros) | **0.0448 ± 0.011** | ~125k | 7561 |
| v2 templated (`--motif-mine-version 2`) | **0.0364 ± 0.001** | ~104k | 7586 |
| v1 exact | 0.0189 ± 0.002 | ~93k | 7554 |

- **v2 − no-motif baseline = −0.008** (v2 sits at the *bottom* of the baseline's
  per-seed range 0.036–0.058; never exceeds it). **Fails the decisive gate.**
- **v2 − v1 = +0.018** → the v2 fixes worked *as designed*: de-fragmentation +
  exposing the slot as content recovered most of v1's −0.026 regression (content n
  93k→104k, i.e. v2 stops "hiding" content in zero-tier `MOTIF_OP`). But that only
  claws back to ≈baseline; it does not beat plain full_macros.
- **No compression either**: v2 mined just 61 templates (min_composers=6 floor),
  enc/song 7586 ≈ baseline 7561; v1's 256 motifs gave ≈0 net either. So neither a
  content lift nor a token saving.

Comparability: v2 from a fresh `:0.2.4` run; baseline + v1 from the prior `:0.2.2`
`motif_mini_body_large` A/B. Spec config is identical (transforms, full_macros cargs,
tkvocab 32768, seq_len 4096, body=large, 60 ep, 3 seeds); the `:0.2.2`→`:0.2.4` image
delta is motif-only code that does not touch the non-motif tokenizer or the
content-tier definition, so the no-motif baseline reproduces (alpha 4255, enc 7561).
Since v2 is *below* baseline, an in-run `:0.2.4` baseline would only widen the gap.

The loop_collapse / prompt-conditioning guards were not run: the decision rule gates
them on a content win, and there was none.

## Known regression (do not lose) — v1 miner spin in tokens 0.23.0

The 0.23.0 frame-guard fix (a motif may never *contain* a FRAME_REG atom) regressed
`mine_motifs` (the v1 exact-match BPE) into an O(k²·N) blowup — a ~2.4 h "hang" at
101% CPU on the mini corpus (not an infinite loop). Root cause: the guard
`has_frame(a) or has_frame(b)` makes frame-advance atoms **permanently unmergeable**;
they are ~20–28% of all adjacent pairs and among the most frequent, so they pile up
at the top of `most_common()` forever. As eligible high-count pairs are consumed, each
of the k iterations linearly scans deeper past the frozen frame wall (scan_depth
3→141+ over 80 iters) and runs an O(N≈2 M) `_ncomposers` re-scan per non-frame
candidate; frame walls also stop run-collapse so N barely shrinks. The old `:0.2.2`
guard (`ends_frame_advance(sym_b)`) let frames be absorbed mid-motif, so N collapsed
fast and 256 merges finished in seconds. Repro: `/scratch/tmp/motif_v1_hang_repro.py`.

**Fix direction if ever revisited** (both cheap, restore near-linear): (a) skip
frame-touching pairs during `cnt.update` so the wall never enters `most_common()`;
(b) accumulate per-pair composer sets in the same counting pass to drop the O(N)
`_ncomposers` re-scans. NOT shipped — the motif pass is refuted, so the spin was only
documented, not fixed.

## Do not revisit without

- A content-tier result (or strong prior) that a motif/chunked vocab beats Unigram on
  the **content tier** at scale — not all-tier val_acc, not compression.
- And the `mine_motifs` perf fix above (or use only the v2 templated miner, which is
  frame-filtered up front and unaffected).

Specs kept: `preframr_experiments/specs/motif_{mini,v2_mini}_body_large.py`.
Parsers: `/scratch/tmp/parse_motif.py`, `/scratch/tmp/compare_motif_v2_vs_prior.py`.
