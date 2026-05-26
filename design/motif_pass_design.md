# Corpus-mined motif pass

**Status:** Shipped in preframr-tokens **0.20.0** (per-block, lossless,
dry-run validated). Framework support (`--motif-pass`/`--motif-dict`, mine CLI)
merged to preframr main → image `anarkiwi/preframr:0.2.2`. **Compression is real
at deployment scale (~11.4% fewer tokens, measured — see Findings)**; whether it
also helps the model is the open question (the A/B's per_class content audit).
OFF by default.

## Problem / motivation

After four content-head architectures were refuted (see
`compound_token_design.md`), the leverage moved to the representation axis. The
question that motivated this pass: *can higher-level tokens that aggregate
recurring multi-atom idioms — above the per-frame FREQ_TRAJ macros — buy either
sequence brevity or better generalization?* A "motif" is a recurring atom
sequence that recurs **across composers** (a shared idiom, not a per-composer
signature), replaced losslessly by one `MOTIF_OP` token.

## Algorithm

`mine_motifs(streams, composers, k, min_count, min_composers)` — a greedy
most-frequent-adjacent-pair merge (i.e. **BPE** on atoms) with two guards:

1. **Boundary guard:** a motif may not *end* on a frame-advance atom
   (`reg == FRAME_REG`) — keeps motifs from straddling a frame boundary on the
   right.
2. **Cross-composer floor:** a pair must occur in `>= min_composers` distinct
   composers before it can merge — biases the dictionary toward *shared idioms*
   rather than per-composer memorization.

Atoms are 5-tuples `(op, reg, subreg, val, diff)` — `diff` (timing) is included
so a merge is timing-exact and `expand` round-trips byte-exact. `MotifDict`
serialises the ordered merges + per-motif expansions to JSON.

## Architecture — the pass runs PER VOICED BLOCK, not at parse-end

This is the load-bearing design decision, established empirically by a
full-tokenize dry-run.

**Parse-end (first attempt) is unsound.** Collapsing motifs at the end of
`RegLogParser.parse` crashes the tokenizer downstream: the frame-based block
slicer + self-containment / back-ref-literalisation machinery
(`iter_self_contained_row_blocks`, `expand_to_literal_form`, the per-block
re-parse) is **not motif-aware**. It rejects `MOTIF_OP` atoms ("unknown op 52")
and underflows on the frame markers a motif swallows ("BACK_REF target frame -2
reaches before output start"). Every motif song gets dropped.

**Per-block (shipped) is correct.** `MotifPass` is applied inside
`iter_voiced_blocks`, to each self-contained voiced block **after** slicing /
self-containment / back-ref literalisation and **before** `merge_token_df` /
Unigram encode. So the frame machinery never sees motif atoms, and both the
vocab build and the encode consume the same collapsed blocks (consistent
alphabet). This is also the original "between the macros and Unigram" intent:
Unigram encodes blocks, so motif-before-Unigram = motif-on-blocks.

**Mining must match.** `mine_dict_from_dumps` mines over the SAME voiced blocks
the encode path collapses (each block one stream tagged by composer), so the
mined merges actually apply at encode time. Mining over the raw parse stream
would not match (voice-reg folding + back-ref literalisation change the atoms).

`MotifTransform` (`LOSS_TIER="zero"`, `DECODES_VIA_DF=True`) carries the
inverse for decode-time expansion.

## Findings — compression (READ THIS BEFORE QUOTING A NUMBER)

The compression story is **regime-dependent**, and conflating the layers/regimes
produced two contradictory numbers earlier:

- **~24% atom-level.** The motif miner absorbs ~24% of atoms into motifs on
  prodlike. This measures *redundancy that exists in the atom stream* — NOT
  additional compression over Unigram.
- **~0.6% post-Unigram (mini dry-run, NOT representative).** On 24 songs at
  vocab 8192, motif-then-Unigram gave only 0.6% fewer tokens than Unigram alone
  — because that regime (tiny corpus, vocab 8192 ≫ alphabet 2841) let Unigram
  *over-merge* and absorb the motif redundancy. This was further muddied by
  seq_len truncation of the block-token count.
- **Deployment regime is different.** The orin doc measured **1.23 atoms/token**
  on prodlike (vocab 32768) — i.e. at deployment scale **Unigram barely merges**
  (the FREQ_TRAJ-class macros do the compressing). Where Unigram is not already
  collapsing atom runs, motifs are NOT redundant, so the marginal compression at
  deployment scale should be **well above 0.6% — plausibly approaching the
  atom-level figure.**

**Measured (deployment regime, this design):** 6 composers / 146 voiced blocks,
**vocab 8192** (the deployment config), k=256, mc=3. Baseline **1.173
atoms/token** — confirms the deployment light-merge regime (≈ the orin doc's
1.23; NOT the over-merged dry-run). With motifs: atom-level collapse **23.1%**
(2.14M → 1.64M atoms) and, crucially, **11.4% fewer UNTRUNCATED encoded tokens**
(1.82M → 1.61M, 1.173 → 1.019 atoms/token). So the real deployment compression
is **~11.4%** — an order of magnitude above the 0.6% small-corpus dry-run
artifact, ~half the 23% atom-level ceiling (Unigram's light merging recovers the
rest). **The compression case holds at deployment scale**; the dry-run's
"redundant with Unigram" reading was an artifact of an over-provisioned vocab on
a tiny corpus.

**Lesson (recorded so it isn't re-derived):** atom-level redundancy ≠ post-
tokenizer compression, and post-tokenizer compression is a function of the
Unigram vocab budget vs corpus diversity. Always measure the **untruncated
encoded-tokens delta in the deployment vocab regime** (cf. the BPE/Unigram
layer-conflation that recurred several times during this work).

## Surviving justification (unproven)

Even if compression is real, the *distinct* thing motifs do that Unigram does
not is the **cross-composer constraint**: Unigram forms whatever pieces maximise
corpus likelihood (can be composer-specific / overfit); motifs are forced to be
idioms shared across `>= min_composers` composers. So the motif vocab is a
**generalization-biased chunking**. Whether that helps the model is the open
question the A/B answers — *not* compression.

## A/B plan

`motif_mini_body_large` (preframr-xpt `feat/motif-spec`): both arms run
`full_macros`; target adds `--motif-pass` with a dict mined by a `pre_run_hook`.
Decisive read: **per_class content-tier val_acc** (motif tokens are loss-tier
zero, so content is measured on the un-collapsed atoms) + **loop_collapse /
prompt-conditioning** (generalization). `encoded_tokens_per_song` reports the
real deployment compression. Needs `PREFRAMR_DATASET_CACHE_DISABLE=1` and image
`anarkiwi/preframr:0.2.2`. GPU frees after STAGE 2.

## Open risks

- **Memorization.** Motifs absorb content (melodic) atoms ~46% vs structural
  ~30%, and a stricter `min_composers` floor does **not** shift this (content
  absorption is ~45% floor-invariant at mc=3/6/12 — it just absorbs less
  overall). So the cross-composer floor cannot tune away melodic-memorization
  risk; a longer horizon of memorized figures is the failure mode to watch in
  loop_collapse.
- **Learnability of denser tokens.** Each motif token packs more information →
  harder per-token prediction. The content audit tests whether the model learns
  them.

## Cross-references

- `orin_inference_optimization_design.md` — the 1.23 atoms/token deployment
  measurement that this pass's compression claim must be reconciled against.
- `compound_token_design.md` — a *different* compression mechanism (parallel
  attribute heads, not dictionary merging); its hypothesis #1 ("sequence-length
  is binding") shares this pass's family, and its Phase-0 audit correctly
  commits to *measuring* sequence compression rather than assuming it.
