# Corpus-mined motif pass

**Status:** **REFUTED 2026-05-27** — the 3-arm content-tier A/B settled the open
question: motif (v1 exact + v2 templated) is neutral-to-negative vs no-motif
full_macros (v2 content 0.036 vs baseline 0.045), and gives no compression. See
`data/refuted/motif_pass.md`. History below.
Shipped in preframr-tokens **0.20.0** (per-block, lossless,
dry-run validated). Framework support (`--motif-pass`/`--motif-dict`, mine CLI)
merged to preframr main → image `anarkiwi/preframr:0.2.2`. **Compression is real
at deployment scale (~11.4% fewer tokens, measured — see Findings)**; whether it
also helps the model is the open question (the A/B's per_class content audit).
OFF by default. **A 2026-05-27 dict+corpus analysis found exact (shape,value)
mining leaves ~54% of motif-shape instances uncaptured (value-shift
fragmentation) → motivates a value-slotted v2 (see Findings + Proposed fix).**

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

## Findings — exact (shape,value) mining leaves the majority uncaptured (2026-05-27)

Mining is exact, ordered, **timing-exact** (`diff` in the key), with **no
canonicalization** — so a motif is a byte-identical contiguous sequence, and any
value-shift (transposition), reorder, voice-swap (`reg` change) or retiming is a
*distinct* sequence. Measured on the live mini dict (256 motifs) + a CPU corpus
scan whose parse matches the dict exactly (104 blocks / 6 composers;
`/scratch/tmp/motif_tail_scan.py`):

- **The 256 motifs collapse to 10 transposition-invariant shapes** (7 len-2 + 3
  len-3): 255/256 differ from a sibling only in `val`; the largest shape was
  mined as **73 separate value-instances**. 64 motifs (32 clusters) are pure
  reorderings.
- **Of all corpus windows matching those 10 shapes, only 45.7% are in-dict;
  54.3% (714,741 occ) are out-of-dict** value-variants. **6,260 distinct
  out-of-dict (shape,value) variants** vs 256 mined. **292 of them (153,639 occ)
  clear the dict's own ≥3-occ/≥6-composer floor** — dropped only by `k=256`.
  The single most frequent escapee occurs **6,780×** across **5** composers —
  dropped only by `min_composers=6`. *(Overlapping windows inflate absolute
  counts; the 46/54 split + the 6,260-vs-256 variant count are robust.)*

So exact-match captures **less than half** of its own shapes' instances. Two
consequences: (1) compression is left on the table (the value axis escapes); (2)
the same gesture is tokenized as a motif token in 46% of cases and as raw atoms
in 54% — **representational fragmentation**, the opposite of a unifying vocab,
and a concrete mechanism for harmed learnability (consistent with the motif
arm's low, flat all-tier val_acc; the A/B's content-tier audit is the real test).
Raising `k` / lowering `min_composers` does not fix this — it just mines more
per-value motifs (vocab grows, fragmentation persists).

## Proposed fix — value-slotted motif templates (MotifDict v2)

Capture the value-shift family with **one template + a value parameter**,
losslessly. A motif becomes a *compound*: structural template + content slot(s).

1. **Template = structural shape.** Key on the `(op, reg, subreg, diff)` sequence,
   not the absolute `val`s. Val positions that are constant across the template's
   occurrences are baked in; positions that vary become **value slots** (one per
   varying position — handles motifs mixing register families with independent
   value axes).
2. **Mine on templates.** `min_count` / `min_composers` apply to the *template*
   (pooled over all its value-instances), so the transposition family qualifies as
   one idiom — e.g. the 6,780× / 5-composer escapee is captured once its shape's
   values pool across ≥6 composers.
3. **Encode.** A window matching a template → `MOTIF_OP[template_id]` + the slot
   value(s) emitted as ordinary value tokens (**reuse the existing `val`
   vocabulary** → no vocab explosion; ~10 templates replace 6,260 instances).
4. **Expand (lossless).** template constants + slot values → exact atoms; keep the
   per-frame oracle + `compare_renders` green (non-negotiable — the pass is
   lossless).
5. **Model view.** Predict the template (structural idiom) **then** its slot
   value(s) (content) — a structure/content factorization that **aligns with the
   loss-tier split** and is exactly `compound_token_design`'s template+attribute
   idea scoped to motifs (CP-Words-style attribute grouping). This is the crux:
   structure and content are predicted *separately* instead of fused into one
   fragmented token.

**Why it fixes both:** compression — collapses 6,260 variants → ~10 templates and
absorbs the 54% tail (well above v1's 11.4%, with a *smaller* motif vocab);
learnability — the same gesture is always the same template token (+ a value
slot), removing the motif-vs-raw fragmentation; the model learns the shared idiom
once and predicts value as content (which it must do regardless).

**What it does NOT solve / caveats.** Templating removes *structural*
fragmentation only — the model still predicts the slot value (content), so expect
gains on sequence/consistency, not a content-acc miracle. A value-shift is
different pitch/content, so slots are **not** free to merge — keep them exact
(lossless); optionally layer `audio_equivalence_normalization` to *quantize* slot
values (lossy, content-tier) as an orthogonal knob. Per-template slot-value
entropy should be monitored: a template whose slot distribution is huge/multimodal
is a weak idiom (conflates distinct figures) and should not merge.

**Validation gates (mirror v1):** byte-exact round-trip (oracle + compare_renders);
untruncated encoded-tokens delta in the deployment vocab regime (target: most of
the 54% tail); template vocab ≪ per-value count (~tens); per_class content-tier
val_acc ≥ baseline + loop/prompt not worse. **Mechanics:** re-mine on shape keys
(cleaner than a post-pass clustering of the exact v1 dict). Could land as a
`MotifDict` v2 in preframr-tokens behind a flag, or fold into the compound-token
tokenizer.

**Implementation design: `motif_templates_v2_impl_design.md`** — data model,
shape-keyed mining, lossless expand, tier wiring, phased work order. It also
**corrects the compression framing above**: separate template+slot tokens are
compression-neutral-to-worse on the len-2-dominated dict; the primary win is
*consistency* (vocab 10 vs 6260) + exposing motif-carried content to the content
tier. Compression is a measured gate, not a premise.

## Cross-references

- `orin_inference_optimization_design.md` — the 1.23 atoms/token deployment
  measurement that this pass's compression claim must be reconciled against.
- `compound_token_design.md` — a *different* compression mechanism (parallel
  attribute heads, not dictionary merging); its hypothesis #1 ("sequence-length
  is binding") shares this pass's family, and its Phase-0 audit correctly
  commits to *measuring* sequence compression rather than assuming it.
