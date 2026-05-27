# MotifDict v2 — value-slotted motif templates (implementation design)

**Status:** **REFUTED 2026-05-27** — built (tokens 0.21.0–0.23.0) and A/B'd. v2
de-fragmented + content-exposed the slot and recovered most of v1's regression, but
content-tier did not beat no-motif full_macros (v2 0.036 vs baseline 0.045). See
`data/refuted/motif_pass.md`. Was: Drafted impl design. Implements the "Proposed fix" in
`motif_pass_design.md`; tokenizer-side (preframr-tokens) + an xpt A/B. **No model
change in Phases 0–2** (the win is tested with the existing single-head model);
the field-factorized variant is Phase 3 = `compound_token_design.md`.

## Goal + honest scope (corrects the proposal's compression framing)

The v1 finding: exact `(shape,value)` mining → 256 motifs over **10 shapes**, and
**54% of shape-matching corpus windows are out-of-dict** value-shifts. The damage
is **representational fragmentation** (the same idiom is motif-token #17 at one
value, #24 at another, raw atoms for the rare tail) — a learnability-harm
mechanism. v2's **primary win is consistency**: one template per shape, value as a
separate slot, so the idiom is always the same token.

**Compression is secondary and must be *measured*, not assumed.** A v1 len-2 motif
is 2 atoms → 1 token. A v2 template+slot is 2 atoms → 2 tokens (template + slot),
so on the len-2-dominated dict (7 of 10 shapes) v2 is roughly **compression-neutral
to slightly worse** at the atom level; it compresses only on len-3+ motifs and via
Unigram re-merging frequent `(template,slot)` pairs. (The proposal's "compression
≫ 11.4%" was over-optimistic — it holds at the *vocab* level, 10 templates vs 6260
variants, not the token-count level.) **Validation gate, not a premise.**

## Current code (grounding)

- `preframr_tokens/macros/motif_pass.py`: `MotifDict(merges, expansions)`,
  `encode`/`expand`, `MotifPass`/`MotifTransform` (`LOSS_TIER="zero"`,
  `DECODES_VIA_DF`). A motif → one atom `_motif_atom(mid) = (MOTIF_OP,0,0,mid,0)`
  (id in `val`); `expand` looks up `expansions[a[3]]`. `MOTIF_OP=52`.
- `preframr_tokens/motif_mine.py`: `mine_motifs` greedy BPE on exact symbols
  (atom-5-tuple or motif-id), boundary guard (`sym_b` not `FRAME_REG`) +
  cross-composer floor. Per-block via `iter_voiced_blocks`.

## Key insight — v2 *exposes* motif-carried content to the content tier

In v1 the motif's atoms (incl. melodic/content vals) are baked into a
**loss-tier-zero** token, so the ~46% content absorbed into motifs is **invisible
to the content-tier audit** (and the model isn't trained to predict it). v2 emits
the value as a **content-tier slot token** → the content tier now properly
accounts for motif-carried content and the model must predict it. This both makes
the A/B's content gate honest and is the mechanism by which v2 can help.

## Data model (v2)

```
template_id -> { "shape": [(op,reg,subreg,diff), ...],   # structure, no val
                 "consts": {pos: val, ...},               # always-constant vals (baked)
                 "slots": [pos, ...] }                     # positions whose val varies
```
Encode emits, per matched window:
- `(MOTIF_OP, 0, 0, template_id, 0)` — the template token (≈10 distinct → small).
- `(MOTIF_ARG, 0, 0, slot_val, 0)` for each slot, in `slots` order — content tokens
  drawn from the existing val space (shared across templates; no per-(shape,value)
  blowup).

`expand`: read `MOTIF_OP[template_id]`, consume `len(slots)` following `MOTIF_ARG`
atoms, fill `shape`+`consts`+slots → exact `(op,reg,subreg,val,diff)` sequence.
**Byte-exact** (the round-trip gate). JSON gains a `"version": 2` discriminator;
`from_json` dispatches v1/v2.

`MOTIF_ARG` is a new `stfconstants` op. Why separate atoms, not one packed atom:
the Unigram alphabet keys on the full 5-tuple, so packing `(template_id, slot_val)`
into `(subreg, val)` re-fragments the alphabet on `(template,value)` — defeating
the point. Separate tokens give a ~10-entry template alphabet + the value alphabet
(which already existed as content), and the model sees the template consistently.

## Mining (v2) — shape-keyed

**Phase-1 approach (recommended): two-pass over the existing BPE.**
1. Run v1 `mine_motifs` exact BPE (tested) to get candidate sequences — but with
   `min_composers`/`min_count` **relaxed** (we re-floor on the template).
2. Cluster the candidates by `shape = (op,reg,subreg,diff)` sequence.
3. Per shape, scan the corpus streams: a val position is a **slot** if it takes ≥2
   distinct values across occurrences, else a **const**. Apply the cross-composer
   floor + `min_count` to the **template** (pooled over all its value-instances) —
   this is what captures the transposition family (e.g. the 6,780×/5-composer
   escapee qualifies once its shape pools ≥6 composers).
4. Cap by `k` templates ranked by pooled occurrence.

(Phase-2 alternative: native shape-BPE — redefine the merge key + `Counter` to
operate on shape-symbols with slot positions wildcarded. Cleaner, more invasive;
do only if the two-pass slot inference proves inadequate.)

## Integration (encode/decode/tiers)

- `MotifPass` runs unchanged **per voiced block** (the load-bearing v1 fix);
  window-match is by template shape, not exact atom-seq. Boundary guard holds:
  neither the template token nor a slot may straddle a frame-advance — keep the
  "no motif ends on `FRAME_REG`" rule and ensure slots stay within the block.
- **Tier map** (`build_tier_map`, used by the per_class audit + training loss):
  `MOTIF_OP` (template) stays **loss-tier zero** (pure structure); `MOTIF_ARG`
  (slot) is classified **content** so it's trained + audited as content. This is
  the only "wiring" change beyond the tokenizer.
- `MotifTransform.expand` consumes the trailing slot atoms; decode stays
  `DECODES_VIA_DF`.

## Phasing

- **P0 — tokenizer:** `MotifDict` v2 (templates, `MOTIF_ARG`, v2 encode/expand) +
  two-pass mining, behind the JSON `version` discriminator. Gate: round-trip
  byte-exact on the fidelity oracle + `compare_renders`; v1 still round-trips.
- **P1 — measure (CPU, no GPU):** re-mine the staged corpus; report template count
  (~tens vs 256), the **untruncated encoded-token delta** in the deployment vocab
  regime (do NOT assume a win), and content-tier coverage (how much
  previously-hidden motif content is now content-tier).
- **P2 — A/B (`motif_v2_mini_body_large`):** 3-arm — `full_macros` (no motif) /
  `full_macros`+v1 / `full_macros`+v2 — mini body=large, `PREFRAMR_DATASET_CACHE_DISABLE=1`,
  image `anarkiwi/preframr:0.2.x` (rebake with v2). **Decisive gate = per_class
  content-tier val_acc + loop_collapse/prompt-conditioning**, NOT all-tier. Pass:
  v2 content ≥ no-motif (and ideally > v1) with loop/prompt not worse.
- **P3 (deferred) — field-factorized model:** embed/predict `(op|template|slot)`
  field-wise so a *single* atom carries template+value (recovers atom-level
  compression + consistency). This is `compound_token_design`; land only if P2
  shows consistency helps but the token-count cost of separate slots bites.

## Validation gates

1. Byte-exact round-trip (oracle + `compare_renders`) — non-negotiable (lossless).
2. v1 dicts still load/round-trip (version dispatch).
3. Template count ≪ per-value (~tens); slot-value entropy per template reported
   (a huge/multimodal slot distribution = weak idiom; flag, don't merge).
4. Encoded-token delta measured in the deployment vocab regime.
5. P2 content-tier gate + loop/prompt.

## Risks / unknowns

- **Unigram re-merge.** Unigram may merge frequent `(template,slot)` into one
  super-token — fine: frequent pairs have data, the *rare* tail stays
  template+slot (consistent template visible), which is exactly the fragmentation
  that hurt. Confirm the rare tail isn't itself merged away.
- **Slot inference.** Two-pass const/slot detection needs corpus stats per shape;
  if a "shape" conflates musically-distinct figures, its slot entropy will be huge
  → treat as not-an-idiom.
- **Token-count regression.** len-2-dominated dict ⇒ v2 may not compress; if P1
  shows a token *increase*, v2 is purely a consistency bet — proceed to P2 only if
  the consistency rationale still holds (it does for learnability).
- **MOTIF_ARG vs frame/back-ref machinery.** Slots must never straddle a frame or
  back-ref boundary; keep collapse strictly within `iter_voiced_blocks`.

## Work order

1. `stfconstants.py`: add `MOTIF_ARG`. 2. `macros/motif_pass.py`: v2 `MotifDict`
(template model, v2 `encode`/`expand`, version dispatch), `MotifPass` shape-match.
3. `motif_mine.py`: two-pass shape templating + template-level floor. 4. tier map:
`MOTIF_ARG`→content, `MOTIF_OP`→zero. 5. tests: v2 round-trip + version dispatch +
slot inference (unit) + the per-frame oracle. 6. preframr-xpt:
`motif_v2_mini_body_large` 3-arm spec. 7. P1 measurement script (reuse
`/scratch/tmp/motif_tail_scan.py` machinery).

## Cross-references

- `motif_pass_design.md` — v1 + the findings that motivate this.
- `compound_token_design.md` — the P3 field-factorized model; v2 P0–P2 is the
  tokenizer half, model-change-free.
- `audio_equivalence_normalization_design.md` — an orthogonal lossy knob to
  *quantize* slot values for further collapse.
