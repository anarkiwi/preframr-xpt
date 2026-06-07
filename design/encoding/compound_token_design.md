# Compound-token tokenizer + parallel-attribute heads (Approach D)

**Status:** Draft, design review pending. Strategic pivot from per-token content-head architectures (mos / entropy / mask / cluster, all refuted) to a multi-attribute-per-token reorganization. Adapts the CompoundWord (Hsiao et al. 2021) and OctupleMIDI (Anticipation, 2023) approaches from MIDI music LLM literature.

**Learnability framing (gate before building).** Sequence-length compression is a learnability win *only if* it lowers per-FRAME h_k or shortens the dependency horizon — packing more per token can instead raise the next-token entropy and add an implicit counter (the `delta` cumulative-timing field is exactly the maintained-counter structure [`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md) Principle 1 warns against; gzip-style compression rewards redundancy the transformer can't exploit). **Before the multi-week build, run `audit/learnability_triage.py` on the compound stream** and gate on per-frame h_k + induction-copy deltas, not token count.

## Problem (re-anchored after four content-head refutations)

Four per-token content-head architectures have been refuted at mini or prodlike:

1. v10 per_tier_heads mos4 (prodlike): criteria 3+4 fail (router saturation).
2. v11/v12 entropy retest (prodlike): criterion 4 fail by 0.077 then deteriorates.
3. mask_structural_loss (mini): diversity_ratio collapses to 0.863.
4. cluster_C256 (mini): diversity_ratio 1.194 vs baseline 1.596.

The pattern: each architecture moves the needle at mini and either stalls or
reverses at prodlike. The shared assumption across all four is **one token per
discrete musical event, with structural-tier tokens (FRAME, VOICE) as flat
scaffolding interleaved with content-tier tokens**. The architectural changes
are all on the content-head output side; the token layout is unchanged.

Music LLM literature suggests this is the wrong layer to iterate. Flat
structural+content tokenization pays ~25% sequence on per-frame and per-voice
delimiters, which the model spends ~5-8% of its loss budget predicting (these
predictions are near-deterministic). Compound-token tokenizers (CP, OctupleMIDI)
pack multiple attributes per token + emit them via parallel softmax heads,
typically shrinking sequence 3-4x with equal or better per-event accuracy.

## Hypothesis

A compound-token reorganization tests three claims simultaneously:

1. **Sequence-length is binding.** Shrinking sequence 3-4x raises effective
   capacity per content decision; if the prodlike attenuation is partly a
   data-floor / context-length issue, this moves the floor.
2. **Structural tokens carry near-deterministic information that doesn't need
   its own token.** Packing FRAME and VOICE into attributes of a single
   compound event removes the "free wins" the model spends loss budget on, and
   refocuses learning on the content-relevant attributes (val, reg).
3. **Parallel attribute prediction breaks the content-head bottleneck.** The
   single 32K-vocab content softmax becomes K small softmaxes (one per
   attribute) whose joint distribution has higher achievable rank than any
   single-softmax variant we've tried.

## Approach

### Token = one compound event

Each token in the sequence represents one (voice, reg, val, op, delta) tuple.
Specifically:

| attribute | size | meaning |
|---|---|---|
| `voice` | 4 (v0, v1, v2, global) | which SID voice this event targets; "global" for filter / volume |
| `reg` | 25 | SID register 0..24 |
| `val` | 256 | register value 0..255 (no bucketing; mel-frequency mapping handled at decoder) |
| `op` | ~20 | macro class: SET / CTRL_BIGRAM / SLOPE_FREQ / HARD_RESTART / etc. (mirrors current tokenizer's op taxonomy) |
| `delta` | 32 buckets | log-spaced cycle-delta since previous event; smallest bucket = 1 cycle, largest = ~1 frame |

Five attributes; comparable to OctupleMIDI's 8. Sequence length collapses from
"FRAME + VOICE + content_token" (3 tokens per event) to "1 compound token per
event" — ~3x compression, matching CP/Octuple literature reports.

### Input embedding (sum of attribute embeddings)

```
emb(token) = voice_embed[t.voice] + reg_embed[t.reg]
           + val_embed[t.val] + op_embed[t.op] + delta_embed[t.delta]
```

Per-attribute embedding tables. Total params: `(4 + 25 + 256 + 20 + 32) * d`
= ~340K at `d=768`. Negligible vs the 125M body.

### Output (parallel attribute heads)

```
out = body(emb_seq)  # (B, T, d)
log_p_voice = log_softmax(W_voice(out))   # (B, T, 4)
log_p_reg   = log_softmax(W_reg(out))     # (B, T, 25)
log_p_val   = log_softmax(W_val(out))     # (B, T, 256)
log_p_op    = log_softmax(W_op(out))      # (B, T, 20)
log_p_delta = log_softmax(W_delta(out))   # (B, T, 32)
```

Five parallel linear heads. Total head params: `d * (4 + 25 + 256 + 20 + 32)`
= ~260K at `d=768`. Much smaller than the single-softmax content head
(~24M for V=32K at d=768).

### Training loss

Per-attribute CE; per-attribute Kendall-Gal weighting (extends the existing
`log_sigma_per_tier` infrastructure):

```
L = sum_a   (exp(-2 * log_sigma_a) * 0.5 * CE(log_p_a, gt_a))
  + sum_a   log_sigma_a
```

Five `log_sigma_a` learnable scalars (one per attribute) replace the current
four `log_sigma_per_tier`. Same machinery, different cardinality.

Per-attribute loss reporting (val / train) per attribute. Per-attribute
accuracy reporting. Full-event accuracy = all five attributes correct;
reported as the headline metric for gate comparisons.

### Inference (sequential sampling per event)

Sample attributes in a fixed order, re-encoding after each pick to condition
subsequent heads. Order: `voice -> reg -> val -> op -> delta`.

```
for t in autoregressive_positions:
    out_t = body.step(prev_emb)
    voice = sample(log_p_voice(out_t), T_sample)
    # re-encode partial event
    partial_emb = voice_embed[voice]
    reg = sample(log_p_reg(body.step_addend(partial_emb)), T_sample)
    # ...continue for val, op, delta
    full_emb = voice_embed[voice] + reg_embed[reg] + val_embed[val] + ...
    yield (voice, reg, val, op, delta)
```

Cost: 5x more body invocations per emitted event vs a single-softmax model.
Mitigated by 3x sequence compression -> ~1.5x net inference cost per event,
which is ~50% slower decoding wallclock. Acceptable.

Alternative (deferred): parallel-sample-then-reject. Sample all attributes
independently in parallel; if the resulting tuple is invalid (e.g.,
`op=SLOPE_FREQ` requires `reg in {0, 7, 14}`), reject and resample. Faster
but adds validity constraints to the design.

### Decoder reconstruction

Each compound token decodes to one or more SID register writes via the existing
macro expansion (`op_expand_lookup`). Cycle timing via `delta` attribute:

```
clock = 0
for tok in tokens:
    clock += delta_bucket_to_cycles(tok.delta)
    expand_op(tok.op)(tok.voice, tok.reg, tok.val) -> writes
    apply(writes, at_clock=clock)
```

The audio bit-exact constraint
(`tests/test_sid_same_value_writes.py` + `test_parser_canonicalisation_audio_invariants.py`)
must hold: `parse -> compound_tokenize -> decode -> writes` round-trip
bit-identical to the original dump. Delta bucketing introduces quantization;
need bucket boundaries fine enough that any two cycles in the same bucket are
acoustically equivalent (within `compare_renders` tolerance with
`max_frame_drift=1`). This is precisely what the preframr-audio v0.3.0
`compare_renders` enhancement we drafted supports.

## Parameter budget

| component | params (prodlike d=768) | notes |
|---|---|---|
| body | 125M | unchanged |
| compound embedding tables | ~340K | sum of (4 + 25 + 256 + 20 + 32) * d |
| 5 parallel heads | ~260K | sum of d * attribute_sizes |
| router | n/a | not needed (no tier routing) |
| learnable Kendall-Gal sigmas | 5 scalars | one per attribute |
| **delta vs per_tier_heads + mos4** | **-25M head params, -1 router head** | dramatically smaller |

Sequence-length impact: ~3x compression. At MAX=8192 today, equivalent to
~24K-write context in the old encoding -- the model now sees ~3x more song
per prompt.

## Flags (additive; no existing default changes)

```
--compound-tokenize           BooleanOptionalAction, default False
                               (tokenizer-level: emit compound tokens vs flat)
--compound-attributes         comma-list, default "voice,reg,val,op,delta"
                               (lets ablations drop attributes for diagnostic)
--compound-delta-buckets      int, default 32
--compound-val-buckets        int, default 256 (no bucketing; matches SID byte)
--compound-attribute-loss-balance   {"uniform", "learnable_sigma", "frequency_weighted"}
                                    default "learnable_sigma"
```

`--compound-tokenize` is at the tokenizer level (parse + tokenize stages must
co-produce compound tokens, not flat). Asserts at arg-parse time that none of
`--per-tier-heads`, `--content-cluster-head`, `--content-diffusion` are active
(compound replaces them; per-tier-heads logic doesn't apply when each token
already factors).

## Phase plan

| phase | scope | wallclock | gate |
|---|---|---|---|
| 0 | Tokenizer prototype. New `preframr_tokens.compound_tokenize` module: parse dump.parquet -> compound event stream -> save as new tokens.csv variant. Decode round-trip: compound -> writes -> render -> compare audio against original. Bit-exact mandatory; quantization tolerance only via the preframr-audio v0.3.1 `compare_renders(max_frame_drift=1, feature_diff_tolerance=...)` knobs. Audit measurement: actual sequence-length compression on prodlike corpus (validate the 3x literature claim). | ~1 week impl + audit | Audio round-trip passes; sequence compression >= 2.5x measured. |
| 1 | Training-side impl. New `preframr/train/model/heads_compound.py` with `CompoundHeads` module (5 parallel linear heads, parallel forward). New `preframr/train/model/losses_compound.py` with per-attribute CE + Kendall-Gal weighting. New `preframr/train/model/embed_compound.py` (or extend body factory) for the sum-of-attribute embedding. Touched: `args.py` (flags), `lightning.py` (training_step branch when compound active). Tests: per-attribute head forward + loss; full training_step end-to-end smoke; embedding sum determinism. | 1-2 weeks impl | Tests pass; smoke training on a 7-SID corpus one-epoch clean. |
| 2 | `compound_mini_body_large` A/B at mini, body=large, 60 max-epochs, 3 seeds. Arms: `compound_5attr` (target), `baseline` (per_tier_heads_mos4 + entropy lambda=0.02 -- best-known mini config). Audits via `audit_checkpoint_per_class.py` adapted for per-attribute reporting + `prompt_conditioning_audit.py` + `loop_detection_audit.py` at T_sample=0.5. | ~3-4 hr training + ~30 min audits | (1) full-event val_acc >= baseline (3-sigma floor on 3-seed std); (2) per-attribute val_acc >= baseline for the 'val' attribute specifically (the content-equivalent); (3) `diversity_ratio` > baseline + 1 sigma at T=0.5; (4) `loop_collapse_rate` <= baseline. Refute if any of (1)-(4) fail. |
| 3 | `compound_prodlike` at prodlike tier, body=canonical, 60 max-epochs, 1 seed. Same arms shape. Eval suite: `eval_a` + 8 `eval_b_*` families. Audits as Phase 2. | ~6-11 hr | Phase 3 prodlike gate from `per_tier_heads_design.md` adapted for the 'val' attribute as content equivalent: (1) eval_a val-acc >= 2x baseline AND >= 0.12 (recalibrated); (2) >= 5 of 8 eval_b_* lifts on val-acc; (3) `loop_collapse_rate` <= baseline; (4) `diversity_ratio` > 1.2; (5) no structural-attribute regression > 1 sigma. |
| 4 | Refuted-tree entry on fail. On PASS: write `landed/compound_token.md` and queue cross-engine prodlike + audio audition. | - | - |

**Total end-to-end: ~3-4 weeks** assuming Phase 2 passes. Tokenizer phase (Phase 0+1) is multi-week; the training phases are comparable to prior bets.

## Decisions taken (rationale)

1. **5 attributes (voice, reg, val, op, delta), not the full 8 of OctupleMIDI.**
   OctupleMIDI has BAR/POSITION/TIMESIG/TEMPO; SID doesn't need TIMESIG/TEMPO
   (constant within a song) and BAR/POSITION are subsumed by `delta`. 5 is the
   minimum that captures all the information a current flat encoding carries.

2. **Parallel heads at training, sequential sampling at inference.** Standard
   CP/Octuple choice. Parallel training is simpler and matches the "all
   attributes are conditionally independent given context" assumption. Sequential
   sampling at inference handles the joint-distribution correlations (e.g.,
   `op=SLOPE_FREQ` only valid with `reg in {0,7,14}`) implicitly by exposing each
   subsequent head to the prior picks.

3. **Per-attribute Kendall-Gal weighting** (uncertainty-weighted multi-task
   loss). Reuses the existing `log_sigma_per_tier` machinery (rename and
   re-cardinalize). Avoids per-attribute weight tuning; the model learns
   relative attribute difficulty.

4. **`val` attribute is the new "content"-tier equivalent.** All gate criteria
   adapted from per-token-content evaluation to per-attribute-val evaluation.
   This makes the gates directly comparable to prior experiments (the val
   attribute is the densest, hardest, and what acoustically distinguishes
   continuations).

5. **Keep macro ops as the `op` attribute** (rather than atomize). Preserves
   the existing tokenizer's macro infrastructure (legato, ctrl_bigram,
   hard_restart, slope) as informative event types. The model can learn "in
   this context, SLOPE is the right op" as a useful signal. Atomization is a
   deferred ablation.

6. **No tier classification.** The `_LOSS_TIER_ORDER` + per-tier-heads
   infrastructure is bypassed. Compound replaces it. The current
   `tier_classify.py` module remains but is unused when `--compound-tokenize`
   is on; future refactor could remove tier infrastructure if compound lands.

7. **Delta as the timing carrier**, NOT FRAME as a structural anchor. Each
   compound token carries the cycle-delta since the previous event. FRAME
   tokens disappear from the stream entirely. The decoder reconstructs
   IRQ-aligned timing from the cumulative delta sum. This is the
   FRAME-elimination piece of the proposal -- audio bit-exactness via delta
   bucketing is the load-bearing constraint validated in Phase 0.

8. **Compound tokens at parser level, not bucket-quantized post-hoc.** The
   parser emits compound events directly; no intermediate flat-token
   representation. Cleaner abstraction; tokenizer pipeline simpler. Cost:
   parser surgery.

## Risk + non-goals

**Risks:**
- **Tokenizer surgery is significant.** The current parse + tokenize pipeline
  is rule-driven and has been refactored once into preframr-tokens. Compound
  tokenization is a new parser path; existing macro logic ports across but
  the surrounding glue is new. ~1-2 weeks impl + audit.
- **Delta bucketing introduces quantization.** Audio bit-exact through
  `compare_renders(max_frame_drift=1)` is required; if bucket boundaries are
  too coarse, perceptually-equivalent renders may diverge. Phase 0 audit
  measures this.
- **Sequential sampling at inference is ~1.5x slower per event** vs
  single-softmax (mitigated by 3x sequence compression). On Jetson Orin NX
  this is acceptable but eats into the inference budget; if the predict path
  becomes load-bearing, parallel-sample-then-reject is the optimization path.
- **The prodlike attenuation pattern that bit prior architectures could bite
  this too.** Compound tokens don't fundamentally fix data sparsity per
  attribute -- the val attribute still has 256 classes that need data
  coverage. Different mechanism than prior bets (parallel heads with
  attribute factorization), so the FAIL evidence isn't directly predictive,
  but it's not a guaranteed win either.
- **Eval semantics shift.** All audit tools currently compare per-token-id
  predictions. Under compound, per-event accuracy requires per-attribute
  comparison + joint-correct rollup. Phase 2 audit-tooling change scope is
  similar to the cluster_log_p sanity check work we did for ClusterHead.

**Non-goals:**
- Atomization of macro ops (deferred ablation; keep current macro set as `op`
  attribute).
- Voice-major or voice-state-pair reorganization (different design direction;
  compound tokens are orthogonal and chosen because of stronger literature
  evidence).
- Audio-equivalence-merged vocabulary (Approach E in our queue; orthogonal --
  can layer on top after compound lands).

## References

- Hsiao, Liu, Yeh, Yang (2021). "Compound Word Transformer: Learning to
  Compose Full-Song Music over Dynamic Directed Hypergraphs." AAAI.
- Thickstun, Hall, Donahue, Liang (2023). "Anticipatory Music Transformer."
  ArXiv. (OctupleMIDI variant.)
- Hsiao et al. (2020). "Pop Music Transformer: Beat-based Modeling and
  Generation of Expressive Pop Piano Compositions." ACMMM. (REMI baseline.)
- Refuted entries in our work that motivated this design:
  - `preframr-xpt:refuted/per_tier_heads_mos_prodlike.md`
  - `preframr-xpt:refuted/per_tier_heads_entropy_prodlike.md`
  - `preframr-xpt:refuted/mask_structural_loss.md`
  - (forthcoming) `preframr-xpt:refuted/cluster_content_head_mini.md`
- Foundational infrastructure: `per_tier_heads_design.md` (Kendall-Gal
  multi-task loss machinery to extend), `content_diffusion_design.md`
  (parallel-head template), `preframr-audio` v0.3.0 fingerprint framework
  (Phase 0 delta-bucketing audit).
