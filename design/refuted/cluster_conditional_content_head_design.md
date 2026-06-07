# Cluster-conditional content head (queue item 2)

**Status: REFUTED** (shelved). The `per_tier_heads_entropy_prodlike_v12` run
this was gated on refuted, and `cluster_conditional_content_head` is itself in
the refuted registry (same ~0.13 eval_a content ceiling, diversity ~1.0–1.2).
Retained for reference; do not reopen without the condition in
`preframr_experiments/data/refuted/`. Originally drafted as anticipatory work
while v12 (lambda=0.02) trained.

**Learnability framing.** A REFUTED model-side content head — it founders on the ~0.13 ceiling that tokenizer-side `full_macros` then lifted. The lever is representation learnability, not the head ([`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md)).

## Problem (re-anchored after v11 + v12)

The entropy-regularisation thread (v10 → v11 → mini sweep → v12)
demonstrates that **the per-tier-heads router architecture can lift
content acc at prodlike scale, but cannot recover prompt-conditioning
to the baseline level by hyperparameter alone**:

| run | tier | lambda | diversity_ratio | content acc (eval_a) |
|---|---|---|---|---|
| v10 | prodlike | 0 | 1.031 | 0.1358 |
| v11 | prodlike | 0.01 | 1.123 | 0.1261 |
| v12 | prodlike | 0.02 | (predicted) 1.14 | (predicted) ~0.12 |
| baseline (plain CE) | prodlike | – | 1.401 | 0.0618 |

Mini sweep shows diversity_ratio peaks at lambda=0.02 (1.596) and
declines at 0.05 (1.400 — over-regularisation). The mini→prodlike
"lift attenuation" (29% transfer for lambda=0.01) caps the entropy
approach below the 1.2 gate at prodlike.

The mask_structural_loss separator probe (`refuted/mask_structural_loss.md`)
ruled out gradient dominance as the framing. **The bottleneck is not
that structural targets dominate the gradient; it is that the
per-position content-tier softmax cannot express the
acoustically-equivalent token set with router-driven sharpening as
its only diversity source.**

Two acoustically-different content tokens get the same penalty as
two acoustically-equivalent ones; the model has no way to spend its
content-position uncertainty budget on "which acoustic region" vs
"which specific token in that region".

## Hypothesis

A hierarchical content head — predict acoustic-equivalence cluster
first, then token within cluster — lets the model commit to a region
of acoustic space without paying CE cost for picking the wrong
specific token within the region. Prompt-driven information flows
into the cluster choice (high-leverage for audio); within-cluster
sampling provides controlled variability that doesn't matter to
audio fidelity.

Expected effects:
1. **Diversity_ratio recovers.** Real prompts push cluster
   distribution towards specific acoustic regions; random prompts
   leave it more uniform. Within-cluster sampling adds noise that
   doesn't reduce real-vs-random distinguishability.
2. **Content acc holds or improves.** Per-cluster heads are smaller
   and easier to learn; the cluster head has C ≈ 256 outputs (vs
   30K) and converges faster.
3. **Loop_collapse stays at baseline.** Two-stage sampling at
   T_sample > 0 prevents argmax-greedy collapse to a single token.

## Approach

### Offline cluster index (Phase 0)

**Goal:** for each `v in V_content`, assign `cluster_id(v) in {0..C-1}`.

**Pipeline (`profile/build_content_clusters.py`, new):**

1. For each content vocab id `v`, synthesise a canonical SID context:
   `[FRAME header] [VOICE 0 init: ADSR + gate-on] [the SET op
   represented by v] [silence until next frame]`. ~50 register writes
   per synthetic dump.
2. Render via `preframr_audio.sidwav.WAVRenderer` → 50 ms PCM @ 44.1 kHz.
3. Extract a fixed-length feature vector: 64-bin log-mel or the
   existing `preframr_tokens.engine_fingerprint.compute_fingerprint`
   feature vector (38-dim) applied to the synthetic dump. (Choice
   tested empirically in Phase 0; default mel.)
4. K-means(C=256) over the feature vectors. Initialise via k-means++.
5. Output: `data/content_clusters/<tokenizer_hash>/cluster_assignments.json`
   with `{vocab_id: cluster_id}` + cluster centroids + per-cluster
   member counts.

**Cost:** ~30K renders × ~10 ms each = ~5 min on a single CPU. K-means
over 30K × 64 = ~30 sec. Total Phase 0 wallclock ~30 min fogbank.

**Validation gates for the cluster index:**
- Silhouette score > 0.3 on the held-out 10% (signal vs noise).
- Each cluster has ≥ 4 members (no tiny clusters that would degenerate
  to per-token CE).
- Largest cluster has ≤ 30% of vocab (no degenerate "uniform" cluster).
- Manual spot-check: 5 random clusters auditioned via preframr_audio
  for perceptual coherence. (Sanity gate; record as audit note.)

### Head architecture

Replaces `MoSHead` in `PerTierHeads` when `--content-cluster-head` is on.
Mutually exclusive with `--per-tier-content-mos-k > 0` (asserted at
arg-parse).

```
ClusterContentHead(torch.nn.Module):
    # shared
    cluster_proj: Linear(d → C)            # cluster predictor
    token_proj: Linear(d → V_content)      # within-cluster token predictor
                                            # (cluster mask applied per-position)
    cluster_id: Buffer(V_content,) long    # vocab → cluster lookup

    def forward(self, h) -> dict:
        cluster_log_p = log_softmax(cluster_proj(h), dim=-1)  # (B, T, C)
        token_logits = token_proj(h)                          # (B, T, V_c)
        return {"cluster_log_p": cluster_log_p,
                "token_logits": token_logits}
```

**Training-time joint log-prob** at a content position with GT token
`v_i`, cluster `c_i = cluster_id[v_i]`:

```
log p(v_i | h) = cluster_log_p[c_i] + log_p_within(v_i | h, c_i)
log_p_within(v_i | h, c_i) = log_softmax(
    token_logits[..., mask_for_cluster_c_i], dim=-1)[local_id(v_i)]
```

The within-cluster softmax is restricted via a per-cluster boolean
mask on `token_logits` (a `V_content`-shape buffer per cluster, stored
as a `(C, V_content)` bool tensor or sparse equivalent).

### Training loss

Joint NLL, computed per content position:
```
L_content = -E_{(h, v)} [ cluster_log_p[c(v)] + log_p_within(v | h, c(v)) ]
```
Both terms backprop through the body and through their respective
linear projections. Plugs into the existing
`_per_tier_training_step` in `lightning.py` — replace the
`MoSHead`-specific `nll_loss(tier_out_selected, local_gt)` branch
with a call to `cluster_content_loss(...)` when the head is
`ClusterContentHead`. Kendall-Gal uncertainty-weighted aggregation
unchanged.

### Inference

**Per-token (unified posterior).** `per_tier_unified_log_p` calls
`head.unified_log_p(h)` which returns `(B, T, V_content)` with the
joint factorised log-prob `cluster_log_p[c(v)] + log_p_within(v | h, c(v))`
for each `v`. Slot directly into the marginal-factorisation scatter
that the existing per-tier router code already does. No router
modifications.

**Two-stage sampling (production path).** New `cluster_sampler.py`:
1. `c ~ Categorical(cluster_log_p[t] / T_sample)`.
2. `v ~ Categorical(log_p_within[t, c] / T_sample)`.
3. Return `v`.

Two-stage is the recommended inference for high diversity_ratio;
per-token unified posterior is the fallback for audits / unified
predict CLI.

### Parameter budget

| component | params | notes |
|---|---|---|
| cluster_proj | d × C | 512 × 256 = 0.13M (mini) |
| token_proj | d × V_content | 512 × 30K = 15.4M (mini) |
| cluster_id buffer | V_content × 1 long | trivial |
| **total content head** | ~15.5M (mini) / ~23M (prodlike d=768) | **smaller than MoS K=4** |

Comparison: MoS K=4 content head at prodlike was ~60M. Cluster-conditional
is ~60% smaller. Fits envelope.

## Flags (additive)

```
--content-cluster-head            BooleanOptionalAction, default False
--content-cluster-c               int, default 256       (number of clusters)
--content-cluster-index           str, default ""        (path to assignments.json)
--content-cluster-feature         {"mel","engine_fp"}, default "mel"
```

Gated on `--per-tier-heads`. Mutually exclusive with
`--per-tier-content-mos-k > 0`.

## Phase plan

| phase | scope | wallclock | gate |
|---|---|---|---|
| 0 | Offline cluster index. `profile/build_content_clusters.py` → `data/content_clusters/<tokenizer_hash>/cluster_assignments.json`. Validation gates above. Manual audio spot-check. | ~1 hr (build + validation + audition) | Silhouette > 0.3; cluster sizes within bounds; perceptual coherence on 5 random clusters. |
| 1 | Impl behind `--content-cluster-head` (default OFF). New: `heads_cluster.py`, `losses_cluster.py`, `cluster_sampler.py`, `profile/build_content_clusters.py`. Touched: `heads.py` (one branch in PerTierHeads init), `lightning.py` (one branch in `_per_tier_training_step`), `args.py` (four flags + mutual-exclusion assertion). Tests: `test_cluster_head_forward.py` (shape, per-cluster mask correctness, joint log-prob = cluster + within), `test_cluster_loss.py` (vs hand-computed reference), `test_cluster_sampler.py` (two-stage decode determinism under seed). | 2 days impl | Tests pass; smoke train (7 SIDs) one-epoch clean. |
| 2 | `cluster_content_mini_body_large` A/B at mini, body=large, 60 max-epochs, 3 seeds. Arms: `cluster_C256` (target), `baseline` (per_tier_heads_mos4 + entropy lambda=0.02 — best-known mini config from the sweep). Audits via `audit_checkpoint_per_class.py`, `prompt_conditioning_audit.py`, `loop_detection_audit.py` at T=0.5. | ~30 min/arm × 3 seeds × 2 arms = **~3 hr** sequential | (1) content acc ≥ mos4+entropy baseline (3σ floor); (2) diversity_ratio > 1.65 at T=0.5 (the sweep's mini target that predicts >1.2 at prodlike); (3) loop_collapse_rate ≤ mos4+entropy; (4) cluster_log_p entropy histogram across epochs shows the cluster head is learning a non-uniform posterior (sanity, not gate). Refute if (1)-(3) fail. |
| 3 | `cluster_content_prodlike` at prodlike, single seed. Same arms shape (target + best-known prodlike comparator = v11 or v12 mos4+entropy). Eval suite as v10/v11. | ~6-11 hr/arm; with `--only-arm` target, ~6-11 hr total | Phase 3 prodlike gate from `per_tier_heads_design.md` with recalibrated absolute floor (0.12 from `refuted/per_tier_heads_mos_prodlike.md` "What this changes" #3). |

**Total wallclock from approval to prodlike verdict:** Phase 0 ~1 hr,
Phase 1 ~2 days agent work, Phase 2 ~3 hr, Phase 3 ~6-11 hr.
**End-to-end:** ~3 days assuming Phase 2 passes.

## Decisions taken (rationale)

1. **C=256 default.** Empirical sweet spot: large enough to give the
   cluster head useful discrimination capacity (~120 tokens/cluster
   on average), small enough that within-cluster softmax remains
   well-conditioned. Tunable in Phase 2 as `cluster_C128`/
   `cluster_C512` ablations if needed.
2. **Mel features default over engine_fp.** Mel captures perceptual
   acoustic similarity directly; engine_fp captures structural-write
   patterns and may cluster tokens that sound different but co-occur
   in similar contexts. The latter would degenerate towards composer/
   style clusters, which isn't the bottleneck. engine_fp kept as
   ablation flag.
3. **Synthesise tokens in a canonical context, not in isolation.**
   Many content tokens require voice ADSR + gate state to render
   audibly (a frequency-LO write with no gate-on produces silence).
   The canonical context fixes that confound at the cost of one
   choice of context being baked into the cluster definition.
4. **Two-stage sampler is the production path; unified posterior
   fallback for audits.** Same rationale as the diffusion design.
5. **Content tier only.** Structural / mid / zero unchanged.
6. **Smaller head than MoS K=4.** The diversity budget moves from
   "K=4 mixture components" to "C=256 cluster commit + within-cluster
   softmax". This is closer to information-theoretic optimum for the
   "acoustically-equivalent alternatives" framing.
7. **No body modifications.** Same shared body. Same per-tier
   infrastructure landed by Approach C.

## Risk + non-goals

**Risks:**
- **Cluster quality.** Bad cluster → bad results. Phase 0 validation
  gates are load-bearing.
- **Canonical-context arbitrariness.** Choice of context affects
  cluster boundaries. Mitigation: ablate two contexts in Phase 0,
  keep the one with higher silhouette score.
- **Within-cluster collapse.** If most content tokens land in 2-3
  large clusters, the within-cluster softmax becomes the new
  bottleneck. Phase 0 size-balance gate catches this.
- **Prodlike attenuation pattern repeats.** Mini lift may not transfer.
  Same risk as every prior architectural bet; no specific mitigation
  beyond the recalibrated Phase 3 gate.

**Non-goals:**
- Online cluster learning. The cluster assignments are fixed at
  offline-build time and treated as data.
- Cross-tier clustering. Structural / mid / zero tokens don't get
  clustered.
- Hierarchical-deeper-than-2 (e.g. cluster of clusters). Diminishing
  returns and exploding implementation complexity.

## Open questions for Phase 0 review

- Does the canonical context bake too much voice-0 specificity into
  the clusters? Quick test: build the cluster index twice (voice-0
  context vs voice-2 context) and check Rand index. If RI > 0.85,
  context choice is robust. If RI < 0.6, clustering needs a
  multi-context render.
- Is C=256 the right starting point? Phase 0 should sweep C ∈ {128,
  256, 512} and pick by silhouette.
- Does the existing `engine_fingerprint` feature vector (38-dim,
  CTRL-bigram-heavy) give comparably-good clusters? Cheap ablation;
  if yes, drop the mel renderer dependency.

## preframr-audio enhancements (scoped to what Phase 0 + the cluster
work surface)

### Feature extraction: defer the library addition until Phase 0 picks

Phase 0 will ablate two feature options for cluster construction:

1. **mel features** (new): 64-bin log-mel, time-pooled (mean + std) to
   a fixed-length vector. Standard psychoacoustic-adjacent default
   for short audio clips.
2. **engine_fingerprint** (existing): the 38-dim feature vector
   already in `preframr_tokens.engine_fingerprint.compute_fingerprint`
   (register-density, delta histogram, CTRL n-grams, filter-touch
   ratio). Originally built for composer/engine clustering at the
   dump level; reused here at the per-token synthetic-dump level.

**Library-add decision deferred to Phase 0 outcome:**

- If `engine_fingerprint` wins: zero preframr-audio change. Cluster
  build script imports the existing preframr-tokens primitive.
- If `mel` wins: add ONE small module
  `preframr_audio/features.py::mel_features(samples, sr, n_mels=64)`.
  ~30 LoC plus tests. Justified by a second consumer down the road
  (audio-audition diagnostics, generation-quality audits). Promotion
  out of the build script is gated on at least one second consumer
  showing up.

**Not adding** (avoiding scope creep): clustering APIs, distance
functions, learned audio embeddings, psychoacoustic / Bark-scale
features, onset / rhythm features. The cluster build is one-off
offline tooling; bespoke is fine.

### Fidelity API: tolerate small SID frame drift + plug-in feature diff

Independent of cluster work, but motivated by the same "two renders
that differ by a frame of timing aren't acoustically different"
intuition. The existing `preframr_audio.fidelity.compare_renders`
fails FRAME_CADENCE_BREAK as soon as cross-correlation lag ≥ one
frame_window — strict bit-exact equivalence, brittle on timing
variation.

Proposed additive parameters (defaults preserve current strict
behaviour):

```python
def compare_renders(
    samples_a, samples_b, sample_rate,
    tolerance: float = FRAME_RMS_TOLERANCE,
    max_frame_drift: int = 0,          # NEW
    feature_diff_fn: Callable | None = None,  # NEW
    feature_diff_tolerance: float = 0.0,      # NEW
) -> AudioFidelityResult: ...
```

- `max_frame_drift=N`: tolerate up to ±N frames of cross-correlation
  lag before failing FRAME_CADENCE_BREAK. When N>0 and lag is within
  bounds, the per-frame RMS comparison runs on the lag-aligned slice
  instead of failing immediately. Default 0 preserves current strict
  semantics. Macro-pass validation tests that legitimately need
  bit-exact stay at 0; cluster / audition / future-generation tests
  opt into N=1 or 2 where small drift is acceptable.
- `feature_diff_fn`: optional callable
  `(samples_a, samples_b) -> float`. When provided, the comparison
  short-circuits the per-frame RMS path and instead computes the
  feature-space distance, comparing against
  `feature_diff_tolerance`. Implementation: same diagnostic
  taxonomy (PASS / *_DIVERGENCE) keyed off whether the feature
  distance exceeds tolerance. The caller plugs in mel / engine_fp /
  whatever distance function they want; the fidelity primitive
  doesn't grow a dependency on librosa or sklearn.
- `feature_diff_tolerance`: only used when `feature_diff_fn` is set.

Use cases enabled:
1. **Cluster validation** in Phase 0: `compare_renders(token_render_a,
   token_render_b, ..., feature_diff_fn=mel_distance,
   feature_diff_tolerance=0.1)` to assert two cluster-mates are
   acoustically close.
2. **Macro-pass tests under timing variation**: legato/loop passes
   that legitimately shift writes by ±1 frame can use
   `max_frame_drift=1` instead of being whitelisted out.
3. **Generation-quality audits** (future): "this generated stream is
   within feature-distance D of the ground-truth prompt continuation"
   becomes a one-line check.

Implementation cost: ~40 LoC in `fidelity.py`, additive (no existing
callers break). New tests for both knobs (drift tolerance edge cases,
feature_diff_fn plug-in correctness, feature_diff_tolerance
threshold). Promoted out of this design doc into a preframr-audio
PR after Phase 0 surfaces the concrete feature_diff_fn we want to
use first.

## References

- Origin: `model_loss_queue.md` item 2.
- Approach C refute trail: `refuted/per_tier_heads_mos_prodlike.md`,
  `refuted/mask_structural_loss.md`, the v11 + v12 commits.
- Template: `content_diffusion_design.md` (this directory).
- Existing infrastructure: `preframr_tokens.engine_fingerprint`
  (compute_fingerprint, ClusterTable), `preframr_audio.sidwav`
  (WAVRenderer), `preframr.train.model.heads.PerTierHeads`,
  `preframr.train.model.lightning._per_tier_training_step`.
