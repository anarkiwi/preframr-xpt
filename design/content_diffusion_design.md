# Content-tier discrete diffusion head (Approach A)

**Status:** Draft, pending review. Fallback for the in-flight
`per_tier_heads_prodlike` Phase 3 if it refutes. Supersedes the
"Approach A" bullet in `multi_modal_objective_design.md`. Reuses
the per-tier head + router infrastructure landed by Phase 1 of
`per_tier_heads_design.md`.

## Problem (re-anchored after Approach C)

Approach C (MoS K=4 content head + tier router + uncertainty-weighted
multi-task loss) passed Phase 2 at sampling temperatures T >= 0.5
(`per_tier_heads_mos_revisited.md`). Phase 3 prodlike is in flight.
**If Phase 3 refutes**, the prevailing explanations will be either:

1. **Mixture collapse at scale.** K=4 components train towards a
   single dominant mode on prodlike, recovering CE-like behaviour.
   The 3-seed mini result (collapse 0/12, diversity 1.223) may not
   survive capacity x data scale-up.
2. **Softmax bottleneck still binds.** Each MoS component is itself
   a softmax over `V_content` (~21K). MoS expands the achievable
   distribution rank from 1 to K=4, but the per-component
   distributions remain parametric softmaxes anchored on linear
   projections of `h`. If the true continuation distribution has
   rank > 4 in regions of state space the body cannot resolve, MoS
   under-fits.
3. **Router still saturates on structural.** Sampling unlocks the
   content head for mini; at prodlike, the body may concentrate
   even more confidently on structural tokens (because structural
   prediction lifts faster with scale than content), pushing the
   router posterior past the temperature-sampling threshold.

The framing remains: the bottleneck is *distribution shape on
content-tier positions*. Approach A replaces the per-position
parametric softmax with a non-autoregressive discrete denoising
objective over content-tier positions. This is the maximalist
intervention from `multi_modal_objective_design.md`; we propose to
take it only after C refutes, with a tractable scope that fits the
project's compute envelope.

## Hypothesis

A discrete-diffusion content head, trained to denoise masked content
tokens conditioned on (a) the same shared transformer body output
and (b) unmasked surrounding content + full structural context,
will model the content-tier distribution at a rank limited by the
*denoising schedule*, not the head parameter count. Multi-step
sampling at inference produces genuinely multi-modal continuations
because each denoising step samples from a learned categorical, and
the joint distribution over the content positions in a block is
factorised across steps rather than per-token.

## Approach

### Architecture (split across new files; existing files untouched)

Shared transformer body (no change). Per-tier heads (no change):
structural / mid / zero keep their linear-CE heads, router keeps
its 4-way linear head. **The content head is the only thing that
changes.** When `--content-diffusion` is on, `PerTierHeads`
constructs `DiffusionContentHead` (new file) in place of
`MoSHead` for the `content` tier.

New files under `preframr/train/model/`:

- `heads_diffusion.py` — `DiffusionContentHead(torch.nn.Module)`,
  encapsulating: time-step embedding, mask-token embedding, the
  denoising projection. Single-step forward: takes
  `(h, x_t, t) -> log_p(x_0 | h, x_t, t)` over `V_content`.
- `losses_diffusion.py` — `discrete_diffusion_content_loss`,
  pure-torch NLL on the sampled mask schedule. Helper:
  `sample_mask_schedule(content_mask, t, schedule="cosine")`.
- `diffusion_sampler.py` — `iterative_unmask(model, h, T, partition,
  ...)`, the K-step inference unmasking loop. Called from a thin
  hook in `predict_lib.py`; imported lazily, no top-level import
  in existing files.

Existing files require minimal edits, all behind the
`--content-diffusion` flag:

- `heads.py` — one branch in `PerTierHeads.__init__`: if
  `content_diffusion=True`, instantiate `DiffusionContentHead`
  (imported locally inside the constructor). The forward path
  already returns a dict; we add a `content_diffusion` key when
  enabled.
- `lightning.py` — one branch in `_per_tier_training_step`: when
  the content head is the diffusion variant, replace the
  `nll_loss(tier_out_selected, local_gt)` call with
  `discrete_diffusion_content_loss(...)`. Uncertainty-weighted
  aggregation is unchanged (the diffusion loss is a scalar in the
  same units as NLL).
- `per_tier_unified_log_p` in `heads.py` — at inference, the
  content head no longer directly emits `log_p(v|h)`. Two options
  (decision below in Phase 1):
  - **Option U1 (chosen):** unified posterior fuses the *one-step*
    fully-masked denoising prediction `log_p(x_0 | h, x_t=mask, t=T)`
    as if it were a per-position softmax. This is what router
    marginalisation expects; the *block-level* iterative sampler
    runs as a separate inference path (see Inference below).
  - Option U2 (rejected): redesign the unified posterior to be
    block-aware. Too invasive; conflicts with the per-token decode
    loop in `predict_lib.py`.

No edits to `bodies.py`, `factory.py`, `tier_map.py`. Existing
`per_tier_heads_*` experiments and tests continue to work
unmodified.

### Training objective (D3PM absorbing-state variant)

Vocabulary for the diffusion process: `V_content` (the existing
content-tier sub-vocab from `_build_tier_vocab_partition`). Add a
single absorbing `[MASK]` token at index `|V_content|`, modelled
explicitly as the (K+1)-th class in the denoising head. Mask token
is *internal* to the diffusion head; it never appears in the main
vocabulary or in the dataset.

Forward (corruption) process at training:

- For each content-tier position in the batch, sample timestep
  `t ~ Uniform(1, T)`.
- Mask probability `alpha_t = cos(pi/2 * t/T)` (cosine schedule;
  swap to linear for ablation).
- With prob `alpha_t`, replace the GT token at that position with
  `[MASK]`. Structural / mid / zero positions are never masked.

Reverse (training) loss:

- Body forward on the *unmodified* input (causal LM context is
  preserved; the diffusion happens only inside the content-head
  forward path).
- Content head receives `(h, x_t, t_emb)` where `x_t` is the masked
  content sequence and `t_emb` is a sinusoidal embedding of `t`.
- Head emits `log_p(x_0 | h, x_t, t)` over `V_content` at every
  masked content position.
- Loss: cross-entropy on the masked positions only, against GT.

```
L_diffusion = -E_{t, mask} [ Sum_{i in masked} log p(x_0_i | h, x_t, t) ]
```

Uncertainty-weighted aggregation (Kendall & Gal) reuses
`log_sigma_per_tier[content]` from the existing `_per_tier_losses`
machinery — no new learnable scalars. The structural / mid / zero /
router losses are unchanged.

Optional auxiliary at any masked position: cross-entropy against
GT *plus* an entropy floor on the predicted distribution
(`-lambda * H(p)`) to discourage early-step mode collapse. Default
`lambda = 0`; tune in Phase 2.

### Inference (two paths, both flag-selectable)

**Per-token path (unified posterior).** When `predict.py` runs in
its current per-token loop, the content head is called with
`x_t = [MASK]` and `t = T` (the maximally-noised state). This emits
`log_p(x_0 | h, x_t=mask, t=T)` — a one-step denoising prediction
*as if* the position were the only masked content token. This slots
into `per_tier_unified_log_p` unchanged. Sampling at T_temp > 0
provides multi-modality across calls.

**Block-level iterative path (new).** A separate entry point
`diffusion_sampler.iterative_unmask(...)` runs the full K-step
unmasking on a block of N content positions:

1. Initialise: all N content positions masked, all structural /
   mid / zero positions filled from the autoregressive decode.
2. For `t = T, T-1, ..., 1`:
   - Compute `h = body(x_t)`.
   - Compute `p_t = head(h, x_t, t)` at masked positions.
   - Sample `x_hat_0 ~ p_t` (temperature `T_sample`).
   - Re-noise: keep `floor(N * alpha_{t-1})` masked, unmask the
     rest. Choice of which to unmask: argmax-confidence (MaskGIT
     greedy) or random (D3PM).
3. After `t = 1`, all content positions are filled.

The block-level path is the production sampler; the per-token path
is a fallback for the existing CLI and for the unified-posterior
audits that don't know about blocks.

Predict-host envelope (Orin NX 15.6 GB):
- Body forward + KV cache: unchanged.
- Diffusion head: 1 linear projection `d -> |V_content|+1` per
  forward; T forwards per block (default T=8). Wallclock impact:
  ~T x current head cost. For T=8 at d=768, |V_c|=21K, this is
  ~0.04 GFLOP * 8 = 0.32 GFLOP per content position vs ~0.02 GFLOP
  for plain CE. Body forward dominates (>0.5 TFLOP per token); head
  remains <5% of inference cost.
- The block-level sampler re-runs the body T times per block, not
  T times per token. At block_size ~ 64 content positions, this is
  8x body forwards amortised over 64 positions = 0.125x per-token
  body cost. Acceptable.

### Parameter budget

| component | params | notes |
|---|---|---|
| body | unchanged (125M at canonical) | — |
| structural / mid / zero heads | unchanged (~9M) | — |
| router | unchanged | — |
| diffusion content head | `d*(|V_c|+1)` + `d*d_t` + `d_t*d` | ~16.5M (d=768, V_c=21K, d_t=128 time-embed) |
| **delta vs Approach C MoS K=4** | **-48M head params** | diffusion head is *smaller* than MoS K=4 |

Diffusion head is materially smaller than MoS K=4 head — the
multi-modality budget shifts from K=4 components to T=8 denoising
steps. Total model size at canonical prodlike: ~150M (vs ~200M
with MoS K=4). Fits the 24 GB envelope comfortably.

## Flags (additive; no existing default changes)

```
--content-diffusion           BooleanOptionalAction, default False
--content-diffusion-T          int, default 8     (denoising steps)
--content-diffusion-schedule  {"cosine","linear"}, default "cosine"
--content-diffusion-entropy-lambda  float, default 0.0  (auxiliary)
--content-diffusion-sampler   {"maskgit","d3pm"}, default "maskgit"
```

All gated on `--per-tier-heads` (the diffusion head is a
content-tier variant, not a standalone model). Asserting at
arg-parse time: `--content-diffusion` requires `--per-tier-heads`
and is mutually exclusive with `--per-tier-content-mos-k > 0`.

## Phase plan

| phase | scope | wallclock | gate |
|---|---|---|---|
| 0 | Diagnostic: re-read Phase 3 mos4 prodlike artefacts (assuming refute). Confirm which of the three explanations above is consistent with the metrics: router posterior on content-acc holdouts; MoS gate entropy histogram across epochs; per-component utilisation. If diagnosis points to **router saturation** rather than **head bottleneck**, prefer the cheap retest `--per-tier-mos-entropy-lambda 0.01` at mini (1 hr) before committing to diffusion. | 1-2 hr analysis + (optional) 1 hr mini retest | No hard gate. Anchors whether to proceed to Phase 1. |
| 1 | Implementation behind `--content-diffusion` (default OFF). New files: `heads_diffusion.py`, `losses_diffusion.py`, `diffusion_sampler.py`. Touched files (minimal one-branch edits): `heads.py`, `lightning.py`, `args.py`. New unit tests under `tests/train/`: `test_diffusion_head_forward.py` (shape + numeric checks; verify `[MASK]` token correctness), `test_diffusion_loss.py` (cosine-schedule sampling determinism under seed), `test_diffusion_sampler.py` (block-level unmask on toy V_c=8). Lint + `run_tests.sh` clean. Smoke train on the smoke-tier corpus (7 SIDs) completes one epoch without NaN. | 2-3 days impl | Tests pass; smoke training one-epoch clean. |
| 2 | `content_diffusion_mini_body_large` A/B at mini, body=large, 60 max-epochs, **3 seeds** (variance bound). Arms: `content_diffusion_T8` (target), `baseline` (per_tier_heads_mos4 -- the best-known mini config, since plain-CE baseline is already established as the floor). Audits via existing `audit_checkpoint_per_class.py` + `prompt_conditioning_audit.py` + `loop_detection_audit.py` on each best ckpt, at both T_sample=0.5 and T_sample=0.7. Block-level sampler used for audit generation. | ~30-40 min/arm x 3 seeds x 2 arms = **3-4 hr** sequential | (1) content acc on `eval` >= mos4 baseline (3-sigma floor on 3-seed std); (2) `router_accuracy` >= 0.7 unchanged; (3) `loop_collapse_rate` at T_sample=0.5 <= mos4; (4) `diversity_ratio` > mos4 baseline + 1 sigma; (5) `content_over_structural` ratio not regressed; (6) (new) `block_diffusion_step_KL`: KL divergence between successive denoising steps remains > 0.05 nats at step 1 (i.e., the sampler is not collapsing to a deterministic answer by step 1). Refute if any of (1)-(5) fail; (6) is observational. |
| 3 | `content_diffusion_prodlike` at prodlike tier, body=canonical, 60 max-epochs, 1 seed (compute budget). Arms: `content_diffusion_T8` target, baseline = best of {mos4, plain CE} at prodlike. Eval suite: `eval_a` + 8 `eval_b_*` families. Audits as Phase 2. | ~10-14 hr/arm x 2 arms = **20-28 hr** sequential | (1) `eval_a` content acc >= mos4 prodlike result AND >= 0.14 (apush4x ceiling); (2) >= 5 of 8 `eval_b_*` families show non-zero content acc lift OR maintain mos4 lift; (3) `loop_collapse_rate` <= mos4; (4) `diversity_ratio` > 1.2 on real-vs-random; (5) no structural regression > 1 sigma. |
| 4 | If Phase 2 refutes -> write refuted entry `data/refuted/content_diffusion.md`; the only remaining queued path is data-scale (melody-transfer augmentation). If Phase 3 refutes -> write refuted entry; same fallback. | - | - |

**Total wallclock from approval to prodlike verdict:**
- Phase 0: 1-3 hr.
- Phase 1: 2-3 days agent work.
- Phase 2: 3-4 hr wallclock.
- Phase 3: 20-28 hr wallclock.
- **End-to-end:** 3-4 days assuming Phase 2 passes.

## Decisions taken (rationale)

1. **D3PM absorbing-state, not generic categorical D3PM.** Absorbing
   state (single `[MASK]` token) is simpler to implement, lower
   memory (no Q transition matrix), and matches the MaskGIT literature
   that has shown the strongest results on discrete sequence
   generation. Generic D3PM (uniform / Gaussian-projection transitions)
   doubles the implementation surface for unclear benefit at our scale.

2. **T=8 denoising steps default.** MaskGIT showed diminishing returns
   past T=8 for vocabularies of this size. T=4 is the floor (less than
   that and the schedule degenerates to one-step prediction); T=16+ is
   tunable in Phase 2 if a single-step ablation shows the per-step
   distribution is unsaturated.

3. **Content tier only.** Structural / mid / zero positions keep CE.
   The bottleneck framing (per `multi_modal_objective_design.md`) is
   specific to content-tier multi-modality; diffusing structural
   positions would burn parameter budget on positions that CE already
   handles well.

4. **Body unchanged.** No body-level modifications (no cross-attention
   on `x_t`, no extra layers). The body is the shared feature extractor
   and must work for the structural / mid / zero CE heads too. Adding
   diffusion-specific body modifications would make this design
   incompatible with the in-place per-tier infrastructure.

5. **3 seeds at Phase 2.** Same rationale as `per_tier_heads_design.md`:
   the InfoNCE refutation was single-seed and noise-bound. Phase 2 at
   mini costs ~3-4 hr; cheap insurance.

6. **Block-level sampler used for audit, per-token sampler used for
   unified-posterior decode.** Two paths because the existing predict
   CLI and audits expect a per-token decode loop. The block-level
   sampler is what the design promises; the per-token path is a
   compatibility shim so the existing inference scripts work without
   rewrites. Eventual landing is to make block-level the default
   on the predict host.

## Risk + non-goals

- **Mask-token leakage.** The `[MASK]` token must never appear in the
  main vocabulary nor in dataset tokenisation. Pinned by an assertion
  in `DiffusionContentHead.__init__` and a test that decodes a
  generated stream and confirms `[MASK]` count is zero.
- **Body bias against masked positions.** The body sees the
  *unmodified* sequence (masking happens inside the head). This means
  the body is never trained on masked input -- consistent with the
  per-token decode path. Risk: at the block-level sampler, the body
  is still being called on the *current partial unmask*, which means
  intermediate body activations differ from training. Mitigation: at
  block-level inference, the body sees the partially-unmasked content
  + structural / mid / zero context; the diffusion head conditions on
  both `h` and `x_t`, so the head learns to compensate for body
  activations on a partially-known sequence. Validated empirically
  in Phase 2 by checking that block-level audit accuracy matches
  per-token unified-posterior audit accuracy within 1 sigma.
- **One-step head fusion miscalibration.** The unified posterior uses
  `log_p(x_0 | h, [MASK], t=T)` as if it were the per-token softmax.
  This is the maximally-uncertain distribution; tier-router fusion
  may down-weight the content head as a result. Mitigation: report
  router marginal on content positions in Phase 2 metrics; if
  consistently < 0.1, calibrate the diffusion head fusion with a
  temperature-rescaling factor learned in Phase 2.
- **Step-count blowup at predict host.** Each block costs T body
  forwards. At block_size ~ 64 content positions, this is 8x body
  forwards over 64 positions = 0.125x per-token cost -- net win.
  At block_size ~ 8, it is 1x per-token cost -- net wash. Mitigation:
  the block-level sampler operates on the largest contiguous span
  of content-tier positions, not on individual tokens.
- **Mode-collapse at T=8.** A diffusion head can collapse to single-
  mode predictions if the body is dominant-mode-confident. The
  entropy regulariser (`--content-diffusion-entropy-lambda`) is the
  hedge; the Phase 2 KL-between-steps metric is the early-warning
  signal.
- **Not a fix for data scale.** Orthogonal to melody-transfer
  augmentation. If both diffusion and augmentation refute, the
  remaining queue is empty and the project hits the "data scale or
  bust" decision point.

## Non-goals

- No body-level changes.
- No structural / mid / zero head changes.
- No predict-host runtime optimisation beyond what falls out of
  block-level sampling.
- No multi-modal *structural* prediction (out of scope per the
  bottleneck framing).
- No swap of the existing per-tier flag default (per_tier_heads
  remains opt-in until and unless Approach C lands at prodlike).

## Acceptance criteria for advancing from this design

Reviewer (user) approves:

1. The six decisions above (D3PM absorbing-state, T=8 default,
   content-tier only, body unchanged, 3 seeds at Phase 2,
   two-path inference).
2. The total wallclock envelope (~3-4 days end-to-end).
3. The split into new files (`heads_diffusion.py`,
   `losses_diffusion.py`, `diffusion_sampler.py`) with minimal
   one-branch edits to `heads.py` + `lightning.py` + `args.py`.
4. The dependency on Approach C Phase 3 refutation as the trigger.
   Implementation does not start while Phase 3 is in flight or has
   passed.

On approval: Phase 0 launches **only after** the in-flight
`per_tier_heads_prodlike --only-arm per_tier_heads_mos4` reports a
verdict. If Phase 3 passes, this design is shelved (logged under
`data/refuted/` as "superseded by Approach C PASS" rather than
refuted, since it was never tested). If Phase 3 refutes, Phase 0
diagnostic runs, then Phase 1 implementation.

## References

- `multi_modal_objective_design.md` -- original B->C->A framing.
- `per_tier_heads_design.md` -- Approach C design; the
  infrastructure this builds on.
- `data/refuted/per_tier_heads_mos_revisited.md` -- Phase 2 re-open
  evidence; the foundation for whether to escalate.
- `data/refuted/contrastive_infonce_auxiliary.md` -- Approach B
  refutation; the failure mode this aims to bypass.
- Austin, J., Johnson, D., Ho, J., Tarlow, D., van den Berg, R.
  (2021). "Structured Denoising Diffusion Models in Discrete
  State-Spaces." NeurIPS. -- D3PM reference.
- Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W. T. (2022).
  "MaskGIT: Masked Generative Image Transformer." CVPR. --
  block-level iterative unmasking + confidence-greedy reveal.
- Yang, Z., et al. (2018). "Breaking the Softmax Bottleneck."
  ICLR. -- MoS reference (what this would replace on the content
  head).
