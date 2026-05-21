# `contrastive_infonce_auxiliary` (InfoNCE on content tier) — REFUTED

**Status:** Refuted at mini sweep (single-seed lift within noise)
and at prodlike epoch 8 (content acc 0.0000). Design replaced by
`design/per_tier_heads_design.md` (Approach C).

## Hypothesis

Per `design/multi_modal_objective_design.md` Approach B: per-token
CE cannot express the multi-modal target distribution on content
positions. Add an InfoNCE-style auxiliary loss at each content
position: GT vs K random distractors from the vocab, train the
model to rank GT above all K. Combined with standard CE as
`L = L_CE + λ · L_InfoNCE`.

## Implementation

Landed in `preframr/train/model.py:233-249` (`_infonce_per_tensor`)
+ `model.py:252-268` (`content_contrastive_loss`); wired via
`--infonce-content-loss-weight λ` + `--infonce-distractors K` flags
in `preframr/args.py:172-180`. K random distractors per content
position sampled `torch.randint(V, ...)` over the full 32K vocab,
cross-entropy with GT as class-0 vs K distractor classes.

## Refutation

### Mini sweep (`contrastive_mini_body_large`, 2026-05-21)

Single-seed body=large mini, 60 max-epochs, three contrastive arms
+ baseline:

| arm | val_acc | content acc | structural acc | c/s ratio |
|---|---|---|---|---|
| baseline (30 ep best) | 0.0779 | 0.0008 | 0.2013 | 0.0042 |
| L0.05_K64 (30 ep best) | 0.0802 | 0.0043 | 0.1879 | 0.0228 |
| L0.1_K32 (30 ep best) | 0.0781 | **0.0001** | 0.1937 | 0.0007 |
| `content_floor_check` baseline (60 ep best, no InfoNCE) | — | **0.0063** | 0.4263 | 0.0147 |

`L0.1_K32` regressed below baseline content acc. `L0.05_K64`'s
"5× lift" (0.0008 → 0.0043) was *smaller* than the baseline-vs-
content_floor_check delta from 30 to 60 epochs (0.0008 → 0.0063)
— meaning the contrastive arm's headline lift was indistinguishable
from "the baseline catches up with longer training". Single-seed
across all arms; no variance estimate.

### Prodlike (`contrastive_prodlike`, 2026-05-21, stopped at epoch ~11)

Body=canonical (16L/768/2048), 60 max-epochs target,
`L0.05_K64`. Audit at best-epoch=8 ckpt:

| eval subset | content acc | structural acc |
|---|---|---|
| eval_a | **0.0000** | 0.2662 |
| eval_b_crisps | 0.0033 | 0.7466 |
| eval_b_daglish | 0.0001 | 0.3518 |
| eval_b_dobek | 0.0005 | 0.5775 |
| eval_b_follin | 0.0010 | 0.3528 |
| eval_b_marquis | 0.0000 | 0.1713 |
| eval_b_mibri | 0.0009 | 0.2796 |
| eval_b_wilson | 0.0024 | 0.3380 |
| eval_b_winterberg | 0.0023 | 0.1589 |

Pass criterion (eval_a content acc ≥ 0.14) missed by an infinite
factor. Stopped before completion to free GPU for Approach C.

## Design-level refutation (independent of empirical result)

The InfoNCE construct is internally inconsistent with the stated
bottleneck:

1. **Soft-classifier on (K+1) classes is the failure mode it
   claims to fix.** `multi_modal_objective_design.md` §B claims
   the contrastive term "teaches 'don't predict obvious wrongs'
   without forcing argmax on the GT token". The implementation
   computes cross-entropy with GT as class-0 — that *is* a softmax
   classifier pulling mass onto GT against K distractors. Same
   mode-seeking behaviour as the CE it augments.
2. **Easy-negative dilution.** K random distractors sampled
   uniformly over V=32K — ~22% are in-content-tier; ~78% are
   out-of-tier (structural / mid / zero) and trivially ruled out
   by local context. K=64 effective in-tier negatives ≈ 14.
3. **No temperature, no normaliser sharing.** Standard
   InfoNCE / NCE pairs with a learned or tuned temperature; here
   it's a raw 1-of-(K+1) softmax. Closer to "label-smoothed CE on
   a random vocab subset" than InfoNCE in the SimCLR / NCE-LM sense.

## Do not revisit without

- A contrastive construct that *spreads* mass across plausible
  alternatives rather than pulling mass onto GT — e.g. a
  distribution-shape loss (KL to empirical conditional, MoS-NLL,
  flow-matching). At that point the construct is no longer InfoNCE.
- Hard-negative mining (distractors drawn from the model's top-K
  predictions, or from a tier-matched vocab subset) — addresses
  the easy-negative dilution but not the soft-classifier
  internal-inconsistency.
- Empirical evidence on a multi-modal target benchmark (not SID)
  that InfoNCE outperforms CE for multi-modal next-token
  prediction with seed-variance bounds.

## Evidence

- Audit JSONs at `/scratch/tmp/audit/cmbl_baseline.json`,
  `cmbl_L0.05_K64.json`, `cmbl_L0.1_K32.json`,
  `content_floor_check.json`, `cprd_contrastive.json`.
- Report JSON at
  `/scratch/tmp/preframr_experiments/results/contrastive_mini_body_large/report.json`.
- Independent review chat 2026-05-21.

## Retained code

`content_contrastive_loss` + `_infonce_per_tensor` in
`preframr/train/model.py` retained as dead code behind
`--infonce-content-loss-weight 0` (default OFF). Keep for now;
may be useful as a benchmark target when Approach C lands.
Removal pending Phase 3 of `per_tier_heads_design.md`.
