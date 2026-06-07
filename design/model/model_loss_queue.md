# Model / loss experiment queue

**Status:** Reference index (decision tree retained; framing closed). The
B→C→A model-side arc is **done**: Approach C (`per_tier_heads_design.md`) and
Approach A (`content_diffusion_design.md`) both **refuted at prodlike** (see
`../data/refuted/`). The content win came **tokenizer-side** (`full_macros` /
FREQ_TRAJ, 2026-05-25) — not from any queue item here. Model-side bets were
re-tested on the new tokenizer in the 2026-05 re-arc; triage reproduces the
refutations (see `../AGENTS.md` STAGE 1 progress). Each entry is a thumbnail;
promote on approval.

**Learnability framing.** These are model-side (loss/head) items; the refutations confirm the lever is tokenizer-side representation learnability ([`learnability_token_ordering_theory.md`](../references/learnability_token_ordering_theory.md)) — keep model-side bets ranked below tokenizer-side ones.

## Why this doc

The B → C → A framing in `multi_modal_objective_design.md` is now
two-deep: B refuted, C in flight, A drafted. If both C and A refute
on metrics we hit the project's "data scale or bust" decision point,
which is a major branch in research strategy. Having the back-pocket
plans queued in concrete form makes that decision less expensive --
we don't cold-start when the verdict lands.

All entries below are **research bets, not certainties**. Promotion
order favours cheapest impl-and-mini-A/B first.

## Queue (cheapest first)

### 1. Router-entropy regularisation retest (mini)

**Scope.** Re-test `per_tier_heads_mini_body_large mos4` with
`--per-tier-mos-entropy-lambda 0.01`. The flag exists; current
default is 0.0. The hypothesis: greedy decode collapses at T=0
because the router posterior saturates; positive entropy lambda
pulls the router toward a more balanced distribution, possibly
making greedy work without sampling. Cheap (~1 hr wallclock).

**Trigger.** Any time. Useful as a single-arm sanity check before
escalating to harder bets. Already enumerated in
`per_tier_heads_mos_revisited.md` "Open questions".

**Promote when.** Phase 3 prodlike mos4 refutes AND the audit shows
router saturation as the dominant failure mode (vs mixture collapse
or softmax bottleneck).

### 2. Cluster-conditional content head

**Scope.** Replace the MoS K=4 content head with a hierarchical
head: (a) Linear(d → C) predicting an audio-equivalence cluster id
(C ~ 256, computed offline by clustering content-tier tokens by
their pyresidfp render fingerprint); (b) per-cluster Linear(d →
|V_c|/C) restricted to tokens in that cluster. Two-stage decoding
at inference: sample cluster, then sample token within cluster.

**Hypothesis.** The bottleneck is "model can't pick the right
specific note among acoustically-equivalent alternatives". Cluster-
first prediction lets the model commit to a region of acoustic
space without being penalised for picking the wrong specific
representative.

**Files.** New `heads_cluster.py`, `losses_cluster.py`. Same
plug-in shape as the diffusion design. Cluster index built offline
(`profile/build_content_clusters.py`, ~1 hr fogbank).

**Cost.** Phase 1 impl ~2 days. Phase 2 mini ~3 hr. Phase 3
prodlike ~20-28 hr. Same envelope as the diffusion design.

**Promote when.** Diffusion phase 3 refutes; OR diffusion phase 2
audit shows distribution-shape is fine but sampler picks
acoustically-irrelevant alternatives.

### 3. Targeted contrastive (cross-composer negatives)

**Scope.** Re-attempt the InfoNCE direction with the failure mode
the original refute identified: random distractors over the full
32K vocab were dominated by trivially-out-of-tier samples,
diluting the within-content discrimination signal. New version
samples distractors from content tokens observed at similar
structural-context fingerprints in OTHER composers (using the
existing structural index). Same loss shape as
`content_contrastive_loss` already in `losses.py`, but with
informed negative-sampling.

**Hypothesis.** Cross-composer targeted negatives teach the model
"don't pick this composer's favourite filler when context implies
a different style", which is the specific generalisation failure
we observe (eval_b composers get the train composer's idioms).

**Files.** New `losses_targeted_contrastive.py`. Reuses the
structural index already built.

**Cost.** Phase 1 impl ~1-2 days. Phase 2 mini ~3 hr. Phase 3
prodlike ~10-14 hr.

**Promote when.** Diffusion or cluster head refutes AND
eval_b_*-vs-eval_a gap is the dominant signal (cross-composer
issue, not within-distribution issue).

### 4. Frame-level structured prediction

**Scope.** Predict the entire next FRAME (the cluster of writes
ending in a FRAME marker) as a structured object, not as a
sequence of tokens. Frame head emits a fixed-length vector
(register state + voice ordering + delay), trained via a
per-register CE plus a soft delay-CE. Inference samples a full
frame in one step.

**Hypothesis.** Per-token CE is mis-factored: many "wrong" content
predictions are actually the same musical decision expressed in a
different write order. Frame-level prediction collapses the
permutation-equivalence quotient at training time.

**Files.** New `heads_frame.py`, new tokenisation pass to extract
frame boundaries (already implicit via FRAME marker), new sampler.
Significant impl surface — most ambitious of the four.

**Cost.** Phase 1 impl ~3-5 days. Phase 2 mini ~5 hr. Phase 3
prodlike ~20-28 hr.

**Promote when.** All three above refute AND audio audition
consistently shows "right notes wrong order".

### 5. Data-scale: melody-transfer augmentation Phase 0

**Scope.** Land Phase 0 of `preframr-aug:design/melody_transfer_augmentation_design.md`
on the existing prototype. Generates A⊗B augmented dumps and
measures whether expanded corpus moves val_acc on a small A/B at
mini.

**Hypothesis.** The bottleneck is data scale (we have ~5K SIDs at
prodlike); augmentation gets us to ~20-50K composite SIDs without
new HVSC ingest.

**Cost.** Phase 0 is the smoke audit (~half day fogbank). Phase 1
A/B at mini ~3 hr.

**Promote when.** Architectural bets above have refuted AND the
remaining hypothesis is "not enough data". Per AGENTS.md OVERRIDING
priority order, this is favoured *less* than tracker-authoring
priors but is the obvious endpoint if every other bet refutes.

## Anti-queue: things explicitly NOT to re-attempt

Per `multi_modal_objective_design.md` and the refuted registry:

- **Plain InfoNCE with random distractors.** Refuted. Re-open only
  with the targeted-negatives variant (queue item 3).
- **Token-juggling macros.** Refuted broadly. Don't add a new
  encoder-level pass and call it a generalisation bet.
- **Static class-weighted CE.** Refuted (`weighted_token_loss`,
  `learnable_class_loss`). Don't add another tier-weight tuning
  knob.
- **Approach D (DPO-style sequence energy).** Refuted in design
  per the multi-modal-objective umbrella: weak per-step gradient,
  expensive inference. Re-open only with a fresh decoding-time
  story.

## Branching summary

```
Approach C Phase 3 prodlike (in flight)
├── PASS → land per_tier_heads as default; cross-engine audits;
│         shelve the queue below (or run them as encore experiments)
└── REFUTE
    ├── Audit shows router saturation → queue item 1 (cheap retest)
    │   ├── PASS → land + ship
    │   └── REFUTE → fall through to next branch
    ├── Audit shows mixture collapse / softmax bottleneck → diffusion
    │   ├── PASS → land + ship
    │   └── REFUTE
    │       ├── Audio shows acoustic-class confusion → queue item 2
    │       ├── Audio shows cross-composer style bleed → queue item 3
    │       ├── Audio shows permutation issues → queue item 4
    │       └── None of the above clean → queue item 5 (data scale)
    └── Audit ambiguous → fall through to diffusion as the default
        next bet
```

The branching is a decision tree, not a strict sequence -- the
audit evidence on the failure mode should pick the next item, not
the queue order.
