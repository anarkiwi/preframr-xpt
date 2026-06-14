# Free-running pathology — remediation ladder for "good first token, poor afterward"

**Status: LIVE — Tier-0 CONFIRMED `exposure_bias` on the v2 baseline (2026-06-14).** The go/no-go
fired: `free_running_gap_audit` on `/scratch/tmp/v2_atoms_baseline.ckpt` (8 held-out blocks, result
`data/audit/v2_baseline_freerun_gap.json`) reads **`exposure_bias`** — teacher-forced accuracy is
**long-horizon healthy** (read-A by distance-from-start rises then plateaus ~0.50–0.535 out to 31k
atoms, so the model genuinely *uses* long context and is NOT short-horizon/context≈0), but
free-running **collapses within ~4 tokens** (read-B: free-run ≈ TF at horizon 1, then drops to ~0.04
content acc while TF holds ~0.5; gap widens to ~0.4–0.58). The pathology was assumed, now observed —
this is the live remediation arc. *(Bears on the context-length null: the model uses long context, so
that null is the matched-epochs step-confound, not short effective context — see
`../encoding/context_length_experiment.md`.)* It is the
remediation counterpart to [`generation_quality_gate.md`](generation_quality_gate.md): the gate
*measures* the pathology, this doc is the prioritised ladder of *fixes* the gate's verdict selects
among. Every arm here is itself gate-promoted (content tier + quality gate), triage-first for
representation changes — nothing flips a default on this doc's say-so.

## The failure mode

Symptom: free-running (self-conditioned) generation is good for the first prediction(s) off a real
prompt, then degrades — drifts, loops, or collapses to the near-silent DELAY/empty-frame drone
already observed at unconstrained `temperature 1.0`
([`../references/framework_architecture.md`](../references/framework_architecture.md)). Equivalently:
the learned next-token map is **short-horizon / copy-dominated** — it predicts continuations of seen
material but cannot *sustain* novel structure once it leaves the prompt's repetitions.

**Why this is a different axis than the closed model-side arc (read before reaching for the
anti-queue).** The umbrella refutation
([`../refuted/multi_modal_objective_design.md`](../refuted/multi_modal_objective_design.md)) and every
bet in its anti-queue were fighting the **teacher-forced content ceiling** — a *single-step,
ground-truth-conditioned* quantity. This pathology is a **train↔inference mismatch / horizon**
quantity. They are nearly orthogonal: a checkpoint can read healthy teacher-forced content (current
v2 atoms-only: eval_a content **0.505**, AGENTS.md) while being a free-running disaster, because
teacher-forced top-1 is carried by induction-copy of the prompt and never exercises self-conditioning.
So the anti-queue does **not** close this space — it closes the *content-distribution-objective*
space. The historically-winning lever (representation + data) is still open and newly motivated here;
the only genuinely-unexplored objective axis is exposure bias (Tier 4).

**Measurement honesty.** Top-1 accuracy *cannot by itself* distinguish this pathology from
irreducible multi-modality: P6 ([`../references/encoding_principles.md`](../references/encoding_principles.md))
concedes absolute-onset targets cap ~0.51 even for a perfect n-gram, and Principle 4.2 deliberately
front-loads the *most-determined* token, so a "good-first / weaker-after" profile is the *expected*
shape even for a healthy model. The pathology is the **gap** between teacher-forced and free-running
as a function of horizon, scored distributionally — not the within-unit accuracy sawtooth.

## Mechanisms → fixes

| # | Mechanism | Primary fix (tier) |
|---|---|---|
| M1 | **Exposure bias** — trained teacher-forced, inferred self-conditioned; never learns to recover from its own errors | Tier 4 (DAgger-on-recanonicalised rollouts); Tier 0 (select against it) |
| M2 | **Multiplex fragility** — frame-major interleave makes per-voice lines long-range & position-unstable; one desync cascades | Tier 2 (lane de-mux, voice- then role-form) |
| M3 | **Mode-collapse under greedy / mis-calibrated sampling** on multi-modal content (low T → drone, high T → grammatical runaway) | Tier 1 (structure-aware sampling + budget caps) |
| M4 | **Copy-dominance** — induction-head circuit aces teacher-forced repeats, collapses with nothing to copy | Tier 3 (transplant/reduction augmentation breaks spurious binding) |
| M5 | **Compounding horizon** — ~85k atoms/tune; errors accumulate across the autoregressive chain | Tier 1 (finer-grained re-anchor chaining); already bounded per-window |

## Tier 0 — Confirm, quantify, and select against it (prerequisite)

Nothing below is tunable without a free-running yardstick at canonical scale. Today selection is
blind: checkpoint is `monitor='val_loss'` (teacher-forced), EarlyStopping effectively never fires
under schedule-free, so the deployed checkpoint is best-teacher-forced-loss at `max_epochs` — chosen
without ever looking at free-running behaviour.

1. **Quick confirmation (hours, reuses existing artifacts).** `validation_step` already returns
   `{preds, gt}`. (a) Bucket teacher-forced accuracy by distance-from-block-start — flat/rising =
   long-horizon healthy, decaying = short-horizon. (b) On one held-out block, greedy free-run from
   the same prompt and overlay against teacher-forced at each position. TF flat + free-run diverging
   ⇒ exposure-bias/distribution (pathology confirmed); TF *also* collapsing ⇒ effective context ≈ 0
   (worse, and a different fix). This is the cheap go/no-go for opening the rest of the ladder.
2. **Land [`generation_quality_gate.md`](generation_quality_gate.md).** It already measures the
   pathology directly: `loop_collapse_rate`, `invalid_rate`, decoded-fraction, distributional
   write-domain metrics, and — for M4 — the **memorization audit** (novel-fraction at n=8/16, longest
   verbatim training match). This is the standing instrument; everything below is scored on it.
3. **Free-running-aware checkpoint selection** (new — not in the gate doc). Don't change the loss;
   change *which* checkpoint ships. Keep `save_top_k=K` on `val_loss`, then choose the deployed
   checkpoint among the K by the quality-gate verdict (or veto gate-failing ones). Cheap (post-train,
   over a handful of checkpoints), anti-queue-safe (selection, not objective). Standing diagnostic:
   log the **teacher-forced − free-running content gap vs. horizon** so any fix's effect is visible.

*Wiring:* items 2–3 touch `run.py`/`report.py`/the audit stage — land them in a window with **no run
in flight** (the mid-run-edit rule, per
[`../measurement/generalization_metric_tracking_design.md`](../measurement/generalization_metric_tracking_design.md)).

## Tier 1 — Decoding-time (no retrain; attacks M3, M5)

Cheapest leverage, targets the *observed* collapse, costs zero training. All scored on the Tier-0
gate's sampling grid (sampling is "the gate's subject").

- **Structure-aware sampling, not a global temperature.** Resolve the bind (low T → drone, high T →
  runaway) by sampling *per tier*: near-greedy on structural/grammar atoms, calibrated sampling
  (top-p / min-p / locally-typical) on content atoms. Add as configs to the gate's grid; it lowers
  single-reference acc, which is exactly why Tier 0 lands first.
- **Hard frame/DELAY budget caps + loop penalty at decode.** `StreamState` already tracks the IRQ /
  `frame_budget`; use it to forbid runaway empty frames, plus an n-gram/repeat penalty for the drone
  mode. Surgical band-aid for the specific failure.
- **Finer-grained decode-and-recompile re-anchoring.** Shorten the chaining interval of
  [`long_range_structure.md`](long_range_structure.md) (its v2 loop-escape — raise T / re-anchor at a
  fresh KEYFRAME on detected tail cycle — is the same lever). **Limit:** this bounds *compounding
  across* windows (M5); it re-grounds exact state but preserves an already-drifted slice — it does not
  fix degeneration *within* a window.
- **Port the constrained-decode mask to the event grammar** (the open item in
  [`../references/framework_architecture.md`](../references/framework_architecture.md);
  `constrained_decode.StreamState` speaks parse-domain). Prevents *grammatical* collapse only —
  necessary, not sufficient (a grammatical stream can still be musically dead; `invalid_rate`
  measures the miss until it lands).

## Tier 2 — Representation (the winning lever; attacks M2)

- **Lane de-mux** ([`../encoding/lane_demux_hypothesis.md`](../encoding/lane_demux_hypothesis.md)) is
  the prime fix for the prime suspect. The frame-major interleave is the most fragile thing under
  self-conditioning — one bad voice-tag/DT desyncs the whole multiplex. Voice-contiguous reordering
  makes per-voice prediction short-range and position-stable; **role-form (accompaniment-before-melody,
  harmony-conditions-melody +0.294 bits)** is the real win but needs the role segmenter. **Sobering
  prior (state it):** pure reorder recovered only ~5% before (`sequence_order_normalization` refuted),
  and the doc itself warns "de-mux helps is not a safe prior."
- **New trigger wiring.** The AGENTS "IF CONTENT NOT LEARNED" branch keys lane-demux off a
  *teacher-forced* content number — which this pathology can pass while free-running collapses. Add a
  complementary trigger: **"IF free-running gap large despite content learned → de-mux."** Tier 0 is
  what makes that trigger observable.
- *Gate (unchanged from lane-demux doc):* `learnability_triage` (seq_len 8192, window mode) on
  frame-major vs voice-major vs role-major **before** any training; then one canonical A/B read on
  `NI_*` per-op content tier **+ no-regression on other lanes + the quality gate**; byte-exactness via
  `encode(verify=True)` on the inverse-permuted stream.
- Keep shrinking the dependency horizon (Principle 3) — the radix per-lane polish (live ~11–12%,
  AGENTS.md) and KEYFRAME-density variants reduce the latent the model must carry across a free-run.

## Tier 3 — Data (attacks M4)

- **Transplant + reduction augmentation**
  ([`transplant_augmentation_design.md`](transplant_augmentation_design.md); reduction in
  [`prompt_interface_design.md`](prompt_interface_design.md)). If the model copies because the corpus
  rewards copying, break the melody×timbre spurious binding so generative structure beats memorised
  spans; the reduction pairs (melody-prefix → full-texture) train the generate-from-a-seed regime
  directly. Register-domain splice + `encode(verify=True)` = zero pipeline change. *Gate:* dosage A/B
  on eval_b content + the quality gate's memorization audit (does novel-fraction rise?), train-split
  leakage rule per the transplant doc. Impl home: preframr-aug.

## Tier 4 — Training objective (gated LAST; anti-queue-aware; attacks M1)

Model-side has a losing record, so only if Tiers 0–3 don't close it. Two options are genuinely
unexplored because they attack exposure bias, not content distribution:

- **DAgger with the exact re-canonicalisation oracle** — the one objective-side bet worth funding, and
  the "fresh decoding-time story" the anti-queue explicitly left open against the refuted Approach-D
  (DPO/energy). Roll the model out, **decode → re-canonicalise the rollout into a valid SID state**
  (the same operation chaining already performs, fidelity-checked), and train it to continue from
  *that*. Because re-canonicalisation is exact, you teach recovery from *plausible* self-generated
  states — not token garbage, which is what blind scheduled sampling cannot guarantee.
- **Scheduled sampling / teacher-forcing decay** — textbook exposure-bias fix, never tried here;
  known-unstable and awkward with KV-cache training. Lower priority than DAgger.

## Do NOT re-attempt (anti-queue alignment)

All refuted at the content ceiling — re-opening any here would re-fight the wrong axis: per-tier /
MoS heads + router-entropy, InfoNCE with random distractors, discrete-diffusion content head,
cluster-conditional content head, static class-weighted CE (`weighted_token_loss` /
`learnable_class_loss`), per-voice auxiliary supervision, naive DPO/energy sequence ranking. **And
not beam search** — it worsens open-ended degeneration and busts the Orin envelope.

## Diagnostic tooling (realized)

The Tier-0/1/2 reads are implemented as CI-tested audits (pure-logic core + GPU-host CLI), in
`preframr_experiments/audit/` (indexed in its README):

- `free_running_gap_audit.py` — the Tier-0 go/no-go: TF-vs-free-running gap by horizon + verdict;
  `--verify-cache` rules out a KV-cache/position bug masquerading as the pathology.
- `copy_novel_audit.py` — copy-vs-novel accuracy split (is the TF number real or induction-copying).
- `effective_context_audit.py` — accuracy vs truncated context k → the model's real horizon
  (disambiguates the `short_context_or_bug` verdict).
- `event_position_audit.py` — accuracy by atom role / offset-in-event (strips the by-design
  front-loading confound, P4.2).
- `voice_interleave_audit.py` — accuracy by voice + interleave gap (the Tier-2 lane-demux trigger).
- `calibration_audit.py` — ECE / over-confidence / entropy (informs the Tier-1 sampling regime).
- `memorization_audit.py` — novel-fraction + longest verbatim match (the quality-gate memorization
  check; output-side copy-dominance, Tier 3).

All single-checkpoint, post-hoc reads; none is a promotion gate on its own — they characterise the
pathology and point to the tier. Wiring them as runner stages is deferred to the
[metric-tracking design](../measurement/generalization_metric_tracking_design.md) (mid-run-edit rule).

## Promotion & sequencing

- **Promotion rule (inherited):** any default flip (encoding, sampling regime, conditioning) requires
  the content-tier verdict **and** the quality gate; representation arms triage-first; cross-tokenizer
  comparisons position-matched or in bits/canonical-atom; calibrate-then-floor for any new threshold
  (as [`../measurement/generalize_min_val_acc_floor_design.md`](../measurement/generalize_min_val_acc_floor_design.md)).
- **Recommended path:** Tier 0 (confirm + gate + free-running-aware selection) → Tier 1 (structure-aware
  sampling + budget caps — free, immediate) → Tier 2 voice-form lane-demux triage. Highest leverage
  for near-zero starting cost, respecting every banked refutation. Tiers 3–4 only if the gap persists.

## Lifecycle

Conditional on the Tier-0 confirmation. If the quick diagnostic shows **no** material gap (free-run
tracks teacher-forced), refute this doc and record the evidence stub (the pathology was assumed, not
observed). If confirmed, this becomes the live remediation arc; land Tier-0 wiring, then promote each
tier's winner per the rule above, moving shipped pieces to `landed/`.
