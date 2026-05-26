# Design notes index

Tracked design docs for ongoing + queued work. Refuted hypotheses
also have one-paragraph stubs at
`preframr_experiments/data/refuted/<exp>.md` (this repo) — start
there before reopening any previously-rejected direction.

## Status legend

- **In flight**: spec + impl shipped; experiment running or pending
  verdict.
- **Drafted, pending trigger**: design approved or proposed; impl
  blocked on a specific event (usually another experiment's verdict).
- **Pending impl**: design approved; implementation queued.
- **Deferred**: design reviewed; implementation deferred until a
  specific condition is met (typically generalization approach
  lands).
- **Refuted**: hypothesis tried and rejected; see the refuted
  registry. Doc retained for reference.
- **Landed**: moved to [`landed/`](landed/) — reference only.

## In flight / drafted

| Doc | Summary | Status |
|---|---|---|
| [`motif_pass_design.md`](motif_pass_design.md) | Corpus-mined, per-block, lossless motif pass (preframr-tokens 0.20.0): mines a cross-composer motif dictionary, collapses motifs into `MOTIF_OP` atoms (loss-tier zero; expanded byte-exact on decode). ~11.4% fewer tokens at deployment vocab; the A/B tests **learnability**, not just compression. | In flight — shipped (tokens 0.20.0); mini A/B (`motif_mini_body_large`) queued, pending GPU after STAGE 2 |
| [`tokenization_vs_music_llms.md`](tokenization_vs_music_llms.md) | Critical comparison of preframr's register-event + macro Unigram tokenization vs symbolic/MIDI, audio-codec, and VQ paradigms; argues the content ceiling is tokenization-induced, motivating compound tokens / acoustic-equivalence / augmentation. | Reference (positioning) |
| [`music_llm_landscape_and_fail_fast_plan.md`](music_llm_landscape_and_fail_fast_plan.md) | Cross-LLM idea survey + ranked cheap fail-fast probes (composer token, vocab pruning, audio-equivalence Phase 0, compound-token prototype). | Reference (strategy) |

## Representation axis (the active leverage)

The confirmed win is tokenizer-side (`full_macros`); these are the queued
representation/tokenization bets that follow it.

| Doc | Summary | Status |
|---|---|---|
| [`compound_token_design.md`](compound_token_design.md) | Approach D: compound-token tokenizer + parallel-attribute heads (CompoundWord, Hsiao et al. 2021 / OctupleMIDI). Strategic pivot from the refuted per-token content-head architectures to a multi-attribute-per-token reorganization. | Draft, design review pending |
| [`audio_equivalence_normalization_design.md`](audio_equivalence_normalization_design.md) | Tokenizer-side data normalization that collapses `(op, reg, val)` tuples producing perceptually-equivalent SID output into canonical forms (parameter-space collapse). | Draft 2026-05-23 |

## Queued model/loss bets

| Doc | Summary | Status |
|---|---|---|
| [`model_loss_queue.md`](model_loss_queue.md) | Back-pocket queue: router-entropy retest, cluster-conditional content head, targeted cross-composer contrastive, frame-level structured prediction, melody-transfer augmentation. Branching-decision tree for picking the next bet from audit evidence. | Reference |
| [`per_voice_aux_supervision_design.md`](per_voice_aux_supervision_design.md) | Per-voice auxiliary classification heads on the body's hidden state. | Scoping |
| `preframr-aug:design/melody_transfer_augmentation_design.md` | Three offline corpus-expansion families: verified-inaudible macro perturbation, within-song voice permutation, cross-song melody/instrument transfer. | Moved to preframr-aug (tooling under `preframr_aug/`; voice-permutation helper + spec stay here) |

## Pipeline coverage holes

| Doc | Summary | Status |
|---|---|---|
| [`generalize_min_val_acc_floor_design.md`](generalize_min_val_acc_floor_design.md) | Calibrate `GENERALIZE_MIN_VAL_ACC` floor as 2/3 × median val_acc once 2-3 canonical baselines run. | Pending impl |
| [`start_seq_rotation_audit_design.md`](start_seq_rotation_audit_design.md) | `predict_load` hard-codes rotation 0; 50% rotations unreachable at max_perm=2. Flat-indexing fix. | Pending impl |

## Framework follow-ups

| Doc | Summary | Status |
|---|---|---|
| [`audio_driver_split_design.md`](audio_driver_split_design.md) | preframr-audio: split the 1499-LoC `audio_driver.py` into a `render.py` core (what fidelity/fingerprint/batch/augmentation import) vs `live.py` (alsa/ASID/MIDI/CLI, all `# pragma: no cover`); back-compat shim keeps imports working. Companion to the landed facade + in-memory equivalence + parallel batch (`feat/audio-augmentation-scaling`). | Drafted, pending review |
| [`runner_iteration_efficiency_design.md`](runner_iteration_efficiency_design.md) | `preframr-xpt` per-run overhead: (1) **LANDED** — key the dataset cache on parse/tokenize-affecting `extra_cargs` (train-only flags denylisted) so macro A/Bs stop force-disabling it; (2) **LANDED (runner-only)** — stop copying 2.7 GB of raw dumps per run via a symlink farm + read-only dump mount (no main-repo change; parse derives output paths by string); (3) drop the post-step chown container (run as runner uid). Resume/early-abort/parallel already deferred-designed. | #1+#2 landed; #3 pending |
| [`flag_stage_routing_design.md`](flag_stage_routing_design.md) | `FLAG_STAGES` registry + `add_stage_args` to enforce stage-aware flag forwarding (parse vs tokenize vs train). | Pending impl |
| [`orin_inference_optimization_design.md`](orin_inference_optimization_design.md) | Predict-host throughput: vocab shrink + GPU-resident constrained-decode (Orin sits at 4% GPU util at predict today). | Pending impl |
| [`streaming_unembed_ce_design.md`](streaming_unembed_ce_design.md) | Stream `output(chunk) + ce_chunk` inside one checkpoint; eliminates 8.6 GiB chunk slab. Restores `batch_size=4` and original prodlike wallclock. | Pending impl |

## Cloud-rental prereqs (deferred)

Open only after generalization approach lands. See AGENTS.md "Multi-GPU rental decision".

| Doc | Summary | Status |
|---|---|---|
| [`auto_early_abort_design.md`](auto_early_abort_design.md) | Spec-declared `decision_rule` callable evaluated after each (arm, seed); writes refutation stub on falsification. | Deferred |
| [`max_parallel_arms_design.md`](max_parallel_arms_design.md) | `concurrent.futures` slot allocator with flock; refuses N>1 on single-GPU hosts. | Deferred |
| [`resume_design.md`](resume_design.md) | Per-stage `_resume.json` manifest for partial-run recovery. Dataset cache (landed in `preframr_experiments/base.py`) already addresses the parse+tokenize portion. | Deferred (partial coverage from dataset cache) |

## Refuted (reference)

The per-token content-objective arc — all rejected at the same ~0.13 eval_a
content ceiling; detailed "do not revisit without" conditions in
`preframr_experiments/data/refuted/`. Leverage proved to be representation, not
the output objective.

| Doc | Summary | Status |
|---|---|---|
| [`multi_modal_objective_design.md`](multi_modal_objective_design.md) | Umbrella framing of the per-token CE bottleneck on the multi-modal content tier. | Refuted (umbrella; B/C/A all rejected) |
| [`per_tier_heads_design.md`](per_tier_heads_design.md) | Shared body + 4 tier output heads + router; MoS-NLL on content head; uncertainty-weighted multi-task loss (Approach C). | Refuted at prodlike (router posterior saturates → outputs ignore prompt content) |
| [`content_diffusion_design.md`](content_diffusion_design.md) | D3PM absorbing-state discrete-diffusion content head (Approach A). | Refuted (sampling-side change didn't move the CE outcome) |
| [`cluster_conditional_content_head_design.md`](cluster_conditional_content_head_design.md) | Cluster-conditional content head (queue item 2). | Refuted (same ceiling, diversity ~1.0–1.2) |

## Process / scope

| Doc | Summary | Status |
|---|---|---|
| [`repo_focus_cleanup_scope.md`](repo_focus_cleanup_scope.md) | Plan to keep `preframr` (main) = core framework, moving experiment orchestration + audits + tier data + design docs to `preframr-xpt`. | Largely executed — see [`landed/experiments_extraction_design.md`](landed/experiments_extraction_design.md) |

## Landed (archived)

Reference docs whose implementations are in HEAD. Index:
[`landed/README.md`](landed/README.md). Includes the
preframr-{audio,tokens,experiments} extractions, the
train/inference container split, the per-tier infrastructure
fragility fixes, the corpus / HVSC / Orin audit landings, the
**FREQ_TRAJ unified-oscillation primitive** (shipped tokens 0.16/0.17), and
the **tokenizer profiling tooling** (shipped tokens 0.20.0).

## Conventions

- **Status header.** Every doc starts with a one-line status bullet
  (Drafted / In flight / Landed / Deferred / Refuted). Edits update
  the status before changing body content.
- **Decision rules.** Mirror prior precedent: 3σ-on-val_acc to flip
  default; capacity-attenuation refuses if prodlike Δ < ¼ × mini Δ;
  per-Eval-B-* breakouts confirm cross-composer transfer.
- **Refuted alternatives.** Move detailed evidence to
  `preframr_experiments/data/refuted/<exp>.md`; the design
  doc retains a status header pointing there. Future re-design
  starts by reading the refuted entry's "do not revisit without"
  condition.
