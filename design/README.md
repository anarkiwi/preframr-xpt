# Design notes index

Organized by **research axis** (the project's priority order), not by lifecycle:
status is a per-row column so refuted/landed work sits next to the live work in
the same thread. Refuted hypotheses also have one-paragraph stubs at
`preframr_experiments/data/refuted/<exp>.md` (this repo) — read those before
reopening a rejected direction.

## How this index is organized (read before adding a doc)

**Five axes** (top-level grouping — mirrors AGENTS.md "Project goal" + priority).
Add a new doc under its **primary** axis; cross-link a secondary axis in prose if
it spans two:

1. **Generalization** (primary goal) — predicting unseen continuations across
   composers/engines: the content objective, representation/tokenization, the
   metrics that detect it.
2. **Correctness & fidelity** — round-trip/decode exactness, audio equivalence,
   eval-leak + version gates, coverage bugs.
3. **Efficiency & deploy envelope** — token budget, training memory/wallclock,
   predict-host (Orin) throughput.
4. **Runner / experiment infra & process** — the docker runner, resume/abort/
   parallelism, repo scope.
5. **Data & corpus** — tier definitions, engine-fingerprint eval sets,
   augmentation, dataset coverage.

**Status column + lifecycle.** Every doc starts with a one-line `**Status:**`
header, and edits update it before the body. Lifecycle:

- **Draft** / **Scoping** → **Pending impl** / **Drafted, pending trigger** →
  **In flight** (spec + impl shipped; running/awaiting verdict) →
- **Landed**: move the file to [`landed/`](landed/) and add a row to
  [`landed/README.md`](landed/README.md) (archive is grouped by *kind*). The
  axis table keeps an inline `Landed` row pointing at `landed/…` so the thread
  stays whole.
- **Refuted**: keep the doc in place, set status to `Refuted`, and add a "do not
  revisit without …" stub at `preframr_experiments/data/refuted/<exp>.md`.
- **Deferred**: reviewed but parked on an external condition (e.g. cloud rental).
- **Reference**: positioning/strategy docs with no single impl.

Directory stays **flat** (no per-axis subdirs — docs span axes and moves break
the relative `[..](..)` / `../data/refuted/` links). Grouping lives here, in the
index.

## 1. Generalization (primary goal)

The content-ceiling thread: the per-token content objective (refuted across the
board) → the pivot to **representation/tokenization**, where the confirmed win
lives (`full_macros`; FREQ_TRAJ landed).

| Doc | Summary | Status |
|---|---|---|
| [`motif_pass_design.md`](motif_pass_design.md) | Corpus-mined, per-block, lossless motif pass (tokens 0.20.0): collapses cross-composer motifs into loss-tier-zero `MOTIF_OP` atoms. ~11.4% fewer tokens at deployment vocab; the A/B tests **learnability**. | In flight (mini A/B `motif_mini_body_large` queued, GPU after STAGE 2) |
| [`compound_token_design.md`](compound_token_design.md) | Approach D: compound-token tokenizer + parallel-attribute heads (CompoundWord / OctupleMIDI). Multi-attribute-per-token reorganization. Also an efficiency bet (token budget). | Draft, design review pending |
| [`audio_equivalence_normalization_design.md`](audio_equivalence_normalization_design.md) | Tokenizer-side normalization collapsing `(op, reg, val)` tuples that render perceptually-equivalent SID output into canonical forms. | Draft 2026-05-23 |
| [`per_voice_aux_supervision_design.md`](per_voice_aux_supervision_design.md) | Per-voice auxiliary classification heads on the body's hidden state. | Scoping |
| [`generalize_min_val_acc_floor_design.md`](generalize_min_val_acc_floor_design.md) | Calibrate `GENERALIZE_MIN_VAL_ACC` as 2/3 × median val_acc once 2-3 canonical baselines run (the generalization gate). | Pending impl (env hook exists, default 0/off) |
| [`model_loss_queue.md`](model_loss_queue.md) | Back-pocket queue + decision tree for picking the next content bet from audit evidence. | Reference |
| [`tokenization_vs_music_llms.md`](tokenization_vs_music_llms.md) | preframr's register-event + macro Unigram tokenization vs symbolic/MIDI, audio-codec, VQ; argues the content ceiling is tokenization-induced. | Reference (positioning) |
| [`music_llm_landscape_and_fail_fast_plan.md`](music_llm_landscape_and_fail_fast_plan.md) | Cross-LLM idea survey + ranked cheap fail-fast probes. | Reference (strategy) |
| [`landed/unified_oscillation_primitive_design.md`](landed/unified_oscillation_primitive_design.md) | Unified `FREQ_TRAJ` op + `FREQ_NUDGE` — the representation win; now the live deployment tokenizer. | Landed (tokens 0.16/0.17) |
| [`multi_modal_objective_design.md`](multi_modal_objective_design.md) | Umbrella framing of the per-token CE bottleneck on the multi-modal content tier. | Refuted (B/C/A all rejected) |
| [`per_tier_heads_design.md`](per_tier_heads_design.md) | Shared body + 4 tier heads + router; MoS-NLL content head (Approach C). | Refuted at prodlike (router saturates) |
| [`content_diffusion_design.md`](content_diffusion_design.md) | D3PM absorbing-state discrete-diffusion content head (Approach A). | Refuted (sampling-side; no CE change) |
| [`cluster_conditional_content_head_design.md`](cluster_conditional_content_head_design.md) | Cluster-conditional content head (queue item 2). | Refuted (same ceiling, diversity ~1.0–1.2) |

## 2. Correctness & fidelity

| Doc | Summary | Status |
|---|---|---|
| [`start_seq_rotation_audit_design.md`](start_seq_rotation_audit_design.md) | `predict_load` hard-codes rotation 0; ≥50% of rotations unreachable at `max_perm>1`. Flat-indexing fix + coverage probe. | Pending impl |
| [`audio_driver_split_design.md`](audio_driver_split_design.md) | preframr-audio: split the 1499-LoC `audio_driver.py` into a `render.py` core (what fidelity/fingerprint/batch import) vs `live.py` (alsa/ASID/MIDI/CLI). | Drafted, pending review |
| [`landed/tokenizer_profiling_tooling_design.md`](landed/tokenizer_profiling_tooling_design.md) | Torch-free tokenizer profiling (`tokenizer_profile` + `audit_primitives` reductions). | Landed (tokens 0.20.0) |
| [`landed/audio_fidelity_helper_design.md`](landed/audio_fidelity_helper_design.md) | Shared render-and-compare helper (`compare_renders`). | Landed (preframr-audio `fidelity.py`) |
| [`landed/tokenizer_alphabet_coverage_bug.md`](landed/tokenizer_alphabet_coverage_bug.md) | RegTokenizer alphabet-coverage bug. | Landed |
| [`landed/hvsc_version_pinning_design.md`](landed/hvsc_version_pinning_design.md) | Runner HVSC-version check wired into preflight. | Landed |

Open (not a design doc): the **~100-song round-trip audio CI gate** (≥95% within
tolerance) — see AGENTS.md "Land any time" (`compare_renders` helper + unit tests
landed; corpus-scale gate pending).

## 3. Efficiency & deploy envelope

| Doc | Summary | Status |
|---|---|---|
| [`streaming_unembed_ce_design.md`](streaming_unembed_ce_design.md) | Stream `output(chunk) + ce_chunk` in one checkpoint; eliminates the 8.6 GiB chunk slab, restores `batch_size=4` + prodlike wallclock. | Pending impl |
| [`orin_inference_optimization_design.md`](orin_inference_optimization_design.md) | Predict-host throughput: vocab shrink + GPU-resident constrained-decode (Orin ~4% GPU util at predict). | Pending impl |

Cross-axis: [`compound_token_design.md`](compound_token_design.md) (token budget;
primary in Generalization). Vocab shrink (tkvocab ~8× to 4096) is queued under
AGENTS.md "Predict-host envelope" (deferred).

## 4. Runner / experiment infra & process

Serves the Generalization axis: `generalization_metric_tracking_design.md` wires
the decisive content-tier audit + scorecard + cross-run ledger.

| Doc | Summary | Status |
|---|---|---|
| [`generalization_metric_tracking_design.md`](generalization_metric_tracking_design.md) | Make the decisive content-tier `per_class` audit a runner stage (not run by hand), add a generalization scorecard (per-eval_b-family content acc + spread + loop/prompt) to the report, and a tokenizer-hash-keyed cross-run ledger that auto-flags confounded comparisons. Reuses existing audits + the metric registry. | Drafted, pending impl (land with no run in flight; tokenizer-health metrics landed) |
| [`runner_iteration_efficiency_design.md`](runner_iteration_efficiency_design.md) | Per-run overhead: (1) cache key on parse/tokenize cargs — **landed**; (2) symlink-farm + RO dump mount (no 2.7 GB copy) — **landed**; (3) drop the post-step chown container — pending. | #1+#2 landed; #3 pending |
| [`flag_stage_routing_design.md`](flag_stage_routing_design.md) | `FLAG_STAGES` registry + `add_stage_args` for stage-aware flag forwarding (parse/tokenize/train). | Pending impl |
| [`auto_early_abort_design.md`](auto_early_abort_design.md) | Spec-declared `decision_rule` evaluated after each (arm, seed); writes a refutation stub on falsification. | Deferred (cloud-rental prereq) |
| [`max_parallel_arms_design.md`](max_parallel_arms_design.md) | `concurrent.futures` slot allocator with flock; refuses N>1 on single-GPU hosts. | Deferred (cloud-rental prereq) |
| [`resume_design.md`](resume_design.md) | Per-stage `_resume.json` manifest for partial-run recovery (dataset cache already covers parse+tokenize). | Deferred (partial coverage landed) |
| [`repo_focus_cleanup_scope.md`](repo_focus_cleanup_scope.md) | Plan to keep `preframr` (main) = core framework, moving orchestration + audits + tier data + design docs to `preframr-xpt`. | Largely executed (see [`landed/experiments_extraction_design.md`](landed/experiments_extraction_design.md)) |

The container split + extraction landings (experiments / tokens / audio
extraction, train/inference split, regdataset decomposition) are archived in
[`landed/README.md`](landed/README.md) under "Library extractions".

## 5. Data & corpus

Active work here is currently in specs/audits, not design docs; the design record
is archived. New tier / eval-set / augmentation / dataset-coverage docs go here.

| Doc | Summary | Status |
|---|---|---|
| [`landed/prodlike_tier_design.md`](landed/prodlike_tier_design.md) | Prodlike data tier (~4.4K train + 385 Eval-A + 8 Eval-B families). | Landed |
| [`landed/engine_fingerprint_evalb_design.md`](landed/engine_fingerprint_evalb_design.md) | 8 cross-engine Eval-B families + engine-family map. | Landed |
| [`landed/corpus_structural_index_design.md`](landed/corpus_structural_index_design.md) | One-shot CPU structural index of full HVSC. | Landed |
| [`landed/stage_dumps_basename_fix_design.md`](landed/stage_dumps_basename_fix_design.md) | Composer-subdir staging; recovers 50 prodlike dumps lost to basename collisions. | Landed |
| `preframr-aug:design/melody_transfer_augmentation_design.md` | Offline corpus-expansion (inaudible perturbation, voice permutation, cross-song transfer). | Moved to preframr-aug |

## Landed (archived)

Reference docs whose implementations are in HEAD, indexed by *kind* (not axis) in
[`landed/README.md`](landed/README.md): library extractions, data tiers +
infrastructure, audit + validation tooling, tokenizer/encoding (FREQ_TRAJ +
profiling), and bug fixes.

## Decision rules

- **Status header.** One-line status bullet on every doc; update it before the
  body.
- **Promotion thresholds.** 3σ-on-val_acc to flip a default; capacity-attenuation
  refuses if prodlike Δ < ¼ × mini Δ; per-Eval-B-* breakouts confirm
  cross-composer transfer.
- **Refuted alternatives.** Detailed evidence + a "do not revisit without …"
  condition go in `preframr_experiments/data/refuted/<exp>.md`; the design doc
  retains only the `Refuted` status header pointing there.
