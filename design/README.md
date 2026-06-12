# Design notes index

**Orientation:** [`architecture_overview.md`](references/architecture_overview.md) — which
part lives in which repo (tokens / audio / framework / xpt / aug) and **why**, the
dependency layering, and how to **derive the release process**. Read it before any
cross-repo change or release.

**THE encoding is the v3 event model** (tokens 0.47.0+, `preframr_tokens/events/`): a fixed
**127-atom alphabet** (varint digits / reg / voice / kind / shape / KEYFRAME) with BPE as the only
dictionary, unconditional (no macro flags), fidelity = `decode(encode(ow)) == canonical_writes(ow)`
self-verified on every encode. **Its authoritative reference is the
[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens)** (alphabet, stream grammar,
fidelity contract, dump format, parse-domain schema); chip facts + pinning tests are in the
[preframr-audio README](https://github.com/anarkiwi/preframr-audio). The docs here are pointers plus
xpt-internal seams: [`tokens_architecture.md`](references/tokens_architecture.md),
[`audio_architecture.md`](references/audio_architecture.md),
[`framework_architecture.md`](references/framework_architecture.md),
[`backlog_tokens_hardening.md`](encoding/backlog_tokens_hardening.md) (the tokens testing
discipline).

## North star: LEARNABILITY (read this first)

The headline goal is **learnability** — a SID tokenization/ordering is good insofar as a bounded
(~TC⁰) autoregressive transformer can *cheaply represent its next-token map*: minimize causal-state
size + dependency horizon, prefer induction-head-expressible structure over implicit per-frame
counters, order tokens by the driver's causal DAG. The lens doc — read before proposing any
representation work — is
[`learnability_token_ordering_theory.md`](references/learnability_token_ordering_theory.md) (theory
+ the training-free `audit/learnability_triage.py` that ranks encodings *without a run*).
**Everything else is subordinate:** correctness/fidelity is the *gate*; compression and
parse/runner/deploy work are *infra* that buy cheaper learnability experiments. The decisive
empirical lesson: **model-side content interventions all refuted at a ~0.13 content ceiling that
tokenizer-side representation then lifted** — most recently the v3 event model's atoms-only baseline
at **eval_a content 0.479** (the architecture was exonerated long ago, `framework_arch_test`).

The priority order (learnability → correctness → efficiency → infra) is the lens for what to work on
next; it is orthogonal to the physical layout below, which groups docs by *subject*. Every doc
carries a one-line `**Status:**` header. Refuted hypotheses have evidence stubs at
`preframr_experiments/data/refuted/<exp>.md` — read those before reopening a rejected direction.

## How this index is organized (read before adding a doc)

Files live in **seven subject-theme subdirs**. Add a new doc under its **primary** theme;
cross-link a secondary theme in prose.

1. **[`references/`](references/)** — orientation maps + domain/literature background (stable,
   read-first).
2. **[`encoding/`](encoding/)** — tokenization & representation research (what's still *open* on
   the representation side; the shipped encoding itself is documented in the tokens README).
3. **[`generation/`](generation/)** — **the generation program (the ultimate goal):** continuation
   is one goal; generation from diverse prompts — including a short musical phrase from a MIDI file
   or keyboard — is the destination. Prompting, whole-tune structure, and quality measurement live
   here.
4. **[`measurement/`](measurement/)** — prediction-side metrics, gates, calibration.
5. **[`performance/`](performance/)** — predict-host (Orin) envelope + pipeline throughput.
6. **[`infra/`](infra/)** — the experiment runner & process.
7. **[`refuted/`](refuted/)** — hypotheses tested and rejected; truncated to status + pointer per
   the decision rules (full designs in git history).

Plus **[`landed/`](landed/)** — archived docs whose implementation shipped (or that record a
superseded arc), indexed in [`landed/README.md`](landed/README.md).

**Cross-doc links are relative** within `design/`. When you move a doc between themes, fix the
inbound links. **Lifecycle:** Draft/Scoping → Pending impl → In flight → **Landed** (move to
`landed/` + index row) / **Refuted** (move to `refuted/`, truncate to status + pointer, write the
evidence stub) / **Deferred** / **Reference**.

## references/ — orientation & domain background

| Doc | Summary | Status |
|---|---|---|
| [`architecture_overview.md`](references/architecture_overview.md) | The repo/dependency/release map. Read before any cross-repo change. | Reference |
| [`learnability_token_ordering_theory.md`](references/learnability_token_ordering_theory.md) | **The north-star lens**: cheap next-token representability + the training-free triage (run at seq_len 8192, window mode, before any training A/B). Track record: predicted the codebook block-scale failure and the generator NO-GO. | Reference + tool |
| [`encoding_principles.md`](references/encoding_principles.md) | The rubric: fidelity (gate) × context-efficiency (bound) × learnability (objective), sub-principles P1–P8, per-change checklist. Evidence re-anchored to v3 (2026-06-12). | Reference |
| [`tokens_architecture.md`](references/tokens_architecture.md) | Pointer → tokens README (the v3 event codec reference). Old revisions document the retired (op,reg,subreg,val) substrate. | Pointer |
| [`audio_architecture.md`](references/audio_architecture.md) | Pointer → audio README + the xpt cross-repo render seam. | Pointer |
| [`framework_architecture.md`](references/framework_architecture.md) | The torch layer: train/predict/model, data path, generation gotchas (incl. the event-vs-parse-domain constrained-decode caveat). | Reference |
| [`sid_render_fidelity_contract.md`](references/sid_render_fidelity_contract.md) | Pointer: chip facts → audio README; v3 canonical form + encode self-verify → tokens README. | Pointer |
| [`verification_and_audits.md`](references/verification_and_audits.md) | **How to verify**: canonical fidelity (tokens) vs canonicalization soundness (audio), + xpt operating rules. | Reference |
| [`voice_encoding_reference.md`](references/voice_encoding_reference.md) | How voices are carried in the v3 stream + the de-mux modeling implication. | Pointer (v3, 2026-06-12) |
| [`sid_driver_ornament_reference.md`](references/sid_driver_ornament_reference.md) | **Domain background:** how C64 drivers generate per-frame ornament (arp/vibrato/PW/filter mechanics, per-driver). | Reference |
| [`digi_detection_reference.md`](references/digi_detection_reference.md) | Digi techniques + detection (refines `is_digi`). | Reference |
| [`release_build_cache.md`](references/release_build_cache.md) | **The one place** for release/build/test/cache process. | Reference (authoritative) |
| [`related_work.md`](references/related_work.md) | Adversarially-verified survey: the raw-register-stream LM + learnability-ordering combination is unaddressed in the literature. | Reference (positioning) |
| [`tokenization_vs_music_llms.md`](references/tokenization_vs_music_llms.md) | v3 register-event scheme vs symbolic/MIDI/codec paradigms: wins fidelity + inductive bias + verified augmentation; pays sequence length, engine specificity, data scale. | Reference (positioning, v3 2026-06-12) |

## encoding/ — what is still open on the representation side

The encoding itself **shipped** (event model; see banner). The macro-pass era and its design stack
(generator-MDL pipeline, melody skeleton work order, superframe/role lane designs, speculative
claims+arbitration, compound tokens, instrument-program codebook, audio-equivalence normalization)
ended 2026-06-12: the generator and pitch docs moved to [`landed/`](landed/) with their absorbed-
into-v3 mapping; the rest were **removed** (melody layer 2 = absorbed as `NI_*` intervals; the
claims/compound/codebook problems were mooted by the unconditional fixed-atom design; full text in
git history). What remains open:

| Doc | Summary | Status |
|---|---|---|
| [`lane_demux_hypothesis.md`](encoding/lane_demux_hypothesis.md) | Cross-voice de-multiplexing (voice-form) + role/causal-DAG ordering (role-form, accompaniment-before-melody) of the event stream; sync/ring wiring constraints; tractability ladder. | Open hypothesis — **trigger only if canonical runs localize content failure to cross-voice interference**; triage-gated |
| [`log_to_swm_recompiler_design.md`](encoding/log_to_swm_recompiler_design.md) | Register log → editable SID-Wizard SWM with re-render-equivalence gate (generated tunes become tracker-editable). | Design 2026-06-06 (IR retargeted 2026-06-12) |
| [`backlog_tokens_hardening.md`](encoding/backlog_tokens_hardening.md) | tokens testing discipline: real-pipeline structural/balance tests through the event codec + the no-skip fixture policy. | Pending impl (retargeted to events 2026-06-12) |

## generation/ — the generation program (the ultimate goal)

**Goal ladder:** G1 continuation from a SID prompt (exists, measured by prediction metrics) → **G2
generation from a musical phrase (MIDI file / keyboard) arranged into a SID tune — the destination**
→ G3 style steering. The program is sequenced after the in-flight canonical learnability run; the
quality gate lands first (it is what makes the rest measurable).

| Doc | Summary | Status |
|---|---|---|
| [`generation_quality_gate.md`](generation/generation_quality_gate.md) | **Land first.** Standard generation cohort + scorecard: pathology flags (loop/diversity/invalid), write-domain structure metrics vs corpus, chip-native fingerprint distance (FAD analogue), **memorization audit** (n-gram novelty + longest verbatim match), minimal blind A/B protocol; picks the sampling regime; event-grammar mask port folded in. Necessary-not-sufficient beside the content tier. | Design 2026-06-12 |
| [`prompt_interface_design.md`](generation/prompt_interface_design.md) | The phrase compiler: MIDI/keyboard phrase → synthetic one-voice dump → `encode(verify=True)` → native prompt block. Distribution shift attacked by **reduction augmentation** (melody-prefix → full-texture pairs from the corpus), scaffolding A/Bs, patch realism; exemplar prompting before conditioning atoms; phrase-adherence gate. | Design 2026-06-12 |
| [`long_range_structure.md`](generation/long_range_structure.md) | Whole tunes via **decode-and-recompile chaining** (re-canonicalize decoded writes into fresh self-contained KEYFRAME blocks — state exact by construction); long-horizon coherence metrics; evidence-gated escalation to section-exemplar conditioning; hierarchical models rejected (envelope). | Design 2026-06-12 |
| [`transplant_augmentation_design.md`](generation/transplant_augmentation_design.md) | **Data side:** donor/host melody & instrument transplants (breaking the melody×timbre spurious binding — distributional P1) + the **mined instrument bank** (P0, feeds the phrase compiler's patch realism). Register-domain splice + `encode(verify=True)` ⇒ zero pipeline changes; train-split-only leakage rule; dosage A/B on eval_b content. Impl home: preframr-aug. | Design 2026-06-12 |

## measurement/ — prediction-side metrics & gates

The live run plan (canonical event-model learnability: atoms-only baseline done at eval_a content
0.479; unigram-dictionary arm in flight; decision branches) is operational state and lives in
**AGENTS.md "Current arc"**. Durable designs:

| Doc | Summary | Status |
|---|---|---|
| [`generalization_metric_tracking_design.md`](measurement/generalization_metric_tracking_design.md) | Make the decisive content-tier audit a runner stage, scorecard the cross-composer signal, tokenizer-hash-keyed ledger auto-flagging confounded comparisons. The generation quality gate wires into the same stage. | Drafted, pending impl |
| [`generalize_min_val_acc_floor_design.md`](measurement/generalize_min_val_acc_floor_design.md) | Calibrate `GENERALIZE_MIN_VAL_ACC` = 2/3 × median once 2–3 canonical event-model baselines settle (first point exists: 0.561). | Pending calibration |

(The generator-era measurement plan is superseded; its surviving result — the triage NO-GO that
motivated the v3 redesign — is recorded in
[`landed/generator_mdl_representation.md`](landed/generator_mdl_representation.md).)

## performance/ — deploy envelope & throughput

| Doc | Summary | Status |
|---|---|---|
| [`orin_inference_optimization_design.md`](performance/orin_inference_optimization_design.md) | Predict-host throughput, measured: cudagraph compile landed (1.9–3.1×); real-time single-stream is ~9× short — offline audition (~6.5 min/song) is the deploy mode. | Measured; real-time deferred |

Open (no doc yet): under v3 the pipeline bottleneck is **`encode(verify=True)` ~33 min / 856 dumps**
(self-verify doubles work by design) — design one only if it actually gates iteration. The old
parse-perf arc (hygiene wins + dead ends) is archived at
[`landed/parse_perf_proposal.md`](landed/parse_perf_proposal.md); the streaming-unembed memory note
was removed (moot at event-model vocab scale; git history).

## infra/ — runner, process & deploy

| Doc | Summary | Status |
|---|---|---|
| [`runner_iteration_efficiency_design.md`](infra/runner_iteration_efficiency_design.md) | Per-run overhead: cache key on cargs + symlink-farm/RO dump mount **landed**; post-step chown drop pending. | #1+#2 landed; #3 pending |
| [`flag_stage_routing_design.md`](infra/flag_stage_routing_design.md) | Stage-aware flag forwarding (macro flags now inert under events; `--tkvocab`/train knobs are the live cases). | Pending impl |
| [`start_seq_rotation_audit_design.md`](infra/start_seq_rotation_audit_design.md) | `predict_load` hard-codes rotation 0; flat-indexing fix + coverage probe. | Pending impl |
| [`audio_driver_split_design.md`](infra/audio_driver_split_design.md) | preframr-audio: split `audio_driver.py` into render core vs live I/O. | Drafted, pending review |
| [`cloud_rental_runner_design.md`](infra/cloud_rental_runner_design.md) | Deferred cloud-rental program: `--resume` → auto early-abort → `--max-parallel-arms`. | Deferred |

## refuted/ — tested & rejected

Docs are truncated to status + pointer (full designs in git history); evidence + "do not revisit
without …" conditions live in `preframr_experiments/data/refuted/<exp>.md`. The model-side
**anti-queue** (the do-not-re-attempt list) is kept live in
[`multi_modal_objective_design.md`](refuted/multi_modal_objective_design.md).

| Doc | Verdict |
|---|---|
| [`multi_modal_objective_design.md`](refuted/multi_modal_objective_design.md) | Umbrella: per-token-CE-bottleneck thesis refuted across B/C/A branches; carries the anti-queue. Stub: `multi_modal_objective.md`. |
| [`per_tier_heads_design.md`](refuted/per_tier_heads_design.md) | Router saturates at prodlike; all-tier lift is structural. Stubs: `per_tier_heads_*.md`. |
| [`per_voice_aux_supervision_design.md`](refuted/per_voice_aux_supervision_design.md) | Refuted by class; never run. |
| [`content_diffusion_design.md`](refuted/content_diffusion_design.md) | Sampling-side; no CE change. Stub: `content_diffusion.md`. |
| [`cluster_conditional_content_head_design.md`](refuted/cluster_conditional_content_head_design.md) | Same ceiling; diversity never recovers. Stub: `cluster_conditional_content_head.md`. |
| [`motif_templates_v2_impl_design.md`](refuted/motif_templates_v2_impl_design.md) | Built + A/B'd; never beat no-motif. Stub: `motif_pass.md`. |
| [`sequence_order_normalization_design.md`](refuted/sequence_order_normalization_design.md) | Reorder is inaudible but recovers ~5%; 84% of repeats are content. Stub: `sequence_order_normalization.md`. |

## landed/ — archived (shipped or superseded-with-record)

Indexed in [`landed/README.md`](landed/README.md). Notable for current work:
[`generator_mdl_representation.md`](landed/generator_mdl_representation.md) (the superseded
generator arc + what v3 absorbed + the triage NO-GO that triggered the redesign),
[`universal_multiresolution_pitch.md`](landed/universal_multiresolution_pitch.md) (the recovered
note-table findings now shipped as `NOTE_TABLE`/`TUNING`/`NI_*`),
[`unified_oscillation_primitive_design.md`](landed/unified_oscillation_primitive_design.md) +
[`trajectory_anchoring.md`](landed/trajectory_anchoring.md) (the FREQ_TRAJ era),
[`prodlike_tier_design.md`](landed/prodlike_tier_design.md) +
[`engine_fingerprint_evalb_design.md`](landed/engine_fingerprint_evalb_design.md) (the data tiers),
[`orinnx_audition_design.md`](landed/orinnx_audition_design.md) (the render smoke harness the
quality gate builds on).

**Elsewhere (not in this repo):**
- `preframr-tokens:pipeline_trace.py` — torch-free pass-by-pass pipeline tracer (parse-domain).
- `preframr-aug:design/melody_transfer_augmentation_design.md` — offline corpus expansion. Its
  cross-song-transfer axis is superseded by
  [`transplant_augmentation_design.md`](generation/transplant_augmentation_design.md) (v3-native);
  inaudible perturbation + voice permutation remain there. The **reduction augmentation** of
  [`prompt_interface_design.md`](generation/prompt_interface_design.md) belongs beside both.

Open (not a design doc): productize the corpus-scale **canonicalization-soundness** perceptual
raw-vs-canonical A/B (see [`verification_and_audits.md`](references/verification_and_audits.md)).

## Decision rules

- **Status header** on every doc; update it before the body.
- **Promotion:** representation/sampling defaults flip only on the **content-tier verdict**
  (all-tier val_acc is confounded across tokenizations) **plus** the
  [generation quality gate](generation/generation_quality_gate.md) once landed; capacity-attenuation
  refuses if prodlike Δ < ¼ × mini Δ; per-eval-B breakouts confirm cross-composer transfer.
- **Triage before training:** any representation A/B runs `learnability_triage` first (seq_len
  8192, window mode).
- **Refuted:** move to [`refuted/`](refuted/), truncate to status + pointer, write the evidence stub
  with a "do not revisit without …" condition.
- **Landed:** move to [`landed/`](landed/) + add an index row. Docs whose subject was *superseded*
  (not shipped) also land there, with the supersession recorded in the status header.
