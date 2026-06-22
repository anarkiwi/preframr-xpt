# Design notes index

**Orientation:** [`architecture_overview.md`](references/architecture_overview.md) — which
part lives in which repo (tokens / audio / framework / xpt / aug) and **why**, the
dependency layering, and how to **derive the release process**. Read it before any
cross-repo change or release.

**THE encoding is the BACC step/tracker codec** (landed): decompile a `.sid` (`recover_from_sid`, white-box)
→ per-voice tracker rows + **pitch-invariant instrument generators** — vibrato/slide/arp/PWM/ADSR/sweeps all
collapse into one **bounded-accumulator (BACC)** primitive (`value += rate every dwell frames`; boundary
wrap/reflect/none; width 8/12-bit; output map absolute/base+offset/note-table-scaled, or table-walk).
Repeated phrases dedup via an inline backward orderlist; a transposed repeat is one backward TRANSPOSE op;
DECODE = render the program → frames. It is `trace = VM(program)` realized: encode the PROGRAM, not the
per-frame TRACE. Whole tunes fit one window, residual-zero: **Monty 1,313 tokens (0.075 tok/frame),
5_Title_Tunes 1,394, Grid_Runner 2,817, A_Mind_Is_Born 496** — ~10× under the retired frame/event codec,
VOCAB=34. The design + the enduring lessons are
[`encoding/sid_player_decompiler.md`](encoding/sid_player_decompiler.md) ("HOW IT LANDED"); the op-set
grounding is [`encoding/sid_opset_inventory.md`](encoding/sid_opset_inventory.md). The codec **shipped in
preframr-tokens** (clean slate — `events/` + `macros/` + the old frame codec DELETED); the public path is
sid-only (`recover_from_sid`, driver=`generic`, via `preframr-sidtrace`). chip facts + pinning tests are in
the [preframr-audio README](https://github.com/anarkiwi/preframr-audio). xpt-internal seams:
[`tokens_architecture.md`](references/tokens_architecture.md),
[`audio_architecture.md`](references/audio_architecture.md),
[`framework_architecture.md`](references/framework_architecture.md).

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
empirical lesson: **model-side content interventions all refuted at a ~0.13 content ceiling because the
frame/event codec signal-fit a dense trace** — the representation-level fix landed as the BACC step/tracker
codec (sparse, generator-level, whole-song under 4096 tokens). The model-side architecture was exonerated
long ago (`framework_arch_test`); the next training read is the atoms-only continuation on the BACC stream.

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
| [`learnability_token_ordering_theory.md`](references/learnability_token_ordering_theory.md) | **The north-star lens**: cheap next-token representability + the training-free triage (run at the whole-song context scale 4096, before any training A/B). Track record: predicted the codebook block-scale failure and the generator NO-GO. | Reference + tool |
| [`encoding_principles.md`](references/encoding_principles.md) | The rubric: fidelity (gate) × context-efficiency (bound) × learnability (objective), sub-principles P1–P8, per-change checklist. Re-anchored to the BACC codec. | Reference |
| [`tokens_architecture.md`](references/tokens_architecture.md) | Pointer → tokens README (the BACC codec). | Pointer |
| [`audio_architecture.md`](references/audio_architecture.md) | Pointer → audio README + the xpt cross-repo render seam. | Pointer |
| [`framework_architecture.md`](references/framework_architecture.md) | The torch layer: train/predict/model, data path, generation gotchas (incl. the event-vs-parse-domain constrained-decode caveat). | Reference |
| [`sid_render_fidelity_contract.md`](references/sid_render_fidelity_contract.md) | Pointer: chip facts → audio README; v3 canonical form + encode self-verify → tokens README. | Pointer |
| [`verification_and_audits.md`](references/verification_and_audits.md) | **How to verify**: canonical fidelity (tokens) vs canonicalization soundness (audio), + xpt operating rules. | Reference |
| [`voice_encoding_reference.md`](references/voice_encoding_reference.md) | How voices are carried (de-muxed by construction in the BACC codec) + the de-mux modeling implication. | Pointer |
| [`sid_driver_ornament_reference.md`](references/sid_driver_ornament_reference.md) | **Domain background:** how C64 drivers generate per-frame ornament (arp/vibrato/PW/filter mechanics, per-driver). | Reference |
| [`digi_detection_reference.md`](references/digi_detection_reference.md) | Digi techniques + detection (refines `is_digi`). | Reference |
| [`release_build_cache.md`](references/release_build_cache.md) | **The one place** for release/build/test/cache process. | Reference (authoritative) |
| [`related_work.md`](references/related_work.md) | Survey (refreshed): the project's angle is byte-exact recovery of a generative program from a deterministic playroutine trace — nearest cousins are trace-driven / re-executable neural decompilation; the trace→generative-program + byte-exact + music + learnability combination is unaddressed. | Reference (positioning) |
| [`tokenization_vs_music_llms.md`](references/tokenization_vs_music_llms.md) | BACC recovered-program scheme vs symbolic/MIDI/codec paradigms: wins fidelity + sparse program + inductive bias; pays recovery coverage, engine specificity, data scale. | Reference (positioning) |

## encoding/ — the landed codec + its grounding

The codec **landed** (step/tracker; see banner) — Monty residual-zero at < 1 token/frame. The
frame/event-codec era and its whole design stack (event/v3 model, generator-MDL pipeline, GoatTracker-
target codec, invented-op-set codec, DEF→REF instrument/phrase banks, melody/timbre factorization, lane-
demux, boundary dictionary, density frontier, context-length sweep, the port deadwood/macros-removal
plans, automated generator recovery, SWM recompiler) all moved to [`landed/`](landed/) with a supersession
banner: every "distinct body" was a few instruments rendered at many pitches (the transposition trap,
HARD RULE #0). What stays live here:

| Doc | Summary | Status |
|---|---|---|
| [`sid_player_decompiler.md`](encoding/sid_player_decompiler.md) | **The landed codec + the durable lessons.** `trace = VM(program)`; op-set = grammar, per-tune program = music, residual→0 the gate, no escape hatch; "HOW IT LANDED" = the step reframe + pitch-invariant instruments + backward orderlist + the transposition trap. | **LANDED 2026-06-20** |
| [`sid_opset_inventory.md`](encoding/sid_opset_inventory.md) | The universal op-set extracted from real drivers (GoatTracker / SID-Wizard / defMON / WEMUSIC + Hubbard/Follin/Galway/Whittaker/Tel/Gray): ~18 ops / 7 primitives, ≈85-90% overlap; the grounding for HARD RULE #0 and residual→0. | Reference (op-set grounding) |
| [`generic_bacc_recovery.md`](encoding/generic_bacc_recovery.md) | Can BACC recovery be made GENERIC (remove the per-driver hand disassembly)? Probed on 4 drivers: note-ons from gate transitions, note-table auto-discovery, and a read/write state classifier auto-infer the RAM map (flavor A, offline). Dump-only (flavor B) hits a wall on dense/free-running tunes. Residual driver-specific cost = the generator arithmetic = a small bounded archetype set → the **generator-fitter** experiment. | Analysis + fitter VALIDATED (seconds/tune) |
| [`cross_driver_note_unification.md`](encoding/cross_driver_note_unification.md) | One note encoding across drivers so a model sees the same note as the same token. (A) decompose GoatTracker's reconstructed .SNG into the same per-voice `(dt, interval, instr_ref, lnth, porta)` + REPEAT rows as Hubbard (retire the raw-bytes path). (B) canonical 12-TET A440 pitch axis — snap onset Fn to an integer grid index, driver-invariant. (C) tuning as a separate decode-side parameter: static per-tune `Δ(n)` table (ET-rounding + base offset, corpus-shared-factored) + dynamic per-onset `micro` (mostly derived); never in the note alphabet. | LANDED (abs grid + Transpose op) |
| [`bpe_unigram_subword.md`](encoding/bpe_unigram_subword.md) | Does BPE/Unigram help on the sparse BACC stream? Measured: vanilla subwording WELDS 63–65% of tokens across field boundaries (note↔instr / whole rows) and collapses induction-copy 0.886→0.114 — a learnability NO-GO. Melodic capture doesn't materialize (the row-LZ already factored phrases). Only hard field-segmented BPE is safe (0% welding, ~1.4×, help-neutral). | Analysis (vanilla REFUTED) |

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
| [`transplant_augmentation_design.md`](generation/transplant_augmentation_design.md) | **Data side:** donor/host melody & instrument transplants (breaking the melody×timbre spurious binding — distributional P1) + the **mined instrument bank** (P0, feeds the phrase compiler's patch realism). Register-domain splice + `encode(verify=True)` ⇒ zero pipeline changes; train-split-only leakage rule; dosage A/B on eval_b content. Impl home: preframr-aug. | Design 2026-06-12 — re-target to the step codec |

(The free-running remediation ladder + DAgger + the abstraction probe + the shipped off-ramp were
event-codec-era; they moved to [`landed/`](landed/) — the free-running ↔ teacher-forced gap was a
dense-stream pathology the step/tracker codec addresses at the representation level. Re-evaluate
generation on the landed stream after the framework rebuild.)

## measurement/ — prediction-side metrics & gates

The live run plan (re-encode the corpus on the step stream + train atoms-only continuation) is
operational state and lives in **AGENTS.md "LIVE ARC"**. Durable designs (the metrics themselves carry
over to the step stream):

| Doc | Summary | Status |
|---|---|---|
| [`generalization_metric_tracking_design.md`](measurement/generalization_metric_tracking_design.md) | Make the decisive content-tier audit a runner stage, scorecard the cross-composer signal, tokenizer-hash-keyed ledger auto-flagging confounded comparisons. The generation quality gate wires into the same stage. | Drafted, pending impl |
| [`generalize_min_val_acc_floor_design.md`](measurement/generalize_min_val_acc_floor_design.md) | Calibrate `GENERALIZE_MIN_VAL_ACC` = 2/3 × median once 2–3 step-codec baselines settle. | Pending calibration (re-baseline on step codec) |

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
| [`tokens_port_release_cascade.md`](infra/tokens_port_release_cascade.md) | Post-port release runbook: tokens tag → preframr rebuild → xpt image repoint → `generalize_continuation` run; flags the >=0.53.0 floor drift. | Prep (pre-port) |

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

Indexed in [`landed/README.md`](landed/README.md). Notable for current work — the **decompiler arc**
(all superseded-with-banner by the landed step/tracker codec, kept for the record):
[`ornament_generator_recovery.md`](landed/ornament_generator_recovery.md) (the central diagnosis: the
event stream isn't sparse because ornaments are per-frame generators),
[`universal_sid_codec.md`](landed/universal_sid_codec.md) +
[`virtual_tracker_codec.md`](landed/virtual_tracker_codec.md) (the invented-op-set / GoatTracker-cage
detours), [`automated_generator_recovery.md`](landed/automated_generator_recovery.md),
[`front_loaded_instrument_encoding.md`](landed/front_loaded_instrument_encoding.md) +
[`phrase_def_ref_triage.md`](landed/phrase_def_ref_triage.md) +
[`lane_demux_hypothesis.md`](landed/lane_demux_hypothesis.md) (the DEF→REF / reorder nulls),
[`encoding_density_frontier.md`](landed/encoding_density_frontier.md) (the BPE-is-not-the-context-lever
record). Earlier eras: [`generator_mdl_representation.md`](landed/generator_mdl_representation.md),
[`universal_multiresolution_pitch.md`](landed/universal_multiresolution_pitch.md) (note-table pitch →
`NOTE_TABLE`/`TUNING`/`NI_*`), [`unified_oscillation_primitive_design.md`](landed/unified_oscillation_primitive_design.md)
+ [`trajectory_anchoring.md`](landed/trajectory_anchoring.md) (FREQ_TRAJ),
[`prodlike_tier_design.md`](landed/prodlike_tier_design.md) +
[`engine_fingerprint_evalb_design.md`](landed/engine_fingerprint_evalb_design.md) (data tiers),
[`orinnx_audition_design.md`](landed/orinnx_audition_design.md) (render smoke harness).

**Elsewhere (not in this repo):**
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
- **Triage before training:** any representation A/B runs `learnability_triage` first (at the
  whole-song context scale, 4096).
- **Refuted:** move to [`refuted/`](refuted/), truncate to status + pointer, write the evidence stub
  with a "do not revisit without …" condition.
- **Landed:** move to [`landed/`](landed/) + add an index row. Docs whose subject was *superseded*
  (not shipped) also land there, with the supersession recorded in the status header.
