# Landed designs (archive)

Reference-only. Implementation has shipped to HEAD or to the
sibling `preframr-{audio,tokens,experiments}` repos. Kept for
historical context; consult git log for the actual commits.

## Library extractions

| Doc | What landed |
|---|---|
| [`preframr_tokens_extraction_design.md`](preframr_tokens_extraction_design.md) | `preframr-tokens` PyPI library (reglog parsing + tokenisation + macros + stfconstants). |
| [`preframr_tokens_corpus_class_design.md`](preframr_tokens_corpus_class_design.md) | `preframr_tokens.corpus.Corpus` (~560 LoC extracted from RegDataset). |
| [`constrained_decode_torch_free_design.md`](constrained_decode_torch_free_design.md) | `preframr_tokens.constrained_decode` numpy-mask + boundary `masked_fill`. |
| [`experiments_extraction_design.md`](experiments_extraction_design.md) | `preframr-experiments` sibling repo (runner + specs + tier lists + refuted registry) at `/scratch/anarkiwi/preframr-xpt`. |
| [`repo_focus_cleanup_scope.md`](repo_focus_cleanup_scope.md) | Scope/plan for the above split (what stays in `preframr` core vs moves to `preframr-xpt`); largely executed. |
| [`model_regdataset_decomposition_design.md`](model_regdataset_decomposition_design.md) | `preframr/train/model/` subpackage + `regdataset.py` slimmed to ~210 LoC adapter. |
| [`train_inference_split_design.md`](train_inference_split_design.md) | `preframr/train/` vs `preframr/inference/` bind-mount split; `Dockerfile.predict` slim eval image. |

## Data tiers + infrastructure

| Doc | What landed |
|---|---|
| [`prodlike_tier_design.md`](prodlike_tier_design.md) | Prodlike data tier (~4.4K train + 385 Eval-A + 8 Eval-B families); pinned in `preframr_experiments/data/prodlike/`. |
| [`hvsc_version_pinning_design.md`](hvsc_version_pinning_design.md) | Runner startup HVSC-version check via `DOCUMENTS/HVSC.txt`; wired into preflight. |
| [`engine_fingerprint_evalb_design.md`](engine_fingerprint_evalb_design.md) | 8 cross-engine Eval-B families pinned in prodlike spec; engine-family map JSON. |
| [`corpus_structural_index_design.md`](corpus_structural_index_design.md) | One-shot CPU audit emitting structural index of full HVSC; output at `data/audit/`. |
| [`stage_dumps_basename_fix_design.md`](stage_dumps_basename_fix_design.md) | Composer-subdir staging in `stage_dumps`; recovers 50 dumps lost to basename collisions in prodlike. |

## Audit + validation tooling

| Doc | What landed |
|---|---|
| [`alphabet_cooccurrence_audit_design.md`](alphabet_cooccurrence_audit_design.md) | `profile/alphabet_cooccurrence_audit.py` for per-tier co-occurrence checks. |
| [`audio_fidelity_helper_design.md`](audio_fidelity_helper_design.md) | Shared render-and-compare helper. Originally landed at `integration_tests/audio_fidelity.py`; subsequently moved to `preframr-audio/preframr_audio/fidelity.py` (where it sits next to the renderer it depends on). |
| [`encodability_metric_design.md`](encodability_metric_design.md) | Per-cluster Eval-B encodability metric extractor. **Retired** — served the refuted `global_instr_ids` Phase A; impl removed in the repo-focus cleanup. |
| [`orinnx_audition_design.md`](orinnx_audition_design.md) | Orin NX predict-host audition harness. |

## Decompiler arc → the STEP / TRACKER codec (the < 1 token/frame win)

All superseded-with-banner by the landed step/tracker codec
([`../encoding/sid_player_decompiler.md`](../encoding/sid_player_decompiler.md) "HOW IT LANDED"); kept
as the record of the arc. The enduring lesson is HARD RULE #0 (the transposition trap recurred 4×).

| Doc | What it recorded |
|---|---|
| [`ornament_generator_recovery.md`](ornament_generator_recovery.md) | **The central diagnosis:** the event stream isn't sparse because ornaments are per-frame generators; recover the per-instrument program → note-rate sparse + byte-exact. Pointed the arc at generator recovery. |
| [`universal_sid_codec.md`](universal_sid_codec.md) | Own-VM / universal op-set codec with a residual *escape lane* — right instinct, wrong mechanism (the landed codec removed the escape hatch). |
| [`virtual_tracker_codec.md`](virtual_tracker_codec.md) | GoatTracker-as-target — refuted (one driver's grid/tempo/table caps = the wrong cage; freq ~85% off-grid). |
| [`automated_generator_recovery.md`](automated_generator_recovery.md) | MDL inference over the op-set grammar (Berlekamp–Massey + periodicity + DP segmentation) to retire hand-coded recognizers; the generators landed in the codec. |
| [`front_loaded_instrument_encoding.md`](front_loaded_instrument_encoding.md) | Tracker-style instrument DEF→REF on the event codec — RAN as a de-confounded null; pitch-invariant instrument banks landed in the step codec instead. |
| [`phrase_def_ref_triage.md`](phrase_def_ref_triage.md) | Phrase/pattern recurrence census — landed as the inline backward orderlist. |
| [`melody_timbre_factorization.md`](melody_timbre_factorization.md) | Track-major melody/timbre split — strongest proxy of the era, but a de-confounded generation null; the step codec is per-voice rows by construction. |
| [`lane_demux_hypothesis.md`](lane_demux_hypothesis.md) | Voice/role-contiguous event-stream ordering — mooted (step codec de-muxes voices structurally). |
| [`event_boundary_dictionary_proposal.md`](event_boundary_dictionary_proposal.md) + [`encoding_density_frontier.md`](encoding_density_frontier.md) + [`context_length_experiment.md`](context_length_experiment.md) | Frame/event-codec density era: BPE / boundary-dictionary refuted as the context lever; < 1 token/frame came from recovering the generator, not compressing a dense stream. |
| [`log_to_swm_recompiler_design.md`](log_to_swm_recompiler_design.md) | Register-log → editable SID-Wizard module — the landed step codec is itself the register-log → program decompiler; export rebuildable on it. |
| [`backlog_tokens_hardening.md`](backlog_tokens_hardening.md) | tokens testing discipline for the event codec; the step-codec port carries its own gates + the no-skip fixture policy. |
| [`macros_removal_refactor_plan.md`](macros_removal_refactor_plan.md) + [`tokens_port_deadwood_manifest.md`](tokens_port_deadwood_manifest.md) | The dead-wood audit + removal sequence the clean-slate port followed (`events/` + `macros/` + frame codec deleted). |
| [`representation_abstraction_probe.md`](representation_abstraction_probe.md) | Local-vs-structural abstraction probe — the structural-locality failure the step codec addresses at the representation level. |
| [`generation_offramp_shipped.md`](generation_offramp_shipped.md) + [`free_running_pathology_remediation_design.md`](free_running_pathology_remediation_design.md) + [`dagger_recanonicalization_design.md`](dagger_recanonicalization_design.md) | The free-running off-ramp + remediation ladder + DAgger — event-codec-era; the free-running ↔ teacher-forced gap was a dense-stream pathology. Re-evaluate on the step stream. |

## Tokenizer / encoding

| Doc | What landed |
|---|---|
| [`generator_mdl_representation.md`](generator_mdl_representation.md) | Generator-MDL encoding (tokens 0.45–0.46 deployed default) → **superseded by the v3 event model** (0.47); records what v3 absorbed + the triage NO-GO that triggered the redesign. |
| [`universal_multiresolution_pitch.md`](universal_multiresolution_pitch.md) | Recovered per-voice note-table pitch model — **absorbed into v3** as `NOTE_TABLE`/`TUNING`/`NI_*`; keeps the universal-grid / chorus-guardrail findings. |
| [`unified_oscillation_primitive_design.md`](unified_oscillation_primitive_design.md) | Unified `FREQ_TRAJ` op (SLOPE + OSCILLATE_ENV + FREQ_VIBRATO + FREQ_RUN) + 2-atom `FREQ_NUDGE`; shipped preframr-tokens 0.16/0.17 (retired substrate era). |
| [`tokenizer_profiling_tooling_design.md`](tokenizer_profiling_tooling_design.md) | Torch-free tokenizer profiling: `python -m preframr_tokens.tokenizer_profile` + `audit_primitives` reductions (`op_atom_profile`, `register_state`, `trajectory_coverage`) + `tokenizer_config` source-of-truth; shipped preframr-tokens 0.20.0. |

## Performance

| Doc | What landed |
|---|---|
| [`parse_perf_proposal.md`](parse_perf_proposal.md) | Parse-perf hygiene wins (PR #49, block path −40%, xdist test gate) + the dead-end registry (structural block slice, diff-attribution, suffix-resume). Remaining levers OBE under v3. |

## Bug fixes

| Doc | What landed |
|---|---|
| [`tokenizer_alphabet_coverage_bug.md`](tokenizer_alphabet_coverage_bug.md) | RegTokenizer alphabet-coverage bug (Int64 + relaxed substitution). |
