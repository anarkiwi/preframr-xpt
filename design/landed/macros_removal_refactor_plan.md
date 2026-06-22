# macros/ removal + refactor plan (gated on the white-box decompiler codec port)

**EXECUTED / SUPERSEDED (2026-06-20):** the concurrent clean-slate port deletes `events/` + `macros/` +
the frame codec wholesale. Kept as the record of the removal sequence the port followed.

**Date:** 2026-06-19. **Mode:** READ-ONLY scoping. No source touched; this file is the only write.
**Builds on:** `design/encoding/tokens_port_deadwood_manifest.md` (bucket analysis). This document does
NOT redo the bucket work; it extends it into an ordered removal+refactor SEQUENCE with the framework
(`preframr/`) and xpt (`preframr-xpt/`) dependents enumerated, and into a responsibility classification
of each live macros entry point.

**Hard gate (repeated up front):** every deletion step below is **GATED ON the decompiler codec port
landing** (the in-progress workstream in `/scratch/tmp/sidemu/`, replacing `events/` and rewiring
`reglogparser`/`regtokenizer`/`corpus`). Do NOT touch `/scratch/tmp/sidemu/`. The ONLY thing safe to
delete ahead of the port is `melody_audit.py` (top-level, 183 LOC, zero importers — manifest bucket 1a),
and that file is OUTSIDE the `macros/` directory.

---

## Part 1 — macros/ inventory + size

The `macros/` directory is **28 `.py` files = 5,759 LOC** (the reclaimable context-clutter). Three
top-level (non-`macros/`) modules are macros-codec adjuncts and are accounted with the tree because they
exist only to feed it: `coarsen_pass.py` (191, a `MacroPass` in `POST_NORM_PRE_VOICE_PASSES`),
`role_lane.py` (93, imports `macros.generator_fit.note_of`), `melody_audit.py` (183, dead-now). Adjunct
subtotal **467 LOC**. **Grand reclaimable ≈ 6,226 LOC** (5,759 macros/ + 467 adjuncts).

| module | LOC | arc it implements | refuted-registry citation |
|---|---|---|---|
| `macros/passes.py` | 632 | Transpose/DedupSet/HardRestart/Legato/Subreg/VoiceBlockOrder passes (live PASSES) | normalization for the event codec; the passes themselves are not a refuted *model* arc but die with the codec |
| `macros/loop_pass.py` | 735 | `LoopPass` (loop DEF→REF / reuse-as-forward-declaration) | AGENTS.md: frozen-table/codebook DEF→REF; "reuse is backward-looking only" |
| `macros/decoders.py` | 458 | `DECODERS` decode table → `generator_fit` + `pitch_grid` | event-codec decode machinery |
| `macros/loops.py` | 343 | `expand_loops`, OVERLAY_BODY_FREQ_DELTA, head-op sets | loop expansion (loop DEF→REF arc) |
| `macros/validators.py` | 320 | PATTERN_REPLAY / back-ref / codebook-ref stream validators | DEF→REF + inline-codebook validation (refuted forms) |
| `macros/op_contracts.py` | 316 | op id↔name tiers, reference-op producers, CODEBOOK_SPECS/TABLES | codebook DEF→REF op contracts |
| `macros/pipeline_check.py` | 315 | static pipeline-spec checker | Transform pipeline_spec form (DEF→REF orderlist) refuted |
| `macros/transform.py` | 281 | Transform base, TransformPipeline, registry, `ensure_default_transforms_registered` | DEF→REF transform/pipeline machinery |
| `macros/state.py` | 239 | `DecodeState`, `_build_decode_state`, FREQ_REGS_BY_VOICE | event-codec decode state |
| `macros/voice_lane.py` | 196 | voice-lane factorization (imports role_lane→generator_fit) | melody/timbre + voice factorization refuted |
| `macros/walker.py` | 186 | `FrameWalker` frame iteration over decode state | event-codec decode iteration |
| `macros/blocks.py` | 175 | `iter_self_contained_row_blocks`, `self_contain_slice` | block-refire / self-contained-window machinery for the codec |
| `macros/macro_contracts.py` | 175 | `RegClass`/`Effect`/`FrameEffect` static reasoner | test-time checker for the macro passes |
| `macros/transforms_bit_exact.py` | 150 | `@register` byte-exact Transform classes (DECOMPOSES_TO_ATOMS) | DEF→REF transform atoms |
| `macros/transforms_audio_bit_exact.py` | 122 | `@register` audio byte-exact Transforms | DEF→REF transform atoms |
| `macros/melody_segment.py` | 115 | `SegmentParams` (no registered class) | melody/timbre factorization refuted |
| `macros/flag_registry.py` | 114 | macro-flag names/conflicts/requires, `resolve_flags`, pkgutil glob | macro-pass flag config for the event codec |
| `macros/codebook.py` | 103 | `CODEBOOK_SPECS` bounded-table proof | codebook DEF→REF refuted as model form |
| `macros/transform_registry.py` | 92 | `ensure_default_transforms_registered` registry wiring | DEF→REF transform registry |
| `macros/pitch_grid.py` | 90 | `q_to_tuning`, `note_freq_at` | superseded by decompiler `pitch_universal_anchor` (LIVE ARC) |
| `macros/generator_fit.py` | 85 | `note_of`/`_tri_seq`/`recon`/`unzig` | superseded by decompiler generator recovery (LIVE ARC) |
| `macros/transforms_parser_stubs.py` | 81 | parser primitive Transform stubs | DEF→REF pipeline-spec validation |
| `macros/roles.py` | 72 | `distance_pair_role`/`frame_weight_role`, DistancePairSpec | distance-pair (DEF→REF distance) roles + frame-weight |
| `macros/decode.py` | 61 | `expand_ops` (called by reglogparser) | event-codec op expansion |
| `macros/freq_lut.py` | 45 | `midi_to_fn`, `fn_to_note_resid` | superseded by decompiler `freq_relative` (LIVE ARC) |
| `macros/default_pipeline.py` | 32 | `default_pipeline_spec()` | pipeline_spec DEF→REF form refuted |
| `macros/__init__.py` | 95 | PASSES / POST_NORM_PRE_VOICE_PASSES chains, `run_passes`, re-exports | live pipeline orchestration for the event codec |
| **macros/ TOTAL** | **5,759** | | |
| `coarsen_pass.py` (top-level adjunct) | 191 | `CoarsenPass` in POST_NORM_PRE_VOICE_PASSES | event-codec coarsening |
| `role_lane.py` (top-level adjunct) | 93 | role-lane via `generator_fit.note_of` | melody/role factorization |
| `melody_audit.py` (top-level adjunct, DEAD NOW) | 183 | melody analysis CLI (`unzig`) | melody/timbre factorization refuted; zero importers |

---

## Part 2 — full dependency map of macros/ across all three repos

### 2.1 preframr-tokens internal — live entry points → macros (the load-bearing edges)

| caller (file:line) | imports | what it does |
|---|---|---|
| `reglogparser.py:9` | `from preframr_tokens import macros` | module handle for `run_passes` / `run_post_norm_pre_voice_passes` |
| `reglogparser.py:14` | `macros.decode.expand_ops` | op→literal expansion on the parse path (`reglogparser.py:292`) |
| `reglogparser.py:1001` | `macros.run_passes(xdf, args)` | applies PASSES (Transpose/DedupSet/HardRestart/Legato/Subreg) — live normalize |
| `reglogparser.py:1005` | `macros.run_post_norm_pre_voice_passes(xdf, args)` | applies VoiceBlockOrder/Loop/Coarsen — live normalize |
| `regtokenizer.py:11` | `macros.loops.{EXTRA_ISOLATION_HEAD_OPS, MULTI_ROW_MACRO_HEAD_OPS}` | head-op sets driving tokenization isolation |
| `regtokenizer.py:395` | `macros.transform.{_REGISTRY, ensure_default_transforms_registered}` | `_decompose_missing_via_registry` walks registry to decompose missing tokens to atoms (`:400`) |
| `constrained_decode.py:28` | `macros.op_contracts.{CODEBOOK_SPECS, CODEBOOK_TABLES, STRUCTURAL_SUBREGS, STRUCTURAL_VALUE_ARRAYS}` | constrained-decode vocab/structural tables (NOTE: constrained_decode is a bucket-3 KEEP module — this edge must be re-provided) |
| `audit_primitives.py:100` | `macros.walker.FrameWalker` | frame iteration in the audit harness |
| `audit_primitives.py:147` | `macros.loops.expand_loops` | loop expansion in audit |
| `audit_primitives.py:148,167` | `macros.state.{_build_decode_state, FREQ_REGS_BY_VOICE}` | decode-state build in audit |
| `audit_primitives.py:168` | `macros.transform.collect_op_loss_tiers` | per-op loss-tier collection |
| `audit_primitives.py:169` | `macros.transform_registry.ensure_default_transforms_registered` | registry warm-up in audit |
| `tokenizer_config.py:48` | `macros.flag_registry.macro_flag_names` | macro-flag enumeration for named configs |
| `vocab_signature.py:7` | `macros.roles.frame_weight_role` | frame-weight role in vocab signature |
| `vocab_signature.py:8` | `macros.transform.{collect_op_loss_tiers, ensure_default_transforms_registered}` | loss tiers + registry warm-up |
| `blocks.py:12` | `macros.{iter_self_contained_row_blocks, self_contain_slice}` | self-contained block iteration (top-level blocks.py is bucket-3 KEEP) |
| `blocks.py:13` | `macros.validators.validate_stream` | stream validation in block builder |
| `coarsen_pass.py:9,10` | `macros.loops.OVERLAY_BODY_FREQ_DELTA`, `macros.passes_base.MacroPass` | CoarsenPass implementation (adjunct) |
| `melody_audit.py:11` | `macros.generator_fit.unzig` | melody audit (DEAD-NOW adjunct) |
| `role_lane.py:11` | `macros.generator_fit.note_of` | role-lane (adjunct) |
| `parse_audit.py:103` | `macros.validators.validate_stream` | parse audit (events-specific top-level, bucket 2c) |

**Internal-only macros modules** (reachable ONLY from within `macros/`, no non-macros importer):
`codebook`, `decoders`, `default_pipeline`, `freq_lut`, `loop_pass`, `macro_contracts`, `melody_segment`,
`passes`, `pipeline_check`, `pitch_grid`, `transforms_audio_bit_exact`, `transforms_bit_exact`,
`transforms_parser_stubs`, `voice_lane`. (These fall when the `macros/` package falls; no external rewire.)

### 2.2 `__init__.py` public re-exports backed by macros (the PyPI surface)

From `preframr_tokens/__init__.py`, the macros-backed `__all__` entries:
- `roles` (`__init__.py:17`): `DISTANCE_PAIR_OPS`, `DistancePairSpec`, `distance_pair_role`, `frame_weight_role`
- `transform` (`__init__.py:50`): `PassBackedTransform`, `PipelineEntry`, `RowExpandingTransform`, `Transform`, `TransformPipeline`, `ensure_default_transforms_registered`, `get_transform_class`, `register`
- `macros/__init__` (`__init__.py:60`): `codebook_live_ids`, `validate_back_refs`, `validate_codebook_refs`, `validate_pattern_overlays`, `validate_stream`
- `op_contracts` (`__init__.py:67`): `op_name_by_id`, `op_name_tiers`

**21 macros-backed public symbols** in `__all__` (12 of them named explicitly in the manifest's caveat #5).

### 2.3 Framework (`preframr/`) consumers

| file:line | symbol(s) | usage |
|---|---|---|
| `preframr/args.py:354` | `macros.flag_registry.{macro_flag_names, resolve_flags}` | `apply_macro_flags_to_args()` — resolves macro-flag config onto `args`, written back as canonical CSV into the checkpoint. **Called from `stftokenize.py:12`, `parse.py:15`, `trainer.py:209`, `predict.py:456,525`.** This is a DEEP framework dependency, broader than the manifest flagged. |
| `preframr/inference/predict.py:25-27` | `validate_back_refs`, `validate_pattern_overlays` (public re-exports) | post-generation "safety net" validating the decoded stream (`predict.py:307-308`) |
| `preframr/train/model/tier_map.py:15` | `op_name_by_id` (public re-export) | per-op-class accuracy reporting (`tier_map.py:133`) |
| `preframr/inference/event_gate.py`, `event_render.py` | `events.*` (sibling codec, not macros) | inference decode/render — dies with the events codec port |
| tests (framework): `tests/test_macro_flags_resolver.py`, `tests/train/test_learnable_class_loss.py`, `tests/train/test_model_ckpt_completeness.py`, `tests/predict/test_event_render.py`, `tests/train/test_regdataset.py` | flag_registry / transform / events | test-suite coupling; retire/port with the codec |

### 2.4 xpt (`preframr-xpt/`) consumers

Live (modules that exist today):
- `preframr_experiments/audit/macros.py:12` — `from preframr_tokens import macros as macros_mod` (whole module)
- `preframr_experiments/audit/residual_set_census.py:118` — `macros.flag_registry.macro_flag_names`
- `preframr_experiments/audit/learnability_triage.py:218,314` — `macros.flag_registry.macro_flag_names`, `macros.blocks.iter_self_contained_row_blocks`
- events-codec audits: `preencode_corpus.py`, `audit/phrase_census.py`, `audit/voice_interleave_audit.py`, `audit/event_position_audit.py`, `audit/abstraction_probe{,_ksweep}.py`, `audit/repeat_control.py`, `audit/instr_census.py`, `tests/test_voice_interleave_audit.py` — all `events.*` (bucket 2c).

**STALE / already-broken** (import macros modules that DO NOT EXIST in the current tree — verified absent:
`macros/skeleton_pass.py`, `macros/wavetable.py`, `macros/motif_pass.py`): all under `audit/probes/` —
`resid_engine_profile.py`, `resid_patch_mutation.py`, `resid_drum_footprint.py`, `resid_tail_profile.py`,
`raw_atom_diag.py`, `resid_patch_codebook.py`, `resid_wavetable_recurrence.py`, `resid_percussion.py`,
`resid_trace.py`, `resid_corpus_survey.py`, `resid_final_accounting.py`, `resid_archetype_survey.py`,
`freqtraj_interval_probe.py`, `resid_drum_codebook.py`. These are **not blockers** — they reference deleted
modules and already fail to import; they are pre-existing dead references to be cleaned opportunistically,
NOT part of the port gate.

---

## Part 3 — responsibility classification of each live macros entry point

Classification of every macros entry point the LIVE (non-test) code calls. Conservative rule: anything not
provably (a) or (c) is (b).

| entry point | callers | class | justification |
|---|---|---|---|
| `run_passes` / `run_post_norm_pre_voice_passes` (`macros/__init__`) | reglogparser | **(a)** event-codec-only | these are the OLD codec's normalize chain (Transpose/Dedup/HardRestart/Legato/Subreg/VoiceBlockOrder/Loop/Coarsen). The new codec parses dump→op-program with its OWN normalization. Dies with the port. |
| `decode.expand_ops` | reglogparser | **(a)** event-codec-only | op→literal expansion of the event codec's op vocabulary; replaced by the decompiler's VM(program) decode. |
| `loops.{EXTRA_ISOLATION_HEAD_OPS, MULTI_ROW_MACRO_HEAD_OPS, expand_loops, OVERLAY_BODY_FREQ_DELTA}` | regtokenizer, audit_primitives, coarsen_pass | **(a)** event-codec-only | loop DEF→REF machinery; reuse-as-forward-declaration is refuted. New codec is inline-streaming (backward-looking reuse only). |
| `transform.{Transform, _REGISTRY, ensure_default_transforms_registered, collect_op_loss_tiers, get_transform_class, register}` + the `transforms_*` registered classes | regtokenizer, vocab_signature, audit_primitives, args(framework), public `__all__` | **(a)** event-codec-only | DEF→REF transform/pipeline-spec form is refuted; `_decompose_missing_via_registry` only exists to decompose the event codec's compressed tokens to atoms. New codec emits atoms directly. **BUT** `collect_op_loss_tiers` / loss-tier metadata (Part 4 note) may need a (b) replacement — flagged in Part 6. |
| `op_contracts.{op_name_by_id, op_name_tiers}` | public `__all__`, framework `tier_map.py` | **(b)** substrate the new codec also needs | the model-facing op→name→tier map is needed for per-op-class accuracy reporting (`tier_map.py`) regardless of codec. The NEW codec has its own op-set (`sid_opset_inventory.md`); this responsibility must be re-provided as an op-id→name→tier table over the new op vocabulary. Post-port home: a new `op_inventory.py` / kept in the codec's public surface. |
| `op_contracts.{CODEBOOK_SPECS, CODEBOOK_TABLES, STRUCTURAL_SUBREGS, STRUCTURAL_VALUE_ARRAYS}` | constrained_decode.py (bucket-3 KEEP) | **(b)** substrate the new codec also needs | `constrained_decode.py` is KEEP substrate (constrained generation / vocab arrays). It needs structural-subreg / value-array tables to constrain decoding. The CODEBOOK_* tables are codec-specific (DEF→REF) and become (a), but the STRUCTURAL_SUBREGS / STRUCTURAL_VALUE_ARRAYS responsibility (which subregs are structural, their legal value sets) must be re-provided by the new codec for constrained decode. Split this entry: codebook tables → (a); structural tables → (b). |
| `roles.{distance_pair_role, frame_weight_role, DistancePairSpec, DISTANCE_PAIR_OPS}` | vocab_signature, public `__all__` | **(a)** event-codec-only | distance-pair roles describe DEF→REF distance ops; `frame_weight_role` weights event-codec ops. The new codec's vocab signature / frame weighting must re-derive weights over its OWN op-set — but that is a fresh implementation, not a port of these functions. Marked (a) for the FUNCTIONS; the *frame-weighting responsibility* is (b) and lives in the new codec's vocab_signature. |
| `state.{_build_decode_state, FREQ_REGS_BY_VOICE}` + `walker.FrameWalker` | audit_primitives | **(a)** event-codec-only | decode-state replay of the event codec's op stream; the audit harness reconstructs frames from event-codec ops. New codec brings its own decode/replay; audit_primitives' decode-dependent helpers get rewired to it. |
| `flag_registry.{macro_flag_names, resolve_flags, FLAG_CONFLICTS, FLAG_REQUIRES}` | tokenizer_config, framework args.py (DEEP), xpt audits | **(b)** substrate — but likely shrinks to near-empty | the macro-flag config system (`apply_macro_flags_to_args`, named configs, checkpoint CSV) is woven through framework train/parse/predict. The new codec has far fewer (ideally zero) pipeline flags. The *responsibility* (resolve a config name to a tokenizer arg set, persist it in the checkpoint) survives; the SET of flags collapses. Re-provide a minimal `macro_flag_names`/`resolve_flags` (or stub returning the new codec's config knobs) so `args.py` keeps compiling. Conservative (b). |
| `validators.{validate_stream, validate_back_refs, validate_pattern_overlays, validate_codebook_refs, codebook_live_ids}` | blocks.py, parse_audit.py, framework predict.py, public `__all__` | **(a)** event-codec-only — BUT the *safety-net responsibility* is (b) | these validate PATTERN_REPLAY / back-ref / inline-codebook DEF→REF structure — the OLD codec's stream grammar. They die with the port. HOWEVER `predict.py:307-308` uses them as a post-generation safety net (reject malformed generated streams). The new inline-streaming codec needs its OWN validator (residual=0 / op-grammar well-formedness). Functions = (a); safety-net responsibility = (b), re-provided by the new codec. |
| `blocks.{iter_self_contained_row_blocks, self_contain_slice}` | top-level blocks.py (KEEP), xpt learnability_triage | **(b)** substrate the new codec also needs | self-contained-window iteration (any prefix is a valid continuable song) is EXACTLY the inline-streaming invariant the new codec must preserve (AGENTS.md: "any prefix is a valid continuable song"). The new codec must provide window iteration. Re-provide in the new codec or in top-level blocks.py. |
| `melody_audit.py` (adjunct) | none | **(c)** pure dead | zero importers anywhere; refuted melody arc. Manifest bucket 1a. Removable now. |

**Tally:** (a) event-codec-only = **8** entry-point families (run_passes/decode, loops, transform-registry,
roles-functions, decode-state/walker, validators-functions, + codebook tables sub-split, + the passes
themselves). (b) substrate the new codec must re-provide = **5** (op_name/op_tiers, structural-subreg /
value-array tables, flag-config responsibility, frame-weighting/vocab-signature responsibility, self-contained
window iteration; plus the validator *safety-net* and loss-tier *metadata* responsibilities flagged within
(a) rows as (b)-responsibilities). (c) pure dead = **1** (`melody_audit.py`).

---

## Part 4 — staged removal plan (each step compiles/tests green)

**STEP 0 — NOW, ungated (the only pre-port deletion).** Delete `melody_audit.py` (183 LOC, top-level,
zero importers). No `__all__` change (it is not re-exported). Optionally delete the 12 stale `audit/probes/`
files in xpt that import non-existent `macros.skeleton_pass`/`wavetable`/`motif_pass` (already broken). Green
by construction (nothing imports them).

**STEP 1 — GATED ON the decompiler codec port landing: provide the (b) substrate in the new codec.**
Before any deletion, the new codec (from `/scratch/tmp/sidemu/`, ported into `preframr_tokens`) must expose:
1. `op_name_by_id()` / `op_name_tiers()` over the NEW op-set (consumed by framework `tier_map.py` + public `__all__`).
2. Structural-subreg / legal-value-set tables for `constrained_decode.py` (replacing `op_contracts.STRUCTURAL_SUBREGS` / `STRUCTURAL_VALUE_ARRAYS`).
3. A minimal flag-config surface — `macro_flag_names()` / `resolve_flags()` (or successor) so framework `args.apply_macro_flags_to_args` keeps resolving + persisting tokenizer config to the checkpoint.
4. Self-contained window iteration (`iter_self_contained_row_blocks` successor) for top-level `blocks.py` + xpt `learnability_triage`.
5. A frame-weighting / vocab-signature path (replacing `roles.frame_weight_role` + `transform.collect_op_loss_tiers` usage in `vocab_signature.py`).
6. A generated-stream safety-net validator (replacing `validate_pattern_overlays`/`validate_back_refs` in framework `predict.py`).

**STEP 2 — rewire the live tokens entry points off macros onto the new codec.** In ONE atomic change with the port:
- `reglogparser.py`: drop `from preframr_tokens import macros`, `macros.decode.expand_ops`, and the two `macros.run_*` calls (`:1001,1005`); replace with the new codec's parse/normalize/decode entry points.
- `regtokenizer.py`: drop `macros.loops` head-op sets and `_decompose_missing_via_registry`'s `macros.transform` registry walk — the new codec emits atoms directly (no registry decomposition).
- `corpus.py`: rewire its tokenize orchestration (currently `events.*`, manifest 2a) onto the new codec; macros is reached only transitively via reglogparser/regtokenizer, so it falls out here automatically.
- `constrained_decode.py`: swap `macros.op_contracts` import for the STEP-1 structural tables.
- `audit_primitives.py`: swap `macros.{walker,loops,state,transform,transform_registry}` for the new codec's decode/replay + op-tier helpers.
- `vocab_signature.py`: swap `macros.{roles,transform}` for the STEP-1 frame-weight/loss-tier path.
- `tokenizer_config.py`: swap `macros.flag_registry.macro_flag_names` for the STEP-1 flag surface.
- top-level `blocks.py`: swap `macros.{iter_self_contained_row_blocks, self_contain_slice, validators.validate_stream}` for the STEP-1 window iterator + new validator.
After STEP 2, no non-`macros/` tokens module imports `macros`.

**STEP 3 — delete buckets.** Delete the `macros/` directory (28 files, 5,759 LOC), the adjuncts
`coarsen_pass.py` (191), `role_lane.py` (93), and (per manifest) the `events/` tree (10 files, 1,969 LOC)
and the events-specific top-levels (`alphabet_projection.py`, `bpe_audit.py`, `render_play.py`,
`sid_frame_diff.py`, `parse_audit.py`, `dump_meta.py` — confirm digi-detect superseded first). Delete the
`tests/macros/` dir (10 files) + the ~22 macros/events integration tests, replaced by the new codec's tests.

**STEP 4 — prune `__all__`.** Remove the 21 macros-backed public symbols from `__init__.py` `__all__`
(lines 17-22, 50-67 import blocks + the matching `__all__` entries). Re-add ONLY the STEP-1 survivors with
their NEW backing module path (e.g. `op_name_by_id`, `op_name_tiers`, the new validator, the flag surface if kept public).

**STEP 5 — PyPI public-API break + version bump.** Removed/relocated public symbols:
`validate_back_refs`, `validate_pattern_overlays`, `validate_codebook_refs`, `validate_stream`,
`codebook_live_ids`, `Transform`, `TransformPipeline`, `PipelineEntry`, `PassBackedTransform`,
`RowExpandingTransform`, `register`, `get_transform_class`, `ensure_default_transforms_registered`,
`DistancePairSpec`, `DISTANCE_PAIR_OPS`, `distance_pair_role`, `frame_weight_role`, and possibly
`op_name_by_id` / `op_name_tiers` (relocated). This is a **breaking change** to the `from preframr_tokens import …`
surface → **minor/major version bump** (current `fallback_version = 0.52.0`; framework floors `>=0.53.0`).
The port release is the version that closes the drift; bump to at least `0.53.0` (the port-release floor the
cascade references — `design/infra/tokens_port_release_cascade.md`), treat the dropped symbols as the documented break.

**GATE:** STEPS 1-5 are ONE atomic PR, GATED ON the decompiler codec port landing. STEP 0 is the only piece
shippable ahead of it.

---

## Part 5 — framework + xpt refactor (concrete edits)

### Framework `preframr/` (do these IN the port PR, gated):

| file:symbol | current | change |
|---|---|---|
| `preframr/args.py:354` (`apply_macro_flags_to_args`) | `from preframr_tokens.macros.flag_registry import macro_flag_names, resolve_flags` | repoint to the STEP-1 flag surface (new public path). If the new codec is flag-free, reduce `apply_macro_flags_to_args` to a near-noop that still writes the codec-config CSV to the checkpoint. **5 call sites** (`stftokenize.py:12`, `parse.py:15`, `trainer.py:209`, `predict.py:456,525`) keep working because the function signature is preserved. |
| `preframr/inference/predict.py:25-27,307-308` | `validate_back_refs`, `validate_pattern_overlays` (public) | replace the two safety-net calls with the new codec's stream validator (STEP-1 #6). These are event-codec-only validators (DEF→REF/back-ref grammar); the new codec needs an inline-streaming well-formedness check. |
| `preframr/train/model/tier_map.py:15,133` | `op_name_by_id` (public) | repoint to the relocated `op_name_by_id` over the NEW op-set (STEP-1 #1). |
| `preframr/inference/event_gate.py`, `event_render.py` | `events.*` | rewire onto the new codec's tokens→writes / render path (events-codec death, manifest 2a). |
| framework tests `test_macro_flags_resolver.py`, `test_learnable_class_loss.py`, `test_model_ckpt_completeness.py`, `test_event_render.py`, `test_regdataset.py` | flag_registry / transform / events | retire or port to the new codec's equivalents. |

### xpt `preframr-xpt/` (gated, except the stale-probe cleanup which is ungated):

| file:symbol | change |
|---|---|
| `preframr_experiments/audit/macros.py:12` (`macros as macros_mod`) | retire or repoint to the new codec; this audit wraps the old macro pipeline. |
| `preframr_experiments/audit/residual_set_census.py:118`, `learnability_triage.py:218,314` | `macro_flag_names` → STEP-1 flag surface; `iter_self_contained_row_blocks` → STEP-1 window iterator. |
| events-codec audits (`preencode_corpus.py`, `audit/phrase_census.py`, `voice_interleave_audit.py`, `event_position_audit.py`, `abstraction_probe{,_ksweep}.py`, `repeat_control.py`, `instr_census.py`, `tests/test_voice_interleave_audit.py`) | retire/port with the events codec (manifest 2c). |
| 14 stale `audit/probes/*` importing non-existent `macros.skeleton_pass`/`wavetable`/`motif_pass` | **UNGATED** opportunistic delete — already broken imports, not part of the port gate. |

---

## Part 6 — risk / uncertainty list (resolve when the port interface freezes)

1. **`op_name_by_id` / `op_name_tiers` shape (b).** Framework `tier_map.py` depends on op-id→name→tier for
   per-op-class accuracy. The new codec's op-set differs; the EXACT id↔name↔tier contract is set by the
   port interface. **Resolve when the port op-inventory freezes.** Until then, do not delete `op_contracts`.

2. **`constrained_decode.py` structural tables (b).** It is bucket-3 KEEP but imports
   `macros.op_contracts.{STRUCTURAL_SUBREGS, STRUCTURAL_VALUE_ARRAYS, CODEBOOK_SPECS, CODEBOOK_TABLES}`.
   Whether the new codec needs all four, or only the structural pair, depends on whether constrained decode
   still references codebook tables. **Resolve when the new codec's constrained-decode interface freezes.**

3. **`flag_registry` reach is deeper than the manifest stated.** It is load-bearing in the FRAMEWORK
   (`args.apply_macro_flags_to_args`, 5 call sites, checkpoint CSV), not just tokens-internal. If the new
   codec is flag-free, `apply_macro_flags_to_args` must still persist the codec config to the checkpoint or
   `predict.load_model` can't reconstruct the tokenizer. **Resolve when the new codec's config surface freezes.**

4. **Validators look (a) but carry a (b) safety-net responsibility.** `validate_back_refs`/`validate_pattern_overlays`
   validate the OLD grammar (die with the port), but `predict.py` uses them as a generation safety net. The
   new codec MUST ship a replacement well-formedness validator or `predict.py` loses its reject-on-malformed
   guard. **Flag: do not drop the safety net silently.**

5. **`transform.collect_op_loss_tiers` loss-tier metadata.** Used by `vocab_signature.py` and `audit_primitives.py`
   to assign per-op loss tiers. Looks event-codec-only, but the loss-tier split (structural vs content) is a
   model-facing concept the new codec likely re-needs. **Resolve when the new codec defines its op loss tiers.**

6. **`frame_weight_role` / vocab signature.** Frame weighting over the new op-set is a fresh implementation,
   not a port of `roles.frame_weight_role`. Confirm the new `vocab_signature` provides frame weights before
   deleting `macros.roles`. **Resolve when the new codec's vocab_signature freezes.**

7. **`dump_meta.raw_is_digi` digi-detect** (manifest caveat #4) — REPLACED by the port's `digi_filter`, not
   deleted. Confirm supersession before removing `dump_meta` (used by `corpus.py`).

8. **Self-contained window invariant (`blocks.iter_self_contained_row_blocks`).** This encodes the
   "any prefix is a valid continuable song" rule (AGENTS.md hard rule). The new codec MUST preserve it. Used
   by top-level `blocks.py` (KEEP) and xpt `learnability_triage`. **Resolve: confirm the new codec's window
   iterator upholds the inline-streaming invariant.**

All eight resolve against the decompiler codec interface still being built in `/scratch/tmp/sidemu/` — none
can be finalized until the port interface freezes. Until then: **delete nothing in `macros/`; only `melody_audit.py`
(STEP 0) and the 14 already-broken xpt probes are safe to remove ahead of the port.**
