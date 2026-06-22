# preframr-tokens dead-wood removal manifest (white-box decompiler port)

**EXECUTED / SUPERSEDED (2026-06-20):** the clean-slate port deletes `events/` + `macros/`. Kept as the
record of the dead-wood audit that scoped it.

**Date:** 2026-06-19. **Mode:** READ-ONLY audit. No source touched.
**Scope:** `/scratch/anarkiwi/preframr/preframr-tokens/preframr_tokens/` plus the xpt
`preframr_experiments/audit/` consumers. Goal: de-risk the LIVE-ARC port
("port codec + digi-detect into preframr-tokens, replacing `events/`, strip dead wood").

## Headline (read this first)

The intuitive plan — "delete the `macros/` tree, it's all refuted arcs (codebook/DEF→REF,
melody, loops, generator_fit)" — is **WRONG / dangerous**. The `macros/` tree is **not** a
parallel refuted experiment that happens to sit beside the live codec; it **IS the current
shipped event codec's normalization + decode + transform-registry machinery**, and it is
**unconditionally imported and called on the live parse/tokenize/decode path** today:

- `reglogparser._parse` calls `macros.run_passes` + `macros.run_post_norm_pre_voice_passes`
  **unconditionally** (`reglogparser.py:1001,1005`).
- `regtokenizer._decompose_missing_via_registry` calls
  `ensure_default_transforms_registered()` (`regtokenizer.py:400`), which glob/explicit-imports
  the `transforms_*` modules into the live Transform registry.
- `reglogparser` decode goes `expand_ops` → `macros.decode` → `macros.decoders` →
  `generator_fit` + `pitch_grid` (`macros/decoders.py:72-74`).
- The public `__init__.py` re-exports 12+ `macros.*` symbols (roles, transform, validators,
  op_contracts) that the framework `preframr` imports (e.g. `validate_back_refs`,
  `validate_pattern_overlays` used in `preframr/inference/predict.py:307-308`).

**Therefore almost all of `macros/` and all of `events/` is "DIES WITH THE PORT" (bucket 2),
not "DEAD NOW" (bucket 1).** It cannot be removed until the new dump→op-program codec lands
and replaces the `reglogparser`→`run_passes`→`regtokenizer` pipeline wholesale. The
refuted-arc labels in AGENTS.md (codebook DEF→REF, melody/timbre, loop_pass, etc.) are
refuted *as model-facing forms / as the future representation* — but the code that implements
them is still wired into the CURRENT byte-exact codec that ships in tokens 0.51.0. Refuted ≠
unreferenced.

**Bucket 1 (DEAD NOW) is small and conservative: 1 file, 183 LOC** (`melody_audit.py`), with
3 more modules (903 LOC including melody_audit) that are *probably* dead-now but carry an
import-time coupling caveat and are listed with that caveat rather than as clean removals.

---

## Evidence method

- `grep -rln` for every `preframr_tokens.<module>` / `from preframr_tokens import <name>`
  across (a) the tokens package itself, (b) the framework `preframr/preframr/`, (c) xpt
  `preframr_experiments/`, and (d) `preframr-tokens/tests/`.
- A module is **DEAD NOW** only if it has **zero importers in (a) the live runtime path**,
  **zero in (b) framework**, is **not** in the live `PASSES` / `POST_NORM_PRE_VOICE_PASSES`
  chains (`macros/__init__.py:39-53`), is **not** reachable from `decoders`/`state` (the live
  decode path), and is **not** pulled in by the `ensure_passes_registered()` /
  `ensure_default_transforms_registered()` import-time glob.
- The "refuted registry" is (i) the single file `data/refuted/unigram_bpe_content_generalization.md`
  and (ii) the **AGENTS.md "Refuted (don't re-propose)" section** + the LIVE-ARC hard rules,
  which is where the per-arc kills (`melody/timbre factorization`, `instrument DEF→REF`,
  `per_tier_heads`, frozen-table/codebook-id DEF→REF) actually live.

---

## Live config gate (decisive for what is "active" vs "registered-but-off")

`tokenizer_config.py` is the source of truth for which macro passes the production codec runs:

- `REGISTERED_MACROS = (hard_restart_pass, legato_pass_c2, legato_pass_c4,
  voice_canonical_block_order, loop_pass, loop_transposed)` — the **production-active** passes
  (`DEFAULT_PIPELINE = "full_macros"`).
- Module docstring, verbatim: *"`full_macros` is the production-registered subset
  (`REGISTERED_MACROS`), NOT every flag — **the rest are experimental/refuted** and corrupt
  FRAME svt combined."*

So `HardRestartPass`, `LegatoPerClusterPass`, `VoiceBlockOrderPass`, `LoopPass` (+ CoarsenPass,
TransposePass, DedupSetPass, SubregPass in the always-on `PASSES` list) are LIVE. The refuted
experimental transforms are *registered but OFF*. **But registered-but-off still means the
module is imported at registry-build time**, so they are bucket 2 (die with the codec), not
bucket 1.

---

## Bucket 1 — DEAD NOW (removable independent of the port)

### 1a. Clean removal (zero importers anywhere — live, framework, tests-as-consumer, xpt)

| file | symbols | LOC | refuted citation | grep evidence |
|---|---|---|---|---|
| `melody_audit.py` | `melody_audit` CLI/probe (uses `generator_fit.unzig`) | **183** | AGENTS.md Refuted: *"melody/timbre factorization"* NULLED on free-running generation; the §7D melody split (`unigram_bpe_content_generalization.md`) found NI_* high-entropy, claim does not narrow | `grep -rln melody_audit` → **0** importers in tokens (`tokens_internal=0`), **0** in framework, **0** in `preframr_experiments/`. No `__main__` consumer wires it. Standalone melody analysis script for a refuted arc. |

**Bucket 1 clean total: 1 file, 183 LOC.**

### 1b. Dead-arc modules with an import-time caveat (recommend removing WITH the port, or only after cutting the registry/glob that reaches them)

These implement refuted arcs and are **not applied** by the live pipeline, but each is reached
by an **import-time side effect** (the `pkgutil` glob in `flag_registry.ensure_passes_registered`
and/or the explicit import in `transform_registry.ensure_default_transforms_registered`, both
of which run on the live tokenizer build). They are safe to delete only if the deleter *also*
prunes those import sites. Classified here (not bucket 2) because the arc is refuted and the
live codec never *uses* the registered class — but flagged, not asserted clean.

| file | symbols | LOC | refuted citation | coupling caveat |
|---|---|---|---|---|
| `macros/melody_segment.py` | `SegmentParams` (no `Transform`/`register`) | 115 | AGENTS.md: melody/timbre factorization refuted | `imported_by=0` direct; pulled in by `ensure_passes_registered` pkgutil glob (`flag_registry.py:85`). 1 test file references it. Defines no registered class, so glob import is inert — strongest 1b candidate. |
| `macros/macro_contracts.py` | `RegClass`, `Effect`, `FrameEffect`, static reasoner | 175 | not a refuted arc per se — a test-time static checker for the macro passes; dies with the macro passes | `imported_by=0` in runtime; 1 test file. Test infra for the codec being replaced. |
| `macros/freq_lut.py` | `midi_to_fn`, `fn_to_note_resid` | 45 | superseded by the decompiler's `freq_relative` / `pitch_universal_anchor` (LIVE ARC); old freq LUT | `imported_by=0` runtime; 2 test files; pkgutil glob only. |
| `macros/default_pipeline.py` | `default_pipeline_spec()` | 32 | the Transform **pipeline_spec** form (DEF→REF/orderlist) is refuted (AGENTS.md: frozen-table/codebook-id DEF→REF) | `default_pipeline_spec` has **0** non-test callers (`grep` outside tokenizer_config/flag_registry → none); 2 test files; pkgutil glob only. |

**Bucket 1b total (caveated): 4 files, 367 LOC.** Combined bucket 1 ceiling (1a+1b) = **5 files,
550 LOC** — but only the 183-LOC `melody_audit.py` is a *clean* dead-now removal.

> Conservative note: `transforms_bit_exact.py` (150) and `transforms_audio_bit_exact.py` (122)
> were initially candidates but are **demoted to bucket 2** — they `@register(...)` the
> Transform classes that `regtokenizer._decompose_missing_via_registry` (`regtokenizer.py:400-408`)
> walks to decompose missing tokens to atoms on the **live** path. Removing them changes live
> tokenizer behavior. See bucket 2. `transforms_parser_stubs.py` (81) likewise registers parser
> primitives used to validate pipeline specs — bucket 2.

---

## Bucket 2 — DIES WITH THE PORT (pending the new codec; do NOT remove until it lands)

### 2a. The `events/` codec (the thing the new codec replaces) — 10 files, 1,969 LOC

`events/__init__.py`, `events/oracle.py` (195), `events/stream.py` (291), `events/inline.py`
(312), `events/dataset.py` (228), `events/instrument.py` (273), `events/seqref.py` (351),
`events/constrained.py` (139), `events/generate.py` (78), `events/pipeline.py` (96).
(Plus the compiled-only `events/varint` — no `.py` source present, only `__pycache__`.)

Evidence it is the current codec, and what blocks removal:
- `corpus.py:27-29` imports `events.dataset`, `events.oracle`, `events.stream` (the tokenize
  orchestrator).
- `bpe_audit.py:11,63,71` uses `events.inline` / `events.dataset.events_alphabet`.
- Framework hard dependency: `preframr/inference/event_render.py` imports
  `events.constrained.EventStreamState`, `events.generate.tokens_to_writes/writes_to_dump_df`;
  `preframr/inference/event_gate.py` imports `events.dataset.unit_starts`,
  `events.generate.tokens_to_writes`. Inference cannot run without these until the new
  decode/render lands.
- `seqref.py` (instrument **DEF→REF** seqref) and `instrument.py` implement the
  instrument-DEF→REF arc — **refuted as a model-facing form** (AGENTS.md) but **still the
  current codec's instrument channel**. Refuted-but-load-bearing.

### 2b. The `macros/` codec machinery (current parse/decode/transform pipeline) — load-bearing TODAY

These are reached unconditionally on the live path and MUST survive until the dump→op-program
codec replaces `reglogparser`+`regtokenizer`:

- **Live PASSES chain** (`macros/__init__.py:39-53`, applied at `reglogparser.py:1001,1005`):
  `macros/passes.py` (632: TransposePass, DedupSetPass, HardRestartPass, LegatoPerClusterPass,
  SubregPass, VoiceBlockOrderPass), `macros/passes_base.py` (131), `macros/loop_pass.py` (735,
  `LoopPass` — in REGISTERED_MACROS `loop_pass`/`loop_transposed`), `macros/loops.py` (343,
  `expand_loops`, imported by regtokenizer/parse_audit/audit_primitives), `coarsen_pass.py`
  (191, CoarsenPass).
- **Live decode chain**: `macros/decode.py` (61, `expand_ops` — called by
  `reglogparser.py:14,292`), `macros/decoders.py` (458, `DECODERS`), `macros/walker.py` (186,
  `FrameWalker`), `macros/state.py` (239, decode state — 10 importers), `macros/generator_fit.py`
  (85, `note_of`/`_tri_seq`/`recon`/`unzig` used by decoders + role_lane), `macros/pitch_grid.py`
  (90, `q_to_tuning`/`note_freq_at` used by decoders).
- **Live transform registry** (reached via `ensure_default_transforms_registered`, a public
  export called from `regtokenizer.py:400` + `vocab_signature.py:32`): `macros/transform.py`
  (281), `macros/transform_registry.py` (92), `macros/pipeline_check.py` (315),
  `macros/transforms_bit_exact.py` (150), `macros/transforms_audio_bit_exact.py` (122),
  `macros/transforms_parser_stubs.py` (81), `macros/flag_registry.py` (114).
- **Live contracts/validators/roles** (public-exported, framework-used): `macros/op_contracts.py`
  (316, `op_name_by_id`/`op_name_tiers` in `__init__`), `macros/codebook.py` (103, `CODEBOOK_SPECS`
  used by validators+op_contracts — codebook DEF→REF refuted as model form but still the codec's
  bounded-table proof), `macros/validators.py` (320, `validate_back_refs`/`validate_pattern_overlays`
  used by `predict.py`), `macros/roles.py` (72, `distance_pair_role`/`frame_weight_role` in
  `__init__`), `macros/blocks.py` (175, self-contained block iteration), `macros/voice_lane.py`
  (196).

### 2c. Events-/codec-specific top-level modules and audits (die with the codec)

- `alphabet_projection.py` (75, used by `corpus.py:17` for event-codec atom projection),
  `dump_meta.py` (244, `raw_is_digi`/`read_meta` — but **digi-detect is part of the NEW codec**;
  the port replaces this with the prototype `digi_filter`, so it is "ported/replaced" rather than
  pure-deleted), `bpe_audit.py` (84, BPE-over-events audit, has `__main__`),
  `render_play.py` (224, event render CLI, has `__main__`), `sid_frame_diff.py` (148, used by xpt
  probe `audit/probes/sid_frame_diff.py`), `role_lane.py` (93, used by `macros/voice_lane.py`),
  `melody_audit.py` is bucket 1 (no consumer), `parse_audit.py` (115, used by reglogparser +
  xpt `hvsc_audit_sweep`).
- xpt event-specific audits (keep until the new codec's audits exist, then port/retire):
  `audit/event_position_audit.py`, `audit/free_running_gap_audit.py`,
  `audit/codebook_coupling.py`, `audit/melody_*` (melody_baseline_corpus, melody_compare_arms,
  melody_features, melody_gap_distribution, melody_predictability, melody_score_generation),
  `audit/repeat_control.py`, `audit/voice_interleave_audit.py`,
  `audit/audit_macro_fidelity_probe.py`, `audit/audit_seq_order_norm.py`. These import
  refuted-arc surfaces and exist to probe the event codec.

---

## Bucket 3 — LIVE / KEEP (substrate the new codec reuses)

These are the parser/tokenizer/audit substrate the white-box codec will sit on top of. Keep.

- **Parser/tokenizer core (reused):** `reglogparser.py` (1018, `RegLogParser` + `combine_reg`,
  `prepare_df_for_audio`, `read_initial_irq`, `remove_voice_reg`), `regtokenizer.py` (502,
  `RegTokenizer`), `blocks.py` (230, `iter_voiced_blocks`, `self_contained_prompt_df`,
  `reg_widths_path`), `corpus.py` (752, `Corpus`/`TokenizeMeta` — orchestrator; will be rewired
  to the new codec but the class/Corpus contract is the substrate), `parse_runner.py` (58,
  `parse_corpus`, framework-used).
  - CAVEAT: `reglogparser`/`regtokenizer`/`corpus` currently *call into* the macros/events
    pipeline. They are KEEP-but-EDIT: the codec internals get swapped, the public surface and the
    raw-dump reading (`_read_dump`) stay. Do not delete; modify in place during the port.
- **Constrained decode / vocab / tiers (model-facing substrate):** `constrained_decode.py`
  (1123, StreamState/VocabArrays/precompute_* — public), `vocab_signature.py` (104),
  `tier_classify.py` (50, public tier ids), `token_weighting.py` (101, framework-used),
  `tokenizer_config.py` (107, public config), `audit_primitives.py` (200, public audit harness —
  `tier_accuracy`, `detect_tail_cycle`, `op_atom_profile`, `register_state`).
- **Low-level shared:** `stfconstants.py` (164, heavily public), `utils.py` (`to_int64_arrays`,
  5 internal importers), `reg_match.py` (83, `reg_class` public), `reg_mappers.py` (2 importers),
  `palette_io.py` (load/dump palettes, public + 4 importers), `engine_fingerprint.py` (242,
  public stable namespace per `__init__` docstring), `train_worker.py` (145, used by
  regtokenizer).
- **xpt audit harness (keep):** `audit/audit_checkpoint_per_class.py`, `content_tier_report.py`,
  `audit/README.md` readers, `audit/parse.py`, `audit/predict.py`, `audit/digi_audit.py`,
  `audit/engine_fingerprint.py`, the `audit/probes/resid_*` byte-exact survey suite (these feed
  the residual-zero discipline of the new codec — explicitly LIVE-ARC infrastructure),
  `audit/learnability_triage.py`. The `specs/generalize*.py` are the live training A/Bs (no
  direct `preframr_tokens` import; they drive the runner).

---

## Totals and operator decisions

- **Bucket 1 (DEAD NOW), clean:** `melody_audit.py` — **1 file, 183 LOC**. Zero importers
  anywhere; refuted melody/timbre arc; safe to delete now.
- **Bucket 1 (DEAD NOW), caveated (delete WITH the import-site prune):** +`macros/melody_segment.py`,
  `macros/macro_contracts.py`, `macros/freq_lut.py`, `macros/default_pipeline.py` — **+4 files,
  +367 LOC** (ceiling 5 files / 550 LOC).
- **Bucket 2 (DIES WITH PORT):** `events/` (10 files, 1,969 LOC) + the live `macros/` codec
  machinery (~22 files, ~6.4k LOC) + events-specific top-level + xpt event audits. Remove only
  when the dump→op-program codec replaces the parse/tokenize path.
- **Bucket 3 (LIVE/KEEP):** parser/tokenizer/constrained-decode/vocab/tier substrate + xpt audit
  harness + residual-survey probes.

### Load-bearing surprises requiring an operator decision BEFORE deletion

1. **The whole `macros/` tree is load-bearing, not dead.** The naive "delete macros/ (it's the
   refuted codebook/melody/loop arcs)" is unsafe: `reglogparser` and `regtokenizer` call it
   unconditionally on the live path. It is bucket 2 (port-coupled), not bucket 1. **This is the
   single biggest correction to the dead-wood assumption.**

2. **`transforms_bit_exact.py` / `transforms_audio_bit_exact.py` / `transforms_parser_stubs.py`
   look refuted (Transform/pipeline_spec = DEF→REF form) but are imported on every live tokenizer
   build** via `ensure_default_transforms_registered()` (public export, called at
   `regtokenizer.py:400`) and their registered classes are walked by
   `_decompose_missing_via_registry`. Do NOT mark dead-now. Demoted to bucket 2.

3. **`seqref.py` (instrument DEF→REF) + `instrument.py` are refuted-as-model-form but are the
   current codec's instrument channel** and are imported by `corpus.py`/framework inference.
   Refuted ≠ removable here. Bucket 2.

4. **`dump_meta.py` digi-detect is REPLACED, not deleted.** The port brings the prototype
   `digi_filter` (raw-write-density threshold). Operator should confirm the new digi-detect
   supersedes `dump_meta.raw_is_digi` (used by `corpus.py:26`) before removing `dump_meta`.

5. **The `macros.*` symbols in the public `__init__.py` `__all__`** (`validate_back_refs`,
   `validate_pattern_overlays`, `op_name_by_id`, `op_name_tiers`, `distance_pair_role`,
   `frame_weight_role`, `codebook_live_ids`, Transform/Pipeline classes,
   `ensure_default_transforms_registered`) are part of the PyPI public API. Removing the backing
   `macros/` modules is a **public-API break** — the port must decide which of these survive into
   the new codec's `__init__` before deletion, or it breaks downstream `from preframr_tokens import`
   sites (the framework `preframr` imports several).

**Recommendation:** treat the port as one atomic change — land the new codec, rewire
`corpus`/`reglogparser`/`regtokenizer`/inference onto it, then delete buckets 1+2 and prune the
public `__init__` in the same PR. The only thing safe to delete *ahead* of the port is
`melody_audit.py` (183 LOC).
