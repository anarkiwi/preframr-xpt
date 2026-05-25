# preframr-tokens extraction design

## Goal

Extract reglog parsing + tokenization + macro transforms + the
constants/helpers they need into a standalone, torch-free PyPI
package `preframr-tokens`. Main `preframr` repo retains the
model + training + loss + predict layer. Mirrors the audio split.

**Why now:** token juggling is refuted. Forward work is on the loss
function and the model. Parsing/tokenization is stable; keeping
~9K LOC of it in the main repo clutters every search, grep,
git-status, and agent-context-window.

## What moves to preframr-tokens

### From `preframr/core/` (~3,500 LOC, 11 files, torch-free)

| module | LOC | role |
|---|---:|---|
| `reglogparser.py` | 838 | dump → parsed df pipeline |
| `regtokenizer.py` | 457 | df → alphabet → unigram tokenizer |
| `stfconstants.py` | 423 | register IDs, op codes, dtypes |
| `engine_fingerprint.py` | 346 | engine clustering for eval-B pinning |
| `coarsen_pass.py` | 209 | tracker-export tool |
| `dump_meta.py` | 177 | per-dump metadata sidecar |
| `reglog_helpers.py` | 120 | palette load / save attrs |
| `alphabet_projection.py` | 69 | eval-set atom projection table |
| `parse.py` | 64 | CLI: dump → parsed parquet |
| `reg_mappers.py` | 37 | FreqMapper (also vendored in preframr-audio) |
| `stftokenize.py` | 19 | CLI: parsed → tokens.csv + tkmodel |

### From `preframr/core/macros/` (~5,700 LOC, 28 files, torch-free)

All of it. Each Transform owns `LOSS_TIER`, `OP_CODES`,
`SUBSTITUTABLE_OPS`, etc. as ClassVars. Model.py consumes these
declarative properties but doesn't define them.

### From `tests/core/` (split, ~half moves)

Moves with the package: `test_reglogparser.py`,
`test_regtokenizer.py`, `test_regtokenizer_more.py`,
`test_merge_token_df.py`, `test_coarsen_pass.py`,
`test_dump_meta.py`, `test_engine_fingerprint.py`,
`test_alphabet_projection.py`, all of `tests/core/macros/`.

Stays in main repo: `test_block_mapper.py`, `test_constrained_decode.py`,
`test_infonce_loss.py`, `test_learnable_class_loss.py`,
`test_model_*.py`, `test_regdataset*.py`, `test_render_play.py`,
`test_structural_loss.py`, `test_tighten_dtypes.py`.

### From `integration_tests/`

Move:
- `profile/aggregate_corpus_index.py`, `build_corpus_structural_index.py`,
  `build_prodlike_4x_list.py`, `digi_audit.py`, `encodability_metric.py`,
  `experimental_parser.py`, `hvsc_version_check.py`, `irq_audit.py`,
  `macros.py`, `parse.py`, `seq_budget_coverage.py`. (~11 files)
- `design/corpus_structural_index_design.md`,
  `hvsc_version_pinning_design.md`, `prodlike_tier_design.md`,
  `engine_fingerprint_evalb_design.md` — the parsing/tokenizer-side
  design docs.

Stay (model/train-side):
- `experiments/*` (specs, harness)
- `profile/audit_checkpoint_per_class.py`, `loop_detection_audit.py`,
  `per_class_acc_audit.py`, `prompt_conditioning_audit.py`,
  `predict.py`, `train_preflight_smoke.py`, `train_prodlike_oom_smoke.py`
- `design/multi_modal_objective_design.md`, `audio_fidelity_helper_design.md`,
  framework follow-ups (`auto_early_abort`, `flag_stage_routing`, etc.)

## What stays in main `preframr` repo

| module | reason |
|---|---|
| `model.py` | torch transformer + Lightning module |
| `regdataset.py` | torch.utils.data.Dataset; bridges tokens → tensors |
| `block_mapper.py` | torch tensor block packing |
| `constrained_decode.py` | torch logit masking during predict |
| `train.py`, `train_worker.py` | training loop |
| `structural_loss.py` | torch loss aux |
| `generalization_gate.py` | pytorch_lightning callback |
| `audit_primitives.py` | shared primitive set for gate + audit (torch-free but tightly coupled to gate) |
| `predict/*` | predict CLI; uses both packages |
| `utils.py` | get_logger; coupled to nothing, lives wherever |
| `args.py` | **split** — see below |

## `args.py` split (the awkward one)

Today `args.py` is one big `add_args(parser)` that defines parse +
tokenize + train + predict + CLI flags together. It imports
`MODEL_GETTERS` from `model.py`, which makes it torch-coupled.

Proposed split:
- `preframr_tokens.args` exposes `add_parse_args(parser)` and
  `add_tokenize_args(parser)`. No torch.
- Main repo's `preframr.core.args` exposes `add_train_args`,
  `add_predict_args`, and a `add_all_args(parser)` that composes the
  preframr-tokens functions plus its own.

CLI tools call the appropriate composer. Flag namespace stays unified.
`--pipeline-spec`, `--tkvocab`, `--reglogs` etc. are defined exactly
once (in preframr-tokens). `--learning-rate`, `--model`, `--tkmodel`-as-train-input
etc. defined in main repo.

## Dependency graph after the split

```
preframr-audio (PyPI v0.1.0)        preframr-tokens (NEW)
        |                                   |
        +------- preframr.predict <---------+
        |              |                    |
        |              v                    |
        |          preframr.core (model/train/gate)
        |              |                    |
        v              v                    |
        + integration_tests/audio_fidelity  +
```

Both packages can be installed standalone for downstream uses.
Main repo orchestrates training. No reverse imports from main repo
into either package.

## Phased migration plan

### Phase 1: scope (this doc) + reserve PyPI name
Decide layout, file the pending-publisher on PyPI for `preframr-tokens`.
Cost: ~1 hr.

### Phase 2: extract no-macros core modules
Copy `reglogparser`, `regtokenizer`, `stfconstants`, `dump_meta`,
`engine_fingerprint`, `coarsen_pass`, `reglog_helpers`,
`alphabet_projection`, `reg_mappers` into new repo. Rewrite imports.
Carve out `args.py` into `add_parse_args` / `add_tokenize_args`.
Move corresponding tests.

Validate: `pip install -e preframr-tokens`, run tests, coverage ≥85%.
Cost: ~1 day.

### Phase 3: extract `macros/`
Big enough to be its own phase. ~5,700 LOC, 28 files. Internal
imports are mostly self-contained (Transform registry).
Tests for macros (`tests/core/macros/*`) move too.

Validate: tests pass standalone. Transform registry registers
identically.
Cost: ~1 day.

### Phase 4: rewrite main repo imports
Change `from preframr.core.reglogparser import ...` to
`from preframr_tokens.reglogparser import ...` across all consumers
in `preframr/core/*`, `preframr/predict/*`, `integration_tests/*`,
`tests/*`. Add `preframr-tokens>=0.1.0` to `requirements.txt`.

Validate: main repo unit tests pass + integration tests pass.
Cost: ~half a day.

### Phase 5: delete from main repo
Remove `preframr/core/{reglogparser,regtokenizer,stfconstants,...}.py`
and `preframr/core/macros/`. Update Dockerfile if needed.

Cost: ~1 hr.

### Phase 6: publish + tag
Same flow as preframr-audio: pending publisher → push → tag `v0.1.0`
→ workflow uploads.

Cost: ~30 min.

**Total: ~3 days wallclock if focused, longer if interleaved.**

## Risks + open questions

1. **`stfconstants.py` is hot.** Almost every other module imports
   from it. Will become `preframr_tokens.constants`. Every import in
   main repo changes. Mechanical but high churn.

2. **Macro Transform registry.** `transforms_*.py` register via the
   `Transform` base class. Registration is module-import-side-effect.
   When preframr-tokens is imported, all transforms register. Main
   repo's `model.py::_registry_op_tier_map` relies on that. Verify
   that `import preframr_tokens` triggers all expected registrations
   (test the registry size).

3. **Pipeline-checker integration.** `validate_pipeline_spec` lives
   in `macros/pipeline_check.py` (moves). It's invoked from `args.py`
   `apply_pipeline_spec_to_args`. If `apply_pipeline_spec_to_args`
   moves to preframr-tokens, main repo's `train.py` calls into it.
   Boundary is clean; just need to confirm the function signature
   doesn't pull in torch.

4. **regdataset.py boundary.** It uses tokenizer outputs (`tokens.csv`,
   `tkmodel`). After the split, `regdataset.py` imports
   `preframr_tokens.regtokenizer.RegTokenizer` and reads the same
   dataframes. Stable interface; no risk.

5. **CLI script entry points.** `parse.py` and `stftokenize.py` are
   `python3 -m preframr.core.parse ...` callers (Dockerfile,
   `validate_branches.sh`, etc.). After move, callers do
   `python3 -m preframr_tokens.parse`. Need to update Dockerfile +
   `validate_branches.sh` + any shell harness that invokes them.

6. **Coverage target.** preframr-tokens has ~9K LOC. Existing tests
   cover most of it (parser + tokenizer + macros all heavily
   tested). Expect coverage in the 75-85% range without writing
   new tests. The 85% gate may be tight; OK to start at 75% and
   ratchet up.

7. **Versioning.** Pre-1.0 semver. Breaking changes to the parser /
   tokenizer alphabet break main repo's saved checkpoints. Tie
   `preframr-tokens` version to checkpoint compatibility — bump
   major when alphabet shape changes.

## Open decisions before Phase 2 starts

1. **PyPI name.** `preframr-tokens` chosen here; alternatives
   `preframr-parse`, `sid-tokens`, `preframr-encoding`. Picking
   determines pyproject + GitHub repo + URLs.
2. **Repo location.** `/scratch/anarkiwi/preframr-tokens` mirrors
   preframr-audio convention. Confirm.
3. **`utils.py`.** Single-function (`get_logger`). Vendor in both
   packages, or extract to a `preframr-shared` micro-package, or
   leave in main and re-import. Recommend: vendor (trivial,
   avoids a third package).
4. **`audit_primitives.py`.** Pure-python but tied to gate consumer.
   Leave in main repo (single consumer), or move to preframr-tokens
   alongside other torch-free? Recommend: leave in main — the gate
   imports it locally, no benefit to splitting.
5. **`engine_fingerprint.py`.** Used by eval-B pinning. Pure SID
   logic (no torch), belongs in preframr-tokens. Confirms.

## Acceptance criteria for the extraction

- `pip install preframr-tokens` works standalone, no torch in deps.
- `import preframr_tokens` triggers all Transform registrations.
- `validate_pipeline_spec` runs without torch.
- Main repo `requirements.txt` includes `preframr-tokens>=0.1.0`.
- `python3 -m preframr_tokens.parse --help` and
  `python3 -m preframr_tokens.stftokenize --help` work.
- Main repo `run_tests.sh` passes end-to-end after the extraction.
- Integration suite (`run_memorize_int_test.sh`,
  `run_generalize_int_test.sh`) passes unchanged behavior.
- Main repo `git ls-files | wc -l` drops by ~80 files; AGENTS.md
  loses the macro landscape detail (preframr-tokens carries it now).
