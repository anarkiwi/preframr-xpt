## Status

**Pending review** (2026-05-21). Pure file decomposition, no behavior
change. Forward-shaped so a future predict-image-shrink (separate
design) is localised to one file instead of cross-cutting surgery.

## Problem

Two source files in `preframr/train/` dominate the repo by size and
mix unrelated concerns:

- `preframr/train/model.py` — 997 LoC. Bundles body factories, tier
  mapping, per-tier heads, MoS, loss functions, frame weighting,
  `Model(LightningModule)`, and runtime factories.
- `preframr/train/regdataset.py` — 903 LoC. Bundles the `RegDataset`
  class, parser-worker helpers, DataLoader factories, prompt
  construction.

Symptoms:

- File length makes targeted change risky — touching the Phase 1
  MoS head means navigating ~1000 lines of unrelated code.
- `Model.__init__` imports `pytorch_lightning`, `schedulefree`,
  `pyarrow`, `tensorboard` at module load. Any consumer of any symbol
  defined in `model.py` (heads, losses, factories) drags that wheel
  set. `predict/predict.py` currently imports `Model` from
  `train/model.py`, so the predict-side import path pulls the full
  train wheel set even though the inference path needs none of it.
- Future predict-image-shrink (separate, deferred design) requires
  separating the `nn.Module` from the `LightningModule` wrapper.
  Today that's surgery across 1000 lines; localising the Lightning
  class to one file makes the future extract a single-file change.

## Goals

Phase A (this design):

- Each new file ≤ 500 LoC.
- One coherent concern per file (body factories, heads, losses,
  Lightning class, factories; dataset class, loader factories,
  helpers, prompt).
- Existing consumer imports remain unchanged across the 30+ sites in
  `preframr/`, `tests/`, `integration_tests/` (back-compat via
  `__init__.py` re-exports).
- Each non-Lightning sub-file imports zero PL machinery (validated
  by test).
- Zero behavior change. Zero checkpoint-compat risk. Zero loss-curve
  change.

Phase B (out of scope here; queued as follow-up):

- Extract `nn.Module` core from `Model(LightningModule)`.
- Split `requirements.txt` into `requirements-runtime.txt` +
  `requirements-train.txt`.
- Add `Dockerfile.predict` + `Dockerfile.predict-jetson` variants.

Phase A delivers no deployment value on its own; it is preparation
that converts Phase B from cross-file surgery into a localised
single-file refactor.

## Non-goals

- No directory restructure outside `preframr/train/`. Modules stay
  in `preframr/train/`; no top-level `preframr/model/` or
  `preframr/data/` directory. (See "Refuted alternatives" below.)
- No symbol renames. Every public symbol keeps its current name to
  preserve the back-compat shim.
- No class-hierarchy changes. `Model` still inherits
  `LightningModule`. `RegDataset` still inherits `torch.utils.data.Dataset`.
- No state-dict key changes. Submodule attributes on `Model`
  (`body`, `head`, `unembed`, …) stay top-level so existing
  checkpoints load unchanged.

## Layout

```
preframr/train/model/                  # was preframr/train/model.py
├── __init__.py        # re-exports: Model, get_model, get_device,
│                      #   build_tier_map, chunked_cross_entropy,
│                      #   content_contrastive_loss, MoSHead,
│                      #   PerTierHeads, per_tier_unified_log_p,
│                      #   _build_vocab_frame_weight, _build_vocab_class_weight,
│                      #   _build_vocab_tier_id, _build_tier_vocab_partition,
│                      #   MODEL_GETTERS, MODEL_PRECISION, OPTIMIZER,
│                      #   SchedulerFreeModelCheckpoint, cpu_compile,
│                      #   cuda_compile
├── lightning.py       # Model(LightningModule). Only file in model/
│                      #   that imports pytorch_lightning. ~400 LoC.
├── bodies.py          # MODEL_GETTERS, get_gemma, get_gemma2,
│                      #   get_llama2, get_llama3_2, get_mistral,
│                      #   get_phi3, get_qwen2, MODEL_PRECISION,
│                      #   OPTIMIZER. ~150 LoC.
├── heads.py           # MoSHead, PerTierHeads, per_tier_unified_log_p,
│                      #   _mos_log_mixture. ~90 LoC.
├── losses.py          # chunked_cross_entropy, _chunked_list_cross_entropy,
│                      #   _cross_entropy_chunk, _cross_entropy_logit_chunk,
│                      #   content_contrastive_loss, _infonce_per_tensor,
│                      #   _build_vocab_frame_weight,
│                      #   _LOSS_TIER_ORDER, _N_LOSS_TIERS,
│                      #   _LOSS_TIER_TO_ID, _CONTENT_TIER_ID. ~250 LoC.
├── tier_map.py        # build_tier_map, _registry_op_tier_map,
│                      #   _vocab_id_to_class_tier,
│                      #   _build_vocab_class_weight,
│                      #   _build_vocab_tier_id,
│                      #   _build_tier_vocab_partition. ~150 LoC.
└── factory.py         # get_model, cpu_compile, cuda_compile,
                       #   get_device, SchedulerFreeModelCheckpoint.
                       #   Imports pytorch_lightning (for
                       #   SchedulerFreeModelCheckpoint). ~80 LoC.

preframr/train/regdataset/             # was preframr/train/regdataset.py
├── __init__.py        # re-exports: RegDataset, get_prompt, get_loader,
│                      #   get_val_loader, glob_dumps, iter_voiced_blocks,
│                      #   parser_worker, materialize_block_array,
│                      #   _reg_widths_path, _self_contained_prompt_df,
│                      #   LowMemoryRandomSampler
├── dataset.py         # RegDataset class. ~500 LoC.
├── helpers.py         # glob_dumps, iter_voiced_blocks, parser_worker,
│                      #   materialize_block_array,
│                      #   _self_contained_prompt_df,
│                      #   _reg_widths_path. ~200 LoC.
├── loaders.py         # get_loader, get_val_loader, _get_loader,
│                      #   LowMemoryRandomSampler. ~80 LoC.
└── prompt.py          # get_prompt. ~50 LoC. Single-purpose so
                       #   predict/predict.py imports from a focused
                       #   module rather than the dataset class file.
```

## Symbol-by-symbol move list

The complete inventory of public + module-private symbols and their
target file. Anything not in this list stays in `model.py` /
`regdataset.py` and is an error (forces re-review).

### `model.py` (current → target)

| current location | symbol | target |
|---|---|---|
| line 17 | `MODEL_PRECISION` | `bodies.py` |
| line 21 | `OPTIMIZER` | `bodies.py` |
| line 24 | `SchedulerFreeModelCheckpoint` | `factory.py` |
| line 47–166 | `get_gemma`, `get_gemma2`, `get_llama2`, `get_llama3_2`, `get_mistral`, `get_phi3`, `get_qwen2` | `bodies.py` |
| line 168 | `MODEL_GETTERS` | `bodies.py` |
| line 179 | `_registry_op_tier_map` | `tier_map.py` |
| line 187 | `_vocab_id_to_class_tier` | `tier_map.py` |
| line 225–228 | `_LOSS_TIER_ORDER`, `_N_LOSS_TIERS`, `_LOSS_TIER_TO_ID`, `_CONTENT_TIER_ID` | `losses.py` |
| line 231 | `_infonce_per_tensor` | `losses.py` |
| line 250 | `content_contrastive_loss` | `losses.py` |
| line 269 | `_build_vocab_class_weight` | `tier_map.py` |
| line 294 | `_build_vocab_tier_id` | `tier_map.py` |
| line 311 | `build_tier_map` | `tier_map.py` |
| line 326 | `_build_tier_vocab_partition` | `tier_map.py` |
| line 338 | `_mos_log_mixture` | `heads.py` |
| line 343 | `MoSHead` | `heads.py` |
| line 360 | `PerTierHeads` | `heads.py` |
| line 389 | `per_tier_unified_log_p` | `heads.py` |
| line 428 | `_build_vocab_frame_weight` | `losses.py` |
| line 481 | `_cross_entropy_chunk` | `losses.py` |
| line 490 | `chunked_cross_entropy` | `losses.py` |
| line 533 | `_chunked_list_cross_entropy` | `losses.py` |
| line 557 | `_cross_entropy_logit_chunk` | `losses.py` |
| line 571 | `Model(LightningModule)` | `lightning.py` |
| line 946 | `get_model` | `factory.py` |
| line 962 | `cpu_compile` | `factory.py` |
| line 971 | `cuda_compile` | `factory.py` |
| line 982 | `get_device` | `factory.py` |

### `regdataset.py` (current → target)

| current location | symbol | target |
|---|---|---|
| line 35 | `_reg_widths_path` | `helpers.py` |
| line 44 | `glob_dumps` | `helpers.py` |
| line 81 | `iter_voiced_blocks` | `helpers.py` |
| line 105 | `materialize_block_array` | `helpers.py` |
| line 146 | `parser_worker` | `helpers.py` |
| line 166 | `get_prompt` | `prompt.py` |
| line 208 | `_self_contained_prompt_df` | `helpers.py` |
| line 228 | `RegDataset` | `dataset.py` |
| line 842 | `LowMemoryRandomSampler` | `loaders.py` |
| line 855 | `_get_loader` | `loaders.py` |
| line 875 | `get_loader` | `loaders.py` |
| line 880 | `get_val_loader` | `loaders.py` |

## Import direction (avoid circularity)

Bottom-up only. Each layer imports from layers below it; nothing
above is imported back.

```
model/factory.py          (LightningModule callback + factories)
   ↓
model/lightning.py        (Model class)
   ↓
model/heads.py   model/losses.py   model/tier_map.py
   ↓                  ↓                  ↓
model/bodies.py         (no internal deps)
   ↓
(torch, torchtune, preframr_tokens)
```

```
regdataset/loaders.py     (DataLoader factories + sampler)
   ↓
regdataset/dataset.py     (RegDataset class)
   ↓
regdataset/helpers.py     regdataset/prompt.py
   ↓                            ↓
(torch, pandas, zstandard, preframr_tokens,
 preframr.train.block_mapper, preframr.args)
```

Verified by inspection — no symbol currently in a "lower" cluster
imports a symbol in a "higher" cluster.

## Back-compat shim

Every consumer site currently writes one of:

```python
from preframr.train.model import Model, get_device, …
from preframr.train.regdataset import RegDataset, get_prompt, …
```

Both forms continue to resolve unchanged because the new
`preframr/train/model/__init__.py` and
`preframr/train/regdataset/__init__.py` re-export every previously-
public symbol. Consumers are not modified in Phase A.

Sample `model/__init__.py`:

```python
from preframr.train.model.bodies import (
    MODEL_GETTERS, MODEL_PRECISION, OPTIMIZER,
    get_gemma, get_gemma2, get_llama2, get_llama3_2,
    get_mistral, get_phi3, get_qwen2,
)
from preframr.train.model.heads import (
    MoSHead, PerTierHeads, per_tier_unified_log_p,
)
from preframr.train.model.losses import (
    chunked_cross_entropy, content_contrastive_loss,
    _build_vocab_frame_weight,
)
from preframr.train.model.tier_map import build_tier_map
from preframr.train.model.lightning import Model
from preframr.train.model.factory import (
    get_model, get_device, cpu_compile, cuda_compile,
    SchedulerFreeModelCheckpoint,
)
```

(Underscore-prefixed private symbols are re-exported only when a
test currently imports them; otherwise they stay file-local.)

## PL-import isolation guarantee (test)

Add `tests/train/test_model_pl_isolation.py`:

```python
import importlib, sys
def test_heads_no_pl():
    sys.modules.pop("pytorch_lightning", None)
    importlib.import_module("preframr.train.model.heads")
    assert "pytorch_lightning" not in sys.modules

def test_losses_no_pl(): ...
def test_bodies_no_pl(): ...
def test_tier_map_no_pl(): ...
```

This test pins Phase A's load-bearing structural property: the PL
import remains localised to `lightning.py` + `factory.py`. Phase B
can later add `test_core_no_pl` once an `nn.Module` core is
extracted. If any future commit accidentally adds a PL import to
heads/losses/bodies/tier_map, this test fails fast.

## Risks

- **Checkpoint compatibility: zero.** Lightning state dicts are
  keyed by parameter names (`body.layers.0.attn.weight`,
  `head.heads.0.weight`, etc.) which are determined by
  `Model.__init__` submodule assignments. Phase A does not touch
  `Model.__init__`; submodule classes (`MoSHead`, `PerTierHeads`)
  move files but keep class names and constructor signatures
  identical. Pickled `MoSHead` instances are not in checkpoints
  (Lightning saves state dicts, not class instances).
  Verified by: existing `tests/train/test_model_ckpt_completeness.py`
  green post-refactor.
- **Coverage gate.** `.coveragerc` currently omits
  `preframr/train/regdataset.py`. Update to
  `preframr/train/regdataset/dataset.py` (the heavy class file). The
  three helper files (`helpers.py`, `loaders.py`, `prompt.py`) are
  small and unit-testable; expect them to clear coverage on their
  own. Net coverage impact: small positive (helpers gain coverage
  via existing tests that already exercise them; the dataset class
  stays at 46% but in a smaller file).
- **Circular imports.** Mitigated by the bottom-up direction above.
  Pre-flight check: grep each new file for `from preframr.train.model`
  / `from preframr.train.regdataset` self-references before commit.
- **Test discovery.** Pytest finds tests under `tests/`; nothing
  about test discovery changes. Black + pylint operate on
  directories; both already recurse into subpackages.
- **Bind-mount mid-run hazard.** `base.py:321` bind-mounts
  `preframr/` into every experiment container spawn. Phase A
  rename of `model.py` → `model/` would break the next arm's import
  if landed mid-experiment. Land only between experiments.
- **Black-formatting churn.** Possible minor reformatting where
  imports cluster; black runs as part of `run_tests.sh` so any
  churn surfaces in CI before commit.

## Success criteria

1. Every new file ≤ 500 LoC measured by `wc -l`.
2. `lightning.py` and `factory.py` are the only files in
   `preframr/train/model/` whose imports include `pytorch_lightning`,
   `schedulefree`, or `tensorboard` (validated by grep + the
   `test_model_pl_isolation.py` test above).
3. `dataset.py` is the only file in `preframr/train/regdataset/`
   exceeding 400 LoC.
4. `./run_tests.sh` green inside the docker image: 392 tests + lint
   + pyright + coverage ≥77%.
5. `./build.sh` green; all three images (`preframr`,
   `preframr-xpu`, `tensorboard`) rebake.
6. `integration_tests/run_memorize_int_test.sh` runs to completion
   on smoke tier (~5–6 min). This validates the runtime import
   paths still wire up under the back-compat shim — the docker
   `run_tests.sh` only covers unit tests, not the runtime
   instantiation path.
7. State dict compatibility: load the existing
   `content_floor_check` checkpoint with the refactored `Model`
   class and confirm `Model.load_state_dict(ckpt['state_dict'])`
   returns `IncompatibleKeys(missing_keys=[], unexpected_keys=[])`.
   This is the load-bearing ckpt-compat check; add as
   `tests/train/test_model_ckpt_load_back_compat.py`.

## Refuted alternatives

- **Move regdataset to `preframr/data/` (top-level).** Cleaner
  layered shape on paper but mechanically heavier: ~30 import-site
  updates with no offsetting deployment value (regdataset doesn't
  pull lightning anyway). Naming-wise `data/` is right;
  organisationally, in-place split delivers the readability win
  without churning unrelated consumers. Re-open this if/when Phase B
  proceeds and a clean shared-layer namespace becomes load-bearing.
- **Move `Model` to top-level `preframr/model/`.** Same reasoning as
  above; the Lightning coupling is the real constraint, not the
  directory. Phase B revisits.
- **Skip the back-compat `__init__.py` shim and update all consumer
  imports.** Adds ~50 mechanical edits across `preframr/`, `tests/`,
  `integration_tests/` for zero behavioral benefit. Defer until a
  consumer-side reason exists.
- **Mega-merge model.py + regdataset.py into one shared `train/core/`
  subpackage.** Conflates two unrelated concerns. Reject.

## Effort + queue order

- **Phase A wallclock:** ~1 day agent work. Dominated by the test
  pass + the `run_memorize_int_test.sh` smoke run (~6 min) + a fresh
  `./build.sh` (~15 min cold, ~5 min warm).
- **Order:**
  1. Phase 2 of `per_tier_heads_mini_body_large` lands (verdict or
     refute).
  2. Phase A executes in a single commit (file moves + `__init__.py`
     shims + `.coveragerc` path update + isolation test +
     ckpt-load back-compat test).
  3. Validate inside the docker image (`./build.sh`).
  4. Validate runtime (`run_memorize_int_test.sh`).
  5. Commit, then re-launch any in-flight follow-up experiment.

## Phase B preview (not in scope here)

Once Phase A lands, Phase B becomes a localised refactor inside
`preframr/train/model/lightning.py`:

1. Extract `ModelCore(nn.Module)` containing the body, heads,
   unembed, forward. Move to `preframr/train/model/core.py` (or a
   sibling location to be decided in the Phase B design).
2. `Model(LightningModule)` retains the same submodule layout but
   delegates `forward` to `ModelCore`. State-dict prefix is preserved
   by registering `core` submodules at the top level (e.g.
   `self._modules.update(core._modules)` or equivalent) — exact
   mechanism pinned in Phase B design.
3. `predict/predict.py` switches `from preframr.train.model import Model`
   to `from preframr.train.model.core import ModelCore` (or similar).
4. `requirements.txt` splits into runtime + train.
5. `Dockerfile.predict` (and `Dockerfile.predict-jetson`) variants
   bake the runtime set only.

Phase A's contribution to Phase B: steps 1 and 3 happen in
`lightning.py` (one file), not across `model.py`'s 1000 lines. The
"register core submodules at top level" trick has a small surface
area to validate.

## References

- `per_tier_heads_design.md` — Phase 1 lifted `MoSHead`,
  `PerTierHeads`, `per_tier_unified_log_p` into `model.py`;
  decomposition makes that block individually unit-testable in
  `tests/train/test_per_tier_heads.py` against
  `preframr.train.model.heads` directly.
- `preframr_tokens_extraction_design.md` — precedent for clean
  extraction with back-compat `__init__.py` shim.
- `../../AGENTS.md` — bind-mount rule
  (`integration_tests/experiments/base.py:321`).
