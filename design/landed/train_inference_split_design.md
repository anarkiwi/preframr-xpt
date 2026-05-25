## Status

**Landed** (2026-05-21, commits `2bfbaeb` + follow-up). Targeted at
the bind-mount mid-edit footgun: previously `preframr/` was
bind-mounted into every container (train, predict, audit), so editing
any file in `preframr/` during a running experiment risked breaking
the next arm's container spawn. This design split the bind surface so
editing inference-side code is always safe.

**Deviation from this doc:** the back-compat `preframr/predict/`
shim described below was reviewed and removed post-impl. Every
consumer of the old paths (`render.sh`, `predict-xpu.sh`,
`predict-nv.sh`, `render-jetson.sh`, `AGENTS.md` mentions, the
Dockerfile back-compat test line) was in this repo, so the shim
served no real downstream; updating callers to canonical
`preframr.inference.*` was cheaper than keeping five permanent shim
files. Below text describing the shim is retained for historical
context but does not reflect HEAD.

## Problem

`integration_tests/experiments/base.py:321` bind-mounts the whole
`preframr/` package into every container:

```python
"-v", f"{repo_preframr_pkg}:/preframr",
```

That mount is shared by:

- Training-stage containers (write checkpoints; iterate fast on
  `train/model.py`, `train/regdataset.py`, etc.).
- Predict / audit / generate-for-audit containers (read checkpoints;
  iterate fast on `predict/*`, `profile/audit_*.py`, etc.).

Consequence: any edit to `preframr/*` mid-experiment risks breaking
the next training arm's import. Today the agent has to remember "don't
touch `preframr/` during a run" — that's the footgun.

The two iteration cycles also have **independent natural cadences**:

- Train-side iteration is rare and high-impact (loss change, head
  shape, regdataset semantics) — by convention you stop the experiment
  before editing.
- Inference-side iteration is frequent and low-impact (audit script
  tweak, predict CLI flag, render pipeline) — there's no reason these
  should be coupled to a training run's container spawns.

## Goal

After this design lands:

1. **Editing inference-side code mid-training is safe** by
   construction. The next training arm container spawn picks up
   exactly the train-side code that was on disk at experiment launch
   (modulo train-side files, which can still be edited mid-run with
   the same care as today).
2. The split is **forward-compatible** with a future predict-image
   shrink (see `model_regdataset_decomposition_design.md` Phase B):
   inference code is already isolated, so a `Dockerfile.predict` that
   bakes only the inference set becomes a one-file addition.
3. Existing CLI entry points (`python3 -m preframr.predict.predict`,
   `python3 -m preframr.predict.render_play`) keep working via a
   back-compat shim — no churn for downstream callers in this commit.

## Non-goals

- **Not splitting the docker image.** One image (`anarkiwi/preframr`)
  still bakes both directories. The change is purely about which
  *bind-mount* gets applied per container role.
- **Not separating shared dependencies** (lightning, schedulefree,
  tensorboard, pyarrow). That's Phase B of the
  `model_regdataset_decomposition_design.md`, separately gated.
- **Not extracting `Model` from `LightningModule`.** Predict still
  imports `Model` from `preframr.train.model`; the bind-mount split
  is structurally orthogonal.
- **Not moving `audit_primitives.py`.** Used by train-time callback
  AND post-hoc audits; stays where it is. The relevant test is
  whether inference-side bind-mount needs the import path resolvable
  — yes, because audit scripts call it. Path stays
  `preframr.train.audit_primitives`; inference containers depend on
  the BAKED train-side copy.
- **Not adding role-aware Dockerfiles for train vs inference.** Both
  paths live in the same image.

## Layout

```
preframr/
├── train/                # bind-mounted into TRAIN containers only
│   ├── model.py          # Model(LightningModule), heads, losses, tier_map, etc.
│   ├── regdataset.py     # RegDataset class + DataLoader factories + get_prompt
│   ├── trainer.py        # Lightning Trainer wiring + train() entrypoint
│   ├── structural_loss.py
│   ├── audit_primitives.py
│   ├── block_mapper.py
│   ├── generalization_gate.py
│   └── __init__.py
├── inference/            # bind-mounted into PREDICT / AUDIT containers only
│   ├── __init__.py
│   ├── predict.py        # was preframr/predict/predict.py
│   ├── predict_lib.py    # was preframr/predict/predict_lib.py
│   ├── constrained_decode.py  # was preframr/predict/constrained_decode.py
│   └── render_play.py    # was preframr/predict/render_play.py
├── predict/              # back-compat shim — re-exports inference.* symbols
│   └── __init__.py       # from preframr.inference.predict import *  (etc.)
├── parse.py              # CLI: read by train-side
├── stftokenize.py        # CLI: read by train-side
├── args.py               # imported by both sides
├── utils.py              # imported by both sides
└── __init__.py
```

Symbol-by-symbol moves (only file moves; no behaviour change):

| current path | new path |
|---|---|
| `preframr/predict/predict.py` | `preframr/inference/predict.py` |
| `preframr/predict/predict_lib.py` | `preframr/inference/predict_lib.py` |
| `preframr/predict/constrained_decode.py` | `preframr/inference/constrained_decode.py` |
| `preframr/predict/render_play.py` | `preframr/inference/render_play.py` |
| `preframr/predict/__init__.py` | `preframr/predict/__init__.py` (shim only — re-exports from `preframr.inference.*`) |

## Bind-mount strategy in `base.py`

`_docker_run` gains a `role: str = "train"` parameter. The mount logic
becomes:

```python
if role == "inference":
    cmd += ["-v", f"{repo_pkg / 'inference'}:/preframr/inference"]
    # train side comes from the baked image — never bind-mounted into inference role
else:
    cmd += ["-v", f"{repo_pkg / 'train'}:/preframr/train"]
    # inference side comes from the baked image — never bind-mounted into train role
```

Top-level files (`args.py`, `utils.py`, `parse.py`, `stftokenize.py`,
`__init__.py`) come from the baked image regardless. Editing them
requires a rebake — acceptable; these are stable.

`run_arm` callers in `base.py` set `role="train"` (default). Audit /
predict / generate-for-audit launchers set `role="inference"`.

The Dockerfile bakes the full `preframr/` package as today. The
bind-mount overlays the relevant subdir only.

## Predict CLI back-compat shim

`preframr/predict/__init__.py` becomes:

```python
"""Back-compat shim: predict moved to preframr.inference.

`python3 -m preframr.predict.predict` still works because
`preframr.predict.predict` resolves to a thin re-export module that
imports everything from `preframr.inference.predict`.
"""
from preframr.inference.predict import *  # noqa: F401,F403
```

Plus per-module shim files:

```
preframr/predict/predict.py       # `from preframr.inference.predict import *; from preframr.inference.predict import main; ...`
preframr/predict/predict_lib.py   # `from preframr.inference.predict_lib import *`
preframr/predict/constrained_decode.py  # `from preframr.inference.constrained_decode import *`
preframr/predict/render_play.py   # `from preframr.inference.render_play import *`
```

Entry-point scripts in `Dockerfile` (`/preframr/predict/predict.py
--help`, `/preframr/predict/render_play --help`) continue to work via
the shim.

## Risks

- **Stale-bake hazard for shared code.** If you edit
  `preframr/train/model.py` and immediately run an audit, the audit
  container gets the **previous** baked version of `model.py` because
  it bind-mounts only `inference/`. Mitigation: same as today — rebake
  before running audits if you changed shared code. New behaviour is
  strictly less footgun-prone than current (which silently picks up
  whatever's on disk at next spawn). Document in `AGENTS.md`.
- **Test discovery.** `tests/predict/test_constrained_decode.py`
  imports `preframr.predict.constrained_decode`. The shim makes this
  resolve via re-export. Pytest discovery unchanged. Verified by
  grepping the test imports against the shim surface.
- **Coverage gate.** `.coveragerc` omits
  `preframr/predict/predict.py`. After the split, this should be
  `preframr/inference/predict.py`. One-line edit; verified before
  commit.
- **Linter directories.** `run_tests.sh` runs `pylint -E preframr` /
  `pylint -E /tests` / `pylint -E /integration_tests` — recursive,
  picks up `inference/` automatically. `black --check` same. No
  changes needed.
- **CI / build path.** `Dockerfile` does `COPY preframr /preframr`
  — recursive, picks up `inference/` automatically. The entry-point
  validation line at `Dockerfile:25`
  (`python3 -m preframr.predict.render_play --help`) keeps working
  via the shim. No Dockerfile change required.
- **Bind-mount mid-edit hazard timing.** This refactor itself is a
  preframr-tree-wide change. **Must land between experiments**, not
  during a running training arm. Phase 2 just finished; no current
  in-flight experiments; safe to execute now.

## Success criteria

1. `find preframr/inference -name "*.py"` returns 5 files (4 moved +
   `__init__.py`).
2. `from preframr.predict.predict import ...` (any prior import) keeps
   working — verified by `pytest tests/predict/` green.
3. `from preframr.inference.predict import ...` (new canonical) works
   — verified by a one-line import smoke in
   `tests/predict/test_constrained_decode.py`.
4. `./build.sh` green: 392 tests + lint + pyright + coverage ≥77%.
5. `run_memorize_int_test.sh` green: validates runtime instantiation
   under the new shim layout.
6. `_docker_run(role="train", ...)` mounts only `preframr/train/`;
   `_docker_run(role="inference", ...)` mounts only
   `preframr/inference/`. Verified by inspection of the cmd list a
   unit test asserts on (new test
   `tests/integration_tests/test_docker_run_mounts.py`).
7. AGENTS.md updated: "Mid-run code edits" rule weakened from
   "`preframr/` is bind-mounted" to "`preframr/train/` is
   bind-mounted into train containers; editing
   `preframr/inference/` mid-experiment is safe."

## Effort

- File moves + import-path edits across `preframr/` + tests: ~30 min.
- `base.py` role parameter + mount-selection logic: ~30 min.
- Shim modules: ~10 min.
- AGENTS.md edit: ~5 min.
- New mount-assertion test: ~15 min.
- `./build.sh` + `run_memorize_int_test.sh` validation: ~25 min.
- **Total: ~2 hours.**

## Execution order

1. Pre-flight: confirm no in-flight experiments (`ps -ef | grep
   experiments.run`).
2. Move predict files → inference/, leave back-compat shim in predict/.
3. Add `role` param to `_docker_run`; thread through call sites.
4. Update `.coveragerc` omit path.
5. Run unit tests (`pytest tests/`) inside the current image —
   pre-rebake — to confirm the shim works against the in-tree code.
6. Rebake (`./build.sh`) to validate the full lint + coverage stack.
7. Run `run_memorize_int_test.sh` on the new image.
8. Commit.

## References

- `model_regdataset_decomposition_design.md` — sibling refactor (file
  decomposition within `train/`); compatible. Either can land first.
- `AGENTS.md` — Mid-run code edits rule; rewritten by this design.
- `Dockerfile:25` — Entry-point sanity check that uses
  `preframr.predict.render_play`; back-compat shim preserves.
