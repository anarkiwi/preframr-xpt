# preframr-experiments

Docker-driven experiment runner + spec registry for the preframr SID model
research stack. Extracted from `preframr/integration_tests/experiments/`;
shares no Python imports with the main repo (it only shells out to docker).

## Install

Editable, against the main repo's docker image:

```bash
pip install -e /scratch/anarkiwi/preframr-xpt
```

Tests:

```bash
pytest /scratch/anarkiwi/preframr-xpt/tests
```

## Running an experiment

```bash
preframr-experiments-run per_tier_heads_prodlike \
    --root /scratch/tmp/preframr_experiments \
    --only-arm per_tier_heads_mos4

# equivalent module-form
python3 -m preframr_experiments.run per_tier_heads_prodlike \
    --root /scratch/tmp/preframr_experiments
```

Spec names resolve under `preframr_experiments.specs.<name>` (e.g.
`per_tier_heads_prodlike`, `memorize`, `generalize`,
`content_floor_check`, `contrastive_mini_body_large`,
`contrastive_prodlike`, `per_tier_heads_mini_body_large`).

## Configuration

| env var | default | what for |
|---|---|---|
| `PREFRAMR_SRC_DIR` | `/scratch/anarkiwi/preframr/preframr` | Host path to the main repo's `preframr/` package; bind-mounted into preflight + parse + tokenize + train containers. |
| `PREFRAMR_DATASET_CACHE_DISABLE` | unset | Set to `1` to disable the parse + tokenize artefact cache at `/scratch/preframr/training-dumps/dataset_cache/`. Useful when a `pre_run_hook` mutates parse inputs. |

## Layout

```
preframr_experiments/
├── base.py            ExperimentSpec, Arm, run_arm, preflight, _robust_rmtree,
│                      dataset cache, docker shelling
├── run.py             CLI entry
├── metrics.py         metric extractors
├── report.py          markdown report renderer
├── rerender_report.py rerender utility
├── hvsc_version_check.py  HVSC release-version pin reader
├── data/              pinned tier lists + refuted registry
│   ├── smoke.list
│   ├── {mini,canonical,prodlike,prodlike_4x}/
│   │   ├── *.list
│   │   └── HVSC_VERSION
│   └── refuted/
│       └── <experiment>.md
├── preflight/         docker-mounted smoke scripts (never imported by runner)
│   ├── train_preflight_smoke.py
│   └── train_prodlike_oom_smoke.py
└── specs/             one module per experiment; each exposes `spec: ExperimentSpec`
```

## Compatibility

Specs reference pipeline-spec transform names registered in
`preframr-tokens` (`{"name": "slope"}` etc.). If a spec names a
transform absent from the main repo's installed `preframr-tokens`
version, the run fails at parse time. The runner itself does not
validate transform names.

## Related repos

- `preframr` — main research repo (model code, training, inference,
  audits).
- `preframr-tokens` — PyPI library for SID register-log parsing +
  tokenisation + macros (consumed by main repo + indirectly by spec
  pipeline-spec transform names).
- `preframr-audio` — PyPI library for SID audio rendering primitives.
