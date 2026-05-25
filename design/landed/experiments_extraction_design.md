# Extract experiment runner into sibling `preframr-experiments` repo

**Status:** LANDED 2026-05-21. Sibling repo at
`/scratch/anarkiwi/preframr-xpt`. No PyPI publish; consumed via
`PYTHONPATH=/scratch/anarkiwi/preframr-xpt` from the main repo's
runner CLI. Final shape diverged from this draft on one point: the
`--profile-dir` injection was unnecessary (metrics.py doesn't
actually subprocess-invoke profile scripts), so the only cross-repo
edge is the `PREFRAMR_SRC_DIR` env var for the preframr/ source
bind-mount. Original Status was "Draft, pending review".

## Problem

`integration_tests/experiments/` is a docker-shelling-out runner
with **zero preframr / torch / pytorch_lightning imports**. Today
it lives inside the main repo and pretends to be "tests", which:

1. Couples runner iteration to main-repo CI cycles (every runner
   tweak rebuilds the full ~2 GB train image to run tests).
2. Conflates two distinct concerns (research code vs research
   orchestration) under one tree, so new contributors don't see
   the boundary.
3. Blocks reuse: the runner shape (ExperimentSpec + docker-driven
   parse/tokenize/train + cached dataset prep + report generation)
   is general; other audio-ML projects could use it unchanged.
4. Makes spec authoring heavier than it needs to be -- adding a
   spec module currently lives in a feature-branch PR against
   the main repo even though it touches no model code.

## Hypothesis

Extract the runner into a sibling repo `preframr-experiments`,
publish to the local PyPI mirror, install in the main-repo
docker image. The split is clean because the runner only talks
to docker (no Python-level coupling to model code); the only
real edges are file-resource sharing (preflight smoke scripts,
profile/ research utilities) which factor cleanly into a small
`--profile-dir` injection point.

## Boundary survey (verified before drafting)

### Cleanly moves to new repo

Pure orchestration -- no `preframr`, `preframr_tokens`,
`preframr_audio`, `torch`, or `pytorch_lightning` imports:

- `integration_tests/experiments/base.py` (933 LoC) -- spec,
  arm, runner core, robust rmtree, dataset cache, docker shelling,
  preflight, TB sidecar mgmt.
- `integration_tests/experiments/run.py` (146) -- CLI entry.
- `integration_tests/experiments/metrics.py` (292) -- metric
  extractors. Uses subprocess to docker-run metric scripts, no
  Python imports of preframr.
- `integration_tests/experiments/report.py` (144) -- markdown
  renderer.
- `integration_tests/experiments/rerender_report.py` (75) --
  rerender utility.
- All 7 spec modules
  (`memorize.py`, `generalize.py`, `content_floor_check.py`,
  `contrastive_mini_body_large.py`, `contrastive_prodlike.py`,
  `per_tier_heads_mini_body_large.py`, `per_tier_heads_prodlike.py`).
  Each is 30-80 LoC; only imports `experiments.base` helpers.
- `integration_tests/profile/hvsc_version_check.py` -- pure
  stdlib + pathlib; imported by `base.py::_hvsc_version_check`.
- `integration_tests/profile/train_preflight_smoke.py`,
  `train_prodlike_oom_smoke.py` -- import torch but run *inside*
  the preframr docker image, not on the runner host. Move as
  docker-resource scripts, not as a Python import dep.
- `integration_tests/data/` pinned tier lists +
  `data/refuted/` registry (essential runner input). ~few MB.
  `data/audit/` (per-experiment generated, 169 MB total) stays
  in main repo -- runner writes its outputs to `--root` (under
  `/scratch/tmp/preframr_experiments/`), not into `data/audit/`.
- Tests:
  - `tests/test_experiments_robust_rmtree.py`
  - `tests/test_experiments_dataset_cache.py`
  - `tests/test_experiment_spec.py`
  - `tests/test_base_micro_mini.py`

### Stays in main repo

Research-side (imports `preframr.*` / `preframr_tokens.*` /
`torch`):

- `preframr/` -- production code, no change.
- `integration_tests/profile/` (minus the 3 moved files) -- the
  research scripts: audits, generators, encodability metrics,
  digi audits, corpus index builders, etc. These import
  `preframr.inference.predict`, `preframr_tokens.audit_primitives`,
  `preframr_tokens.macros.*`, etc.
- `integration_tests/design/` -- research design docs (per_tier,
  multi_modal, melody_transfer, this doc itself).
- `integration_tests/fixtures/` -- test fixtures (hvsc_authored
  sample reglogs).
- `data/audit/` -- per-experiment audits + generated streams.
- Research-side `tests/` (loop_detection, prompt_conditioning,
  per_class_acc, audio_fidelity, parser_canonicalisation,
  aggregate_corpus_index, etc.).

### Edges that need to dissolve

| edge | direction | resolution |
|---|---|---|
| `base.py::_hvsc_version_check` imports `from integration_tests.profile.hvsc_version_check` | runner -> profile (Python) | Move `hvsc_version_check.py` into new repo. It's pure stdlib. |
| `base.py::preflight_check` bind-mounts `INTEGRATION_TESTS_DIR / "profile"` into smoke container, runs `train_preflight_smoke.py` | runner -> profile (file resource) | Move the two smoke scripts into new repo's `preflight/` subdir; runner mounts that. |
| `metrics.py` shells out to `integration_tests.profile.audit_engine_fp_palette_eval_encodability` via subprocess | runner -> profile (subprocess) | Runner accepts `--profile-dir` arg (default to a sentinel) that points at the main repo's `profile/`. Specs / metrics that name a profile script reference it relative to that dir. |
| `data/refuted/<name>.md` registry referenced by design docs | shared concept | Move with the runner. New design docs in main repo link to the new-repo path. |
| Pipeline-spec transform names (`{"name": "slope"}` etc.) reference transforms registered in `preframr_tokens.macros.*` | spec -> preframr_tokens (semantic) | Soft coupling, no Python import. Document as version-compat note in new repo's README; pin a `preframr-tokens >= X.Y` in metadata only. |

## Target shape

```
preframr-experiments/                       (new sibling repo)
├── README.md
├── pyproject.toml                          PyPI publishable
├── preframr_experiments/
│   ├── __init__.py
│   ├── base.py                             ExperimentSpec, Arm, run_arm,
│   │                                       _robust_rmtree, _dataset_cache_*,
│   │                                       preflight_check, ...
│   ├── metrics.py
│   ├── report.py
│   ├── rerender_report.py
│   ├── run.py                              CLI: `python -m preframr_experiments.run`
│   ├── hvsc_version_check.py
│   ├── data/                               pinned .list tiers + refuted registry
│   │   ├── smoke.list
│   │   ├── mini/...
│   │   ├── canonical/...
│   │   ├── prodlike/...
│   │   ├── prodlike_4x/...
│   │   └── refuted/...
│   ├── preflight/                          docker-mounted (never imported)
│   │   ├── train_preflight_smoke.py
│   │   └── train_prodlike_oom_smoke.py
│   └── specs/                              `python -m preframr_experiments.run <name>`
│       ├── memorize.py
│       ├── generalize.py
│       ├── content_floor_check.py
│       ├── per_tier_heads_mini_body_large.py
│       ├── per_tier_heads_prodlike.py
│       ├── contrastive_mini_body_large.py
│       ├── contrastive_prodlike.py
│       └── (future spec modules land here)
└── tests/
    ├── test_robust_rmtree.py
    ├── test_dataset_cache.py
    ├── test_experiment_spec.py
    └── test_base_micro_mini.py
```

Main repo after extraction:

```
preframr/
├── preframr/                              unchanged
├── integration_tests/
│   ├── design/                            research design docs
│   ├── fixtures/                          test fixtures
│   ├── profile/                           research utilities
│   │                                      (minus 3 moved files)
│   └── data/
│       └── audit/                         per-experiment audits
│                                          (data/<tier>/ moves out)
└── tests/                                 research-tied unit tests only
```

New CLI shape (no functional change for the user):

```
# old
python3 -m integration_tests.experiments.run per_tier_heads_prodlike \
    --root /scratch/tmp/preframr_experiments --only-arm per_tier_heads_mos4

# new
python3 -m preframr_experiments.run per_tier_heads_prodlike \
    --root /scratch/tmp/preframr_experiments --only-arm per_tier_heads_mos4 \
    --profile-dir /scratch/anarkiwi/preframr/integration_tests/profile
```

The `--profile-dir` arg points the runner at the main repo's
profile scripts (for `audit_engine_fp_palette_eval_encodability`
etc.). Default: cwd-relative `integration_tests/profile` so the
existing host layout keeps working without explicit args.

## Decisions taken (rationale)

1. **Separate sibling repo, not a subpackage of preframr.** Same
   pattern as `preframr-tokens` / `preframr-audio` -- the user
   has already validated this shape twice. Keeps build / test /
   CI / release independent. PyPI mirror install is a one-liner
   in `requirements.txt`.

2. **Move the data tier lists (`data/<tier>/`) with the runner.**
   They are runner inputs (`smoke.list`, `mini/train.list`, etc.).
   Specs reference them via tier name; `base.py::*_paths()`
   resolves them relative to `DATA_DIR`. Keeping them in the
   main repo would require a `--data-dir` injection that no one
   ever changes from the default.

3. **Leave `data/audit/` in the main repo.** It's per-experiment
   generated content (audits, streams, gate JSONs) that
   research design docs reference relatively. Moving it would
   make the new repo grow unboundedly with every audit run, and
   would force a cross-repo update for every audit landing.
   Runner writes its outputs to `--root` already, not into
   `data/audit/`.

4. **Leave `data/refuted/<name>.md` -- decision: MOVE with the runner.**
   Refuted entries are spec-adjacent: each refers to a refuted
   experiment by spec name. They live with the specs in the new
   repo. Main-repo design docs that reference refuted entries
   get updated to the new path on extraction.

5. **`--profile-dir` injection, not a vendored profile/ copy.**
   The runner only needs `profile/` for two things: docker
   bind-mount during preflight (already file-resource), and
   subprocess-invoked metric scripts. Both work with a path
   arg. Vendoring `profile/` into the new repo would either
   (a) duplicate research code, or (b) split it across two repos
   along an arbitrary line.

6. **No back-compat shim during migration.** Same pattern as
   the recent preframr-tokens 0.7.0 cutover -- one PR moves
   everything, all importers update in lockstep. Avoids the
   trap of dual import paths.

7. **Pin `preframr-experiments >= X.Y` in main repo's
   `requirements.txt`.** Bumping a spec's required transform
   (e.g., a new macro added to preframr-tokens) means bumping
   the spec module in the new repo, releasing a new version,
   and bumping the pin in the main repo. Standard chain.

## Phase plan

| phase | scope | wallclock | gate |
|---|---|---|---|
| 0 | Create `/scratch/anarkiwi/preframr-experiments` skeleton (pyproject, README, package layout); confirm local PyPI mirror publish works (`pip install --index-url http://192.168.5.1:5001/index/ preframr-experiments`). | ~2 hr | Empty package installs cleanly into the preframr docker image. |
| 1 | Move runner + specs + tests + data tier lists + refuted registry + hvsc_version_check + 2 preflight scripts. Update all imports (`integration_tests.experiments.*` -> `preframr_experiments.*`). Run unit tests in the new repo's lighter container. | ~4-6 hr | New-repo pytest passes (4 test modules); old import sites in main-repo `tests/` updated. |
| 2 | Main-repo cleanup: delete moved files, slim `integration_tests/` to design/profile/fixtures + research-side audit data, add `--profile-dir` plumbing for runner subprocess calls. Update `requirements.txt` to install `preframr-experiments`; update `Dockerfile` if needed. Verify `./build.sh` clean + `run_tests.sh` clean. | ~2-3 hr | Full main-repo CI green; one prodlike-spec smoke launch from new-repo CLI succeeds. |
| 3 | Update AGENTS.md (new repo location, new CLI shape, `--profile-dir` convention); update `design/` docs that reference `integration_tests/experiments/*`. Update sibling-repo notes block (analogous to the `preframr-tokens` / `preframr-audio` mention). | ~1 hr | Docs reflect new shape. |
| 4 | (Optional, follow-on) Run smoke on a fresh clone of the new repo to validate no hidden coupling. | ~1 hr | Cold install + smoke spec passes. |

**Total wallclock:** ~10-13 hr (~1.5 days agent work).

## Risk + non-goals

- **Spec / transform-name coupling.** Spec modules reference
  pipeline-spec transform names that exist in `preframr-tokens`.
  If the new repo lands a spec naming a transform that doesn't
  exist in the main repo's installed `preframr-tokens` version,
  the run fails at parse time. Mitigation: pin
  `preframr-tokens >= X.Y` in the new repo's metadata; document
  the dual-version concern in the new repo's README.
- **Refuted registry path moves.** Existing main-repo design
  docs reference `integration_tests/data/refuted/<name>.md`. The
  extraction PR updates each link. Future readers grepping the
  old path get a clean miss.
- **`--profile-dir` default brittleness.** Default cwd-relative
  path works only when the user runs the CLI from the main repo
  root. Mitigation: explicit error message if the default path
  doesn't exist and no `--profile-dir` was passed, listing the
  expected path.
- **Lost git blame across the move.** `git log --follow` works
  for individual files but blame on the diff PR will be loud.
  Mitigation: do the move in a single commit with `git mv`
  (where shapes match) and document in commit body.
- **No back-compat shim during migration.** If a long-lived
  branch is mid-flight when the move lands, it has to be
  rebased. Mitigation: coordinate -- this is a quiet moment in
  the project (the only in-flight work is the Phase 3 prodlike
  babysit, which doesn't touch `integration_tests/experiments/`).
- **Risk of premature factoring.** The runner has been touched
  6+ times this session for new features (dataset cache, rmtree
  fixes, NFS hygiene). Extracting now means PR-velocity
  rises across a repo boundary for the next few rounds of
  runner-fragility work. Mitigation: do the extraction after
  Phase 3 verdict lands, when the runner has settled.

## Non-goals

- No change to the docker images, the train / inference pipeline,
  or any preframr-side code.
- No reorganisation of `profile/`. It stays in the main repo with
  the existing layout; only the 3 runner-adjacent scripts move.
- No move of `design/` -- research designs stay where the model
  code is.
- No move of `fixtures/` -- they're parse-test fixtures, not
  runner inputs.
- No new framework features (resume, parallel arms, ddp scaling)
  -- those are queued separately under "Framework follow-ups".

## Acceptance criteria for advancing from this design

Reviewer (user) approves:

1. The boundary: runner / specs / runner-data move out;
   profile / fixtures / design / audit-data / preframr stay.
2. The `--profile-dir` injection point as the only cross-repo
   subprocess edge.
3. The 4-phase plan + ~1.5 day wallclock envelope.
4. The timing: extraction happens AFTER the in-flight Phase 3
   prodlike returns a verdict (passing or refuting). Not while
   the GPU is mid-run, not before a fragility shakeout commit
   lands.
5. The decision to NOT keep a back-compat import shim.

On approval: Phase 0 launches when Phase 3 verdict lands.

## References

- `preframr-tokens` PyPI extraction (committed 2026-05-21):
  similar shape -- pure-Python library extracted, main repo
  installs from mirror.
- `preframr-audio` v0.1.0 extraction (committed 2026-05-20):
  same pattern.
- `integration_tests/experiments/base.py` -- the runner this
  doc extracts.
- `AGENTS.md` `Packages` + `Environment` sections -- conventions
  the new repo would inherit.
