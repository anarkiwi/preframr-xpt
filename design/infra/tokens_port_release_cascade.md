# Release / floor cascade for the decompiler-codec port

Runbook to make the post-port release mechanical. The white-box decompiler / BACC codec has **LANDED**
(it replaced `events/` in preframr-tokens); this cascade is **largely executed**. Kept as the
port-specific release reference — three packages bump in order. Authority for the general build/cache
rules stays `design/references/release_build_cache.md`.

## Cascade order (each link gates the next)

1. **preframr-tokens** (`/scratch/anarkiwi/preframr/preframr-tokens`) — port lands, then release.
   - Version is `setuptools_scm` dynamic from git tags; `pyproject.toml` `fallback_version = "0.52.0"`.
   - Release = push a `v*` git tag → `.github/workflows/release.yml` builds + publishes to PyPI
     (`pypa/gh-action-pypi-publish`), with a built-in PyPI-propagation wait.
   - **Tag ≥ `v0.53.0`** (see drift note below). Bump `fallback_version` to match the tag.
   - Gate before tagging: `run_tests.sh` green (CI runs it inside the docker build), and the new
     codec's corpus byte-exact + token-economy validation passed (the sidemu prototype gates).

2. **preframr** (`/scratch/anarkiwi/preframr/preframr`) — rebuild against the new tokens wheel.
   - `requirements.txt` **already floors `preframr-tokens>=0.53.0`** (and `preframr-audio>=0.5.9`),
     so no edit is needed there *if* tokens tags exactly 0.53.0; raise the floor if tokens tags higher.
   - `Dockerfile` `ARG BASE` is the upstream pytorch base; tokens is pulled via the pip floor at build.
   - Release = merge to `main` (`release.yml` on main-push + `vX.Y.Z` tag) → `:VERSION` + `:latest`.
     Bump the preframr version past `0.2.30` (e.g. `0.2.31`) and `git tag -a`.

3. **preframr-experiments (xpt)** (this repo) — repoint the image.
   - `Dockerfile:6` `ARG BASE=anarkiwi/preframr:0.2.30` → the new preframr tag (e.g. `:0.2.31`).
   - The dataset cache key folds the image's tokens version (`base.py _image_tokens_version`), so a
     tokens bump **auto-invalidates** stale parse/tokenize artefacts — no manual cache purge.
   - Then run the continuation spec: `specs/generalize_continuation.py` (atoms-only;
     `CONTINUATION_TKVOCAB` = the codec's final vocab — **VOCAB=34** (the "55 in the prototype" figure
     is stale)).

## Drift to reconcile at port time
- `preframr/requirements.txt` floors `preframr-tokens>=0.53.0`, but the last tokens tag is behind
  (`fallback_version 0.52.0`). The floor is already ahead of shipped — the port release (tag ≥0.53.0)
  closes the gap. Until then a clean `pip install preframr` cannot satisfy the floor.
- AGENTS.md still records `tokens 0.51.0 / preframr 0.2.30` — stale vs the current 0.53.0 floor; refresh
  the "Shipped substrate" / "Packages" version lines as part of the port commit.

## One-glance edit list
| pkg | file | from | to |
|---|---|---|---|
| tokens | `pyproject.toml` `fallback_version` | `0.52.0` | match new tag (≥`0.53.0`) |
| tokens | git tag | — | `vX.Y.Z` (≥`0.53.0`) |
| preframr | `requirements.txt` | `preframr-tokens>=0.53.0` | raise only if tokens tags >0.53.0 |
| preframr | version + git tag | `0.2.30` | next (e.g. `0.2.31`) |
| xpt | `Dockerfile:6` `ARG BASE` | `anarkiwi/preframr:0.2.30` | new preframr tag |
| xpt | run | — | `generalize_continuation` (set `CONTINUATION_TKVOCAB`=34) |
