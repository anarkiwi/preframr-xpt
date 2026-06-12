# Release, build, test, cache — the one place

**Status:** Reference — the single authoritative copy. Supersedes the scattered
"Release process" notes; `architecture_overview.md` and the repo `AGENTS.md`s point
here. Covers: which host runs what, the pip cache + how to bust it, the per-repo
release procedure, and the local build/test commands.

## Hosts — run non-GPU work off the GPU box

| Host | Role | Use for |
|---|---|---|
| **defroster** (`192.168.5.x`, this repo's GPU box) | GPU / training | ONLY model training (the `trainer.py` stage). Everything else competes with training for CPU. |
| **fogbank** (`192.168.5.2`, 72 cores, no GPU) | non-GPU workhorse | **builds, parse/tokenize, audits, pytest, lint, image bakes.** Shared `/scratch` (same repos + dumps), its own docker daemon + preframr images, reaches the proxpi mirror. `ssh fogbank` (passwordless). |
| **proxpi mirror** (`192.168.5.1:5001`) | PyPI cache | local pip/docker builds install through it (fast, offline-ish). Caches a stale index after a release — see **Cache**. |

**Rule of thumb:** if it doesn't need the GPU, run it on fogbank (`ssh fogbank '...'`
or `docker -H ssh://fogbank ...`). defroster's load spikes to 60+ during a parse;
keep it for training. The runner stages dumps under the shared `/scratch`, so a
parse/audit launched on fogbank sees the same data.

## The proxpi pip cache

Local docker/pip builds install through the proxpi mirror for speed. `build.sh`
sources gitignored `.env` (template `.env.example`) → `PIP_OPTS`:

```
PIP_OPTS="--index-url http://192.168.5.1:5001/index/ --trusted-host 192.168.5.1"
```

Builds must pass `--network host` so the container can reach the mirror, e.g.
`docker build --network host --build-arg PIP_OPTS="$PIP_OPTS" ...`.

**Busting it (REQUIRED after a PyPI release).** The mirror caches the package index,
so a fresh release is invisible to local builds until busted:

```
preframr_experiments/bust_release.sh <pkg> <version>     # busts + polls until served
# or manually:
curl -X DELETE http://192.168.5.1:5001/cache/<pkg>        # then confirm at .../index/<pkg>/
```

Busting the mirror does NOT prove **public** PyPI has propagated (a few-minutes lag);
downstream **CI** installs from public PyPI, so for CI poll `pypi.org/pypi/<pkg>/json`
or just re-run the CI once it propagates. For **local** builds, the mirror bust is
enough.

## Release process — two mechanisms, never conflate

### A. PyPI libs — `preframr-tokens`, `preframr-audio`

Torch-free wheels. The deliberate release is a **`v*` tag** (not a merge):

1. Land the change on `main` (CI green). Bump `fallback_version` in `pyproject.toml`
   to the target (setuptools-scm derives the real version from the tag).
2. `git tag -a vX.Y.Z -m "..." <sha> && git push origin vX.Y.Z` → `release.yml` fires
   on `v*` → builds the wheel → publishes to PyPI via trusted-publisher OIDC.
   (Merging to `main` publishes nothing — CI only.)
3. Confirm live: `curl -s pypi.org/pypi/<pkg>/json | jq -r .info.version`.
4. **Bust the mirror** so local builds see it: `bust_release.sh <pkg> X.Y.Z`.

Build/test before releasing (on fogbank): `pytest tests`. The fidelity tests need the
SID fixture cache — set `PREFRAMR_SID_FIXTURE_CACHE=/scratch/preframr/sid_fixture_cache`
(else they render from HVSC, or skip). Never let them silently skip.

### B. Docker apps — `preframr` (framework), `preframr-xpt`

Images, not wheels. **The release is a VERSION/base bump + push to `main`** (CI builds
+ publishes to Docker Hub, tagged `:latest` + `:${VERSION}` from the **VERSION file**,
not the git tag).

- **preframr:** bump `VERSION`, floor the lib in **all** req files (`requirements.txt`,
  `predict-requirements.txt`, `jetson/predict-requirements.txt`), commit, push `main`
  → `release.yml` publishes cuda + slim `-predict`/`-xpu`/`-jetson`. **Then tag the
  release: `git tag -a vX.Y.Z <released-main-sha> -m ... && git push origin vX.Y.Z`** —
  every shipped VERSION gets a matching git tag (a version with no tag is useless). The
  tag re-fires `release.yml` and re-pushes the identical image; that's expected/harmless.
- **preframr-xpt:** bump `ARG BASE=anarkiwi/preframr:<v>` in the `Dockerfile` + the
  per-spec `image=` pins, push `main` → `docker.yml` builds (`push:false`, build IS the
  test gate; the runnable xpt image is baked locally).

**ALWAYS build the image locally, in parallel with the push (the standing rule).**
Don't wait for CI to publish and then pull a multi-GB image over the network — that
serializes you behind CI + a slow pull. Kick off the local bake on fogbank the moment
you push; a failed local build is discardable (CI is the source of truth), the point is
to keep moving. Local build commands (run on fogbank):

```
# preframr (cuda; runs run_tests.sh — the build IS the gate):
cd /scratch/anarkiwi/preframr && . ./.env && DOCKER_BUILDKIT=1 docker build --network host \
  --build-arg PIP_OPTS="$PIP_OPTS" -f Dockerfile \
  -t anarkiwi/preframr:<v> -t anarkiwi/preframr:latest .

# preframr-xpt (FROM preframr:<v>; runs pytest tests):
cd /scratch/anarkiwi/preframr-xpt && DOCKER_BUILDKIT=1 docker build --network host \
  -f Dockerfile -t anarkiwi/preframr-xpt:<v> -t anarkiwi/preframr-xpt:latest .
```

(`build.sh` bakes the full matrix incl. tensorboard/jetson; for a fast iterate just
build the one cuda image as above.)

## Cross-repo order when a tokenizer change ripples up

A tokens op/flag change is a 3-repo release:

1. **preframr-tokens** → PyPI (tag, bump fallback) → `bust_release.sh` + wait for public
   PyPI propagation.
2. **preframr** → floor `preframr-tokens>=X.Y.Z` (all req files) + bump `VERSION` →
   build locally on fogbank **while** pushing `main` → image `:VERSION` + `:latest`.
3. **preframr-xpt** → `ARG BASE` bump + per-spec `image=` pins → build locally on fogbank
   while pushing `main`.

Removing a tokens `*_OP` constant breaks preframr's train tests AND the Docker build's
`run_tests.sh` — see memory `cross-repo-release-ordering`. Worked example (v0.42.1):
tokens `per_reg_burst` fix + walker perf → PyPI 0.42.1 (tag `v0.42.1`) → preframr 0.2.18
(floor `>=0.42.1`, VERSION bump, local cuda build on the proxpi mirror while pushing) →
xpt `ARG BASE=0.2.18` + spec `image=` pins.
