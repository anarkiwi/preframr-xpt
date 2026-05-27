# Architecture overview — which part lives where, and why

**Status:** Reference (orientation). The map for deciding *which repo a change
belongs in* and *deriving the release process* from that. Read this before
touching a cross-repo change (it would have prevented mis-scoping the motif tier
fix and the PyPI-propagation CI race).

## The five repos (sibling dirs under `/scratch/anarkiwi/`)

| Repo | Ships as | Holds | Why here |
|---|---|---|---|
| **preframr-tokens** | **PyPI wheel** (`preframr_tokens`) | reglog parse, tokenization, macros (FREQ_TRAJ, motif pass `motif_pass.py`/`motif_mine.py`), **tier classification** (`vocab_signature.py`/`tier_classify.py`), `RegTokenizer`, `blocks`, `stfconstants`, `audit_primitives` (`tier_accuracy`, `detect_tail_cycle`), `Corpus`, `constrained_decode`, `render_play`, tokenizer profiling | **torch-free** → small, fast-iterating library; **anything about atom/token/tier *semantics* lives with the tokenizer that defines them** (that's why MOTIF_ARG→content is a tokens change, not framework) |
| **preframr-audio** | **PyPI wheel** | SID render (`pyresidfp`), `fidelity`/`compare_renders`, engine fingerprint | torch-free audio primitives, separable |
| **preframr** | **Docker image** (`anarkiwi/preframr` + slim `-predict`/`-xpu`/`-jetson`) | the **trainable/inferable** package: `train/` (trainer, model, `regdataset`, heads, `tier_map`), `inference/`, `args`, `parse`, `stftokenize`, the `mine_motifs` CLI, `utils` | torch-heavy runtime; shipped as an **image** (the env + heavy deps), `pip install`s tokens+audio (floored in `requirements.txt`) |
| **preframr-xpt** (this repo) | **nothing published** (runs from source; `docker.yml` build+test only, `push:false`) | the **experiment surface**: runner (`base.py`/`run.py`), `specs/`, `audit/` (per_class, prompt/loop, …), `design/`, tier `.list` data, refuted registry | high-churn orchestration kept OUT of the framework so main stays lean; runner core is **pure host orchestration (no torch import)**; coupled audits run in the layered image (`Dockerfile` `FROM ${BASE}` = a `preframr` image) |
| **preframr-aug** | repo (+ tooling) | offline corpus expansion (melody transfer, inaudible-perturbation probe, voice permutation) | separable augmentation research |

## Dependency / layering (bottom → top)

```
preframr-tokens (PyPI)  preframr-audio (PyPI)
            \               /
             v             v
        preframr  (framework; pip-installs both; → anarkiwi/preframr image)
             |
             v
       preframr-xpt  (Dockerfile FROM a preframr image; per-spec image= pins;
                      runner orchestrates docker runs of that image)
```
`preframr-aug` consumes tokens+audio; the experiment **arms** run in the per-spec
`image` (default `:latest`), not the xpt image (which is just the runner/audit gate).

## Deciding *where a change goes*

- Atom / token / tier / macro / motif **semantics** → **preframr-tokens**.
- Trainable model, loss, heads, args, parse/tokenize CLI, inference → **preframr**.
- Audio render / fidelity / fingerprint → **preframr-audio**.
- Experiment runner, specs, audits, tier-data lists, design docs → **preframr-xpt**.
- Augmentation → **preframr-aug**.

## Release process (the authoritative copy — derived from the topology)

Two mechanisms — don't conflate them:

- **PyPI libs** (`preframr-tokens`, `preframr-audio`): `release.yml` fires on a
  **`v*` tag ONLY** → PyPI (trusted-publisher OIDC). Version is **dynamic**
  (setuptools-scm from the tag); bump `fallback_version` in `pyproject.toml` to
  match. **Merging to `main` publishes nothing** (CI only) — only a `v*` tag
  releases, and `release.yml` does PyPI only (make the GitHub Release object
  separately with `gh release create`).
- **Images** (`anarkiwi/preframr` + slim `-predict`/`-xpu`/`-jetson`; preframr-xpt
  *builds* via `docker.yml` `push:false`): `release.yml` fires on **push to `main`
  AND `v*` tags** → Docker Hub, tagged `:latest` + `:${VERSION}` from the
  **VERSION file** (NOT the git tag name). Auth `secrets.DOCKER_TOKEN` (renamed
  from `DOCKER_PASSWORD` — the old name fails login). Merge-to-`main` republishes
  the versioned image (intended). **Gotcha:** the image tag is the VERSION file,
  not the tag name — **bump VERSION in the release commit**, or a `vX.Y.Z` tag
  ships a stale `:VERSION` (the v0.2.1→`:0.2.0` episode; fixed by VERSION→0.2.2).
  Local build beats waiting on GHA:
  `docker build -f Dockerfile . -t anarkiwi/preframr:<v>`.

**Merging a PR is always safe** — everything is versioned and new code is opt-in.
The deliberate *release* is the **tag** (PyPI) or the **VERSION bump + push**
(images); gate THAT on the experiment verdict, not the merge.

**Public-PyPI propagation (the CI gotcha).** Downstream image CI installs the libs
from **public PyPI**, which propagates a release with a **few-minutes lag**.
`preframr_experiments/bust_release.sh <pkg> [version]` busts + polls the **local
proxpi mirror** (for local builds, where `build.sh` sources `.env` → `PIP_OPTS`)
— it does NOT prove public PyPI is ready. So after a lib release, **wait for
public-PyPI propagation before the downstream image PR/CI** (poll
`pypi.org/pypi/<pkg>/json`), or just re-run the CI once it propagates (this caused
a real CI race). Manual mirror bust: `curl -X DELETE
http://192.168.5.1:5001/cache/<pkg>`, confirm at `.../index/<pkg>/`, then rebake.

**Cross-repo order when a tokenizer change ripples up:** preframr-tokens PyPI
(tag, bump fallback) → *wait for public-PyPI propagation* → preframr (floor
`preframr-tokens>=X.Y.Z`, bump VERSION, merge/tag → image) → preframr-xpt
(`ARG BASE` bump + per-spec `image=` pins). Worked example (motif v2): tokens
0.21.0 (v2) → tokens 0.22.0 (tier fix) → preframr 0.2.3 (flag + floor) → xpt v2 A/B.
