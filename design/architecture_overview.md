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

## Deriving the release process (two mechanisms — see AGENTS.md "Releasing")

- **PyPI libs** (tokens, audio): release on a **`v*` tag only** (setuptools-scm
  dynamic version; bump `fallback_version` to match). Merging to main publishes
  nothing. **Published to public PyPI**, which **propagates with a few-minutes
  lag** — downstream image CI installs from **public PyPI** (not the local proxpi
  mirror), so **wait for public-PyPI propagation before opening/CI-ing the
  downstream image PR**. `preframr_experiments/bust_release.sh` only refreshes the
  *local mirror* (for local builds); it does **not** prove public PyPI is ready
  (this caused a CI race — fix by re-running CI once PyPI propagates, or polling
  `pypi.org/pypi/<pkg>/json` before the downstream PR).
- **Images** (preframr + slim; xpt builds but `push:false`): release on **push to
  main + `v*` tags**; the image tag is the **VERSION file** (not the git tag).
  Merge republishes the versioned image — **bump VERSION in the release commit**.

**Cross-repo order for a tokenizer change that ripples up:** preframr-tokens PyPI
(tag, bump fallback) → *wait for public-PyPI propagation* → preframr (floor
`preframr-tokens>=X.Y.Z`, bump VERSION, merge/tag → image) → preframr-xpt
(`ARG BASE` bump + per-spec `image=` pins). The motif v2 work is the worked
example: tokens 0.21.0 (v2) → tokens 0.22.0 (tier fix) → preframr 0.2.3 (flag +
floor) → xpt v2 A/B spec.
