# Operational notes for agents (preframr-xpt)

This repo (`preframr-xpt`) is the **experiment surface**: the docker-driven
runner + spec registry + audits + design docs + tier data + refuted registry.
The framework, libraries, and corpus live elsewhere; this is where the
high-churn research work happens.

## Packages (as of 2026-05-25)

- **`preframr` 0.1.0** — framework only (train / inference / model / args /
  parse / stftokenize / utils). First *versioned* release. Image
  `anarkiwi/preframr:0.1.0` (+ `:latest`). No PyPI package; it ships as the
  docker image. `integration_tests/` (audits, probes, fixtures, design docs)
  was moved OUT to this repo; main carries only the framework.
- **`preframr-tokens` 0.19.0** (PyPI) — torch-free parser/tokenizer +
  macros + `render_play` (the parse→wav/live utility, moved here from main;
  run `python3 -m preframr_tokens.render_play`). Main floors `>=0.19.0`.
- **`preframr-audio` 0.5.1** (PyPI) — SID audio rendering primitives.
- **`preframr-experiments`** (this repo; editable / PYTHONPATH, no PyPI) —
  runner + specs + `audit/` (moved from `integration_tests/profile/`) + tests.
  Pure orchestration on the host; the audits import preframr/torch and run
  inside the **xpt image**.

Sibling source repos: `/scratch/anarkiwi/preframr-{audio,tokens,xpt}` and
`/scratch/anarkiwi/preframr` (framework). Libraries install from the PyPI
mirror; `preframr_experiments` runs from source
(`PYTHONPATH=/scratch/anarkiwi/preframr-xpt`, CLI on the host, not docker).

## Images

- **`anarkiwi/preframr`** (`:latest` + `:0.1.0`) — train + test, full deps.
  Builds run `./run_tests.sh`. Entry points: trainer, parse, stftokenize,
  predict. (`render_play` now lives in tokens.)
- **`anarkiwi/preframr-predict` / `-xpu` / `-jetson`** — slim eval/predict
  images (`Dockerfile.predict`, + xpu / jetson base overrides). All four
  publish `:latest` + `:${VERSION}`.
- **`anarkiwi/preframr-xpt`** — layers the runner + audits on top of
  `anarkiwi/preframr:0.1.0` (pinned `ARG BASE`). Build runs `pytest tests`;
  the arc runs in this image.

Release: `release.yml` publishes on push to `main` + `v*` tags; each image
tagged `:latest` + `:${VERSION}` (VERSION file in main; currently `0.1.0`).
Auth via `secrets.DOCKER_TOKEN` (renamed from `DOCKER_PASSWORD` — workflows
referencing the old name fail login). Local build is faster than waiting on
the GHA publish: `docker build -f Dockerfile . -t anarkiwi/preframr:0.1.0`.

`build.sh` sources a gitignored `.env` (template `.env.example`) for
`PIP_OPTS` (proxpi mirror). After a new `preframr-{audio,tokens}` release the
mirror serves a stale index until busted:
`curl -X DELETE http://192.168.5.1:5001/cache/<pkg>`, confirm at
`.../index/<pkg>/`, then rebake.

## Project goal (OVERRIDING)

Train a SID model that **generalises** — predicts unseen continuations from
arbitrary mid-song prompts, across composers (primary `val_acc`) and ideally
across engines (stretch) — inside a fixed envelope:

- **Train:** single RTX 4090, 24 GB. Specs needing >~50M body to show Δ are
  out-of-envelope; refute in design, don't A/B.
- **Predict:** Jetson Orin NX (15.6 GB) at PROMPT=2048 / MAX=8192. KV cache at
  prodlike dims ~16 KiB/token → 128 MiB at MAX; bounded by seq_len, not VRAM.

## Open problem + active result (OVERRIDING)

**Token juggling is refuted** — every model-side macro/loss move shifted
structural accuracy without lifting content (5 interventions, all at the same
~0.13 eval_a content ceiling; see Refuted). The ceiling was
**tokenization-induced**, not model-capacity.

**Active pivot — strict-no-diff tokenizer (preframr-tokens):** every token is
a structured trajectory primitive or an enumerated carveout. Unified
`FREQ_TRAJ` op + 4 CLI-only absorber macros
(CTRL_TRIPLE / FREQ_NUDGE / RELEASE_UPDATE / lonely_catch_all). Per-frame
fidelity oracle byte-exact; 0.18.0 fixed a duration-drop bug (macro round-trip
0–0.5% of baseline).

**PASSED (2026-05-25): `full_macros_prodlike`.** First non-refuted,
content-confirmed intervention — and it is tokenizer-side. Verdict via the
**content-tier per_class audit** (the decisive, un-confounded gate):
- eval_a content **0.160→0.287 (+0.127)**; structural only +0.040.
- eval_b content **8/8 non-negative** (mean +0.0997).
- All-tier val_acc is CONFOUNDED (different tokenizations inflate structural
  tokens) — only the content-tier audit settles content vs structural.
  **Always run it before calling an encoder A/B a win.**

Priority: (1) generalization detection — per-tier accuracy, loop-collapse,
prompt-conditioning (audit suite); (2) encoding efficiency is secondary;
(3) predict-host envelope; (4) tracker-authoring prior favoured over
signal-only compression.

## RUNNING — vocab-trimmed re-arc STAGE 1 (launched 2026-05-25)

Re-run ALL baselines + the 5 refuted model-side specs from scratch on the
post-FREQ_TRAJ tokenizer at the deployment-efficient config (no metrics
transfer — vocab/op-alphabet/seq-len changed). STAGE 1 (mini) launched from
this repo; status via `check_overnight_batch.sh` (done marker
`/scratch/tmp/preframr_experiments/overnight_batch.done`). Everything staged:

- **Base:** `anarkiwi/preframr:0.1.0` (tokens 0.19.0, leaned framework, dep
  bumps pandas 3.0.3 / pyarrow 24.0.0 / black 26.5.1), built + verified.
- **xpt image:** rebuilt FROM `:0.1.0`, audits + runner green (90 passed).
- **STAGE 1 (mini triage):** `preframr_experiments/run_overnight_batch.sh` —
  7 specs (`content_floor_check`, `per_tier_heads_mini_body_large`,
  `content_diffusion_mini_body_large`, `contrastive_mini_body_large`,
  `mask_structural_loss_mini`, `voice_permutation_mini_body_large`,
  `cluster_content_mini_body_large`), `--tkvocab 8192` (UNK=0, provably; vocab
  is ~91% dead, 2929/32768 used), frozen baked code. cluster spec runs with
  `PREFRAMR_DATASET_CACHE_DISABLE=1`.
- **STAGE 2 (prodlike):** follows once mini is clean — `B=4 / accum=8`
  (effective batch 32; `B=8` OOMed at 23.3 GiB).

Launch (host-side, drives the xpt image):
```
cd /scratch/anarkiwi/preframr-xpt && nohup bash preframr_experiments/run_overnight_batch.sh & disown
```
The runner now sets `REPO_ROOT` to this repo, so `python3 -m
preframr_experiments.run` resolves from cwd (no PYTHONPATH); src-bind + tier
data paths are absolute and unaffected.
Dataset cache (`/scratch/preframr/hvsc/dataset_cache/`, keyed on spec inputs
+ image tokens-version) re-tokenizes on the 0.18→0.19 bump; net re-arc cost
negligible (mini ~29M; prodlike tokenizes fresh regardless).

NOT bundled: effective-batch change, GPU rental, tokenizer-default flips.

### STAGE 1 progress (2026-05-26, mid-batch — TRIAGE ONLY, `per_class` audit NOT run)

5/7 specs done, no FAILED, healthy. Cleared `content_diffusion` (where the
prior run was stopped); `voice_permutation` running — its
`augment_voice_permutation.py` pre_run_hook (kept here, NOT moved to
preframr-aug) emits 750 variants/seed clean; `cluster_content` (cache-disabled)
last. Identical tokenization across all specs (alphabet 3703), so the deltas
are clean model-side A/Bs — but **all-tier val_acc, not content**:

- `per_tier_heads_mos4`: +0.057 all-tier val_acc (0.114→0.171), val_loss
  9.8→5.1 — the **structural-inflation confound** (refuted at prodlike for
  ignoring content); NOT a content signal.
- `content_diffusion`: flat (−0.002 vs mos4_entropy baseline).
- `contrastive` (InfoNCE): flat (+0.003).
- `mask_structural_loss`: negative (val_acc 0.018) — re-confirms structural
  supervision is load-bearing.
- `content_floor_check`: body=large baseline content acc ~0.006.

Read so far: the model-side specs **reproduce their refutations on the
corrected tokenizer** — no content lift on top of the tokenizer win; leverage
is representation/data, not architecture. This is val_acc triage, NOT the
verdict: run `audit_checkpoint_per_class` on these ckpts before promoting
anything to prodlike or touching the Refuted registry. Data-side
`voice_permutation` is the one still worth watching (augmentation, not arch).

## Tests + runner

- Framework tests (in `anarkiwi/preframr`): `./run_tests.sh` (black, pytest
  `/tests`, pylint curated, pyright, coverage ≥77).
- Runner + audits (in `anarkiwi/preframr-xpt`): `pytest tests` runs at image
  build; host CLI is
  `PYTHONPATH=. python3 -m preframr_experiments.run <spec> --root <work> --tkvocab 8192`.
- **Content-tier audit (decisive gate):** run
  `python3 -m preframr_experiments.audit.audit_checkpoint_per_class` in the
  xpt image; `audit_*.json` land under `/scratch/tmp`.
- Outputs under `/scratch/tmp/preframr_experiments/`. Status:
  `check_overnight_batch.sh`; done marker `overnight_batch.done`.

## Conventions

- **Code = frozen baked image by default.** Runs use baked `preframr/`; rebake
  to pick up edits. Working-tree bind-mount is opt-in (`run.py --bind-src` /
  `$PREFRAMR_BIND_SRC=1`) and runs un-gated code — don't use without asking.
- **Background runs:** `nohup`+`disown`; don't poll, use `ScheduleWakeup`.
- **Comments:** no session narration / dev-local paths / PR numbers;
  `tests/test_lint.py` rejects narrative `#` and >5-line docstrings.
- **NFS hygiene:** no lingering `tail -f` on workdir files (silly-renames);
  stop `preframr_tb` before deleting tb_logs subtrees.
- **Arm ordering:** target arm first in `spec.arms`, baseline last (sequential
  runner; audition wants target ckpt first).
- **Renaming a transform** silently disables it in stale specs (no error) —
  grep specs on any pass/transform rename.

### Wallclock anchors

mini body=large ~12-20 min/arm · canonical 60-120 min/arm · prodlike ~6-11 hr
per (arm, seed). parse+tokenize ~25 min/prodlike uncached.

## Forward-looking work

### Land any time
- **Profile + optimize preframr-tokens parsing** — correct but slow; big share
  of uncached run setup. Keep the per-frame fidelity oracle green.
- **Recover control-write-rejected dumps** — characterize dumps rejected for
  too many control writes; relax/absorb to grow the corpus.
- **12-SID WAV audition cohort** — non-negotiable gate before flipping any
  tokenizer default + re-cutting training data.
- **Per-primitive round-trip audio gate** — wire `compare_renders` over ~100
  songs into CI (≥95% within tolerance).

### Predict-host envelope (queued, post multi-day-training)
Lead with the deployment envelope: **vocab shrink** (tkvocab ~8× to 4096 —
~91% dead) → GPU-resident constrained-decode → full-context audition. Orin is
~4% GPU util at predict. Re-open alongside the Multi-GPU rental decision
(deferred until a generalising approach lands: prodlike result + DDP-4 ≥60% +
`--resume`/`auto_early_abort`/`--max-parallel-arms` + refute rate ≤40%).

### Framework follow-ups
- Streaming unembed-CE — recovers prodlike 2× wall.
- Generalization-gate thresholds — `content_over_structural_min=0.05` misfires
  pre-saturation (body=large mini baseline ~0.015); recalibrate per tier.
- Augmentation tooling + design moved to **`preframr-aug`** (`preframr_aug/`:
  inaudible-perturbation probe, melody/instrument transfer + Phase-0 audit;
  `design/melody_transfer_augmentation_design.md`). Voice-permutation helper
  (`audit/augment_voice_permutation.py`) + its spec stay here — wired into the
  arc runner. Melody-transfer Phase-0 smoke still pending.
- **Autocast fp32-promotion trap:** any new `Module` in
  `preframr/train/model/heads*.py` must cast log_softmax/logsumexp back to
  input dtype or per-position buffers stay fp32 and OOM at prodlike
  (pinned by `tests/train/test_per_tier_heads.py::test_bf16_input_preserves_*`).

### Content tier deliberately lossy
slope/preset/transpose are cent-binned (lossy by design; content-tier-OFF is
byte-perfect vs raw). Lossless rework deferred.

### xpt cleanup pending (post-render_play-move)
`encodability_metric.py` still mounts main's (gone) `integration_tests` + runs
`integration_tests.profile.audit_engine_fp_palette_eval_encodability` — needs
repointing to run inside the xpt image. `build_prodlike_4x_list.py` default
paths + some docstrings still reference moved-out paths.

### Fixtures move-out pending
SID songs must NOT be tracked here. Build a helper that creates + caches
fixtures locally (untracked) from HVSC; use Goto80 (not Commando). A 116M
`engine_fp_palettes.json` was removed from main's working tree; live copies
are in `data/{canonical,mini}/`.

## Refuted alternatives

Registry: `preframr_experiments/data/refuted/<exp>.md`. All 5 model-side
interventions concentrated at the same ~0.13 eval_a content ceiling:
- `per_tier_heads_mos_prodlike` — router posterior saturates at prodlike,
  outputs ignore prompt content (crit 3+4 FAIL).
- `per_tier_heads_entropy_prodlike` — lambda non-monotonic, peak 0.01, can't
  reach gate 1.2 (v11 1.123, v12 worse).
- `mask_structural_loss` — masking structural CE collapsed diversity to 0.863
  (< baseline 1.220): structural supervision is load-bearing for
  prompt-conditioning.
- `cluster_conditional_content_head` — same ceiling, diversity ~1.0-1.2.
- `content_diffusion` — sampling-side change didn't move the CE outcome.
- Earlier nulls: `legato_ab`, `palette_merge`, `head_row_class`,
  `adsr_equivalence`, `macro_coarsening`, `b2_unblock`, `palette_pwm_prereqs`,
  `global_instr_ids_phase_a`, `weighted_token_loss`, `learnable_class_loss`,
  `voice_trajectory` (all variants), `set_to_diff`,
  `contrastive_infonce_auxiliary`.

## Deferred deploy-stage efficiency (post-generalization)

~25% token-budget wins, both refute as generalization bets:
- **FRAME subsumes VOICE** — header token encodes voice order + write counts
  (+50% FRAME alphabet; partial tracker analogue).
- **Drop VOICE_REG** — reg space already disambiguates voice (zero inflation;
  loses `voice_block_order`).

## Resolved log (compact; details in git log)

- **2026-05-25** — Lean-core + 0.1.0 release. `integration_tests/` (audits,
  fixtures, design docs) moved to this repo's `audit/` + `tests/` + data tree;
  main is framework-only. `render_play` moved to preframr-tokens 0.19.0.
  preframr versioned (0.1.0, all 4 images `:latest`+`:VERSION`, release on
  main+`v*`, `DOCKER_TOKEN`). xpt base pinned to `:0.1.0`. Dep bumps folded
  (pandas 3.0.3 / pyarrow 24.0.0 / black 26.5.1; jetson pins held). Workflow
  nuisance-runs reduced (release on main only).
- **2026-05-25** — re-arc STAGE 1 (mini) launched from this repo;
  `run_overnight_batch.sh` REPO_ROOT repointed framework→xpt (the documented
  launch was dying on ModuleNotFoundError from the framework cwd).
- **2026-05-25** — `full_macros_prodlike` PASSED (content-confirmed; see Open
  problem). tokens 0.18.0 frame-timing duration fix. Re-arc staged.
- **2026-05-24** — strict-no-diff tokenizer rework shipped (FREQ_TRAJ unified
  op + absorbers, tokens 0.16.0/0.17.0). Motivating A/B: full macro set lifted
  eval_a content 0.150→0.274 (~1.83×) — proved the ceiling was
  tokenization-induced.
- **2026-05-21..23** — entropy/cluster/diffusion threads refuted at prodlike;
  `preframr-experiments` extracted to this repo; libraries split to PyPI
  (`preframr-tokens`, `preframr-audio`); preframr restructured
  (train/inference/model split, Corpus extraction).
- **earlier** — see git log + `data/refuted/`.
