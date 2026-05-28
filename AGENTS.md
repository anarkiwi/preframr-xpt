# Operational notes for agents (preframr-xpt)

This repo (`preframr-xpt`) is the **experiment surface**: the docker-driven
runner + spec registry + audits + design docs + tier data + refuted registry.
The framework, libraries, and corpus live elsewhere; this is where the
high-churn research work happens.

## Packages (as of 2026-05-26)

- **`preframr` 0.2.2** — framework only (train / inference / model / args /
  parse / stftokenize / utils + `mine_motifs.py`). Image
  `anarkiwi/preframr:0.2.2` (+ `:latest`). No PyPI package; it ships as the
  docker image. Adds the motif-pass CLI wiring (`--motif-pass` / `--motif-dict`
  in args; `preframr/mine_motifs.py` → `preframr_tokens.mine_dict_from_dumps`).
  `integration_tests/` (audits, probes, fixtures, design docs) was moved OUT to
  this repo; main carries only the framework.
- **`preframr-tokens` 0.20.0** (PyPI) — torch-free parser/tokenizer +
  macros + `render_play` + the **corpus-mined motif pass** (per-block, lossless;
  `MotifPass`/`MotifDict`/`mine_motifs`/`mine_dict_from_dumps`/`get_motif_dict`,
  OFF by default). Main floors `>=0.20.0`. See `design/motif_pass_design.md`.
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

- **`anarkiwi/preframr`** (`:latest` + `:0.2.2`) — train + test, full deps.
  Builds run `./run_tests.sh`. Entry points: trainer, parse, stftokenize,
  predict, mine_motifs. (`render_play` now lives in tokens.)
- **`anarkiwi/preframr-predict` / `-xpu` / `-jetson`** — slim eval/predict
  images (`Dockerfile.predict`, + xpu / jetson base overrides). All four
  publish `:latest` + `:${VERSION}`.
- **`anarkiwi/preframr-xpt`** — layers the runner + audits on top of
  `anarkiwi/preframr` (pinned `ARG BASE` = `:0.2.2`; override at build to track
  `:latest`). Build runs `pytest tests`; the audits + orchestrator run in this
  image. Experiment **arms** run in their per-spec `image` (default `:latest`;
  the motif spec pins `:0.2.2` for `mine_motifs.py`).

**Release process** — the authoritative copy is
[`design/architecture_overview.md`](design/architecture_overview.md) ("Release
process"): the PyPI-tag (libs) vs image-VERSION-file (images) mechanisms, why
merging is always safe, the cross-repo order, the **public-PyPI-propagation CI
gotcha**, and the proxpi cache-bust (`preframr_experiments/bust_release.sh`).
`build.sh` sources a gitignored `.env` (template `.env.example`) for `PIP_OPTS`
(the proxpi mirror) for local builds.

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
a structured trajectory primitive or an enumerated carveout. The tokenizer is
the unified `FREQ_TRAJ` op + 4 CLI-only absorber macros
(CTRL_TRIPLE / FREQ_NUDGE / RELEASE_UPDATE / lonely_catch_all). Per-frame
fidelity oracle byte-exact; 0.18.0 fixed a duration-drop bug (macro round-trip
0–0.5% of baseline).

**PASSED — `full_macros_prodlike` (single seed 2026-05-25; ×3-seed CONFIRMED
2026-05-27).** First non-refuted, content-confirmed intervention, tokenizer-side.
**What the A/B isolates (verified in every seed's `pipeline_spec.json`):**
FREQ_TRAJ rides in the shared base pipeline — it is in **both** arms — so this A/B
does NOT test FREQ_TRAJ. The sole arm difference is the target's four absorber
macros (`--ctrl-triple-pass --freq-nudge-pass --release-update-pass
--lonely-catch-all`). The win is attributable to those absorbers stacked on an
already-FREQ_TRAJ baseline — **not** to FREQ_TRAJ, whose own content contribution
this A/B does not measure. Verdict via the **content-tier per_class audit** (the
decisive, un-confounded gate):
- eval_a content **0.219→0.324 (+0.105), ×3 seeds, seed-stable** (baseline
  0.228/0.207/0.223; full_macros 0.318/0.323/0.330). Single-seed precursor was
  0.160→0.287.
- eval_b content **8/8 non-negative** (Δ +0.010…+0.135).
- **Not compression:** alphabet grew 4628→5492 and encoded_tokens/song was ~flat
  (9095→9137, slightly up). The absorbers re-type residual SETs, they don't
  shrink the stream; the lift is representation/learnability, not fewer tokens.
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
- **STAGE 2 (prodlike): LAUNCHED 2026-05-26** — reoriented to the representation
  axis (mini found no model-side candidate). `B=4 / accum=8` (effective batch
  32; `B=8` OOMed at 23.3 GiB). See "STAGE 2 RUNNING" below.

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

### STAGE 1 COMPLETE (batch finished 2026-05-26 01:39; `per_class` audit run on per_tier_heads)

7/7 specs ran. Identical tokenization across all (alphabet 3703), clean A/Bs
(mini body=large, 30 epochs, 3 seeds). **No content signal worth promoting to
prodlike from any spec.**

- `per_tier_heads_mos4`: +0.057 all-tier val_acc (0.114→0.171), val_loss
  9.8→5.1. **per_class content audit (seed0): mostly structural** (struct
  0.46→0.60, +0.144) **with a small but real content lift** (content
  0.0043→0.0375, +0.033 — ~8×, not pure inflation). BUT this is the arch already
  **refuted at prodlike** (router saturates → content collapses at scale), so
  the mini content gain is not expected to survive. No new prodlike A/B
  warranted. (Earlier "structural-only" call was too strong — corrected by the
  audit.)
- `content_diffusion`: flat (−0.002 vs mos4_entropy baseline).
- `contrastive` (InfoNCE): flat (+0.003).
- `mask_structural_loss`: negative (val_acc 0.018) — structural supervision
  load-bearing.
- `voice_permutation_K5` (data aug): **flat** (+0.0007, below its +0.005 bar).
  The one data-side bet didn't help at mini either.
- `content_floor_check`: body=large baseline content acc ~0.006.
- `cluster_content`: **INCONCLUSIVE — deferred by decision** (low-value: an
  already-refuted model-side bet; a mini number wouldn't move the verdict).
  Blocked on a code fix, not a data regen — documented here, not actioned.
  Rebuilt the structural C256 index against the new tokenizer
  (range now correct `[0,3702]`, gates PASS) — fixed the gross stale mismatch,
  but the rerun `cluster_C256` still fails all 3 seeds:
  `load_cluster_assignments` ValueError `vocab id 60 (local 32) ... missing`.
  Root cause: `build_content_clusters._content_vocab_ids` classifies a
  different subset as "content" (2417 of 3703 vids) than the model's
  content-tier definition demands — the two diverged post-FREQ_TRAJ. Fix
  options: align the builder's content-vid selection with the model's content
  tier; OR cluster every vid so none is ever missing (the model only looks up
  the ids it needs); OR a fallback in `load_cluster_assignments`. Index at
  `data/content_clusters/mini_freqtraj_structural_c256.json`
  (`PREFRAMR_CONTENT_CLUSTER_INDEX`). Does NOT change the STAGE 1 read —
  cluster was already a refuted model-side bet; baseline (mos4_entropy,
  val_acc 0.170) is the comparator.

Read: every model-side AND the data-side (voice_permutation) intervention
reproduces its refutation on the corrected tokenizer — no lift on top of the
tokenizer win; leverage is representation. The per_tier_heads mini content lift
is real but small and dies at prodlike. **STAGE 1 concluded** (cluster cell
deferred, above). Next prodlike effort goes to the **representation/tokenizer
axis** where the win lives (`full_macros`), not more model-side A/Bs.

### STAGE 2 RUNNING (launched 2026-05-26 02:47)

`full_macros_prodlike` **×3 seeds** at the deployment config
(`--tkvocab 8192 --batch-size 4 --accumulate-grad-batches 8`) — a **variance
bound** on the single-seed confirmed win (content 0.160→0.287), moved onto the
shipping-efficient config. The passed run was 1 seed at tkvocab 32768 /
B=2/accum=16; this is the new info (seed variance + stable B=4 training), not
the vocab trim (prodlike alphabet ~4628 < 8192, so 8192 drops no tokens —
content should ≈ the 32768 run).

- `--root /scratch/tmp/preframr_stage2` (preserves the passed seed0 artifacts
  under `…/preframr_experiments/results/full_macros_prodlike/`). Log:
  `/scratch/tmp/preframr_stage2/run.log`. Per-seed content audit via runner gate.
- **ETA ~36-66h** (6 arm-seeds × 6-11h; tokenizes fresh ~25 min, then cached).
  GPU fully booked — keep foreground work non-GPU.
- Launch (host-side from this repo):
  ```
  nohup python3 -m preframr_experiments.run full_macros_prodlike \
    --root /scratch/tmp/preframr_stage2 --seeds 3 \
    --tkvocab 8192 --batch-size 4 --accumulate-grad-batches 8 \
    > /scratch/tmp/preframr_stage2/run.log 2>&1 & disown
  ```
- **2 arms × 3 seeds = 6 arm-seeds** (`full_macros` target + `baseline`); OOM
  gate passed at B=4 (steady ~17.7/24 GiB vs B=8's OOM).
### STAGE 2 COMPLETE (2026-05-27 00:33) — full_macros content win CONFIRMED ×3 seeds

6/6 done. **DECISIVE content-tier per_class audit** (run on all 6 best ckpts,
cuda, whole eval set; `audit_per_class.json` per seed dir + parser
`/scratch/tmp/parse_per_class.py`):

- **eval_a CONTENT-tier acc: full_macros 0.324 ± 0.006** (0.318/0.323/0.330) vs
  **baseline 0.219 ± 0.011** → **Δ +0.105, seed-stable.** content_over_structural
  0.479 vs 0.322. The single-seed pass (0.160→0.287, tkvocab 32768) **HOLDS** at
  the deployment config (8192/B=4) — both arms sit higher, full_macros content
  (0.324) exceeds the old 0.287. **This is the program's first confirmed,
  multi-seed, un-confounded content/generalization win — and it is tokenizer-side.**
- All-tier (confounded, for ref): eval_a val_acc 0.384±0.005 vs 0.313±0.009.
- **Content-tier eval_b stratification (CONFIRMS the all-tier read on the content
  tier):** full_macros beats baseline on **all 8** families (Δ +0.010…+0.135), but
  content is sharply **family-specific** — 0.031 (marquis) → 0.479 (winterberg),
  stdev 0.121. Laggards **marquis (0.031), dobek (0.172), wilson (0.236)**.
  marquis's all-tier 0.245 was almost all *structural* — its content is ~zero,
  visible only on the content tier → top **targeted-augmentation** target
  (preframr-aug). Parser `/scratch/tmp/parse_per_class.py`.
- **Next:** **melody learnability** is the active research arc → `design/melody_learnability.md`.
  **6 mini A/Bs** converged: melody onset is **predictable** (trigram 0.79–0.82, not aleatoric),
  model has **capacity** (freq_core: SET 0.08→0.42; freq_onset_channel: SET 0.15→0.83), melody
  is **~13.4% of stream** (not rare), yet V0-onset = **0.000 every time at mini** → scale-bound
  hard cross-song prediction (only prodlike has moved it, op45=0.067). The freq_onset_channel
  mini gave the biggest SET-cleanup content lift (content-tier 0.076→0.249) — still SET-carried.
  Landed encoding/loss stack: anchoring (0.25.0/0.2.6) + interval V0 (0.26.0/0.2.7) +
  FREQ_ONSET channel (0.27.0/0.2.9) + `--onset-loss-weight` (0.2.8). **RUNNING:**
  `melody_stack_prodlike` (full_stack vs full_macros, 3 seeds, deployment config on `:0.2.9`)
  — the deferred decisive test, dual-purpose (melody at scale + SET-cleanup scale-confirmation),
  seed-major so a 1-seed cross-arm signal comes ~6–11h in not 30h. Reserve: distributional/
  perceptual melody metric pivot. Augmentation (preframr-aug) sits behind a learnable
  substrate. LESSON: always read the content tier — via reusable `audit.content_tier_report
  --onset` (op-aware
  unified `melodic_onset_bucket`) + `melody_predictability` (the predictability ceiling).

### Motif pass — REFUTED 2026-05-27 (both v1 exact + v2 templated)

The corpus-mined motif pass does NOT help on the decisive content-tier gate.
3-arm mini A/B (full_macros base): **v2 templated content 0.036 ± 0.001 vs no-motif
baseline 0.045 ± 0.011 vs v1 exact 0.019** (subset `eval`). v2 recovers most of v1's
−0.026 regression (de-fragmentation + MOTIF_ARG→content exposure stop v1 "hiding"
content in zero-tier MOTIF_OP, content n 93k→104k) but lands *at/below* plain
full_macros — never beats it. No compression either (v2 61 templates, enc/song 7586 ≈
baseline 7561). Comparability: v2 from `:0.2.4`, baseline+v1 from the prior `:0.2.2`
`motif_mini_body_large` run; identical spec config, image delta is motif-only code.
loop/prompt guards not run (gated on a content win; none). Stub
`data/refuted/motif_pass.md`; designs `design/motif_pass_design.md`,
`design/motif_templates_v2_impl_design.md`.

**Known regression captured in the stub:** tokens 0.23.0's frame-guard fix regressed
`mine_motifs` (v1 BPE) into an O(k²·N) ~2.4 h spin (frame atoms become permanently
unmergeable → pile up at the top of `most_common()`; scan + O(N) `_ncomposers`
re-scans blow up). NOT fixed (motif refuted); fix direction in the stub. Repro
`/scratch/tmp/motif_v1_hang_repro.py`. The v2 templated miner is frame-filtered up
front and unaffected.

**Follow-on probe — DIAGNOSED → trajectory anchoring (A/B live):** the content-tier
per_class audit showed the full_macros win is carried by **SET register values** (acc
0.44, one class SET val=21 at 0.835), NOT melody — **FREQ_TRAJ (op 45) ~0.026 acc, 0.6%
of content hits despite 9% of positions**, and a *different* op is predicted 73% of the
time at FREQ_TRAJ positions. Tolerance-banding the metric did NOT rescue it
(`/scratch/tmp/audit_freqtraj_tolerance.py`, prodlike win ckpt) → not a metric artifact.
Root cause: the **un-anchored encoding** — `FreqTrajectoryPass` segments on value-runs
and discards the gate/sweep anchor, so the note-onset pitch reaches the model as noise.
Fix landed: `TrajectoryAnchorPass` (tokens 0.25.0) + opt-in `--trajectory-anchor-pass`
(preframr 0.2.6). **Mini A/B `trajectory_anchor_mini` (2026-05-27): content WIN but NOT
melody.** It lifted **content-tier 0.036→0.080 (+0.044, seed-stable), val_acc 0.113→0.137**
— but **SET-carried (op0 0.063→0.175); op45 stayed at the floor (0.001→0.002).** Token
distribution confirms the mechanism: anchoring is **anti-compressive** (tok/song 7561→8039,
+6.3%; alphabet 4255→4211, slightly cleaner) and its biggest atom-level effect is **+9% SET
anchor tokens (op0)**; corpus **FREQ_TRAJ atom count is ~flat (−1.7%)** — NOT the dramatic
consolidation the design predicted (the eval standalone-op45 drop 29.5k→16.0k is post-merge
displacement, not fewer FREQ_TRAJ). Mini can't test melody: op45 ~0 in *both* arms (prodlike baseline 0.067).
**ROOT CAUSE → INTERVAL-V0 pivot:** probes localised the V0-onset failure — the onset line is
highly predictable (cond. entropy 2.2 bits, trigram **0.79**) yet the model predicts the **V0
onset exactly 0.000** (subreg-split + `audit.melody_predictability`). Cause: V0 is **absolute
pitch**, so a motif at a different key is a different token sequence → no cross-song transfer.
**Fix implemented:** interval-coded freq V0 (`--freq-v0-interval`, opt-in, byte-exact) — tokens
**PR #19** (fallback 0.26.0) + framework **PR #139** (floor 0.26.0, VERSION 0.2.7);
See `design/melody_learnability.md` (active arc) + the landed encoding/loss stack
(anchoring → interval V0 → FREQ_ONSET channel → onset-loss-weight). The absolute-encoding
`trajectory_anchor_prodlike` was **STOPPED** (~2h, 0 ckpts; result predictable). Current read
via `audit.content_tier_report --onset` with the unified `melodic_onset_bucket`
(op45 V0 + op48 FREQ_ONSET + op47 NUDGE pitch on freq regs). Complementary:
`freq_core_ablation_mini` (PW/filter noise removed → confirmed mini capacity, not melody).
`design/trajectory_anchoring.md`.

## Tests + runner

- Framework tests (in `anarkiwi/preframr`): `./run_tests.sh` (black, pytest
  `/tests`, pylint curated, pyright, coverage ≥77).
- Runner + audits (in `anarkiwi/preframr-xpt`): `pytest tests` runs at image
  build, gated in CI by `.github/workflows/docker.yml` (builds `Dockerfile` on
  push to `main` + every PR — the build runs the test gate, then the full suite
  re-runs explicitly in the image; no proxpi mirror needed, base is public +
  pyproject is dep-free). Locally, `docker build -f Dockerfile .` reproduces it.
  Host CLI (orchestrates docker;
  the host needs only `preframr_experiments` on `PYTHONPATH`, not torch):
  `PYTHONPATH=. python3 -m preframr_experiments.run <spec> --root <work> [--seeds N --tkvocab 8192 ...]`.
  One spec module per A/B under `preframr_experiments/specs/`; the runner stages
  data → parse → tokenize → train per (arm, seed), each in a `docker run` of
  `spec.image` (default `anarkiwi/preframr` = `:latest` = 0.2.2; pin per-spec
  via `image=`). `nohup … & disown` for long runs.
- **Spec-dependent tokenization** (motif / cluster_content / voice_permutation —
  anything whose `pre_run_hook` mutates staged dumps or mines a per-spec
  artifact): launch with `PREFRAMR_DATASET_CACHE_DISABLE=1` so the hook runs
  every seed (the dataset cache key would otherwise reuse stale artefacts).
- **Content-tier audit (decisive gate):** per arm-seed, run
  `audit.audit_checkpoint_per_class --ckpt … --work-dir … --out audit_per_class.json`
  in the xpt image (forwards eval blocks → per-class/per-tier acc). Then read the A/B
  with **`python3 -m preframr_experiments.audit.content_tier_report --results-root <dir>`**
  (torch-free, host-runnable): per-tier content/structural Δ vs baseline, `eval_b_*`
  per-family, and a **by-op** breakdown (FREQ_TRAJ op45 first-class). All-tier val_acc is
  CONFOUNDED across tokenizations — only the content-tier read settles a representation
  A/B. Always run it before calling a win. The audit suite (reusable tested modules vs the
  `audit/probes/` reference archive) is indexed in
  [`preframr_experiments/audit/README.md`](preframr_experiments/audit/README.md) — use the
  tested readers, not bespoke `/scratch/tmp` scripts.
- Outputs under `/scratch/tmp/preframr_experiments/` (or the `--root` given).
  Status: `check_overnight_batch.sh`; done marker `overnight_batch.done`.

## Conventions

- **Code = frozen baked image by default.** Runs use baked `preframr/`; rebake
  to pick up edits. Working-tree bind-mount is opt-in (`run.py --bind-src` /
  `$PREFRAMR_BIND_SRC=1`) and runs un-gated code — don't use without asking.
- **Background runs:** `nohup`+`disown`; don't poll, use `ScheduleWakeup`.
- **Comments:** no session narration / dev-local paths / PR numbers;
  `tests/test_lint.py` rejects narrative `#` and >5-line docstrings.
- **NFS hygiene:** no lingering `tail -f` on workdir files (silly-renames);
  stop `preframr_tb` before deleting tb_logs subtrees.
- **Arm ordering:** target arm first in `spec.arms`, baseline last. The runner is
  **seed-major**: it loops `for seed: for arm: run`, so each seed runs target→baseline
  back-to-back, and a 1-seed cross-arm comparison is available the moment seed0 finishes
  both arms (early-signal-friendly, audition still gets target ckpt first per seed).
- **Renaming a transform** silently disables it in stale specs (no error) —
  grep specs on any pass/transform rename.
- **Design docs** live in `design/`, indexed by **research axis** in
  `design/README.md` (Generalization / Correctness & fidelity / Efficiency &
  deploy / Runner & infra / Data & corpus), with status as a per-row column. A
  new doc gets a one-line `**Status:**` header + a row under its primary axis;
  on ship it **moves to `design/landed/`**, on rejection it gets a
  `data/refuted/<exp>.md` stub. See that index's "How this index is organized".

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
- **Per-primitive round-trip audio gate** — the `compare_renders` helper +
  fidelity unit tests landed in preframr-audio (`fidelity.py`,
  `test_fidelity.py`/`test_dfs_equivalent.py`); STILL PENDING is the
  corpus-scale CI gate (run it over ~100 songs at ≥95% within tolerance).

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

### xpt path cleanup (RESOLVED 2026-05-26)
No `integration_tests` refs remain in xpt `.py`. `encodability_metric.py` was
removed (unused; served the refuted `global_instr_ids` Phase A; its audit module
was not carried through the extraction and is unrecoverable from git).
`build_prodlike_4x_list.py` + `pick_mini_stratified.py` defaults, the
`run_eval_per_composer_8k.py` docker mount, and the spec/audit docstrings now
point at xpt paths. Remaining repo-focus items: the fixtures move-out (below) +
the data/audit tracking decision — see `design/repo_focus_cleanup_scope.md`.

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
- `motif_pass` (tokenizer-side: v1 exact + v2 templated) — content-tier
  neutral-to-negative vs no-motif full_macros (v2 0.036 vs baseline 0.045);
  no compression. `data/refuted/motif_pass.md`.
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

- **2026-05-26** — re-arc STAGE 1 (mini) concluded: no model-side or data-side
  content signal on the corrected tokenizer (per_class audit on per_tier_heads —
  +0.033 content at mini, dies at prodlike); `voice_permutation` flat;
  `cluster_content` deferred (builder/model content-vid mismatch). Augmentation
  tooling + design moved to `preframr-aug` (new self-contained repo: Dockerfile,
  tests, CI). STAGE 2 prodlike launched — `full_macros_prodlike` ×3-seed
  variance bound at the deployment config (see STAGE 2 RUNNING).
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
