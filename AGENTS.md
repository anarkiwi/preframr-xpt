# Operational notes for agents working on this repo

## Packages

Main repo: research + training-side code only. Stable layers
carry the non-torch foundation:

- `preframr-audio` ≥0.1.0 (PyPI) — SID audio rendering primitives.
- `preframr-tokens` ≥0.6.0 (PyPI) — torch-free library. Modules:
  - `reglogparser`, `regtokenizer`, `macros`, `stfconstants`,
    `reglog_helpers`, `alphabet_projection`, `reg_mappers`,
    `engine_fingerprint`, `coarsen_pass`, `dump_meta`
  - `constrained_decode` — sampling-time mask state machine (numpy
    bool; caller applies via single `masked_fill` at the boundary)
  - `audit_primitives` — `tier_accuracy`, `detect_tail_cycle`,
    `distinct_n`
  - `blocks` — block iteration / materialization + `SeqMeta` +
    `parse_eval_reglogs` + `LEGACY_EVAL_SUBSET_NAME`
  - `corpus.Corpus` — parse + tokenize + cache + load orchestration
    (~560 LoC formerly inside `RegDataset`)
- `preframr-experiments` (sibling source at
  `/scratch/anarkiwi/preframr-xpt`, editable-install / PYTHONPATH;
  no PyPI yet) — docker-driven experiment runner + spec registry
  formerly at `integration_tests/experiments/`. Pure orchestration:
  no preframr / torch imports.

Sibling source repos at `/scratch/anarkiwi/preframr-{audio,tokens,xpt}`
exist for library-side dev. The docker image installs the
`preframr-{audio,tokens}` libraries from the PyPI mirror;
`preframr-experiments` is run from its source dir via
`PYTHONPATH=/scratch/anarkiwi/preframr-xpt` (CLI runs on the host,
not in docker).

## Layout

```
preframr/
├── train/
│   ├── model/      # model package (lightning + bodies + heads + losses + tier_map + factory)
│   ├── trainer.py
│   ├── regdataset.py    # thin Corpus + BlockMapper adapter (~210 LoC)
│   ├── block_mapper.py  # torch.utils.data.Dataset wrapper around blocks files
│   ├── generalization_gate.py  # Lightning callback consuming audit_primitives
│   └── structural_loss.py
├── inference/      # predict, predict_lib, render_play (canonical CLI home)
├── args.py         # argparse + add_args; re-exports parse_eval_reglogs
├── parse.py        # CLI shim
├── stftokenize.py  # CLI shim
└── utils.py        # get_logger
```

Entry points:
- `python3 -m preframr.train.trainer`
- `python3 -m preframr.parse`
- `python3 -m preframr.stftokenize`
- `python3 -m preframr.inference.predict`
- `python3 -m preframr.inference.render_play`

## Project goal (OVERRIDING)

Train a SID model that **generalises** -- predicts unseen
continuations from arbitrary mid-song prompts, across composers
(primary `val_acc`) and ideally across engines (stretch) -- inside
a fixed deployment envelope:

- **Train:** single RTX 4090, 24 GB. Specs needing >~50M body to
  show Δ are out-of-envelope; refute in design, don't A/B.
- **Predict:** Jetson Orin NX (15.6 GB) at PROMPT=2048 / MAX=8192.
  KV cache at prodlike dims (16L × 4 kv_heads × 64 head_dim, GQA,
  bf16) is ~16 KiB/token, MAX=8192 → 128 MiB; bounded by trained
  seq_len, not VRAM.

## Open problem (OVERRIDING)

**We do not yet have an approach that demonstrably generalises.**
Each intervention is a research bet, not a tuning question.

**Token juggling is refuted** — every encoder-level macro / loss
move shifts structural accuracy without lifting content. Aggregated
in "Refuted alternatives".

**Active hypothesis (2026-05-25):** strict-no-diff tokenizer rework — every
token is a structured trajectory primitive or an enumerated carveout.
**SHIPPED in preframr-tokens 0.16.0**: the unified `FREQ_TRAJ` op (45, SUBTYPE
MONOTONE_RAMP/OSCILLATE/RUN) folds SLOPE+OSCILLATE_ENV+FREQ_VIBRATO+FREQ_RUN;
`FREQ_NUDGE`→2-atom delta; + torch-free profiling tools (`tokenizer_profile`,
`trajectory_coverage`). **Validated**: per-frame fidelity oracle byte-exact,
coverage 0.522, efficiency confirmed; main repo floors `>=0.17.0`, images rebaked.
**0.17.0**: `MACRO_FLAGS` derives from the passes (`GATE_FLAGS`/`REQUIRES_ARGS`
self-registration) not a hand-list; dropped 2 phantom flags, surfaced 4.
**0.18.0**: frame-timing duration fix — the parse pipeline was dropping up to
~67% of wall-clock duration (degenerate-first-frame `frame_diff=0`,
non-cycle-preserving `_consolidate_frames`, truncating `_add_frame_reg`); macro
round-trip now within 0–0.5% of the no-macro baseline. Main repo floors
`>=0.18.0`, images rebaked.
**PASSED (2026-05-25, 0.18.0):** `full_macros_prodlike` complete. The 4 CLI-only
absorber macros (CTRL_TRIPLE/FREQ_NUDGE/RELEASE_UPDATE/lonely_catch_all) on the
shared FREQ_TRAJ base, vs the registered baseline. Verdict via the **content-tier
per_class audit** (the decisive, un-confounded gate — `audit_*.json` in
`/scratch/tmp`): **content lift is genuine, not structural inflation.**
- eval_a content **0.160→0.287 (+0.1265)**; structural only +0.040.
- **eval_b content 8/8 non-negative** (mean +0.0997; mibri 2.5×, daglish +0.181);
  structural deltas ~±0.04, several negative.
- All-tier eval_a 0.2391→0.3377; alphabet +18.7% (5492 vs 4628); compression
  FAILED (tokens/song 9118 vs 9042 — no token-count win, but efficiency is
  secondary). Learnability is the criterion and it passes decisively.
First non-refuted, content-confirmed intervention in the program, and it is
**tokenizer-side** — vindicates the strict-no-diff pivot. **Method note:** all-tier
val_acc was confounded (different tokenizations, structural-token inflation); only
the content-tier audit settles content vs structural — always run it before
calling an encoder A/B a win.
**Next:** the vocab-trimmed re-arc (see Forward-looking) — re-run ALL baselines +
the 5 refuted model-side specs from scratch at the deployment-efficient config.

Priority order:
1. Algorithmic generalization detection — per-tier accuracy
   (`per_class_acc_audit.py` + `audit_checkpoint_per_class.py`),
   loop-collapse, prompt-conditioning. Audio review is supplementary.
2. Encoding efficiency — secondary; accept inflation if it might
   help generalize.
3. Predict-host envelope — capacity the predict host can't carry
   is not a win.
4. Tracker-authoring prior — macros that map to how SID was authored
   start favoured; signal-only compression with no analogue starts null.

## Environment

- **PyPI cache + release:** `build.sh` sources a gitignored `.env` (template
  `.env.example`) for `PIP_OPTS`; with no `.env` it resolves from upstream PyPI.
  After publishing a new `preframr-{audio,tokens}` release, the proxpi mirror
  serves a STALE index until busted — before rebaking on the cache host:
  `curl -X DELETE http://192.168.5.1:5001/cache/<pkg>` then
  `.../cache/list`; confirm the new version at `.../index/<pkg>/`.
- **Jetson predict host:** Orin NX (15.6 GB) NFS-mounts `/scratch`.
  `build.sh` builds `anarkiwi/preframr-jetson`. Pass `--runtime=nvidia`.
- **CPU compute host `fogbank`:** same NFS `/scratch`, 72 cores, no GPU.
  Route CPU-only workloads here.

## Images

`./build.sh` produces in parallel:

- **anarkiwi/preframr** — train + test image; full deps incl.
  tensorboard, pylint, black, pyright. Entry points: trainer, parse,
  stftokenize, predict, render_play. Runs `./run_tests.sh` at build
  time.
- **anarkiwi/preframr-xpu** — same content, intel xpu base.
- **anarkiwi/preframr-predict** — slimmed eval/predict image built
  from `Dockerfile.predict` + `predict-requirements.txt`. Drops
  tensorboard + test/lint deps; keeps Lightning runtime (Model is
  still a LightningModule). Validates with `--help` smoke against
  predict + render_play + the four audit scripts in
  `integration_tests/profile/`. Use this for eval experiments and
  any predict-only workload.
- **anarkiwi/tensorboard** — TB sidecar.

Rebuild (`./build.sh`) on `requirements.txt`,
`predict-requirements.txt`, or any Dockerfile change.

## Tests + lint

```bash
docker run --rm \
    -v /scratch/anarkiwi/preframr/preframr:/preframr \
    -v /scratch/anarkiwi/preframr/tests:/tests \
    -v /scratch/anarkiwi/preframr/integration_tests:/integration_tests \
    -v /scratch/anarkiwi/preframr/run_tests.sh:/run_tests.sh \
    -v /scratch/anarkiwi/preframr/.coveragerc:/.coveragerc \
    anarkiwi/preframr /run_tests.sh
```

## Integration suite

Experiment runner + specs + pinned tier lists live in the sibling
`preframr-experiments` repo (`/scratch/anarkiwi/preframr-xpt`).
Invoke via:

```
PYTHONPATH=/scratch/anarkiwi/preframr-xpt python3 -m preframr_experiments.run <spec> ...
```

The main repo retains:

- **Pre-launch gates:** `preframr_experiments/validate_branches.sh`
  (in the sibling repo), `integration_tests/profile/train_prodlike_oom_smoke.py`
  is no longer here — both pre-launch smokes live in
  `preframr_experiments/preflight/` now.
- **Generalization gates** (post-train, CPU-friendly):
  `integration_tests/profile/{loop_detection,per_class_acc,prompt_conditioning,audit_checkpoint_per_class}_audit.py`.
- **Tier metadata** at `integration_tests/data/<tier>/`:
  `engine_fp_palettes.json`, `picker_summary.json`, `leak_audit.json`,
  `engine_*.json`, `eval_b_family_map.json`, `irq_audit.csv`. The
  `.list` files + `HVSC_VERSION` pins themselves moved to the sibling
  repo. Eval-leak audit `profile/audit_eval_leak.py` before any new
  corpus pin (results write back to main repo's `data/<tier>/`).
- **Audit artefacts** at `integration_tests/data/audit/` — per-experiment
  generated audit JSONs + streams.
- **Refuted registry** moved to
  `/scratch/anarkiwi/preframr-xpt/preframr_experiments/data/refuted/`.
- **Dump cache:** `/scratch/preframr/training-dumps/`. Parse+tokenize
  artefact cache at `/scratch/preframr/training-dumps/dataset_cache/`
  (auto-populated; `PREFRAMR_DATASET_CACHE_DISABLE=1` to bypass).
- **`PREFRAMR_SRC_DIR` env var** (default
  `/scratch/anarkiwi/preframr/preframr`) tells the sibling runner
  where the main repo's `preframr/` package is, for bind-mounts into
  preflight + parse + tokenize + train containers.

### Wallclock anchors

- micro_mini: ~5 min/arm. mini body=large: ~12-20 min/arm.
- canonical: 60-120 min/arm. prodlike: ~6-11 hr per (arm, seed).
- `run_memorize_int_test.sh`: ~5-6 min.
- `run_generalize_int_test.sh`: ~75-115 min.

## Conventions

- **Background runs.** `nohup`+`disown`. Don't poll; use `ScheduleWakeup`.
- **Code = frozen baked image by default.** Runs use the tested baked
  `preframr/` code; rebake (`./build.sh`) to pick up edits. So a long run
  can't be perturbed by edits and the working tree stays free to change
  mid-run — consistency/stability over iterate-without-rebake. The
  working-tree bind-mount is **opt-in, the exception**: `run.py --bind-src`
  (`$PREFRAMR_BIND_SRC=1`) mounts `preframr/{train,inference}/` over the bake
  per container role (top-level CLIs always bake-only). Don't use `--bind-src`
  without checking first; it runs un-gated code. See
  `design/landed/train_inference_split_design.md`.
- **Comments.** No session narration, no dev-local paths, no PR
  numbers. `tests/test_lint.py` rejects narrative `# ...` and
  >5-line docstrings.
- **NFS hygiene.** Don't leave `tail -f` on workdir files between
  runs (NFS silly-renames to `.nfs*`). Stop `preframr_tb` container
  before deleting tb_logs subtrees.
- **TensorBoard.** `preflight_check` restarts `preframr_tb` on
  `<root>/results` each launch. `tb.sh` is the manual equivalent.
- **Arm ordering.** Target arm first in `spec.arms`, baseline last.
  Runner is sequential; downstream audition wants target ckpt first.

## Overnight batches

Sequential under `/scratch/tmp/preframr_experiments/` via
`preframr_experiments/run_overnight_batch.sh` (in the sibling repo).
Status: `preframr_experiments/check_overnight_batch.sh`. Done marker:
`overnight_batch.done`.

## Multi-GPU rental decision

Deferred until generalization approach lands. Re-open when:
prodlike result; DDP-4 ≥60%; `--resume` / `auto_early_abort` /
`--max-parallel-arms` landed; last 5 specs refutation rate ≤40%.

## Forward-looking work

### Shipped: FREQ_TRAJ tokenizer rework (preframr-tokens 0.16.0) — training A/B in flight

The strict-no-diff rework shipped as the unified `FREQ_TRAJ` op (45, SUBTYPE
MONOTONE_RAMP/OSCILLATE/RUN folding SLOPE+OSCILLATE_ENV+FREQ_VIBRATO+FREQ_RUN;
`FREQ_NUDGE`→2-atom delta) + torch-free profiling tools (`tokenizer_profile`,
`trajectory_coverage`). **Validated 2026-05-25:** per-frame fidelity oracle
byte-exact (run on a docker-capable host — it SKIPs in nested docker, so ensure
CI runs it there), `trajectory_coverage` captured_frac **0.522**, efficiency
confirmed. Main floors `>=0.17.0`; images rebaked. Design + validation results:
`integration_tests/design/unified_oscillation_primitive_design.md`.

**Why it mattered (motivating A/B, 2026-05-24, prodlike 1 seed 60 ep, SAME
model):** pre-rework full macro set vs registered baseline lifted eval_a content
val_acc **0.150 → 0.274 (~1.83×)**, all 8/8 eval_b families up — proving the ~0.13
content ceiling that refuted all 5 model-side interventions was
**tokenization-induced**, not model-capacity. The win was *strictness +
recognition*, not compression. FREQ_TRAJ makes the structural primitives actually
fire (0.522 coverage vs the old ~0.24% of atoms).

**In flight: `full_macros_prodlike`** (see Open problem → LAUNCHED). NB FREQ_TRAJ
rides the shared base in both arms, so this isolates the 4 absorber macros, not
the FREQ_TRAJ lift itself. Broader arc: re-run ALL baselines + the 5 refuted
model-side specs from scratch (no metrics transfer; vocab/op-alphabet/seq-len
changed). The dataset-cache `extra_cargs` collision is fixed (keyed on the
parse/tokenize cargs slice, landed in preframr-xpt), so the cache stays on — no
`PREFRAMR_DATASET_CACHE_DISABLE`.

**Content tier deliberately lossy** (slope/preset/transpose cent-binned) — by
design, not a bug; content-tier-OFF is byte-perfect vs raw; lossless rework deferred.

### Land any time

- **Per-primitive round-trip audio gate** — wire `compare_renders`
  (preframr-audio 0.4.0) over ~100 songs into CI (≥95% within tolerance);
  the per-frame register oracle (`test_full_pipeline_fidelity.py`) already
  covers the zero/structural/mid tiers byte-exact.
- **12-SID WAV audition cohort** — eventual non-negotiable gate before
  flipping any tokenizer default + re-cutting training data.
- **Profile + optimize preframr-tokens parsing** — correct but slow; parse
  is a big share of uncached run setup (parse+tokenize ~25 min/prodlike).
  Profile the `reglogparser`/`regtokenizer` hot paths, optimize, keep the
  per-frame fidelity oracle green.
- **Recover control-write-rejected dumps** — analyze songs rejected for
  too many control writes; characterize the writes and see if we can admit
  them (raise/relax the threshold, or absorb via a macro) to grow the corpus.
- **Mechanical preframr-tokens cleanup** (`preframr-tokens:API_SURFACE.md`).
- **Audit docker mount convention** — `work_dir → /scratch/preframr` inside
  parse/tokenize/train/audit containers; document in
  `integration_tests/profile/README` or a helper.

### Refuted alternatives

Registry at `preframr-xpt:preframr_experiments/data/refuted/<exp>.md`. Entries:
`legato_ab`, `palette_merge`, `head_row_class`, `adsr_equivalence`,
`macro_coarsening`, `b2_unblock`, `palette_pwm_prereqs`,
`global_instr_ids_phase_a`, `weighted_token_loss`,
`learnable_class_loss`, `voice_trajectory` (insertion/replace/distributed),
`set_to_diff`, `contrastive_infonce_auxiliary` (mini single-seed
lift within noise; prodlike epoch 8 content acc 0.0000; design
self-undermines per `per_tier_heads_design.md` rationale),
`per_tier_heads_mos_prodlike` (Phase 3 v10: criterion 1 borderline
FAIL at absolute 0.14 floor; criteria 3+4 FAIL — router posterior
saturates at prodlike scale, outputs ignore prompt content),
`mask_structural_loss` (option-1 separator probe at mini: masking
structural-tier CE collapsed diversity_ratio to 0.863, **worse than
baseline 1.220** — structural supervision was load-bearing for
prompt-conditioning; refutes the gradient-dominance framing and
confirms the router-architecture framing tested by entropy retest),
`per_tier_heads_entropy_prodlike` (v11 lambda=0.01 borderline FAIL
crit 4 diversity_ratio 1.123 vs gate 1.2; v12 lambda=0.02 WORSE on
every active criterion — div_ratio 1.038, collapse 42%, content acc
slipped under 2.0× ratio. **Lambda is non-monotonic at prodlike**
with peak at 0.01; tuning cannot get past gate 1.2),
`cluster_conditional_content_head` (Phase 2 prodlike: same content-tier
ceiling ~13% eval_a, diversity_ratio ~1.0-1.2 — same shape as the
prior interventions),
`content_diffusion` (Phase 3 prodlike: same ceiling; sampling-side
intervention did not change the outcome from CE training).
All 5 model-side interventions concentrated at the same ceiling;
strategic pivot to tokenizer-side strict-no-diff rework
(`preframr-tokens:TOKEN_IMPROVEMENTS.md`).

### Pipeline coverage holes

- `GENERALIZE_MIN_VAL_ACC` floor — needs 2-3 canonical baselines.
- `start-seq` rotation — `predict_load` hard-codes rotation 0;
  50% unreachable at max_perm=2.
- Audio-neutrality invariants pinned by
  `tests/test_sid_same_value_writes.py` +
  `tests/test_parser_canonicalisation_audio_invariants.py`.

### Runner / infra fragility (new, post-2026-05-21 babysitting)

- Dataset cache (`/scratch/preframr/training-dumps/dataset_cache/`)
  short-circuits parse + stftokenize on retries with identical
  spec inputs; ~25 min/run win at prodlike. Disable with
  `PREFRAMR_DATASET_CACHE_DISABLE=1`. `pre_run_hook` is skipped
  on cache hit (warning logged). Key includes the
  parse/tokenize-affecting slice of `arm.extra_cargs` (train-only flags
  denylisted), so macro A/B arms no longer collide
  (`runner_iteration_efficiency_design.md` #1), **and the image's
  preframr-tokens version** (`_image_tokens_version`) so a tokenizer upgrade
  self-invalidates instead of silently serving stale tokenization.
- `_robust_rmtree` retries Errno 39 + Errno 16; bounces
  `preframr_tb` on first failure to release NFS handles. `run_arm`
  restarts TB after the rmtree succeeds.
- Autocast fp32-promotion trap on new heads: any new
  `torch.nn.Module` in `preframr/train/model/heads*.py` must cast
  log_softmax / logsumexp outputs back to input dtype, or the
  per-position output buffers stay fp32 and OOM at prodlike
  (B=2, T=8192). Pinned by
  `tests/train/test_per_tier_heads.py::test_bf16_input_preserves_*`.

### Framework follow-ups

- Cloud-rental prereqs DEFERRED: `--resume`, `auto_early_abort`,
  `--max-parallel-arms`, `profile/ddp_scaling.py`.
- Streaming unembed-CE — recovers prodlike 2× wall.
- Orin inference optimisation — 4% GPU util at predict; vocab
  shrink + GPU-resident constrained-decode unblock full-context
  audition. **Vocab is ~91% dead**: full_macros 0.18.0 train split uses
  only **2929 / 32768** pieces (8.9%), usage-weighted **1.23 atoms/token**
  (top-15 all single atoms — compression is atom-level macros, not Unigram
  merges). `tkvocab` can drop ~8× (→~4096) to shrink embed/unembed tables
  with no train coverage loss; re-measure per arm before flipping.
- **Predict-performance push (queued, contingent on this A/B):** once
  multi-day training is the norm, lead with the predict-host envelope —
  vocab shrink (above) → GPU-resident constrained-decode → full-context
  audition. Re-open alongside the Multi-GPU rental decision.
- Melody-transfer augmentation — `profile/augment_melody_transfer.py`;
  Phase 0 smoke pending. Orthogonal to multi-modal objective.
- Generalization gate thresholds — default
  `content_over_structural_min=0.05` misfires at pre-saturation
  tiers (body=large mini baseline ~0.015). Recalibrate per tier.

### Deferred deploy-stage efficiency (post-generalization)

Both refute as generalization bets; queued as ~25% token-budget
wins once a generalising model lands.

- **FRAME subsumes VOICE** — header token encodes voice ordering
  + per-voice write counts. Alphabet inflation +50% on FRAME;
  partial tracker analogue (order yes, counts no).
- **Drop VOICE_REG** — SID reg space already disambiguates voice
  (regs 0-20). Zero alphabet inflation. Loses `voice_block_order`
  pass.

### Resolved log (compact; details in git log)

- **2026-05-24 (latest)** — Strict-no-diff tokenizer rework spec
  landed (`preframr-tokens:TOKEN_IMPROVEMENTS.md`, canonical
  implementer section at top). Architectural pivot following all 5
  model-side interventions refuted at prodlike. 12 empirical probes
  across two sessions, 100% accounting of unmodelled rows under
  strict-no-diff (58,087 residual mapped to 61 signatures), six
  primitives + universal trajectory-anchor extension rule absorb
  everything. Locked detector defaults (vibrato_min=2, portamento_min=3,
  NOTE_ON_window=12, VOICE_TRACK MIN_TRACK_DURATION=10). New op set:
  OSCILLATE_ENV (subsumes items 1/8/9 + FREQ ARP via parametric
  envelope families), VOICE_TRACK (pre-per-voice-split cross-voice
  tuning, 16-entry interval table), FREQ_NUDGE, RELEASE_UPDATE,
  CTRL_TRIPLE. WAV audition gate on a 12-SID representative cohort
  is the non-negotiable final acceptance. Implementer needs no further
  probes; spec covers every parameter. Next-session arc rewritten —
  cluster Phase 0 work shelved (cluster + diffusion both REFUTED at
  Phase 2/3). Re-run from scratch is required post-rework (no
  metrics transfer across vocab change).
- **2026-05-23 (latest)** — Entropy thread definitively REFUTED at
  prodlike. v11 (lambda=0.01) landed 3-of-4 PASS, criterion 4
  diversity_ratio 1.123 (gate 1.2, off by 0.077). Mini sweep at
  {0.01, 0.02, 0.05} showed inverted-U with peak at 0.02 (mini
  div_ratio 1.596), predicted ~1.14 at prodlike under v11's 29% lift
  attenuation. v12 (lambda=0.02 prodlike) actual: div_ratio 1.038
  (WORSE than v11's 1.123), collapse_rate 42% (vs v11's 8%), content
  acc 1.98× (slipped under the 2.0× ratio criterion). Lambda is
  non-monotonic at prodlike too, peaking at 0.01; tuning cannot reach
  the 1.2 gate. Refuted entry:
  `preframr-xpt:refuted/per_tier_heads_entropy_prodlike.md`. Pivot to
  cluster-conditional content head (queue item 2; design committed,
  Phase 0 in flight). Audit artefacts:
  `integration_tests/data/audit/per_tier_heads_entropy_prodlike{,_v12}/`.
- **2026-05-22** — Two probes for the v10 prodlike refute
  ran at mini in succession. **Entropy retest** (lambda=0.01) at
  `per_tier_heads_entropy_retest`: val_acc 0.1483 ± 0.0016 (within
  1σ of lambda=0 0.1494 ± 0.0028), collapse 0/12 at T=0.5,
  diversity_ratio **1.535** (vs lambda=0 1.220) — **PASS** all three
  criteria. **Mask-structural-loss probe** (separator option 1 at
  mini, plain CE + `--mask-structural-tier-loss`) at
  `mask_structural_loss_mini`: diversity_ratio **0.863** (BELOW
  baseline 1.220, refuted). The contrast separates frames cleanly:
  the prodlike failure is router-posterior shape, NOT structural-tier
  gradient dominance. Structural supervision is load-bearing for
  prompt-conditioning at mini (refuted entry
  `preframr-xpt:refuted/mask_structural_loss.md`). Entropy escalating
  to prodlike as v11 next-session item #1. Companion infra: bumped
  `requirements.txt` floor to `preframr-tokens>=0.9.0`, rebaked all
  4 images (`anarkiwi/preframr{,-predict,-xpu}`, `tensorboard`) with
  the 0.9.0 cutover + the new `--mask-structural-tier-loss` flag.
- **2026-05-22** — Phase 3 prodlike per_tier_heads + MoS
  K=4 (v10) **REFUTED**. Mixed-tier eval_a content acc 0.1358 vs
  baseline 0.0618 (2.20× ratio, but 3% short of the 0.14 absolute
  floor — which was a mini-tier number). 8/8 eval_b families
  positive content lift. Criterion 3 collapse_rate at T=0.5: target
  33% (4/12) vs baseline 8% (1/12). Criterion 4 diversity_ratio:
  target 1.031 (`prompt_ignored`) vs baseline 1.401 (`prompt_used`).
  Signature: both regressions concentrate on **random prompts**;
  the per-tier router posterior dominates output statistics and
  outputs ignore prompt content (router-saturation failure mode
  queued in `per_tier_heads_mos_revisited.md` "Open questions").
  Refuted entry: `preframr-xpt:refuted/per_tier_heads_mos_prodlike.md`.
  Audit artefacts: `integration_tests/data/audit/per_tier_heads_phase3/`.
  Companion fix: `preframr/train/regdataset.py` `reg_widths` setter
  (predict.load_model:393 crashed AttributeError post-Corpus-extraction;
  no test exercised end-to-end load_model).
- **2026-05-22** — Cutover to preframr-tokens 0.8.0 + preframr-audio
  0.2.0 (`a7d8cf3`). New helpers consumed: `tier_classify.{vocab_id_tier,
  build_vocab_tier_ids, build_vocab_tier_map}`, `token_weighting.vocab_frame_weights`,
  `reglog_helpers.read_initial_irq`, `constrained_decode.{frame_marker_count,
  tail_charge_for_prompt}`, `macros.transform.LOSS_TIER_NAMES`,
  `stfconstants.DEFAULT_IRQ_CYCLES`. Main repo `-87 LoC` net; the
  (op, reg, subreg) → decision switches now live in preframr-tokens
  next to the data they classify. Remaining stfconstants imports in
  main are `PAD_ID` + `MODEL_PDTYPE` only (boundary constants).
  Audio-side cutover: `preframr-audio/fidelity.py::_irq_from_df` now
  routes through `read_initial_irq` with a sentinel (preserves the
  "raise on no FRAME rows" contract). Phase 3 prodlike v10 ran to
  completion using a snapshot of the OLD image (cutover landed
  after the in-flight run started); the new image is in place for
  the post-train audits and the next round.
- **2026-05-21 (latest)** — `preframr-tokens` 0.8.0 release (narrowed
  API surface, +24 tests). `preframr-experiments` sibling repo
  extracted to `/scratch/anarkiwi/preframr-xpt`. Carries
  `integration_tests/experiments/` + tier `.list` files +
  `HVSC_VERSION` pins + `refuted/` registry + 3 runner-side profile
  scripts (`hvsc_version_check`, `train_preflight_smoke`,
  `train_prodlike_oom_smoke`) + runner shell wrappers
  (`validate_branches`, `run_overnight_batch`, etc.) + runner
  tests. Pure orchestration, no preframr / torch imports. Single
  cross-repo edge: `PREFRAMR_SRC_DIR` env var. New CLI:
  `PYTHONPATH=/scratch/anarkiwi/preframr-xpt python3 -m preframr_experiments.run`.
  Companion fragility commit: MoSHead bf16 fix (resolves the
  prodlike OOM), `_robust_rmtree` Errno 16 + TB bounce, dataset
  artefact cache at `dataset_cache/` (~25 min/retry win),
  `Dockerfile.jetson` entry-point fix, regression tests on all of
  the above.
- **2026-05-21 (late)** — preframr-tokens v0.3.0 → v0.4.0 → v0.6.0
  released to PyPI in succession; main repo follow-on cutovers
  landed. Extracted surfaces: `constrained_decode` (torch-free,
  numpy mask; one-line torch glue at the boundary),
  `blocks` (`SeqMeta`, `parse_eval_reglogs`, `iter_voiced_blocks`,
  `materialize_block_array`, `parser_worker`, `glob_dumps`,
  `reg_widths_path`, `self_contained_prompt_df`), `corpus.Corpus`
  (~560 LoC formerly inside `RegDataset`), `audit_primitives`
  (`tier_accuracy`, `detect_tail_cycle`, `distinct_n`). Main repo:
  `preframr/inference/` split landed, `preframr/predict/` shim
  dropped, `preframr/train/model/` subpackage created (lightning +
  bodies + heads + losses + tier_map + factory; PL imports
  isolated to lightning + factory and pinned by
  `tests/train/test_model_pl_isolation.py`), `regdataset.py` 753 →
  ~210 LoC as a thin Corpus + BlockMapper adapter. `Dockerfile.predict`
  + `predict-requirements.txt` produce `anarkiwi/preframr-predict`
  for eval / serving workloads. Phase 2 verdict on per_tier_heads
  re-opened from "refuted at greedy" to "viable at T ≥ 0.5"
  (3-seed multi-seed audit at T=0.5: collapse 0/12, diversity
  1.223 ± 0.029).
- **2026-05-21** — preframr-tokens v0.2.0 released (PyPI), main repo
  restructured (`preframr/core/` → `preframr/train/` + flat CLIs);
  ClusterTable refactor removes corpus data from library;
  tokenizer alphabet-coverage bug fixed (Int64 + relaxed
  substitution); `content_floor_check` confirms body=small was the
  floor (content acc 0 there, 0.63% at body=large);
  `contrastive_prodlike` stopped at epoch ~11 after independent
  review flagged single-seed mini lift was within seed-variance
  noise (baseline 60-ep content acc 0.0063 exceeded contrastive
  30-ep 0.0043) and prodlike epoch 8 content acc was 0.0000;
  Approach B refuted, pivot to per-tier heads + MoS (Approach C,
  `design/per_tier_heads_design.md`); Phase 0 audits ran (PASSED
  framing: greedy 100% loop-collapse, sample 0%, diversity_ratio
  1.075); Phase 1 impl landed (`--per-tier-heads`, MoSHead,
  PerTierHeads, marginal-factorization unified-posterior fusion);
  Dockerfile astroid workaround applied (symlink-based
  `/code/preframr` for `pylint -E preframr` resolution) plus
  pre-existing lint cleanup across `preframr/train/model.py`,
  `tests/train/{test_model_ckpt_completeness,test_learnable_class_loss}.py`,
  `tests/test_generalization_gate.py`,
  `integration_tests/profile/{generate_for_audit,build_prodlike_4x_list}.py` —
  rebake in-flight at session end (killed before completion; resume
  with `./build.sh` post-reboot).
- **2026-05-20** — Distributed voice_traj + set_to_diff PASS at
  mini, refuted at prodlike. InfoNCE auxiliary loss landed.
  Generalization gate + audit suite landed. Stage-C `micro_mini`
  tier. Strategic pivot to multi-modal objective.
  preframr-audio v0.1.0 extracted to PyPI.
- **2026-05-19** — Local-context macro mandate. CtrlBigramPass +
  ArpeggioPass + FilterTripleSetPass landed. Corpus structural
  index built (87K SIDs).
- **2026-05-18** — VoiceBlockOrderPass + GateSlopeShiftPass +
  tokenizer-concat fix (atoms/token 1.003→1.801).
- **2026-05-13 → 16** — `legato_per_cluster` per-cluster verdicts.
  `hard_restart_ab` borderline PASS. `engine_fingerprint_evalb`
  re-pin (8 cross-engine families). `loop_lookahead_prodlike`
  PASS → default 3.
- **2026-05-10 → 12** — `mini_baseline_seeds` σ. `instrument_pass_ab`
  null. `cents_sweep` 50 wins. `mini_capacity_diag` ceiling.
- **2026-05-15** — Torch 2.12 migration. Train-only container
  refactor.
