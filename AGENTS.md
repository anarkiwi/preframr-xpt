# Operational notes for agents (preframr-xpt)

This repo is the **experiment surface**: docker-driven runner + spec registry
+ audits + design docs + tier data + refuted registry. Framework, libraries,
corpus live elsewhere.

## Packages

- **`preframr` 0.2.19** — framework only (train / inference / model / args /
  parse / stftokenize / utils). Image `anarkiwi/preframr`. No PyPI; ships as the
  docker image (`0.2.19` + `:latest` published on main-push). Carries the
  per-op-accuracy gate. Floors `preframr-tokens>=0.44.0` (in **all** req files:
  `requirements.txt`, `predict-requirements.txt`, `jetson/predict-requirements.txt`
  — `op_name_by_id` is on the predict import path). `tier_map.build_op_map` reads
  op→name from tokens' `op_name_by_id()`. **Macro passes are supplied as ONE
  validated list** — `apply_macro_flags_to_args` resolves `--macro-flags` /
  `--macro-config` off the tokens registry (default all-OFF); the old per-flag
  `--foo-pass` args + `_PIPELINE_NAME_TO_FLAG` + `--pipeline-spec` are gone.
- **`preframr-tokens` 0.44.0** (PyPI) — torch-free parser/tokenizer + macros
  + `render_play`. **Byte-exact** (corpus dirty ~8%→0; 1 known load-dependent
  outlier `Ascension.1`, debug deferred); STAMP/PATCH/SWEEP/held-ARP/WAVETABLE
  codebooks; toggleable parse audit (`PREFRAMR_PARSE_AUDIT`). **0.44.0 (residual-SET
  drain):** raw `SET`s (unmodeled driver mechanisms) on the digi-excluded stride sample
  drained to **0** — `onset_def` (define-on-first via CTRL_WT), `env_multiload` (AD/SR
  hard-restart), `pre_gate_freq` (drop/relocate inaudible pre-gate freq — the first
  AUDIO-exact atom, in `parse_audit._LOSSY_RESETS`), `nibble_wavetable` (CTRL_WT subreg
  0/1 lanes). All default-OFF, OUT of `REGISTERED_MACROS`. **0.43.0:** codebook ids are
  pure define→ref ordinals never value-snapped + a `register_state` decode memo (~6.6%
  parse). 0.42 added PW/filter sweep mining + `op_name_by_id()`/`op_name_tiers()`.
  0.42.1 fixed the `per_reg_burst` empty-cand+barrier crash + a `FrameWalker` parse speedup.
- **`preframr-audio` 0.5.6** (PyPI) — SID audio rendering primitives.
- **`preframr-experiments`** (this repo; editable / PYTHONPATH, no PyPI) —
  runner + specs + `audit/` + tests. Pure orchestration on the host; audits
  import preframr/torch and run inside the **xpt image**.

Sibling source repos: `/scratch/anarkiwi/preframr-{audio,tokens,xpt,aug}` and
`/scratch/anarkiwi/preframr` (framework).

## Images

- **`anarkiwi/preframr`** (`:latest` + `:VERSION`) — train + test. Entrypoints:
  trainer, parse, stftokenize, predict, mine_motifs.
- **`anarkiwi/preframr-predict` / `-xpu` / `-jetson`** — slim eval/predict.
- **`anarkiwi/preframr-xpt`** — layers runner + audits on `anarkiwi/preframr`
  (pinned `ARG BASE`; override at build to track `:latest`). Build runs
  `pytest tests`. Experiment **arms** run in their per-spec `image`.

Release, build, test, cache — one authoritative doc: **`design/release_build_cache.md`**
(per-repo release procedure, the proxpi cache + how to bust it, local build/test
commands). Two standing rules from it: **run non-GPU work (builds, parse, audits,
pytest, lint) on `fogbank`** — `ssh fogbank`, 72 cores, shared `/scratch`, its own
docker + the preframr images — keep defroster for training; and when releasing a Docker
app **build the image locally in parallel with the push** so you never wait on CI + a
slow image pull (a failed local build is discardable).

## Project goal (OVERRIDING)

Train a SID model that **generalises** — predicts unseen continuations from
arbitrary mid-song prompts, across composers (primary `val_acc`) and ideally
across engines (stretch) — inside:

- **Train:** single RTX 4090, 24 GB. Specs needing >~50M body to show Δ are
  out-of-envelope; refute in design, don't A/B.
- **Predict:** Jetson Orin NX (15.6 GB) at PROMPT=2048 / MAX=8192. KV cache
  at prodlike dims ~16 KiB/token → 128 MiB at MAX; bounded by seq_len.

**Operational lens — LEARNABILITY** (`design/learnability_token_ordering_theory.md`, the
design north-star): generalisation is won when the *encoding* lets a bounded (~TC⁰)
transformer cheaply predict the next token — minimise causal-state + dependency horizon,
prefer induction-head DEF→REF copy over implicit per-frame counters, order by the driver
causal DAG. Correctness is the *gate*; compression / parse-perf / deploy are *infra*. A
training-free triage (`audit/learnability_triage.py`) ranks encodings before a run — run it at
**prodlike `seq_len=8192`** (the real block/predict scale), not mini. **Mini (4096) is not a
research dimension**: it mode-collapses in training (`loop_collapse_rate` ~1.0) AND distorts the
static read via its window size; it's plumbing/cost only. The triage's value is the
prodlike-*scale* learnability read at mini-*cost* (static, minutes) — reserve training runs for the
collapse→learning *threshold*. Model-side content interventions were refuted at the ~0.13 ceiling
that tokenizer-side `full_macros` then lifted — the lever is tokenizer-side representation.

## Current arc — PITCH REALIGNMENT (universal recovered-table) is the active encoding work; this session owns it (2026-06-06)

**The encoding direction = universal recovered-table pitch** (`design/universal_multiresolution_pitch.md`).
Replaces the generator's single-scalar per-tune `ref` (`tune_ref`/`note_of`/`recon`) with: shared **NOTE INDEX**
(note 49≈C5 everywhere, the universal prediction target) + per-voice **recovered table** (exact tracker entries)
+ per-voice **tuning** (chorus/detune) + **cents** modulation (tuning-invariant effects); base-pitch cross-tune
transfer via **intervals (Δnote)**, the melody `MELODY_INTERVAL` op. **Why:** the deployed generator encoding's
block-scale induction-copy regressed (0.916 ≤ baseline 0.930, alphabet 3.7×) = pitch refragmentation; this fixes it.
**Validated EXACT (not within-cents):** SWM/defMON/Hubbard/Galway recover the trackers' literal pitches+notes —
recovered note-index == the trackers' own `FREQTBL`/`NOTE_PITCH` table index (40/40 SWM voice-traces 100%).

**Branch `feat/universal-pitch-grid` (10 commits on `main`, all default-OFF, byte-exact gated):** foundation
`pitch_grid.py` (note_index/voice_tuning/note_freq/recover_table/decompose, lossless) + the gated `universal_pitch`
wiring — **step 1** intervals keyed on per-voice `note_index` (correct under chorus/detune, transferable);
**step 2** per-voice `GEN_TUNING`/`gen_ref` → static melody notes pure (resid→0); **step 3** `role_lane`→
`lane_rank`→`voice_lane` causal ordering (accompaniment→melody, byte-neutral). Each verified `parse_audit=raise`
clean + tests. **Step 4 = the go/no-go:** `learnability_triage --mode window` on the wired encoding vs current
generator vs atomic — does block-local note/interval copy clear 0.930? Default flips only if it does.
Deferred: GEN_TABLE arps on the universal index (§3 de-frag), octave+pitch-class re-keying probe (a lossless
bijective A/B for cross-tune transfer; the layer-4 scale-degree hypothesis).

**TAKEOVER (2026-06-06):** the melody-skeleton agent is **stopped**; this session owns pitch + melody. Layers 2
(interval) + 3 (`voice_lane` de-mux + `role_lane` + `lane_rank`) landed default-OFF (#69/#70); the wiring above
connected them. Melody base designs: `melody_skeleton_impl.md`, `superframe_voice_lane_design.md`,
`role_lane_factorization.md`. The harmony-conditions-melody +0.294-bits + de-mux +0.10-transfer pre-screens are the
melody levers.

**GENERATOR LANDING (context):** the generator pipeline (one self-verifying `{HOLD,ACCUM,SWEEP,TABLE}` model over
all value channels, lossless + residual-zero, `design/generator_mdl_representation.md`) merged on tokens `main`
(#62–#68): `generator_pass` default, per-pass zoo deleted, `GEN_*` ops + reused `SWEEP_OP=64` live. Waveform is
NEVER read to route pitch (Facemorph guardrail). **NOT on PyPI** (0.44.0) — held for breaking **0.45.0** (memory
`tokens-0.45.0-release-pending`). Measurement plan: `design/generator_measurement_readiness.md`.

### Measurement state (the decisive go/no-go is the canonical training run)
- **Byte-exactness is DONE** (corpus dirty ~8%→0); the generator pipeline is byte-exact + residual-zero. The
  lever is **re-encoding**; training is gated behind a byte-exact encoding.
- **`learnability_triage`:** use **`--mode window`** (`--mode blocks` chokes on `GEN_*` → 0/90). Block-local
  result (30 tunes): the **generator+melody+universal_pitch family is NOT a within-tune learnability win** —
  copy ~0.92 (≤ atomic baseline 0.930), h∞/frame 4.13 (> baseline 3.84), `universal_pitch` ≈ no metric change
  (it only re-keys sparse melody onsets + the minority detuned voices). **Caveat: within-tune copy is the wrong
  metric** — it credits trivial redundancy the generator compresses, and can't see the *cross-tune transfer* the
  note/interval encoding is for (interval transfer 0.307 ≫ absolute 0.090, already shown). So the triage is a
  cheap proxy, not the verdict.
- **The verdict is the canonical-tier training run** (mini mode-collapses regardless, `loop_collapse_rate`~1.0).
  Generalise the `full_macros`-vs-atomic A/B to a canonical spec; gate on per-tier `content_over_structural` +
  per-op `op_acc` (all-tier val_acc is confounded across tokenizations). This is **release-gated** (needs the
  baked image on released tokens). Plan: `design/generator_measurement_readiness.md`.
- The earlier per-substrate codebook-swap blowup (PW/filter +16/+19/+6pp) is **gone by construction** — PW/cut/
  res/modevol are ordinary generator channels now. Model-side content interventions were refuted at the ~0.13
  ceiling that tokenizer-side `full_macros` then lifted — the lever is tokenizer-side representation.

### NEXT
1. **0.45.0 release prep (in progress this session):** the bundle (generator pipeline #62–#68 + instrument
   collapse #57/#58 + melody #69/#70, all on tokens `main`) is ready. Gate the publish on the **12-SID WAV
   audition** (non-negotiable; `audit/audition_cohort_render.py`, baseline vs full_macros), then the cross-repo
   sequence (memory `cross-repo-release-ordering`): tag `v0.45.0` → framework floor `>=0.45.0` (all 3 req files)
   → rebuild images → re-cut datasets → **canonical run** (the decider). 0.45.0 *enables* the decisive test; it
   is not a proven-win release.
2. **Pitch realignment** (`feat/universal-pitch-grid`, Current arc): `universal_pitch` is correct + validated but
   moves aggregate metrics ~0 because it only touches melody onsets + detuned voices. The real lever, if pursued,
   is applying the universal note-index/interval/cents realignment to the **bulk freq stream** (all freq atoms,
   not just onsets) — a larger change, decide after the canonical run. Then the octave+pitch-class re-keying
   probe (lossless bijective A/B for cross-tune transfer).

### Prior arc (compacted; details in `design/landed/` + git log)
**Architecture exonerated** — `framework_arch_test` (torchtune llama3_2, mini) gets val 0.903 on UNSEEN synthetic
motifs: the body generalizes, the SID failure is downstream/representational. Canonical content win confirmed
×3 seeds (`full_macros` eval_a 0.324 vs 0.219). Mini is plumbing-only (mode-collapses). Op-distribution read on
the released encoding (`audit_checkpoint_per_class` → `content_tier_report`) precedes the canonical run.

## Tests + runner

- **Framework** (`anarkiwi/preframr`): `./run_tests.sh` (black, pytest, pylint
  curated, pyright, coverage ≥77).
- **xpt** (`anarkiwi/preframr-xpt`): `pytest tests` at image build, gated by
  `.github/workflows/docker.yml` (push to main + every PR). Locally
  `docker build -f Dockerfile .` reproduces. Host CLI (host needs only
  `preframr_experiments` on `PYTHONPATH`, not torch):
  `PYTHONPATH=. python3 -m preframr_experiments.run <spec> --root <work> [--seeds N --tkvocab 8192 ...]`.
  One spec module per A/B under `preframr_experiments/specs/`; runner stages
  data → parse → tokenize → train per (arm, seed) in a `docker run` of
  `spec.image`. `nohup ... & disown` for long runs.
- **Macro passes = one list.** An arm declares its tokenizer passes via
  `Arm(macro_flags=(...), macro_config="full_macros"|"baseline")` — names from
  `preframr_tokens.tokenizer_config.MACRO_FLAGS` (the `macro_flag_names()` registry).
  The runner renders these to `--macro-flags`/`--macro-config`; `apply_macro_flags_to_args`
  validates each name, adds deps + rejects conflicts (`resolve_flags`), and sets the
  per-flag attrs. **Default = all passes OFF** (a bare/`baseline=True` arm is truly atomic);
  `full_macros` = `REGISTERED_MACROS`. There is no more `pipeline_spec` / `--foo-pass`
  surface (the old hand-maintained argparse flags + `_PIPELINE_NAME_TO_FLAG` map are gone).
- **Spec-dependent tokenization** (motif / cluster_content / voice_permutation /
  any `pre_run_hook` that mutates staged dumps or mines a per-spec artifact):
  launch with `PREFRAMR_DATASET_CACHE_DISABLE=1`.
- **Content-tier audit (decisive gate):** per arm-seed, run
  `audit_checkpoint_per_class --ckpt ... --work-dir ... --out audit_per_class.json`
  in the xpt image (emits `vocab_atom`). Then host-side, torch-free:
  `python3 -m preframr_experiments.audit.content_tier_report --results-root <dir>`
  (+ `--onset` for V0-onset bucket). All-tier val_acc is CONFOUNDED across
  tokenizations — the content-tier read settles representation A/Bs.
  Tested readers indexed in `preframr_experiments/audit/README.md`; use them,
  not bespoke `/scratch/tmp` scripts.
- Outputs under `/scratch/tmp/preframr_experiments/` (or `--root`). Status:
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
- **Arm ordering:** target arm first in `spec.arms`, baseline last. Runner is
  seed-major (`for seed: for arm: run`) — 1-seed cross-arm comparison
  available as soon as seed0 finishes both arms.
- **Renaming a transform** silently disables it in stale specs (no error) —
  grep specs on any pass/transform rename.
- **Design docs** live in `design/`, indexed by **research axis** in
  `design/README.md` with status as a per-row column. New doc: one-line
  `**Status:**` header + row under primary axis. On ship → `design/landed/`;
  on rejection → `data/refuted/<exp>.md` stub.

### Wallclock anchors

mini body=large ~12-20 min/arm · canonical 60-120 min/arm · prodlike ~6-11 hr
per (arm, seed). parse+tokenize ~25 min/prodlike uncached.

## Forward-looking work

### Land any time
- **FOLLOW-UP — move the staged tracker round-trip tests into `preframr-tokens/tests/`.**
  `staging/tokens_tests/` (in THIS repo) holds **written + verified** SWM/defMON forward
  round-trip tests (`module → register log → generator_pass → decode == player output`,
  equivalence via `parse_audit='raise'`). These are the §7B Tier-1 / §7.2 tests the generator
  work order specified but PRs #62–#68 shipped WITHOUT (the agent never wired pysidwizard/
  pydefmon). Verified green on tokens `main` @ 632f498: **11 passed (6 SWM + 5 defMON), 1
  skipped (non-`$1800` fixture), 1 xfail** (the unbuilt reverse `log → SWM → log` recompiler,
  `design/log_to_swm_recompiler_design.md`). To land: add pysidwizard+pydefmon as **test-only**
  deps, drop the `conftest.py` source-path shim, provision fixtures via
  `pysidwizard.tests._swm_cache` + pydefmon's bundled `build/fixtures` (no SID binaries in git),
  keep the reverse xfail as the recompiler's tracking test. See `staging/tokens_tests/README.md`.
- **Profile + optimize preframr-tokens parsing** — correct but slow; big share
  of uncached run setup. Keep the per-frame fidelity oracle green.
- **Recover control-write-rejected dumps** — characterize dumps rejected for
  too many control writes; relax/absorb to grow the corpus.
- **12-SID WAV audition cohort** — non-negotiable gate before flipping any
  tokenizer default + re-cutting training data.
- **Per-primitive round-trip audio gate** — corpus-scale CI run of
  `compare_renders` (~100 songs ≥95% within tolerance).

### Predict-host envelope (queued, post-generalization)
Lead with: **vocab shrink** (tkvocab ~8× to 4096 — ~91% dead) →
GPU-resident constrained-decode → full-context audition. Orin is ~4% GPU
util at predict. Re-open alongside the Multi-GPU rental decision (deferred
until a generalising approach lands).

### Framework follow-ups
- Streaming unembed-CE — recovers prodlike 2× wall.
- Generalization-gate thresholds — recalibrate per tier.
- Augmentation tooling + design in **`preframr-aug`**; melody-transfer
  Phase-0 smoke still pending.
- **Autocast fp32-promotion trap:** any new `Module` in
  `preframr/train/model/heads*.py` must cast log_softmax/logsumexp back to
  input dtype or per-position buffers stay fp32 and OOM at prodlike
  (pinned by `tests/train/test_per_tier_heads.py::test_bf16_input_preserves_*`).

### Content tier deliberately lossy
slope/preset/transpose are cent-binned (lossy by design; content-tier-OFF is
byte-perfect vs raw). Lossless rework deferred.

### Fixtures move-out pending
SID songs must NOT be tracked here. Build a helper that creates + caches
fixtures locally (untracked) from HVSC; use Goto80 (not Commando). A 116M
`engine_fp_palettes.json` was removed from main's working tree; live copies
are in `data/{canonical,mini}/`.

## Refuted alternatives

Registry: `preframr_experiments/data/refuted/<exp>.md`. Model-side
interventions concentrated at the same ~0.13 eval_a content ceiling (since
lifted by tokenizer-side `full_macros`):

- `per_tier_heads_mos_prodlike`, `per_tier_heads_entropy_prodlike` — router
  saturates / lambda non-monotonic.
- `mask_structural_loss` — diversity collapse; structural supervision
  load-bearing.
- `cluster_conditional_content_head`, `content_diffusion`,
  `contrastive_infonce_auxiliary` — same ceiling.
- `motif_pass` (v1 exact + v2 templated) — content-tier neutral-to-negative
  vs no-motif full_macros; no compression.
- Earlier nulls: `legato_ab`, `palette_merge`, `head_row_class`,
  `adsr_equivalence`, `macro_coarsening`, `b2_unblock`,
  `palette_pwm_prereqs`, `global_instr_ids_phase_a`, `weighted_token_loss`,
  `learnable_class_loss`, `voice_trajectory` (all variants), `set_to_diff`.

## Deferred deploy-stage efficiency (post-generalization)

~25% token-budget wins, both refute as generalization bets:
- **FRAME subsumes VOICE** — header token encodes voice order + write counts.
- **Drop VOICE_REG** — reg space already disambiguates voice.

## Resolved log (compact; full detail in git log + design/landed/ + data/refuted/)

- **2026-06-06** — generator pipeline (#62–#68) + melody layers 2+3 (#69/#70) + instrument collapse (#57/#58)
  all landed on tokens `main` (unreleased, 0.45.0 held). This session: validated the universal recovered-table
  pitch model exact on 4 trackers, wired `universal_pitch` (steps 1–3, default-OFF, byte-exact), took over the
  stopped melody agent, ran the `--mode window` triage (generator family ≤ atomic baseline within-tune; verdict
  deferred to the canonical run), started 0.45.0 release prep.
- **2026-06-04** — instrument-program collapse shipped (tokens #56/#57/#58); residual-SET drain → 0 on the
  sample; tokens 0.44.0 + framework 0.2.20 released. The ~10 note-associated passes collapsed into one
  define-on-first instrument codebook (residual==0 by construction).
- **2026-06-02** — byte-exactness DONE (corpus dirty ~8%→0, tokens 0.41.1); tokens 0.42.0 (PW/filter SweepPass,
  `op_name_by_id`); framework 0.2.17 unified macro-flag surface (`--macro-flags`/`--macro-config`, default all-OFF).
- **2026-05-21..28** — architecture exonerated (`framework_arch_test` val 0.903 on unseen motifs); substrate
  ablation lifted FREQ_TRAJ 2.4×; entropy/cluster/diffusion threads refuted at prodlike; libs split to PyPI.
- **earlier** — see git log + `data/refuted/`.
