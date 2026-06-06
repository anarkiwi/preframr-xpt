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

## Current arc — the GENERATOR-MDL pipeline has LANDED on tokens `main` (unreleased); measurement is the active work (2026-06-06)

**LANDING (2026-06-06):** the generator pipeline is **merged on `preframr-tokens` `origin/main`** (PRs
#62/#64/#65/#66/#67/#68): `generator_pass` is the deployed default in `REGISTERED_MACROS`, the per-pass macro
zoo is **deleted**, ops `GEN_TRI=83`/`GEN_TUNING=84`/`GEN_TABLE_{DEF=85,STEP=86,END=87,REF=88}` + reused
`SWEEP_OP=64` are live. It is **NOT yet on PyPI** (latest 0.44.0) — held on `main` to bundle whole-chip-zero
into one breaking **0.45.0** (memory `tokens-0.45.0-release-pending`). **The measurement plan is now the active
work: `design/generator_measurement_readiness.md`** — what to run now (cheap static triage, runnable against
local `main` source) vs what is release-gated (the decisive canonical training run).

### NOW — one self-verifying generator model of every SID write (supersedes the per-pass macro zoo)
The whole pitch/ornament/residual-SET/whole-chip-zero line of work has **converged** onto one design: model
every value-channel's per-frame series as a sequence of generators `{HOLD, ACCUM, SWEEP, TABLE}` reused via a
block-local DEF→REF bank, with pitch over a **unified per-tune semitone LUT** (no cents quantization). It is
**lossless + residual-zero by construction** (a self-verifying longest-wins fitter ⇒ every claim is byte-exact;
arbiter `validate=True` never drops) and **provenance-invariant**. Prototyped + scale-validated: **byte-exact +
0 unexplained on 1580 corpus tunes, every historically-hard engine (Baggis/JCH, SoundMonitor, System6581,
Commando, Camerock), and SID-Wizard (91 modules) + defMON (9) rendered through their own players.** Key
simplification: two of the three ops already exist — `SWEEP_OP`=ACCUM, the GlobalOsc cycle=TABLE — so only the
triangle SWEEP + tuning/codebook are new; the generator pass UNIFIES `SweepPass`+`GlobalOscPass` over all
channels. Waveform is NEVER read to route pitch (the Facemorph guardrail: noise accents pitched notes, pulse
plays percussion).

- **Design (canonical):** `design/generator_mdl_representation.md`. The former per-pass pitch/ornament/melody
  stack (unified-pitch, ornament-transfer, sweep-oscillation, melody-channel/skeleton/gap-ladder, macro-zoo
  triage, residual-SET workorders, voice/role-lanes) was **removed from `design/` 2026-06-05** — generator-MDL
  subsumes it; do not resurrect those approaches.
- **Implementation — handed to a preframr-tokens agent:** `preframr-tokens/AGENT_TASK_generator_pipeline.md`
  (self-contained, fully unambiguous: embedded fitter, exact 13 channels, exact op-ids 83+, the codebook, the
  digi check, the module↔macros round-trip tests, through to a pushed PR). **xpt expects that agent's output:**
  a new default tokenizer where `generator_pass` replaces `freq_trajectory`/`skeleton`/`sweep`/`gradient`/
  `global_osc`/`preset`/`stamp`/`wavetable`/`per_reg_burst`/`note_off`/`init` (deleted; op-ids freed as holes),
  `SWEEP_OP=64` reused (new producer), new ops `GEN_TRI=83`/`GEN_TUNING=84`/`GEN_TABLE_DEF..REF=85-88`,
  `GLOBAL_OSC_OP=82` retired; `InstrumentProgramPass` KEPT for ctrl/AD/SR; `loop`/`hard_restart`/`legato`/
  `voice_block` KEPT. **This LANDED on tokens `main` 2026-06-06** (`generator_pass` is the deployed default;
  the zoo is deleted). **xpt work now:** the cheap static measurements
  (`design/generator_measurement_readiness.md` §1–§3) run against local `main` source today; the release-gated
  chain (re-floor `preframr-tokens>=0.45.0` all req files → rebuild xpt image → re-cut datasets → canonical
  learnability A/B) waits on the cross-repo 0.45.0 release + the 12-SID WAV audition gate (memory
  `cross-repo-release-ordering`, `tokens-0.45.0-release-pending`).

### Background arc — byte-exact + PW/filter sweep + unified macro-flags (2026-06-02)

The lever is **re-encoding**; training is **gated** behind a byte-exact encoding. **Byte-exactness is DONE
and released — preframr-tokens v0.41.1, image 0.2.16 — corpus dirty rate ~8% → 0.** Root-fixes this arc (all
register-exact, never a fallback): arbiter `validate=True` (a register-exact pass drops any claim that
changes the decoded `register_state` — fixes the ctrl/patch tick-drain clobber); a **digi/multispeed pre-load
gate** (`RegLogParser._admit_dump` rejects 10M+-row dumps before `_read_df`); and the **lead-frame render**
(`FrameWalker._walk_lead` — content before the first FRAME marker is the tune's note-on, was silently
dropped; the `register_state` oracle now holds it at `snaps[0]` **replace-not-append** so frame indices stay
aligned with `frame_reg`). Encoder dedup also landed (`emit_recurring` / `run_collapse` / `make_row` /
`_SetEquivalentDecoder`, all byte-neutral). Verify byte-exactness with `/scratch/preframr/cb_div_audit.py`
(corpus `parse_audit='raise'`, currently dirty=0). Cross-repo gotcha: removing a tokens `*_OP` breaks
preframr's train tests AND the Docker build's `run_tests.sh` (memory `cross-repo-release-ordering`).

### Learnability read (mini, on byte-exact v0.41.1) — INCONCLUSIVE, scale-bound

`learnability_full_macros_mini` (full_macros vs atomic baseline, gen-gate, 3 seeds) on byte-exact tokens:
**all 6 runs mode-collapsed** (`loop_collapse_rate` ~1.0, gate abort epoch 15) — same as the pre-byte-exact
result, so the collapse is a **scale** problem, not a tokenization one. full_macros is directionally better
(val_acc 0.067 vs 0.052, alphabet −7%, tokens −2.4%) but inside the collapsed regime → not trustworthy. **The
actual go/no-go needs the canonical/prodlike tier** (where loop_collapse_rate drops). Mini distribution:
FREQ_TRAJ 49%, loop/reuse PATTERN_* 19%, SET 13% (86% of which is frame markers), ctrl-collapse 2.8%;
**codebooks ~0% because stamp/patch/wavetable are NOT in REGISTERED_MACROS** (experimental, default-OFF).

### (Resolved by the generator pipeline) the old codebook SWAP blowup
The earlier codebook swap (skeleton/wavetable, conflicting with `freq_trajectory_pass`) dropped FREQ_TRAJ for
SKEL+ORN+STAMP/WAVETABLE but **PW/filter reverted to SET/PWM_PRESET/FC_PRESET (+16/+19/+6pp)**. The
generator-MDL pipeline removes this whole tension: PW/cutoff/res/modevol are ordinary generator channels
(HOLD/ACCUM/TRI/TABLE), so there is no per-substrate blowup and no skeleton↔freq_trajectory conflict — all of
those passes are deleted.

### SHIPPED + RELEASED — tokens 0.42.0, framework 0.2.17, unified macro-flags (2026-06-02)
- **tokens 0.42.0** (PyPI): `SweepPass` mines PW (regs 2/9/16, `pw_sweep`) + filter cutoff (reg 21,
  `filter_sweep`), default-OFF sub-flags under `sweep_pass`, `note_aligned=False`, register-exact (one
  `Claim`/run, `validate=True`) — a constant-delta ramp that was one `PWM_PRESET`/`FC_PRESET`/`SET` per frame
  now collapses to one `SWEEP`. Plus `op_name_by_id()`/`op_name_tiers()`.
- **framework 0.2.17** (image `anarkiwi/preframr:0.2.17` + `:latest`, published cuda/predict/xpu/jetson via
  `release.yml`; `docker-test` + `docker-release` both green): `tier_map.build_op_map` reads tokens
  `op_name_by_id` (local dir-scan deleted). **Unified macro-flag surface (breaking):** the three
  hand-maintained surfaces (per-flag `--foo-pass` args, `_PIPELINE_NAME_TO_FLAG`, `--pipeline-spec`) collapsed
  to one — `apply_macro_flags_to_args` resolves `--macro-flags`/`--macro-config` via the tokens registry
  (validate → `resolve_flags` deps/conflicts → per-flag attrs; default all-OFF, so a bare/`baseline=True` arm
  is truly atomic; `full_macros` = `REGISTERED_MACROS`). predict recovers `args.macro_flags` from the ckpt.
  All req files floored `>=0.42.0` (jetson's was missed first → release failed → fixed). No tokens release
  needed (reused 0.42.0 primitives).
- **xpt:** all 31 specs migrated to `Arm(macro_flags=..., macro_config=...)`; runner renders the CLI; dataset
  cache key hashes them. Merged to main (origin's dead-wood PR #5 removed the refuted motif specs — folded in).
  Image `anarkiwi/preframr-xpt:0.2.17` baked on the 0.2.17 base.
- Codebook+PW/filter pipeline verified reachable + runs end-to-end (`PARSE_AUDIT=raise` on the truncated
  `sid_fixture_cache/*_20s` dumps trips, but so does `full_macros` — a fixture property; the real byte-exact
  gate is the corpus sweep `cb_div_audit.py`).

### NEXT — generator pipeline LANDED on tokens `main`; measurement plan is `design/generator_measurement_readiness.md`
The generator pipeline **landed on tokens `origin/main` 2026-06-06** (unreleased; 0.45.0 held). **The active
experiment plan is `design/generator_measurement_readiness.md`** — §1 the cheap static learnability triage
(`learnability_triage --configs baseline,full_macros --mode blocks --seq-len 8192`, runnable now vs local
`main` source; generator induction-copy vs the historical 0.718 = the queue-or-not go/no-go), §3 the
residual-in-key refragmentation check (runnable now), and §4 the **release-gated** canonical learnability A/B
(generator vs atomic; needs 0.45.0 → image rebuild → re-cut). The two summarized points below (op-distribution
read; canonical go/no-go) are detailed there.

**The melody layer is the NEXT tokens work order (now unblocked): SELF-DIRECTING
`design/melody_skeleton_impl.md`** — tell its agent only "execute this .md"; its §A start-gate polled tokens
`origin/main` for `generator_pass` deployed-default + zoo-deleted (**now satisfied**), then
executes the melody-learnability layers in preframr-tokens with no further help/decisions: **layer 2** (note
segmentation + interval-from-previous onset encoding) AND **layer 3** (`voice_lane` de-mux + **causal-DAG lane
ordering: accompaniment roles before melody** so melody is predicted with its harmony in-context — the DOMINANT
lever; role identification is the mechanism, plain physical lanes can backfire; deployed melody-onset ≈ 0 vs
~0.34 ceiling; triage (lane-order variants + no other-content regression) + canonical-run gated; designs
`superframe_voice_lane_design.md` / `role_lane_factorization.md`, reinstated). **Layer 4** (rhythmic/harmonic
determinants + scale-degree anchoring) is a named deferred hypothesis if layer 3 plateaus. It stays out of tokens until that gate passes, so the in-flight
generator agent is never confused. Once the generator pipeline is the default + released (0.45.0) and the xpt
image is rebuilt on it, the experiment program runs:
1. **Op-distribution read on the new encoding** — `audit_checkpoint_per_class` → `content_tier_report`:
   confirm the stream is generator atoms (`SWEEP_OP`/`GEN_TRI`/`GEN_TABLE` DEF→REF + the kept loop/instrument
   ops) with raw `SET` ~0, and that PW/filter are SWEEP/TABLE (the old +16/+19/+6pp `PWM_PRESET`/`FC_PRESET`
   blowup is gone by construction). Read distribution, not val_acc (mini mode-collapses regardless).
2. **Canonical-tier learnability go/no-go** — the real test. Mini collapses regardless of vocab
   (`loop_collapse_rate` ~1.0); only canonical/prodlike settles whether the generator vocab's PAYLOAD learns.
   Generalise the `full_macros`-vs-atomic A/B to a canonical spec on the new default; gate on per-tier
   `content_over_structural` + per-op `op_acc`. The learnability prediction: provenance-invariant DEF→REF
   generators + a transposition-invariant pitch LUT are induction-head-friendly (see
   `design/learnability_token_ordering_theory.md`).

### Prior arc (compacted; details in `design/landed/` + git log)
Substrate ablation (2026-05-28, `melody_substrate_iter_mini`) lifted FREQ_TRAJ 0.085→0.206 (2.4×); absorber
macros add zero on the clean substrate; V0 onset stays ~0 (model learns trajectory STRUCTURE, not pitch).
**Architecture exonerated** — `framework_arch_test` (torchtune llama3_2, mini) gets val 0.903 on UNSEEN
synthetic motifs: the body generalizes, the SID failure is downstream. Write-up
`design/landed/substrate_ablation_v1.md`. (Per-op deltas before 2026-05-28 are ~58% mis-assigned —
`content_tier_report.id_to_op` bug, since fixed; ignore them.)

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

## Resolved log (compact; details in git log + design/landed/ + data/refuted/)

- **2026-06-04 (SHIPPED — instrument collapse done tokens #56/#57/#58; whole-chip-zero work order handed off; live state is the Current-arc NOW block)** — **residual→0 pivoted from
  point-fixes to the instrument-program COLLAPSE.** After driving the corpus residual census to ~97.6%
  and chasing the tail, classified the remaining residual: ~half is pre-/never-gated FREQ (pitch
  channel), ~half post-gate ctrl/AD/SR singletons. Root realization (user-led, "too much complexity vs
  the drivers; some sequences are note-associated, some not"): the ~10 note-associated passes
  (`stamp`/`patch`/`preset`/`ctrl_wavetable`+nibble/`onset_def`/`ctrl_osc`/`ctrl_triple`/`ctrl_bigram` +
  `hard_restart`/`note_off` markers) are all **fragments of ONE driver concept** — the per-frame
  *instrument program* (waveform/AD/SR walk) a note-onset fires, a small bank reused by id. The residual
  tail IS the gaps between each pass's escape condition (`MINREP≥2`, `fr_reg_count==1`, onset floor,
  osc-period, nibble-lane id). **Collapse them into one define-on-first codebook → residual==0
  structurally.**
  - **Empirically validated** (861,098 spans, `register_state`, `/scratch/tmp/empirical_checks.py`):
    AD/SR constant within a gate-held span **97.0/96.3%** (onset-anchored span ✓), waveform-walk mean
    **1.91** frames (short program ✓), program `(ctrl-walk,AD,SR)` **exact-recurrence 98.0%** within a
    tune (small reused bank ✓ → exact-REF + define-on-first ⇒ residual==0 by construction).
  - **Design doc:** `design/instrument_program_codebook_design.md` (supersedes
    `instrument_state_codebook_design.md`; the 3 contracts — span=gate/HR boundary, program↔sweep
    set-vs-delta, exact REF — are DECIDED + VERIFIED). Scope: this collapse is the **timbre** channel
    (ctrl/AD/SR); pitch stays with the ornament stack, PW/filter with the sweep channel.
  - **Executable impl doc handed to the other agent:** `preframr-tokens/design/instrument_program_pass_impl.md`
    — self-contained inside preframr-tokens (StampPass is the template; new ops 78–81; new `"instrument"`
    CodebookFamily + codec; new `InstrumentProgramPass` run **inline on actual voice regs** = the
    voice-confusion guardrail; flag `instrument_program` default OFF; in-repo residual gate + xdist tests).
  - **Interim point-fixes this session:** `ctrl_wt` lane-keying id-collision fix (committed, tokens
    branch `resid/ctrl-wt-lane-keying`, NOT released); never-gated-voice FREQ drop in `pre_gate_freq`
    (was REVERTED — pitch channel, the collapse/ornament handles it; do not re-add).
  - **PICK UP AFTER THE AGENT:** (1) verify their gate — `instrument_program=True` ⇒ ctrl/AD/SR residual
    **0** corpus-wide `reparse=True` digi-excluded (their §6.1 script) + byte-exact `register_state` +
    xdist green; (2) run a reject-claim audit to confirm no new divergences; (3) **12-SID WAV
    audio-equivalence audition** before flipping any default; (4) only then flip `instrument_program` into
    `REGISTERED_MACROS` and ship tokens **0.45.0** cross-repo (per `design/release_build_cache.md`);
    (5) the DELETION release (remove the ~10 subsumed passes/ops/decoders) comes LAST, once the unified
    path is default + green. Standing gate: **ZERO is non-negotiable; always `reparse=True`; validate on
    the corpus not a sample; progress markers in every sweep.**
- **2026-06-04** — **residual-SET drain COMPLETE on the sample; tokens 0.44.0 shipped (PyPI).** Raw
  `SET`s on the digi-excluded stride sample driven 444 → 0 across five mechanisms (GRADIENT + INIT
  prior; then `onset_def` 215→20, `env_multiload` 20→11, `pre_gate_freq` 11→6, `nibble_wavetable`
  6→0). `onset_def`/`env_multiload`/`nibble_wavetable` are register-state-exact by arbiter
  construction (reuse the CTRL_WT codebook + HARD_RESTART_OP — no new ops/families); `pre_gate_freq`
  is the **first AUDIO-exact (not register-state-exact) drain atom** — a freq before a voice's first
  gate-on is inaudible (proven in preframr-audio `test_freq_write_audibility`), DROP it if the first
  note sets its own freq else RELOCATE into the gate frame; it sits in `parse_audit._LOSSY_RESETS`,
  audible region preserved. Byte-exact verified `parse_audit=raise` (cb config, no preset) 56/57 clean
  (1 filtered). All default-OFF, OUT of `REGISTERED_MACROS`. Merged tokens PRs #54 (drain) + #55 (md
  cleanup, README-only); released **v0.44.0** (tag → `release.yml` OIDC → PyPI, run green, live).
  **Cross-repo release DONE:** framework **0.2.20** floors `preframr-tokens>=0.44.0` (all 3 req files;
  `run_tests.sh` green, images published) + xpt image **0.2.20** rebuilt on it (fogbank, 169 tests).
  **Full-corpus census (step 10, 8705 tunes, reparse=True): 8186 clean / 199 dirty / 400 residual SETs
  / 320 digis = 97.6% of non-digi tunes fully clean.** CAUGHT A CENSUS-TOOL BUG: `residual_set_census`
  omitted `reparse=True`, so it read STALE pre-drain tokenization caches (falsely reporting ~715k
  residual / ~33% dirty); fixed (xpt `9ed9cd2`) — the drain itself is corpus-effective. The remaining
  400 is the real tail (NOT zero): recurring CTRL gate/waveform bytes `(4,-1,{65,33,129,17})` that
  escaped `ctrl_wavetable`/`onset_def`, FREQ words `(0,-1,*)` (startup/non-recurring), a few AD/SR +
  `(24,-1,31)`. That's the genuine next-drain work-queue. ALWAYS pass `reparse=True` for residual/
  byte-exact corpus measurements (parse() returns the stale cache otherwise).
- **2026-06-02** — tokens **0.42.0** shipped (PW/filter sweep mining + `op_name_by_id()`/`op_name_tiers()`
  API; PyPI). Framework owner-cleanup landed on `feat/per-op-accuracy`: `tier_map.build_op_map` swapped to
  tokens `op_name_by_id` (local dir-scan deleted), `requirements.txt` floored `>=0.42.0`; tier_map/onset/
  learnable-class-loss tests green in the `0.2.16` image, black clean. Same day earlier: byte-exact tokenizer
  COMPLETE at 0.41.1 (corpus dirty ~8%→0); mini learnability INCONCLUSIVE (scale-bound, all 6 runs collapsed).
  **Macro-flag surface unified (breaking)**: the three hand-maintained surfaces (22 argparse `--foo-pass` flags
  + `_PIPELINE_NAME_TO_FLAG` map + `--pipeline-spec` JSON; 11 flags incl `skeleton_pass`/`pw_sweep`/`filter_sweep`
  reachable by none) collapsed to ONE — `preframr.args.apply_macro_flags_to_args` resolving `--macro-flags`/
  `--macro-config` off the tokens registry (default all-OFF; deps/conflicts via `resolve_flags`). Specs now use
  `Arm(macro_flags=..., macro_config=...)`; predict recovers `args.macro_flags` from the ckpt. All 31 specs
  migrated (60 arms resolve clean), framework 239 + xpt 164 tests green. **Merged + released**: framework main
  `bf07d9e`, image **0.2.17** published (cuda/predict/xpu/jetson; `docker-release` + `docker-test` green —
  first release attempt failed on jetson because `predict-requirements.txt` + `jetson/predict-requirements.txt`
  still floored tokens 0.35.0, fixed to 0.42.0). xpt main `343a741`, image 0.2.17 baked (origin PR #5 removed
  the refuted motif specs — folded into the merge). No tokens release (reused 0.42.0 primitives). Codebook+PW/
  filter pipeline now reachable from a spec → unblocks the `codebook_distribution_mini` experiment (NEXT #1).
- **2026-05-28** — `melody_substrate_iter_mini` PASSED ×3 seeds:
  substrate ablation lifts op45 0.085 → 0.206; macros add zero on the
  clean substrate. `framework_arch_test` PASSED — torchtune llama3_2
  generalizes at mini scale on synthetic (train 1.000, val 0.903 on
  UNSEEN motifs). Melody-features automation landed
  (`melody_features`/`melody_baseline_corpus`/`melody_score_generation`/
  `melody_compare_arms` + muspy in the xpt image + framework
  `--predict-dump` flag). Full write-up:
  `design/landed/substrate_ablation_v1.md`. Earlier same day:
  `content_tier_report` uid→op fix landed (`vocab_atom` sidecar; all
  pre-fix per-op deltas unreliable). Next: scale
  `substrate_no_macros` to prodlike.
- **2026-05-27** — full_macros content win CONFIRMED ×3 seeds (content-tier
  eval_a 0.324 ± 0.006 vs 0.219 ± 0.011, Δ +0.105). Melody-stack landed
  (anchoring + interval V0 + FREQ_ONSET + onset-loss-weight). Motif pass
  REFUTED (v1 + v2). See `design/melody_learnability.md`.
- **2026-05-26** — re-arc STAGE 1 (mini) concluded: no model-side or data-side
  content signal on the corrected tokenizer; STAGE 2 (full_macros_prodlike
  ×3-seed) launched. Augmentation moved to `preframr-aug`.
- **2026-05-25** — Lean-core + 0.1.0 release. `integration_tests/` moved to
  xpt's `audit/` + `tests/` + data tree; main is framework-only.
  `full_macros_prodlike` PASSED single-seed (content-confirmed).
- **2026-05-24** — strict-no-diff tokenizer rework shipped (FREQ_TRAJ unified
  op + absorbers, tokens 0.16.0/0.17.0). Motivating A/B: full macro set lifted
  eval_a content 0.150→0.274 (~1.83×) — proved the ceiling was
  tokenization-induced.
- **2026-05-21..23** — entropy/cluster/diffusion threads refuted at prodlike;
  `preframr-experiments` extracted to this repo; libraries split to PyPI;
  preframr restructured (train/inference/model split, Corpus extraction).
- **earlier** — see git log + `data/refuted/`.
