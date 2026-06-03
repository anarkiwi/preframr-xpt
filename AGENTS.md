# Operational notes for agents (preframr-xpt)

This repo is the **experiment surface**: docker-driven runner + spec registry
+ audits + design docs + tier data + refuted registry. Framework, libraries,
corpus live elsewhere.

## Packages

- **`preframr` 0.2.19** — framework only (train / inference / model / args /
  parse / stftokenize / utils). Image `anarkiwi/preframr`. No PyPI; ships as the
  docker image (`0.2.19` + `:latest` published on main-push). Carries the
  per-op-accuracy gate. Floors `preframr-tokens>=0.43.0` (in **all** req files:
  `requirements.txt`, `predict-requirements.txt`, `jetson/predict-requirements.txt`
  — `op_name_by_id` is on the predict import path). `tier_map.build_op_map` reads
  op→name from tokens' `op_name_by_id()`. **Macro passes are supplied as ONE
  validated list** — `apply_macro_flags_to_args` resolves `--macro-flags` /
  `--macro-config` off the tokens registry (default all-OFF); the old per-flag
  `--foo-pass` args + `_PIPELINE_NAME_TO_FLAG` + `--pipeline-spec` are gone.
- **`preframr-tokens` 0.43.0** (PyPI) — torch-free parser/tokenizer + macros
  + `render_play`. **Byte-exact** (corpus dirty ~8%→0; 1 known load-dependent
  outlier `Ascension.1`, debug deferred); STAMP/PATCH/SWEEP/held-ARP/WAVETABLE
  codebooks; toggleable parse audit (`PREFRAMR_PARSE_AUDIT`). **0.43.0:** codebook
  ids are pure define→ref ordinals never value-snapped (`is_codebook_id_atom` guard
  + alphabet id-range coverage; STAMP_DEF char-drop) + a `register_state` decode memo
  (~6.6% parse). 0.42 added PW/filter sweep mining + `op_name_by_id()`/`op_name_tiers()`.
  0.42.1 fixed the `per_reg_burst` empty-cand+barrier crash (unblocked the codebook pipeline on the
  real corpus) + a `FrameWalker` parse speedup (~7-12%, byte-exact).
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

## Current arc — byte-exact + PW/filter sweep + unified macro-flags ALL SHIPPED; codebook distribution read is the next experiment (2026-06-02)

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

### The codebook pipeline is a SWAP, not an add-on
`skeleton_pass` (WavetablePass needs it) **conflicts with `freq_trajectory_pass`** — alternative freq
substrates. Swapping to the codebook pipeline drops FREQ_TRAJ (49%→0) for SKEL+ORN+STAMP/SWEEP/WAVETABLE, and
PATCH takes recurring envelopes from RELEASE_UPDATE — **but PW/filter lose trajectory compression and revert
to SET/PWM_PRESET/FC_PRESET (+16/+19/+6pp)**. That blowup motivates the handoff below.

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

### NEXT — proposed immediate experiment, then the strategic go/no-go
1. **`codebook_distribution_mini` (write + run this next)** — the payoff read for the whole 0.42 + plumbing
   arc, now finally launchable. Mini, `--tkvocab 0`, 2 arms:
   - `codebook`: `macro_flags=(<base> + "skeleton_pass","held_arp","zero_plain","slide_wide","slide_landing",
     "stamp_pass","sweep_pass","sweep_loop","pw_sweep","filter_sweep","wavetable_pass","wt_short","wt_oneshot","patch_pass")`
     (NOT `freq_trajectory_pass` — `resolve_flags` rejects skeleton+freq_trajectory). `<base>` = preset/
     hard_restart/legato c2/c4/voice_block/ctrl_bigram/loop/loop_transposed.
   - `full_macros` baseline: `macro_config="full_macros"`.
   **Read = the op DISTRIBUTION, not val_acc** (mini training mode-collapses regardless — established). Run
   `audit_checkpoint_per_class` → `content_tier_report`; confirm (a) the PW/filter SET/PWM_PRESET/FC_PRESET
   blowup (+16/+19/+6pp) becomes SWEEP, and (b) STAMP/WAVETABLE codebooks now register (were ~0% because they
   weren't in REGISTERED_MACROS). This proves the encoding payoff before spending the canonical budget.
   (NOTE: the `codebook_coupling.py` triage tool + `macro_learnability_triage.md` were removed by PR #5 —
   read the distribution from `content_tier_report` directly.)
2. **Canonical-tier learnability run** — the real go/no-go. Mini collapses regardless of vocab
   (`loop_collapse_rate` ~1.0); only canonical/prodlike (where collapse drops) settles whether the compressing
   vocab's PAYLOAD learns. `learnability_full_macros_mini` (now `macro_config="full_macros"` vs atomic
   baseline) generalises to a canonical spec; gate on per-tier `content_over_structural` + per-op `op_acc`.

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
