# Operational notes for agents (preframr-xpt)

This repo is the **experiment surface**: docker-driven runner + spec registry
+ audits + design docs + tier data + refuted registry. Framework, libraries,
corpus live elsewhere.

## Packages

- **`preframr` 0.2.15** — framework only (train / inference / model / args /
  parse / stftokenize / utils + `mine_motifs.py`). Image `anarkiwi/preframr`.
  No PyPI; ships as the docker image. 0.2.15 carries the per-op-accuracy gate.
- **`preframr-tokens` 0.40.0** (PyPI) — torch-free parser/tokenizer + macros
  + `render_play`. **Byte-exact** on the lossless path; STAMP/PATCH/SWEEP/held-ARP/
  WAVETABLE codebooks; toggleable parse audit (`PREFRAMR_PARSE_AUDIT`).
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

Release process: authoritative in `design/architecture_overview.md`
("Release process"). `build.sh` sources gitignored `.env` (template
`.env.example`) for `PIP_OPTS` (the proxpi mirror) for local builds.

## Project goal (OVERRIDING)

Train a SID model that **generalises** — predicts unseen continuations from
arbitrary mid-song prompts, across composers (primary `val_acc`) and ideally
across engines (stretch) — inside:

- **Train:** single RTX 4090, 24 GB. Specs needing >~50M body to show Δ are
  out-of-envelope; refute in design, don't A/B.
- **Predict:** Jetson Orin NX (15.6 GB) at PROMPT=2048 / MAX=8192. KV cache
  at prodlike dims ~16 KiB/token → 128 MiB at MAX; bounded by seq_len.

## Current arc — byte-exact tokenizer SHIPPED; next = learnability audit (2026-06-02)

The lever is **re-encoding** (the skeleton+ornament tokenizer in preframr-tokens); training is **gated**
behind a deterministically-sound encoding (no training/audition until the pre-training tests are green).
**Shipped (preframr-tokens up to v0.40.0, macro passes gated default-OFF):** control-aware foundation,
claim/arbiter framework, STAMP (drum codebook), PATCH (instrument codebook), SWEEP, held-ARP, WAVETABLE
codebook. Universal-driver collapse verified (zero per-driver branching; provenance-invariance test #11.4).
**v0.40.0 makes the tokenizer byte-exact** on the lossless path (PerRegBurst DIFF bases on the decoder's
carried value; transposed loops only when lossless; STAMP barrier; lossless `_cap_delay`) plus a toggleable
parse-consistency audit (`PREFRAMR_PARSE_AUDIT=raise|warn`). The byte-exact / RESID→0 work orders are
retired (shipped); the residue is documented wavetable engines (note-relative offset cycles recur ~80–89%,
codebook-able), NOT an irreducible floor. Release flow: memory `cross-repo-release-ordering`.

### NEXT STEP — learnability audit on the rebake (the gate is no longer correctness)

Correctness is done (v0.40.0 byte-exact; constrained decode guarantees ref VALIDITY). The open question
is **learnability**: does each pattern-compressing token's PAYLOAD actually learn. Plan:

1. **Rebake** the corpus with preframr-tokens 0.40.0 — corrected tokens. NEVER measure on pre-0.40.0
   tokens (the prior "full_macros = only content win ×3 seeds" was on partly-wrong tokens).
2. **Train `full_macros` @ `--tkvocab=0` (Unigram OFF)** + an **atomic baseline**, with
   `--generalization-gate`. Keep Unigram OFF: (a) the rebake changed token freqs → stale vocab; (b) it
   obscures the per-op signal; (c) it would desync the constrained-decode mask, which operates on the
   atomic op-grammar — reintroduce only later, mask-aware, if sequence length actually hurts.
3. **Read the learnability readout** (preframr ≥0.2.15): per-tier `gate/content_over_structural` + the
   per-op `gate/op_acc/{op}` (build_op_map / GeneralizationGate, this session). Structural (which-op)
   should be high; watch the CONTENT payloads — DIFF delta vs BACK_REF distance vs STAMP_REF / WAVETABLE
   **codebook-id**. A high-cardinality / per-tune-unique payload (esp. codebook ids) is the unlearnable
   risk; verify id reuse/transferability — if low, the fix is encoding-side (lower-card / codebook-
   relative), NOT model-side. Don't mistake mask-guaranteed validity for learned selection.
4. **Go/no-go**: re-confirm full_macros vs atomic (×seeds, byte-exact rebake) on content_acc + tier/op
   accuracy — the decision on whether the compressing vocab is the right substrate.

### Byte-exact corpus verification (10% HVSC sweep — running DETACHED at handoff)

v0.40.0's byte-exact path is verified by a 10% HVSC audit sweep (every 10th dump, skeleton path,
`PREFRAMR_PARSE_AUDIT=raise`); script `preframr_experiments/audit/probes/hvsc_audit_sweep.py`. A run is
LIVE in a **detached container that survives a context-clear** — container `preframr_hvsc_sweep_v040`,
durable log **`/scratch/preframr/sweep_v040.log`**.

- **Fresh agent: `tail /scratch/preframr/sweep_v040.log`.** All-clear = a final `DONE sample=8705 ...
  DIRTY=0 ERR=0`. A `DIRTY [i/N] <tune>: <pass> ... reg R V->V'` line (streamed live) names the
  tune+pass+reg. The prior session was clean through ~55%, INCLUDING every tune the pre-fix sweep
  flagged (Aria/Day_Tripper/Sleigh_Ride/Pocket_Rockets/Uninvited/Mini_Melodies/1394).
- If the container is gone (`docker ps`) and the log lacks `DONE`, re-run: `docker run --rm --network host
  -v /scratch:/scratch -e PYTHONPATH=/scratch/anarkiwi/preframr-tokens:/scratch/anarkiwi/preframr-audio
  -e PREFRAMR_SID_FIXTURE_CACHE=/scratch/preframr/sid_fixture_cache anarkiwi/preframr-xpt:0.2.13
  python3 -u preframr_experiments/audit/probes/hvsc_audit_sweep.py 10 20`.
- A NEW divergence class is a tokens bug to ROOT-FIX (never a fallback — training must not see wrong tokens).

### Prior arc — substrate is the lever; V0 pitch (2026-05-28, superseded by the skeleton re-encoding)

**Decisive 2026-05-28: `melody_substrate_iter_mini` (×3 seeds).** PW+filter
substrate ablation lifts content acc 0.056 → 0.089 (Δ +0.033) and op45
(FREQ_TRAJ) 0.085 → 0.206 — **2.4× on the melody primitive**, seed-stable.
Absorber macros add **zero** on the clean substrate (`substrate_no_macros`
≡ `substrate_full_macros` at 0.089 on every content metric, slightly better
on the audition headline). V0 onset (the pitch atom) stays near zero in
all three arms — model learns trajectory STRUCTURE not melody PITCH. Full
write-up: `design/landed/substrate_ablation_v1.md`.

**Architecture exonerated.** `framework_arch_test` (torchtune llama3_2 at
mini body=large, 5.5M params) trained on a synthetic deterministic-motif
task gets train acc 1.000, val acc 0.903 on UNSEEN held-out motifs, gap
0.097. The core body can generalize; SID failure is downstream.

**Next experiment (high-confidence): `melody_substrate_prodlike`.** Take
`substrate_no_macros` to prodlike. Deployment config (`--tkvocab 8192
--batch-size 4 --accumulate-grad-batches 8`). Macros explicitly OFF (they
add nothing on substrate). Reserved budget ~30–66h on the 4090.

**Open: V0 absolute pitch.** Interval-V0 (`--freq-v0-interval`) was on in
all substrate arms — V0 onset still 0. Either V0 interval is bugged (e.g.
resets per-block), or pitch is genuinely scale-bound at mini and needs
prodlike. Two cheap probes before prodlike:
1. Grep `pipeline_spec.json` per arm-seed to confirm interval flag landed.
2. Run `audit.melody_predictability` on the ablated corpus. If V0 trigram
   ceiling rises from the noisy-corpus 0.79 baseline, scale is the issue;
   if flat, V0 encoding is.

**Automation landed.** `audit.melody_features` + `melody_baseline_corpus`
+ `melody_score_generation` + `melody_compare_arms` give an end-to-end
audio-side musicality score on any prediction dump (muspy-backed:
pitch_class_entropy, scale_consistency, pitch_in_scale_rate, plus
SID-specific gate density / note duration / interval distribution).
Framework patch: `predict.py --predict-dump <path>` captures the
prediction-window audio_df as parquet. Replaces per-WAV ear audits.

**Closed audit bugs (history, design-history pre-this-date is suspect):**
- `content_tier_report.id_to_op` row-index proxy — fixed via `vocab_atom`
  emission in `audit_checkpoint_per_class`. Per-op deltas pre-2026-05-28
  are ~58% mis-assigned; ignore them.
- `iter_self_contained_row_blocks` stripped freq passes — fixed by
  `freq_passes_re_fire_on_blocks` (commit `a71f676`).

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
