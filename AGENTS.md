# Operational notes for agents (preframr-xpt)

This repo is the **experiment surface**: docker-driven runner + spec registry
+ audits + design docs + tier data + refuted registry. Framework, libraries,
corpus live elsewhere.

## Packages

- **`preframr` 0.2.23** — framework only (train / inference / model / args /
  parse / stftokenize / utils). Image `anarkiwi/preframr`. No PyPI; ships as the
  docker image (`0.2.23` + `:latest` published on **`v*` tag via release.yml**, NOT
  main-push — main-push docker.yml is `push:false`/test-only). Carries the
  per-op-accuracy gate. Floors `preframr-tokens>=0.46.1` (only **one** req file —
  `requirements.txt` — floors tokens; the "all 3 req files" lore was wrong).
  `tier_map.build_op_map` reads
  op→name from tokens' `op_name_by_id()`. **Macro passes are supplied as ONE
  validated list** — `apply_macro_flags_to_args` resolves `--macro-flags` /
  `--macro-config` off the tokens registry (default all-OFF); the old per-flag
  `--foo-pass` args + `_PIPELINE_NAME_TO_FLAG` + `--pipeline-spec` are gone.
- **`preframr-tokens` 0.46.1** (PyPI) — torch-free parser/tokenizer + macros
  + `render_play`. **0.46.1 (universal pitch + table-resid-split):** merged the per-voice
  universal note-index pitch model (PR #72) + GEN_TABLE NOTE_UNIV residual-split as default-OFF
  macro flags (`universal_pitch` / `universal_freq` / `table_resid_split`); byte-exact; + a parse fix.
  These ARE the `pitch_resid_prodlike` target arm. **0.45.1 (generator-op loss tiering):** the generator/codebook ops were
  MacroPass-emitted (no `Transform` class) so absent from `collect_op_loss_tiers` → all
  defaulted to `content`, biasing `content_over_structural`. Fixed via
  `op_contracts.MACRO_OP_LOSS_TIERS` (GEN_TABLE_DEF/END/REF + GEN_TUNING → structural;
  SWEEP/GEN_TRI/MELODY_INTERVAL/STEP → content). **0.45.0 (generator-MDL default):** the
  generator pipeline (#62–#68) became the default, per-pass zoo deleted (`freq_trajectory_pass`/
  `preset_pass`/`ctrl_bigram_pass`/`wavetable_pass`/`skeleton_pass`/… GONE), `GEN_*` ops +
  reused `SWEEP_OP=64`, instrument-program collapse (#57/#58), melody layers 2 (`MELODY_INTERVAL`)
  + 3 (`voice_lane`/`role_lane`, #69/#70) **default-OFF**. **Byte-exact** (corpus dirty ~8%→0; 1 known load-dependent
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

Release, build, test, cache — one authoritative doc: **`design/references/release_build_cache.md`**
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

**Operational lens — LEARNABILITY** (`design/references/learnability_token_ordering_theory.md`, the
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

## Current arc — CANONICAL RUN LAUNCHED: `pitch_resid_prodlike` on defroster (2026-06-07)

### Run reliability (live) — 2026-06-07
**Goal right now is RELIABILITY, not results.** The generator abstraction mis-encodes repeated patterns
(user troubleshooting), so the current run's tokenized logs are WRONG and the experiment will be re-cut after
the next tokens (generator) fix — do NOT treat val_acc/eval_b as a deliverable. Three robustness bugs were
found + fixed + released this session: tokens **0.46.2** (note_freq/base_note overflow guard — universal_pitch
parse hang), tokens **0.46.3** (`DIFF_PDTYPE` UInt16→Int32 — macro-arm tokenize crashed on the `diff=-1`
voice-lane marker sentinel), framework **0.2.25** floors 0.46.3 → image `anarkiwi/preframr:0.2.25`.
**Relaunched** `pitch_resid_isolated_prodlike` (3-arm) on 0.2.25, runner pid 266042, root
`/scratch/tmp/preframr_experiments/pitch_resid_isolated_v2`, started 16:08. Reliability milestone to confirm:
all 3 arms clear parse→tokenize→train-start cleanly (esp. tokenize now succeeds for pitch_resid + full_macros).
Status: pitch_resid parse in progress. Await user signal for the generator-fix upgrade + restart.

**THE CANONICAL RUN is live.** `preframr_experiments/specs/pitch_resid_prodlike.py`: prodlike A/B, target =
`full_macros + universal_pitch + table_resid_split` (per-voice universal note-index for melody onsets AND
GEN_TABLE arps), baseline = the atomic control (all passes OFF), on `anarkiwi/preframr:0.2.23` (tokens 0.46.1,
1 seed, tkvocab 32768, seq_len 8192, 60 epochs). Decisive gate = `audit.content_tier_report` (per-tier
`content_over_structural` + per-op `op_acc`) on the corrected 0.46.1 tier map; secondary = `eval_b_*` held-out
composer families. Launch: `preframr-experiments-run pitch_resid_prodlike --root <root>` on the GPU host
(defroster, RTX 4090 — NOT fogbank). ~6–11h/(arm,seed) to a 1-seed cross-arm signal. It **replaces** the stale
`melody_skeleton_prodlike` (0.45.1 / no residual-split) and the older zoo-macro prodlike specs (NONE run under
0.46.1).

**Cross-repo propagation DONE.** tokens **0.46.1 on PyPI** (PR #72 universal-pitch merged + table_resid_split +
#73 generator-op tiering) → framework floor `>=0.46.1` → **`anarkiwi/preframr:0.2.23`** → spec image.

**NFS-hang recovery (2026-06-07).** A first launch died at 99.7% of parse when **fogbank (the `/scratch` NFS
server) flaked under concurrent load** — a CPU transfer-audit running *on* fogbank + defroster's parse writing
~34k files *to* it; defroster's `hard` mount turned the server stall into a D-state hang → reboot. Recovered:
cleaned the partial run (no checkpoint existed — died in parse), relaunched. See NFS hygiene below + memory
`nfs-hard-mount-hang`. The transfer-audit (`/scratch/tmp/transfer_audit_evalb.py`) was then parallelized
(multiprocessing Pool, default min(48, nproc) workers) — capped to leave cores for `nfsd`.

**Pitch model (`universal_pitch`/`universal_freq`/`table_resid_split`) IS in 0.46.1** (PR #72 merged). The universal
recovered-table pitch model (shared NOTE INDEX + per-voice recovered table + per-voice tuning + tuning-invariant
cents, transfer via intervals; `design/encoding/universal_multiresolution_pitch.md`) is the `pitch_resid_prodlike`
TARGET arm. Default-OFF flags: `universal_pitch` (re-key melody onsets), `universal_freq` (the bulk-freq probe:
re-key every sounding HOLD/ACCUM atom on melodic voices, 4.5× more interval atoms), `table_resid_split` (GEN_TABLE
NOTE_UNIV residual-split). Transfer-audit signal: melup vs mel lifted absolute transfer 0.094→0.157 (+67%) by
re-keying onsets alone; interval_transfer 0.40 sits below the 2-gram ceiling 0.53 (~0.13 headroom). The CPU mirror
of `eval_b_*` is `transfer_audit_evalb.py` (now parallelized). Validated EXACT on SWM/defMON/Hubbard/Galway
(recovered note-index == trackers' own FREQTBL/NOTE_PITCH).

### Why the within-tune triage is NOT the verdict
- `learnability_triage --mode window` (`--mode blocks` chokes on `GEN_*`): the generator+melody family copy ~0.92
  ≤ atomic baseline 0.930 — but **within-tune copy is the wrong metric**. It credits trivial redundancy the
  generator compresses and is blind to the **cross-tune transfer** the note/interval encoding is for
  (interval transfer 0.40 ≫ absolute 0.09). The verdict is the canonical run, gated on per-tier
  `content_over_structural` + per-op `op_acc` (all-tier val_acc is confounded across tokenizations).
- The earlier per-substrate codebook-swap blowup (PW/filter +16/+19/+6pp) is **gone by construction** — PW/cut/
  res/modevol are ordinary generator channels now. Model-side content interventions were refuted at the ~0.13
  ceiling that tokenizer-side `full_macros` then lifted — the lever is tokenizer-side representation.

### NEXT
1. **Canonical run is LAUNCHED** (`pitch_resid_prodlike` on defroster, recovered after the NFS-hang reboot). Read via
   `audit.content_tier_report` (per-tier `content_over_structural` + per-op `op_acc`); secondary = `eval_b_*`
   held-out composer generalization. ~6–11h/(arm,seed) to a 1-seed cross-arm signal.
2. **After the result:** if the pitch arm wins, decide whether to add a `+universal_freq` bulk-freq arm (re-keys the
   whole pitched-freq stream, not just sparse onsets — the deeper lever) and/or a 3-seed confirm; if it loses,
   reconcile against the within-block triage being flat (memory `within-block-triage-exhausted`).

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
- **NFS hygiene:** **fogbank IS the `/scratch` NFS server**; defroster mounts it
  `hard`, so heavy fogbank-local load (parallel audits / builds) overlapping a
  defroster parse/stage can saturate `nfsd` → defroster D-state hang → reboot
  (cost a 2026-06-07 reboot mid-parse, losing a 99.7%-done parse). Cap fogbank
  pools to leave cores for `nfsd`; canary defroster with `stat -f /scratch`. No
  lingering `tail -f` on workdir files (silly-renames); stop `preframr_tb`
  before deleting tb_logs subtrees.
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
  `design/encoding/log_to_swm_recompiler_design.md`). To land: add pysidwizard+pydefmon as **test-only**
  deps, drop the `conftest.py` source-path shim, provision fixtures via
  `pysidwizard.tests._swm_cache` + pydefmon's bundled `build/fixtures` (no SID binaries in git),
  keep the reverse xfail as the recompiler's tracking test. See `staging/tokens_tests/README.md`.
- **Profile + optimize preframr-tokens parsing** — correct but slow; big share
  of uncached run setup. Keep the per-frame fidelity oracle green.
- **Recover control-write-rejected dumps** — characterize dumps rejected for
  too many control writes; relax/absorb to grow the corpus.
- **Register-log equivalence gate** — non-negotiable before flipping any
  tokenizer default + re-cutting training data: the decoded register stream must match
  the source dumps (same regs/order/delay; control regs exact, FREQ/PW/filter within
  `freq_tol`) corpus-wide via `cb_div_audit.py`. Same registers/order/delay ⟹ same render
  by construction — no WAV audition needed (see `design/references/verification_and_audits.md`).
- **Corpus-scale register-equivalence CI** — run `cb_div_audit.py` over the corpus
  (within-`freq_tol`), not a WAV `compare_renders` pass.

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
