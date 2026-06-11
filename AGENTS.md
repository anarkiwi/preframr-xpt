# Operational notes for agents (preframr-xpt)

This repo is the **experiment surface**: docker-driven runner + spec registry
+ audits + design docs + tier data + refuted registry. Framework, libraries,
corpus live elsewhere.

> **STAGED FOR THE EVENT-MODEL TOKENS RELEASE (>=0.47.0 assumed) — branch `mdl-transition-staging`.**
> The tokens encoding is being rewritten wholesale to an **event / tracker token model**
> (design: `gen2-preframr-tokens/REDESIGN_optionB.md`, which **supersedes** the earlier MDL
> gesture-codebook on `feat/mdl-optimal-parser`). The model reads/writes a typed **event stream**
> (`preframr_tokens/events/`: `NOTE_ON`/`NOTE_STEP`/`MOD_FREQ`/`MOD_PW`/`MOD_CTRL`/`MOD_CUTOFF`/
> `FILTER_CTL`/`MOD_VOL` + `TICK`/`TUNING`/`NOTE_TABLE` headers), **not** register-write rows.
> Key properties: **no dictionary ids / DEF / REF / codebook at all** — every field is a
> *complete value* over a small fixed alphabet and **BPE/Unigram over the event stream IS the
> corpus-global dictionary** (escape-free; no per-tune singletons); **no note-off** (derived at
> `onset + duration`); **one mixed-radix `frame→tick` time encoding** for every time quantity;
> the **fidelity oracle is the exact ORDERED register-write stream** (intra-frame write order
> matters), NOT settled `register_state`; `constrained_decode` → an **event-grammar mask**.
> RETIRED by Option B: `MdlGesturePass`, `_GestureCodec`, all `GESTURE_*` ops, `codebook_emit`,
> the per-tune bank, the arbiter/Claim flow. KEPT as internal encoder primitives only: `mdl_core`
> (HOLD/POLY/PERIOD), `pitch_grid` (note-table recovery), `audit_primitives` (secondary check).
> **Status: Option B is IN FLIGHT, substantially BUILT, and the PRODUCTION SWAP IS DONE** (`preframr_tokens/
> events/`, see its `STATUS.md`): the events codec **is the tokenization pipeline now** — it replaced the
> `parse → (op,reg,subreg,val) → merge_token_df` substrate, and the gesture/codebook machinery is retired
> (`CODEBOOK_FAMILIES = {}`, GESTURE ops de-registered). v1 factored codec is byte-exact on 5 drivers +
> 200-tune corpus + 300/300 corpus tunes; invariants pass; note/attack + mixed-radix tick + derived gate-off
> + combined PW lane landed; **3.77 bits/write (4.24×)** on 5 drivers. The tokens suite is **676 passed / 3
> failed (all 3 pre-existing, zero events-swap regressions)**. NOT yet done: §7.1 generation grammar-mask,
> §3.1 semantic `MOD_*`/`GLOBAL` typing, §4.1 span inheritance, §9 corpus-scale bits, cosmetic dead-code
> cleanup, and — crucially — the **downstream model train/generate end-to-end run on the event tokens (NOT
> done)**. Two avenues were tried + **rejected on measurement**: §8.4 joint freq/note DP (4.23→4.42 bits) and
> §2.7 mixed-radix ORDER-`DT` (gap entropy≈0). **INTEGRATION (this branch, 2026-06-11):** xpt `base.py`
> tokens source repointed to `/scratch/anarkiwi/gen2-preframr-tokens` (the event model; canonical clone is
> stale) — all framework imports + the data/blocks contract resolve against it. **Event-aware tier split
> LANDED tokens-side** (`events/dataset.events_alphabet`): value-digit atoms → `content`, structural-marker
> atoms → `structural` (via a structural-tier op, NOT `FRAME_REG` — that would re-trigger BPE splitters),
> frame-weights unchanged, byte-exact roundtrip intact. This flows **registry-driven** through `tier_map` →
> `audit_checkpoint_per_class` → `content_tier_report`, so the decisive `content_over_structural` gate works
> on event tokens with **no framework/xpt code change**. Remaining: the train/generate end-to-end run (needs
> the image; host has no torch) + rebuild the image so the dataset-cache version bump fires (see Conventions).
> **This branch is staged, not merged**: until tokens ships, live state is the 0.46.x generator
> line in git history. The dead-encoding flags (`generator_pass`, `instrument_program`,
> `melody_skeleton`, `universal_pitch`, `universal_freq`, `table_resid_split`, `freq_trajectory_pass`,
> `trajectory_anchor_pass`, `freq_onset_pass`, `freq_v0_interval`) are GONE. ⚠️ Option B's v1 is
> **BPE-only** (loop/pattern/voice-block/coarsen structural passes are NOT ported), so the SURVIVING
> macro-flag set is **uncertain and possibly empty** — the staged framework tests are therefore
> registry-driven (pick/skip on whatever `macro_flag_names()` exposes), not pinned to flag names.
> **The "0.47.0" floor is an assumed number — bump it in one place (`preframr/requirements.txt`,
> + any spec `_IMAGE` pin) when the real tag is known.**

## Packages

- **`preframr` 0.2.26** (staged; 0.2.25 is the live release) — framework only (train /
  inference / model / args / parse / stftokenize / utils). Image `anarkiwi/preframr`.
  No PyPI; ships as the docker image (`:VERSION` + `:latest` published on **`v*` tag via
  release.yml**, NOT main-push — main-push docker.yml is `push:false`/test-only). Carries
  the per-op-accuracy gate. **0.2.26 floors `preframr-tokens>=0.47.0`** (the event-model release;
  only **one** req file — `requirements.txt` — floors tokens). The flag/tier registry consumption
  (`macro_flag_names()`/`resolve_flags()`/`NAMED_CONFIGS`, `op_name_by_id()`, `tier_classify`,
  `op_contracts.MACRO_OP_LOSS_TIERS`) stays registry-driven, so the only framework edits the swap
  *currently* needs were two tests that hardcoded flag NAMES (`tests/test_macro_flags_resolver.py`,
  `tests/train/test_model_ckpt_completeness.py`) — rewritten **registry-driven** (pick a flag from
  `macro_flag_names()` or skip if empty). No source op-string was hardcoded. **Framework coupling —
  RESOLVED (no framework source change needed):** the dataset/training contract is **PRESERVED** —
  `corpus.preload` (tokens-side, which the framework's `regdataset`/`trainer`/`predict` consume) still
  writes per-dump `.0.blocks.npy` + `tokens.csv` + reg-widths and `iter_block_seqs` serves blocks unchanged;
  the event stream rides the alphabet-agnostic `RegTokenizer`+BPE → data-loading needs no change. The
  tier/op instrumentation, which would otherwise go degenerate (the event alphabet was all-`op=SET`/`reg=0`),
  is now **event-aware via a tokens-side fix in `events/dataset.events_alphabet()`**: value-digit atoms →
  `content`, structural-marker atoms → `structural` (mapped to a structural-tier op, with `reg=0` so they
  are NOT counted as `FRAME_REG` BPE splitters; frame-weights stay 1.0). Because `tier_map`/`tier_classify`
  (`_row_tier` over `tokens.csv`) → `audit_checkpoint_per_class` → `content_tier_report` are all
  registry-driven, the decisive `content_over_structural` gate + per-tier heads + structural-loss adapt for
  free. Residual (minor): `content_tier_report`'s by-op spotlight defaults to FREQ_TRAJ op 45 (absent in
  events) — the per-tier read is correct; the by-op breakdown is event-coarse until first-class event op
  names land (gen2 §3.1). **Still untested:** the actual train/generate run on event tokens (needs the
  image; host has no torch). **Macro passes are supplied as ONE validated list** —
  `apply_macro_flags_to_args` resolves `--macro-flags` / `--macro-config` off the tokens registry
  (default all-OFF); the old per-flag `--foo-pass` args + `_PIPELINE_NAME_TO_FLAG` + `--pipeline-spec`
  are gone.
- **`preframr-tokens` >=0.47.0** (PyPI; staged target — gen2, design `REDESIGN_optionB.md`) —
  torch-free parser/tokenizer + macros + `render_play`. **0.47.0 (event/tracker model — Option B):**
  the register-write token model is replaced by a typed **event stream** (`preframr_tokens/events/`:
  `schema`/`encoder`/`decoder`/`tokenize`/`grammar`/`oracle`). Event kinds: `NOTE_ON` (gate-on edge,
  carries zig-zag note-interval + an ordered interleaved CTRL/AD/SR **attack** write-sequence +
  mixed-radix duration; **no note-off**, derived at onset+duration), `NOTE_STEP` (pitch change under
  the held gate = arp/legato/slide), `MOD_FREQ`/`MOD_PW`/`MOD_CTRL` (per-voice freq-delta / pulse-width
  / body-waveform gestures), and a `GLOBAL` lane `MOD_CUTOFF`/`FILTER_CTL`/`MOD_VOL` (SID has ONE shared
  filter + master volume), plus `TICK`/`TUNING`/`NOTE_TABLE` headers. **Escape-free, no ids:** every
  field is a *complete value* over a small fixed alphabet; **BPE/Unigram over the event stream IS the
  corpus-global dictionary** — frequent field-sequences fuse into learned tokens, rare ones stay as
  digits in the same alphabet (no DEF/REF, no per-tune bank, no singleton glitch-tokens, no literal/escape
  path). **One mixed-radix `q·tick + r` time encoding** for every time quantity (duration, gesture span,
  LFO/arp rate, rest); decoder does all arithmetic, model selects/copies low-cardinality fields. **Fidelity
  oracle = the exact ORDERED register-write stream** (frame order + intra-frame write order, audibly
  significant for hard-restart/gate timing), NOT settled `register_state` (now a secondary pre-filter only);
  same-frame writes sequenced by a complete per-frame **order descriptor**. `constrained_decode` →
  **event-grammar mask** (finite-state over event kinds+fields, registry-driven, completeness-tested; no
  REF-liveness). **§7.1 PRODUCTION SWAP DONE** — the events codec IS the tokenization pipeline (`events/
  pipeline.py` + `events/dataset.py` + `corpus.preload` + `events/generate.py`), replacing the
  `parse → (op,reg,subreg,val) → merge_token_df` substrate; RETIRED: `MdlGesturePass`, `_GestureCodec`, all
  `GESTURE_*` ops, `codebook_emit`, the per-tune bank, the arbiter/Claim flow (gesture was the last codebook
  family → `CODEBOOK_FAMILIES = {}`, GESTURE ops de-registered from `op_contracts`/`macro_contracts`) — plus
  the earlier `GeneratorPass`/`InstrumentProgramPass`/cents-quantization and all dead-encoding flags (see
  banner). **v1 is BPE-only:** loop/pattern/voice-block/coarsen structural passes are NOT ported, so the
  surviving macro-flag set is uncertain/possibly empty. KEPT as internal encoder primitives: `mdl_core`
  (HOLD/POLY/PERIOD, in the per-register gesture lanes + scalar parse), `pitch_grid` (note-table for the freq
  two-tiling), `audit_primitives` (secondary check). **Status: substantially built + green** (`events/STATUS.md`):
  v1 factored codec byte-exact on 5 drivers + 200-tune corpus + 300/300 corpus tunes; note/attack + mixed-radix
  tick + derived gate-off + §8.5 combined PW lane landed (**3.77 bits/write, 4.24×**); invariants pass; tokens
  suite **676 pass / 3 fail (pre-existing, zero swap regressions)**. Remaining before release: §7.1 generation
  grammar-mask (decoder already validates), §3.1 semantic `MOD_*`/`GLOBAL` typing, §4.1 span inheritance, §9
  corpus-scale bits, cosmetic dead-codebook-code cleanup, and the **downstream model train/generate run on
  event tokens (not done)**. **Measured + rejected** (kept simpler form): the §8.4 joint freq/note DP (4.23→4.42
  bits — greedy two-tiling wins on static notes) and §2.7 mixed-radix ORDER-`DT` (gap entropy≈0). (The interim
  gesture model on `feat/mdl-optimal-parser` is byte-exact but the wrong substrate — 2.4× expansion, per-tune
  polysemous ids, verbatim ctrl/ADSR — hence Option B.)
  **Floor moves only after the event model finishes those items, stays green, and releases.**
  (History — gesture model 0.46.x, generator pipeline 0.45.x, instrument collapse 0.44.x, byte-exact 0.41–0.43 —
  is in git log + `design/landed/`; Option B supersedes all of it.)
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

## Current arc — EVENT-MODEL (Option B) TRANSITION (staged; canonical run deferred)

**The arc is the encoding rewrite, not a training run.** The whole pitch_resid / generator /
residual line is RETIRED, and so is the MDL **gesture-codebook** that briefly replaced it: Option B
(`gen2/REDESIGN_optionB.md`) supersedes the gesture overlay with an **event/tracker token model**
(see banner + Packages). The old reliability saga — the arbiter per-claim re-decode soft-hang
(`parse-slow-decode-walker`, O(claims×df)) and the "generator mis-encodes repeated patterns" bug —
is dissolved at the root: the event model has **no arbiter/Claim flow and no per-tune codebook at
all**; repetition is handled by BPE over a complete-value event stream. The gesture model's own
failure modes (2.4× expansion, per-tune polysemous ids, verbatim ctrl/ADSR) are exactly why Option B
replaced it.

### What is staged in this branch (`mdl-transition-staging`)
- **framework** (`preframr`): VERSION 0.2.25→0.2.26, floor `preframr-tokens>=0.47.0`, and the two
  tests that hardcoded flag NAMES rewritten **registry-driven** (pick-or-skip on `macro_flag_names()`)
  — because Option B's BPE-only v1 makes the surviving flag set uncertain/possibly empty. Everything
  else is registry-driven and adapts on the version bump. The §7.1 production swap (tokens-side) preserves
  the framework's data-loading contract (blocks/tokens.csv/reg-widths via `corpus.preload`); the event-aware
  tier split is now landed tokens-side (`events_alphabet`, see Packages `preframr`), so the content-tier gate
  works registry-driven. The remaining open item is the untested train/generate path (needs the image).
- **xpt `base.py`**: tokens source repointed `/scratch/anarkiwi/preframr-tokens` → `/scratch/anarkiwi/gen2-preframr-tokens`
  (where the event model lives; canonical clone is stale). The image bind-mounts this over the installed
  package, so event runs MUST disable the dataset cache (`PREFRAMR_DATASET_CACHE_DISABLE=1`) until the image
  is rebuilt with event tokens installed — the cache key folds the *installed* dist version, which the
  bind-mount does not bump, so a stale 0.46.x parse could otherwise be reused.
- **xpt specs**: 30 specs deleted — the 29 dead-flag specs (refuted model-side threads, the
  dead-encoding generator/melody/freq zoo minis, the dead canonical prodlike `pitch_resid_*` /
  `full_macros_prodlike` / `melody_stack` / `trajectory_anchor_*`) PLUS `learnability_full_macros_mini`
  (its full_macros-vs-baseline macro A/B goes **degenerate** under BPE-only v1 — full_macros may resolve
  empty). Surviving runnable specs: only `generalize` + `memorize` (infra/build-gate, `baseline=True`,
  encoding-agnostic).
- **audits**: the `audit/probes/resid_*` scripts don't crash but the **residual concept is dead** under
  Option B (no residuals; fidelity is the ordered-write oracle) — historical, don't extend.

### The canonical run is DEFERRED until the event model is built + released (user decision)
The decisive experiment is **intentionally not designed yet.** Under Option B the meaningful levers
shift: BPE vocab/merge count, the perceptual ADSR embedding-tying (§5.2), and (only if BPE leaves
long-range repetition) optional loop/pattern ops — NOT a macro-pass A/B. Gate it as before on per-tier
`content_over_structural` + per-op `op_acc` (now over event KINDs) + `eval_b_*` held-out composers.
There is no surviving runnable A/B scaffold; a new event-model spec is written post-release.

### Carry-over context that survives the rewrite
- **Within-tune triage is NOT the verdict.** `learnability_triage --mode window` credited trivial
  redundancy and was blind to cross-tune transfer; the verdict is a canonical run on per-tier
  `content_over_structural` + per-op `op_acc` (all-tier val_acc is confounded across tokenizations).
  This holds under the event encoding; re-point the triage at the event-token stream.
- Model-side content interventions were refuted at the ~0.13 ceiling that tokenizer-side representation
  then lifted — the lever remains tokenizer-side, which is exactly what the event model sharpens
  (escape-free complete-value fields + BPE-as-dictionary + perceptual envelope tying).

### NEXT
1. **Gate:** the event model (`preframr_tokens/events/`, Option B) is **built + green with the §7.1 production
   swap DONE** (`events/STATUS.md`: events codec IS the pipeline, gesture/codebook retired, byte-exact on 5
   drivers + 300-tune corpus, 3.77 bits/write / 4.24×, tokens suite 676/3-pre-existing). REMAINING before
   release: §7.1 generation grammar-mask, §3.1 semantic `MOD_*`/`GLOBAL` typing, §4.1 span inheritance, §9
   corpus-scale bits, cosmetic dead-code cleanup, and the **downstream model train/generate run on event
   tokens (not done)**; then **release to PyPI** (§8.4 joint DP + §2.7 ORDER-`DT` were tried and rejected —
   not blockers). Bump the assumed `>=0.47.0` floor when the real tag is known.
2. **First event-model run (host has no torch — needs the image):** rebuild `anarkiwi/preframr:0.2.26` + the
   xpt image with the event tokens installed, OR run now against the bind-mounted gen2 source with
   `PREFRAMR_DATASET_CACHE_DISABLE=1` (the bind-mount doesn't bump the cache-key version). Run
   `generalize`/`memorize` to confirm tokenize→train→generate end-to-end on the event encoding. The event-aware
   tier split already flows through the gate (landed tokens-side); confirm `content_tier_report`'s per-tier
   numbers look sane on a real run (its by-op spotlight default op 45 is event-irrelevant — secondary).
3. **Then:** design + launch the deferred canonical generalization run on the event model.

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
  validates each name, adds deps + rejects conflicts (`resolve_flags` — `FLAG_REQUIRES` /
  `FLAG_CONFLICTS` are empty, so resolution is a no-op pass-through but the machinery stays), and
  sets the per-flag attrs. **Default = all OPTIONAL passes OFF.** ⚠️ Under the Option B event model
  the encoding is unconditional and **v1 is BPE-only** — loop/pattern/voice-block/coarsen structural
  passes are NOT ported, so the surviving macro-flag set is **uncertain and possibly empty**, and a
  `full_macros`-vs-`baseline` macro A/B may be **degenerate** (both the same event stream). The
  meaningful v1 levers are tokenizer-side (BPE vocab, perceptual envelope tying), not macro passes.
  Treat any macro-flag arm as TBD until the event model defines its flag surface. There is no more
  `pipeline_spec` / `--foo-pass` surface (the old argparse flags + `_PIPELINE_NAME_TO_FLAG` are gone).
- **Spec-dependent tokenization** (motif / cluster_content / voice_permutation /
  any `pre_run_hook` that mutates staged dumps or mines a per-spec artifact):
  launch with `PREFRAMR_DATASET_CACHE_DISABLE=1`.
- **Content-tier audit (decisive gate) — event-aware as of the tokens-side tier fix.** Per arm-seed, run
  `audit_checkpoint_per_class --ckpt ... --work-dir ... --out audit_per_class.json`
  in the xpt image (emits `vocab_atom`). Then host-side, torch-free:
  `python3 -m preframr_experiments.audit.content_tier_report --results-root <dir>`
  (+ `--onset` for V0-onset bucket). All-tier val_acc is CONFOUNDED across
  tokenizations — the content-tier read settles representation A/Bs.
  Tested readers indexed in `preframr_experiments/audit/README.md`; use them,
  not bespoke `/scratch/tmp` scripts. **On event-model runs** the per-tier
  `content_over_structural` split is meaningful again (value-digit atoms = content, structural-marker atoms
  = structural, set tokens-side in `events_alphabet` and flowed through `tier_map`→`audit_checkpoint_per_class`).
  Caveat: the **by-op spotlight** (`--spotlight-op`, default FREQ_TRAJ op 45) is event-irrelevant — the event
  structural atoms share one borrowed op, so the by-op breakdown is coarse until first-class event op names
  land (gen2 §3.1); rely on the per-tier read, not by-op, for event A/Bs.
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
- **FOLLOW-UP (port to event model) — staged tracker round-trip tests in `staging/tokens_tests/`.**
  These SWM/defMON forward round-trip tests assert `module → register log → parse → decode ==
  player output`. The principle (byte-exact round-trip on real tracker output) is MORE relevant under
  Option B, not less — but Option B **strengthens the oracle**: the target becomes the exact ORDERED
  register-write stream (`events/oracle.py: ordered_writes(df)`), not settled `register_state`/
  `parse_audit='raise'`. They were written against the retired `generator_pass` and must be re-pointed
  at the **event encoder/decoder** before landing in `preframr-tokens/tests/`. Provisioning is the
  same: pysidwizard+pydefmon as **test-only** deps, drop the `conftest.py` source-path shim, fixtures
  via `pysidwizard.tests._swm_cache` + pydefmon `build/fixtures` (no SID binaries in git). See
  `staging/tokens_tests/README.md` (its generator-specific claims are stale).
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

### Content tier (Option B: byte-exact, no lossy cent-binning)
Under the event model the cents-quantization path is **gone** — every field is a
complete value over a fixed alphabet, byte-exact by construction (no slope/preset/
transpose cent-binning, no lossy content tier, no escape path). The old "content-tier
deliberately lossy" caveat no longer applies; the fidelity oracle is the exact
**ordered register-write stream** (`events/oracle.py`), with settled `register_state`
only a secondary pre-filter.

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

- **2026-06-08** — **event-model (Option B) transition staged** (branch `mdl-transition-staging`, both
  repos). First reconciled against the MDL gesture-codebook (`feat/mdl-optimal-parser`); the approach
  then changed again to the **event/tracker model** (`gen2/REDESIGN_optionB.md`, supersedes the gesture
  overlay: typed event stream, no codebook/ids, BPE-as-dictionary, ordered-write oracle, no note-off,
  mixed-radix time). Re-pointed the staging: framework floor→0.47.0, VERSION→0.2.26, 2 flag-name tests
  rewritten **registry-driven** (surviving flag set now uncertain under BPE-only v1); xpt: 30 specs
  deleted (29 dead-flag + `learnability_full_macros_mini` whose macro A/B goes degenerate), only
  `generalize`+`memorize` remain; AGENTS.md + memory reconciled to the event model. The event model is
  **substantially built + green with the §7.1 production swap DONE** (`events/STATUS.md`: events codec IS
  the pipeline, gesture/codebook retired, byte-exact on 5 drivers + 300-tune corpus, 3.77 bits/write /
  4.24×, tokens suite 676/3-pre-existing); **gated on §7.1 grammar-mask / §3.1 typing / §4.1 span
  inheritance / §9 corpus bits / dead-code cleanup / the untested downstream train-generate run, then
  release** (§8.4 joint DP + §2.7 ORDER-DT mixed-radix tried + rejected on measurement). Key framework
  finding: the swap preserves the data-loading contract but the synthetic 68-atom alphabet makes the
  content-tier gate degenerate (needs event-aware rework). Canonical run deferred (user decision).
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
