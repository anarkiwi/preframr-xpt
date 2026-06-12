# Operational notes for agents (preframr-xpt)

This repo is the **experiment surface**: docker-driven runner + spec registry
+ audits + design docs + tier data + refuted registry. Framework, libraries,
corpus live elsewhere.

> **EVENT-MODEL (Option B → v3 canonical) TOKENS — BUILT, MEASURED, CHIP-VERIFIED, and now
> SHIPPED + train→generate→decode GREEN (2026-06-12).** tokens 0.47.0 / audio 0.5.8 (PyPI),
> preframr 0.2.26 (Docker Hub `:0.2.26`+`:latest`, tagged `v0.2.26`), xpt on `main` based on
> `:0.2.26`. The `memorize` build-gate runs train→generate→decode end-to-end (event-native
> `preframr/inference/event_gate.py`, wired as its `predict_gate`): last run mean greedy acc 0.929,
> mean decoded-gen fraction 0.960, PASS. NEXT-1 and NEXT-2 below are DONE; the open arc is the
> canonical learnability run (NEXT-3). Canonical tokens repo is `/scratch/anarkiwi/preframr-tokens`
> (gen2 merged in — forget the gen2 path).
> The tokens encoding IS the event/tracker model now (`preframr_tokens/events/`,
> design `REDESIGN_optionB.md` as corrected by `events/STATUS.md` — STATUS supersedes the design doc
> where they disagree). The production swap is DONE and the encoding is **unconditional** (no macro
> flags gate any of it): `stream.encode` is the tokenizer (`corpus.preload` → per-dump event blocks →
> alphabet-agnostic `RegTokenizer`+BPE), `events/generate.py` decodes generated ids to render-ready
> writes. **The v3 canonical fidelity contract (2026-06-11) replaced the §2.8 byte-order oracle**:
> the oracle is `stream.canonical_writes(dump)` — settled freq/PW first per voice, globals last,
> same-value rewrites dropped (chip latch no-ops, reSID-verified), **no NOTE OFF** (gate 1→0 always
> derived at onset+duration), NOTE_ON owns the envelope lifecycle (onset AD/SR + hard-restart prep
> pair + duration as NOTE_ON fields), and — measured against reSID this session — **gate-edge
> crossings are content**: folded envelope writes re-emit on the *recorded side* of the gate edge
> (driver conventions split; a fixed order is audibly wrong). `encode` self-verifies
> `decode == canonical_writes` (fail loudly), and raw-vs-canonical **renders at the reSID noise
> floor on all 5 drivers** (perceptual A/B). Chip semantics are pinned as a 24-test canonical
> reference in preframr-audio (`test_gate_adsr_reference` / `test_adsr_write_liveness_matrix` /
> `test_release_write_position`): the ADSR bug is compare-change associated (not gate associated),
> the (phase × nibble) write-liveness matrix decides what is relocatable, raising sustain mid-note
> kills the note. **Vocab = 127 fixed atoms** (32 BE-varint digits, 25 regs, 4 voices, 17
> kinds/shapes, 48 typed value nibbles, KEYFRAME); 96 occur on the corpus sample, the <1% tail is
> fully explained (headers bounded-by-construction, rare chip features, the proven-irreducible
> mid-note S/D envelope events). **Measured (59-65 in-scope tunes + 5 drivers): atomic 1.71
> tok/write, post-BPE 0.21–0.23 tok/write at ~1.8–2.0 bits/write order-0 — 7.8× (order-0) / 23×
> (order-1) vs the 16-bit raw floor, past the §9 10–14× target.** Learnability layer is in: typed
> value nibbles (timbre bits are single embeddings), big-endian varints, KEYFRAME chunk conditioning
> (`dataset.encode_block_array` leads every training chunk with decoder state — mid-song prompts can
> interpret durations/intervals). Scope: single-speed non-digi (~92% of corpus). ⚠️ **Wire format
> changed twice on 2026-06-11** (NOTE_ON lifecycle fold; gate-edge side flags): ALL cached parses /
> blocks / trained BPE merges from before then are stale — bust or disable the dataset cache.
> Event kinds (actual names): `NOTE_ON`/`CTRL`/`AD`/`SR` (cas lane), `NI_STEP`/`NI_RAMP` (pitch
> intervals), `FD_STEP`/`FD_RAMP` (freq residual), `PW_STEP`/`PW_RAMP`, `G_STEP`/`G_RAMP` (globals,
> reg-tagged), `POLY`/`PERIOD` shapes, `TUNING`/`NOTE_TABLE`/`TICK` headers — the design doc's
> `MOD_*` naming never shipped (§3.1 semantic labels remain cosmetic-open). RETIRED: the whole
> (op,reg,subreg,val) substrate, ORDER descriptor, PRE primitive, literal/escape paths, gesture
> codebook, arbiter/Claim, all dead-encoding flags. v1 factored codec remains in-tree as the
> byte-exact measurement baseline only.
> **Train→generate→decode is GREEN and shipped** (2026-06-12, `memorize` gate via `event_gate.py`);
> tokens 0.47.0 + preframr 0.2.26 (tag `v0.2.26`) are released. The open arc is the canonical
> learnability run (see Current arc / NEXT-3).

## Packages

- **`preframr` 0.2.26** (live release; published 2026-06-12) — framework only (train /
  inference / model / args / parse / stftokenize / utils). Image `anarkiwi/preframr`.
  No PyPI; ships as the docker image. **The release is the merge to `main`**:
  `release.yml` fires on **main-push AND `v*` tags** with `push: true`, publishing
  `:${VERSION}` (from the VERSION file) + `:latest`. `docker.yml` is the separate
  `push: false` *validation* build (PRs + main). **Convention: also tag each release
  `git tag -a vX.Y.Z <released-main-sha>` and push it** (a version number with no tag is
  useless; the tag re-fires `release.yml` and re-pushes the identical image — harmless).
  Carries the per-op-accuracy gate. **0.2.26 floors `preframr-tokens>=0.47.0`** (the event-model release;
  only **one** req file — `requirements.txt` — floors tokens). Framework coupling is RESOLVED with
  no framework source change: `corpus.preload` (tokens-side) still writes per-dump `.0.blocks.npy`
  + `tokens.csv` + reg-widths and `iter_block_seqs` serves blocks unchanged; the event stream rides
  the alphabet-agnostic `RegTokenizer`+BPE. Tier/op instrumentation is event-aware via the
  tokens-side `events/dataset.events_alphabet()` (value-digit atoms → `content`, structural-marker
  atoms → `structural`, no `FRAME_REG` re-triggering, frame-weights 1.0) and flows registry-driven
  through `tier_map` → `audit_checkpoint_per_class` → `content_tier_report`, so the decisive
  `content_over_structural` gate works on event tokens. Residual (minor): the by-op spotlight
  defaults to FREQ_TRAJ op 45 (absent under events) — rely on the per-tier read until first-class
  event op names land (gen2 §3.1). Two tests that hardcoded flag NAMES are rewritten
  registry-driven (pick-or-skip on `macro_flag_names()`); macro passes remain ONE validated list
  via `apply_macro_flags_to_args`, default all-OFF — but see the flag-surface warning under
  Tests + runner.
- **`preframr-tokens` 0.47.0** (PyPI; LIVE — canonical repo `/scratch/anarkiwi/preframr-tokens`,
  design `REDESIGN_optionB.md` corrected by `events/STATUS.md`) — torch-free parser/tokenizer + macros + `render_play`.
  The event model: see banner for the full current state (v3 canonical contract, NOTE_ON
  envelope lifecycle, recorded gate-edge sides, typed nibbles, KEYFRAME, 127-atom vocab,
  measured collapse). Key invariants: escape-free complete-value fields over one fixed alphabet
  (BPE IS the corpus-global dictionary — no DEF/REF, no per-tune bank, no literals); no note-off
  (derived); mixed-radix `q·tick + r` durations (tempo-invariant; deliberately NOT applied to
  event-gap DTs — measured harmful); voice-grouped frames (`[DT]([VOICE][events]*)*` — patches are
  voice-portable for BPE); `encode` self-verifies against `canonical_writes`. Measured + rejected
  (don't re-propose): §8.4 joint freq/note DP, §2.7 mixed-radix ORDER-DT, DT-in-ticks, POLY degree
  cap, mid-note R-only fold into NOTE_ON (valid on-chip but buys 0.8% of AD/SR changes — the
  standalone mid-note bucket is ~94% proven irreducible). Remaining pre-release items —
  §7.1 generation-time grammar mask (decoder already validates; needed for clean *sampling*, not
  for training/tier metrics), §3.1 semantic `MOD_*`/GLOBAL labels (cosmetic; unblocks by-op
  spotlight), §4.1 span inheritance (optimization), dead-codebook code deletion (cosmetic) —
  **none block the first training run.**
  **Floor moves when 0.47.0 tags.** (History — gesture model 0.46.x, generator pipeline 0.45.x,
  instrument collapse 0.44.x, byte-exact 0.41–0.43 — git log + `design/landed/`; superseded.)
- **`preframr-audio` 0.5.6+** (PyPI) — SID audio rendering primitives + the **chip-semantics
  canonical reference** (2026-06-11): `tests/test_gate_adsr_reference.py` (the ADSR bug exactly:
  compare-change associated, write-only freezes in all phases, one-directional, internal handoffs
  never stall, gate-edge position is content at single-write granularity),
  `tests/test_adsr_write_liveness_matrix.py` (the (phase × nibble) relocation matrix; equality
  sustain-hold: raising S mid-note kills the note), `tests/test_release_write_position.py`
  (R-fold placement rules). Envelope/canonicalization questions are answered from these tests,
  not by new ad-hoc probes; methodology notes inside (write-count-matched variants, ENV3 verdicts,
  per-write clocking — collapsed-timing A/Bs MASK placement effects).
- **`preframr-experiments`** (this repo; editable / PYTHONPATH, no PyPI) —
  runner + specs + `audit/` + tests. Pure orchestration on the host; audits
  import preframr/torch and run inside the **xpt image**.

Sibling source repos: `/scratch/anarkiwi/preframr-{audio,tokens,xpt,aug}`,
`/scratch/anarkiwi/gen2-preframr-tokens` (the event model — canonical
`preframr-tokens` clone is stale; xpt `base.py` points here), and
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
prefer induction-head copy over implicit per-frame counters, order by the driver
causal DAG. Correctness is the *gate*; compression / parse-perf / deploy are *infra*. The event
model is the direct product of this lens, and its learnability layer is already measured in
(typed nibbles: H1 5.92→5.81 bits/write with each token more predictable; KEYFRAME gives every
chunk its interpretive state — the mid-song-prompt goal, attacked at the encoding). A
training-free triage (`audit/learnability_triage.py`) ranks encodings before a run — run it at
**prodlike `seq_len=8192`**, re-pointed at the event-token stream. **Mini (4096) is not a
research dimension** (mode-collapses; window distorts the static read) — plumbing/cost only.
Model-side content interventions were refuted at the ~0.13 ceiling that tokenizer-side
representation then lifted — the lever is tokenizer-side, which is now the event model itself.

## Current arc — CANONICAL EVENT-MODEL LEARNABILITY RUN (smoke green + shipped; the canonical run is open)

The encoding rewrite AND the end-to-end pipeline are **done and shipped** (2026-06-12): production
swap done, v3 canonical contract chip-verified, corpus-scale collapse measured past target, tier
split landed — and now tokens 0.47.0 / audio 0.5.8 (PyPI), preframr 0.2.26 (Docker Hub + tag
`v0.2.26`), xpt on `main`. The `memorize` build-gate validates train→generate→decode end-to-end via
the event-native `event_gate.py` (acc 0.929, decoded-gen-frac 0.960, PASS). The old reliability saga
(arbiter soft-hang, generator mis-encode) is dissolved at the root. What has **never happened** is
the canonical learnability run on event tokens — that is the open arc, scientific not operational.

### NEXT (concrete, in order)
1. **DONE — smoke the pipeline end-to-end.** `memorize` trains→generates→decodes on current-wire
   event tokens on the baked `anarkiwi/preframr:0.2.26` image (no bind, no cache-disable);
   `event_gate.py` is wired as its `predict_gate`. **Methodology that matters:** the gate decodes
   COMPLETE self-contained blocks (`--gen-tokens ≥ block_len`), NOT a truncated window —
   `stream.decode` raises or returns empty on a mid-frame cut and frame boundaries are sparse in
   dense songs, so a fixed window gives confounded zeros. Per-tier `content_tier_report` capture
   still applies (ignore the by-op spotlight until §3.1).
2. **DONE — release.** tokens 0.47.0 + audio 0.5.8 (PyPI), preframr 0.2.26 (Docker Hub `:0.2.26` +
   `:latest`, tagged `v0.2.26`), xpt `ARG BASE`→0.2.26 on `main`. The framework was reconciled with
   0.47.0 (`args.py` `NAMED_CONFIGS` is now a TUPLE → `named_config()`; 3 stale regdataset/macro
   tests). Runs no longer need the bind-mount / cache-disable pair.
3. **NOW — run `generalize`, triage, then design + run the canonical learnability spec.** First run
   the `generalize` spec (held-out composers + Eval-A; canonical tier, its own val_acc
   `predict_gate`). Then re-point `learnability_triage` at the event stream at seq_len 8192 for the
   static read, and write the canonical event-model spec. Meaningful levers: **BPE merge count /
   trained vocab size** (the 127-atom alphabet is fixed; merges are the dictionary),
   **typed-nibble embedding treatment** (NIB_ENV may already deliver what §5.2 perceptual ADSR-tying
   wanted — check before building it), and KEYFRAME conditioning variants. NOT a macro-pass A/B (no
   flag surface exists). Gate on per-tier `content_over_structural` + per-op acc over event KINDs +
   `eval_b_*` held-out composers.

### Carry-over context that survives
- **Within-tune triage is NOT the verdict** (`--mode window` credited trivial redundancy, blind to
  cross-tune transfer); the verdict is the canonical run on per-tier `content_over_structural`.
  All-tier val_acc is confounded across tokenizations.
- **Spec surface**: 30 specs were deleted in the staging cut (dead-flag threads + the degenerate
  macro A/B); surviving runnable specs are `generalize` + `memorize` (infra/build-gate,
  `baseline=True`, encoding-agnostic). The canonical event-model spec is written post-release.
- **audits**: `audit/probes/resid_*` don't crash but the residual concept is dead (no residuals;
  fidelity is the canonical-writes oracle) — historical, don't extend.

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
- **Macro passes = one list — and under the event model the list is empty in practice.** The
  `Arm(macro_flags=..., macro_config=...)` machinery and `apply_macro_flags_to_args` validation
  survive registry-driven, but the event encoding is **unconditional**: no optional passes gate
  any primitive, `FLAG_REQUIRES`/`FLAG_CONFLICTS` are empty, and a `full_macros`-vs-`baseline`
  A/B is **degenerate by construction** (same event stream). The experiment levers are BPE
  vocab/merges and embedding/conditioning treatments, not macro flags. There is no
  `pipeline_spec` / `--foo-pass` surface.
- **Spec-dependent tokenization** (motif / cluster_content / voice_permutation /
  any `pre_run_hook` that mutates staged dumps or mines a per-spec artifact):
  launch with `PREFRAMR_DATASET_CACHE_DISABLE=1`.
- **Content-tier audit (decisive gate) — event-aware.** Per arm-seed, run
  `audit_checkpoint_per_class --ckpt ... --work-dir ... --out audit_per_class.json`
  in the xpt image (emits `vocab_atom`). Then host-side, torch-free:
  `python3 -m preframr_experiments.audit.content_tier_report --results-root <dir>`
  (+ `--onset` for V0-onset bucket). All-tier val_acc is CONFOUNDED across
  tokenizations — the content-tier read settles representation A/Bs. On event-model runs the
  per-tier `content_over_structural` split is meaningful (value-digit atoms = content,
  structural-marker atoms = structural, set tokens-side in `events_alphabet`). Caveat: the
  **by-op spotlight** (default FREQ_TRAJ op 45) is event-irrelevant — per-tier read only,
  until gen2 §3.1 lands first-class event op names. Tested readers indexed in
  `preframr_experiments/audit/README.md`; use them, not bespoke `/scratch/tmp` scripts.
- Outputs under `/scratch/tmp/preframr_experiments/` (or `--root`). Status:
  `check_overnight_batch.sh`; done marker `overnight_batch.done`.

## Conventions

- **Code = frozen baked image by default.** Runs use baked `preframr/`; rebake
  to pick up edits. Working-tree bind-mount is opt-in (`run.py --bind-src` /
  `$PREFRAMR_BIND_SRC=1`) and runs un-gated code — don't use without asking.
  The 0.47.0/0.2.26 bind+cache-disable exception is RETIRED: the baked
  `anarkiwi/preframr:0.2.26` image is event-model-current, so runs use it
  directly (no bind, no cache-disable).
- **Background runs:** `nohup`+`disown`; don't poll, use `ScheduleWakeup`.
- **Comments:** no session narration / dev-local paths / PR numbers;
  `tests/test_lint.py` rejects narrative `#` and >5-line docstrings (gen2 enforces
  the same gate — it cost a CI round on 2026-06-11).
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
per (arm, seed). parse+tokenize ~25 min/prodlike uncached (pre-event numbers;
re-anchor on the first event-model run).

## Forward-looking work

### Land any time
- **FOLLOW-UP (port to event model) — staged tracker round-trip tests in `staging/tokens_tests/`.**
  SWM/defMON forward round-trip: `module → register log → parse → decode == player output`. Under
  v3 the target is `stream.canonical_writes` (the audibly-faithful canonical form), checked by
  `stream.encode(verify=True)` — re-point the staged tests at the event encoder/decoder before
  landing in tokens `tests/`. Provisioning unchanged (pysidwizard+pydefmon test-only deps, fixtures
  from caches, no SID binaries in git). `staging/tokens_tests/README.md` generator-era claims stale.
- **Profile + optimize event parsing/encode** — `encode(verify=True)` doubles work by design;
  measure whether corpus builds need a verified-once-then-fast path. Keep the self-verify in tests.
- **Recover control-write-rejected dumps** — characterize dumps rejected for
  too many control writes; relax/absorb to grow the corpus. Also revisit the multi-speed (~5%)
  and digi (~3%) exclusions once the single-speed model trains.
- **Fidelity gate under v3** — superseded machinery: `parse_audit` / `cb_div_audit` /
  residual-zero census belonged to the retired substrate. The v3 gates are (a) `encode`
  self-verification (every encode, fail-loudly), (b) the events test suites (5 drivers + 200-tune
  corpus roundtrip), (c) the chip-semantics reference suites in preframr-audio, (d) the
  perceptual raw-vs-canonical A/B (tmp probe → worth productizing as a corpus-wide audit).
  See the updated `design/references/verification_and_audits.md`.

### Predict-host envelope (queued, post-generalization)
The 127-atom fixed alphabet + chosen BPE vocab replaces the old "~91% dead tkvocab" problem —
vocab size is now a dial, not a cleanup. Remaining: GPU-resident constrained decode (needs gen2
§7.1 grammar mask) → full-context audition. Orin is ~4% GPU util at predict. Re-open alongside
the Multi-GPU rental decision (deferred until a generalising approach lands).

### Framework follow-ups
- Streaming unembed-CE — recovers prodlike 2× wall.
- Generalization-gate thresholds — recalibrate per tier.
- Augmentation tooling + design in **`preframr-aug`**; melody-transfer
  Phase-0 smoke still pending.
- **Autocast fp32-promotion trap:** any new `Module` in
  `preframr/train/model/heads*.py` must cast log_softmax/logsumexp back to
  input dtype or per-position buffers stay fp32 and OOM at prodlike
  (pinned by `tests/train/test_per_tier_heads.py::test_bf16_input_preserves_*`).

### Content tier (v3: canonical-exact, no lossy band at all)
Every field is a complete value over the fixed alphabet; there is no cent-binning, no
`freq_tol` band, no escape path, and no residual concept. "Lossless" means **canonical**:
within-frame settled freq/PW + ordered cas activity + derived gate-offs, with the
sub-frame transients it canonicalizes away measured masked (−27 dB) and same-value writes
chip-inert (reSID-verified). The old "content-tier deliberately lossy" caveat is gone.

## Refuted alternatives

Registry: `preframr_experiments/data/refuted/<exp>.md`. Model-side
interventions concentrated at the same ~0.13 eval_a content ceiling (since
lifted by tokenizer-side representation):

- `per_tier_heads_mos_prodlike`, `per_tier_heads_entropy_prodlike` — router
  saturates / lambda non-monotonic.
- `mask_structural_loss` — diversity collapse; structural supervision
  load-bearing.
- `cluster_conditional_content_head`, `content_diffusion`,
  `contrastive_infonce_auxiliary` — same ceiling.
- `motif_pass` (v1 exact + v2 templated) — content-tier neutral-to-negative.
- Earlier nulls: `legato_ab`, `palette_merge`, `head_row_class`,
  `adsr_equivalence`, `macro_coarsening`, `b2_unblock`,
  `palette_pwm_prereqs`, `global_instr_ids_phase_a`, `weighted_token_loss`,
  `learnable_class_loss`, `voice_trajectory` (all variants), `set_to_diff`.
- Tokens-side, measured + rejected in gen2 (see its STATUS): §8.4 joint freq/note DP,
  §2.7 mixed-radix ORDER-DT, DT-in-ticks, POLY degree cap, mid-note R-only NOTE_ON fold.

## Deferred deploy-stage efficiency (post-generalization)

The old FRAME/VOICE_REG token-budget items are obsolete under the event model (no FRAME/VOICE_REG
markers; voice tokens fell 18.2%→11.1% of the stream via voice-grouped frames). Revisit deploy
token budget only from real event-model predict traces.

## Resolved log (compact; full detail in git log + design/landed/ + data/refuted/)

- **2026-06-12** — **event model SHIPPED + train→generate→decode GREEN (NEXT-1/2 done).** Found the
  framework's generate/decode path unported (`predict.py`→old macros walker, crashes on event
  tokens); built event-native `preframr/inference/event_gate.py` (greedy gen → `events/generate.py`
  decode of COMPLETE self-contained blocks; mean greedy acc 0.929 / decoded-gen-frac 0.960, PASS),
  wired as `memorize`'s `predict_gate`. Reconciled the framework with tokens 0.47.0 (`args.py`
  `NAMED_CONFIGS` is now a tuple → `named_config()`; 3 stale regdataset/macro tests). Released:
  preframr 0.2.26 (Docker Hub `:0.2.26`+`:latest`, tag `v0.2.26`), xpt `ARG BASE`→0.2.26 on `main`.
  Doc fix: framework releases on **main-push** (`release.yml` `push:true`), not tag-only; tag every
  release going forward. Open arc → canonical learnability run (NEXT-3).
- **2026-06-11** — **v3 canonical contract + chip-semantics verification + encoding conformance.**
  The fidelity contract corrected to `canonical_writes` (canonical, not byte-order; PRE primitive
  removed; NOTE_ON owns the envelope lifecycle; learnability layer measured in: typed nibbles, BE
  varints, KEYFRAME; voice-grouped frames). The ADSR mechanism was fully characterized in
  preframr-audio as a 24-test canonical reference (compare-change associated, not gate associated;
  (phase × nibble) liveness matrix; equality sustain-hold) after a mid-note-AD/SR relocation
  question escalated into measurement; the audit proved the standalone mid-note bucket ~94%
  irreducible. The encoding was then made to match: gate-edge crossings are content → folded onset
  envelope re-emits on the RECORDED side of the gate edge (driver conventions split — a fixed
  canonical order was audibly wrong on grid_runner at 0.15% samples >500/max Δ 6722) + HR prep on
  the gate=0 side; raw-vs-canonical now renders at the reSID noise floor on all 5 drivers.
  Token distribution audited (96/127 atoms, tail explained, 1.706 tok/write atomic). Wire format
  changed (cache-busting required). AGENTS.md + fidelity/encoding design references rewritten to v3.
- **2026-06-08** — event-model (Option B) transition staged (branch `mdl-transition-staging`, both
  repos): framework floor→0.47.0, VERSION→0.2.26, 2 flag-name tests registry-driven; xpt: 30 specs
  deleted, `generalize`+`memorize` remain; `base.py` tokens source → gen2. Production swap done
  tokens-side; event-aware tier split landed (`events_alphabet`). Canonical run deferred.
- **2026-06-06** — generator pipeline + melody layers + instrument collapse landed on the retired
  substrate (history); `--mode window` triage verdict deferred to canonical.
- **2026-06-04** — instrument-program collapse shipped; tokens 0.44.0 + framework 0.2.20.
- **2026-06-02** — byte-exactness DONE on the old substrate (tokens 0.41.1/0.42.0; framework 0.2.17).
- **2026-05-21..28** — architecture exonerated (`framework_arch_test` val 0.903 on unseen motifs);
  substrate ablation lifted FREQ_TRAJ 2.4×; libs split to PyPI.
- **earlier** — see git log + `data/refuted/`.
