# Operational notes for agents (preframr-xpt)

This repo is the **experiment surface**: docker-driven runner + spec registry +
audits + design docs + tier data + refuted registry. Framework, libraries, corpus
live elsewhere (sibling repos under `/scratch/anarkiwi/`).

## State (2026-06-12)

The **event/tracker model is SHIPPED and end-to-end GREEN**: tokens 0.48.0 / audio
0.5.9 (PyPI), preframr 0.2.27 (Docker Hub `:0.2.27`+`:latest`, tag `v0.2.27`), xpt on
`main` (`ARG BASE=anarkiwi/preframr:0.2.27`). The `memorize` build-gate runs train→generate→decode via event-native
`preframr/inference/event_gate.py` (decodes COMPLETE self-contained blocks, not
truncated windows). The encoding **is** the event model now (`preframr_tokens/events/`,
design `REDESIGN_optionB.md` as corrected by `events/STATUS.md` — STATUS wins on conflict)
and is **unconditional** (no macro flags gate any primitive). `stream.encode` is the
tokenizer (alphabet-agnostic `RegTokenizer`+unigram BPE-as-dictionary); `events/generate.py`
decodes generated ids to render-ready writes.

**v3 canonical fidelity contract**: the oracle is `stream.canonical_writes(dump)` —
settled freq/PW first per voice, globals last, same-value rewrites dropped, **no NOTE OFF**
(gate 1→0 derived at onset+duration), NOTE_ON owns the envelope lifecycle, gate-edge
crossings are content (folded envelope re-emits on the *recorded* side). `encode` self-verifies
`decode == canonical_writes` (fail-loud); raw-vs-canonical renders at the reSID noise floor on
all 5 drivers. Vocab = **127 fixed atoms** (value-digit/reg/voice/kind/shape + KEYFRAME);
measured collapse 7.8× (order-0) / 23× (order-1) vs the 16-bit raw floor. Chip semantics are
pinned as a 24-test reference in preframr-audio. Scope: single-speed non-digi (~92% of corpus).
Details: `design/references/{verification_and_audits,learnability_token_ordering_theory}.md`.

## Current arc — CANONICAL EVENT-MODEL LEARNABILITY RUN (in progress)

The encoding + pipeline are done/shipped; the open arc is the **canonical learnability run
on event tokens** (scientific, not operational). The atoms-only baseline is DONE and the
BPE-dictionary run is in flight. Findings so far:

- **TOKENIZER CRASH + FIX (load-bearing).** Any `tkvocab>0` run hit a hard **SIGSEGV** in the
  `tokenizers` 0.23.1 **UnigramTrainer** — root cause (gdb-confirmed): recursive
  `alloc::rc::Rc::drop_slow` **stack overflow** when dropping the per-sentence lattice on long
  SID sentences (35K–150K tokens). **NOT** esaxx, **NOT** vocab size (256 and 131072 both crash;
  the esaxx_fast/#1698 lead was a red herring an `esaxx_fast`-off rebuild did NOT fix).
  **Fix = `RUST_MIN_STACK=2000000000` plumbed into `_docker_run` (base.py)** — gives spawned
  threads a 2 GB stack. Verified at full 856-dump scale (tkvocab=2048 trains + writes tkmodel
  where it used to die). `tkvocab=0` **skips the trainer entirely** (`corpus.py` `if not tkvocab`),
  so the atoms-only path never crashed — that's why `memorize` was green.
- **`generalize` spec `tkvocab` was a stale 131072** (pre-event "dead tkvocab" carry-over) —
  both unrealizable for the 127-atom alphabet and the first to trip the crash. Vocab is now a
  **dial** (the 127 atoms are fixed; merges are the dictionary). Pass realizable values via
  `--tkvocab N` (folds into the dataset-cache key → re-tokenizes).
- **Atoms-only `tkvocab=0` baseline DONE** (stopped epoch 99/100, val_acc 0.561). Content-tier
  (the decisive gate): eval_a **0.479**, eval_b_daglish **0.559**, eval_b_follin **0.416**;
  content/structural 0.72–0.88. ~3.7× the old **~0.13 eval_a content ceiling**, and held-out
  composers track/beat in-distribution (daglish > eval_a) — **content is learnable and generalises
  in the event model, even at the no-dictionary floor.** Artifacts: `/scratch/tmp/v3c_final.ckpt`,
  `audit_per_class_{,2,3,final}.json`. (1 seed, 24-block sample; content-tier defn differs from the
  old substrate, so vs-0.13 is directional.)
- **Live vocab ~98%** at unigram tkvocab=2048 (2015/2048 ids used; the 33 dead are rare base atoms
  absent from the single-speed corpus). Demolishes the old "~91% dead tkvocab" problem — vocab is a
  **dial with near-full utilisation**. Tunes avg ~30k tokens; **82% exceed seq_len 8192** (mean 4.16
  KEYFRAME-led windows/tune) — the model trains on self-contained windows, never whole tunes.
- **EARLY-STOP IS EFFECTIVELY DISABLED under schedule-free (load-bearing gotcha).** Optimizer is
  `AdamWScheduleFree` (no LR schedule/warmup); its averaged eval-iterate val_loss decreases steadily
  (~0.005/epoch) for 100+ epochs, so `EarlyStopping(min_delta=0.01, patience=5)` re-counts an
  "improvement" every ~2 epochs and **never fires** → runs hit `max_epochs`. Stops only when
  improvement < min_delta/patience (≈0.002/ep). For a real stop: raise `min_delta` (≈ target_rate ×
  patience, e.g. 0.05) per run, or set a deliberate `max_epochs`. (Aside: the plateau→steep-drop in
  val_loss is a *real* learning transition — present in the un-averaged train iterate too — only
  ~6× amplified by the averaging, not a schedule artifact.)

### NEXT — the dictionary run is in flight; branch on its content-tier outcome
**RUNNING (2026-06-12, on the 0.2.27 / tokens 0.48.0 stack):** `generalize --tkvocab 2048` — the
**canonical ~14M body** (8L-d320-im896) + unigram **tkvocab=2048**, same corpus/holdouts as the
atoms-only baseline → directly comparable. Root `/scratch/tmp/preframr_experiments/unigram_canonical_v2`
(fresh, re-tokenized against 0.48.0; the earlier `_v1` run was stopped + superseded by this rebuild).
The RUST_MIN_STACK UnigramTrainer fix is plumbed via `_docker_run`. **Decision metric = content-tier
on `eval_b` held-out composers vs the baseline** (eval_a 0.479 / daglish 0.559 / follin 0.416), NOT
all-tier val_acc. (The ~107M `generalize_prodlike_unigram` prodlike variant remains a runnable spec
for the scale-up branch, but the canonical learnability run is the 14M body.)

**IF CONTENT LEARNED** — BPE + capacity ≥ baseline on eval_b content (esp. lifts follin / narrows the
held-out gap; content/structural holds) → the tokenizer-side lever works, **scale it**:
1. De-confound: multi-seed + full-eval (`--max-blocks 0`) audit; confirm the eval_b gain is
   generalisation, not memorisation.
2. **Sweep the BPE-vocab dial** (2048 → 4096 → 8192…) for where content saturates (98% live ⇒
   headroom); pick the knee and write the canonical event-model spec at it.
3. Stack embedding/conditioning treatments: typed-nibble embeddings (NIB_ENV may already deliver
   §5.2 perceptual ADSR-tying — check first) + KEYFRAME variants.
4. Then the **stretch**: cross-*engine* generalisation; re-open the Orin **offline** predict path
   (grammar-mask constrained decode — real-time single-stream is out of reach, see
   `design/performance/orin_inference_optimization_design.md`).

**IF CONTENT NOT LEARNED** — eval_b content ≤ baseline / collapses / the content circuit never forms
→ BPE/capacity isn't the content lever; **diagnose + re-represent**, don't add capacity (model-side
is refuted at the 0.13 ceiling):
1. Localise: per-op acc over event KINDs (which content fails — `NI_*` pitch intervals, `FD_*` freq
   residual, `PW_*`); eval_b-vs-eval_a gap (memorisation vs generalisation); the per-tier transition
   shape (did the copy circuit ever turn on?).
2. Rule out artifacts: enough training (the schedule-free early-stop gotcha), all-tier confounding,
   the window-triage trap.
3. **Tokenizer-side interventions** (the only live lever): does BPE merging content+structural atoms
   muddy the content tier? revisit learnability ordering (DEF→REF / causal-DAG); is the
   note_index/note_table pitch two-layer learnable; are KEYFRAME segments sufficient for
   durations/intervals.
4. Re-run `learnability_triage` at seq_len 8192 to rank encodings before more training. If
   tokenizer-side *also* can't lift content past the baseline, escalate to a content-layer
   representation redesign (or re-scope cross-composer content as out-of-envelope).

Carry-over: **all-tier val_acc is CONFOUNDED** across tokenizations — content-tier is the verdict.
Within-tune `--mode window` triage credits trivial redundancy. Runnable specs: `generalize`,
`generalize_prodlike_unigram`, `memorize`.

## Packages

- **`preframr` 0.2.27** — framework (train/inference/model/args/parse). Docker image
  `anarkiwi/preframr` (no PyPI). **Release = merge to `main`** (`release.yml` fires on main-push
  AND `v*` tags, `push:true` → `:VERSION`+`:latest`); also `git tag -a vX.Y.Z` each release.
  Floors `preframr-tokens>=0.48.0` + `preframr-audio>=0.5.9` (in `requirements.txt`,
  `predict-requirements.txt`, `jetson/predict-requirements.txt`). Tier instrumentation is event-aware
  via tokens-side `events/dataset.events_alphabet()` (value-digit atoms→content, markers→structural).
- **`preframr-tokens` 0.48.0** (PyPI; canonical repo `/scratch/anarkiwi/preframr-tokens`,
  gen2 merged in) — torch-free parser/tokenizer. Event model per the banner. 0.48.0 = ~36%
  faster warm parse + dead-wood removal (old generator/macro path trimmed; event API intact). Measured + rejected
  (don't re-propose): §8.4 joint freq/note DP, §2.7 mixed-radix ORDER-DT, DT-in-ticks, POLY degree
  cap, mid-note R-only NOTE_ON fold.
- **`preframr-audio` 0.5.9** (PyPI) — SID rendering + chip-semantics canonical reference
  (`test_gate_adsr_reference`, `test_adsr_write_liveness_matrix`, `test_release_write_position`).
  Envelope/canonicalization questions are answered from these tests, not ad-hoc probes.
- **`preframr-experiments`** (this repo; editable/PYTHONPATH, no PyPI) — runner + specs + `audit/`.
  Orchestration runs on the host (no torch); audits import preframr/torch and run in the xpt image.

Sibling repos: `/scratch/anarkiwi/preframr-{audio,tokens,xpt,aug}` + `/scratch/anarkiwi/preframr`.
Release/build/test/cache authority: **`design/references/release_build_cache.md`**. Two standing
rules: **run non-GPU work (builds, parse, audits, pytest, lint) on `fogbank`** (72 cores, keep
defroster for training); when releasing the Docker app, build locally in parallel with the push.

## Project goal (OVERRIDING) + learnability lens

Train a SID model that **generalises** — predicts unseen continuations from arbitrary mid-song
prompts, across composers (primary `val_acc`), ideally across engines. Envelope: **train** single
RTX 4090 24 GB (specs needing >~50M body to show Δ are out-of-envelope — refute in design, don't A/B);
**predict** Jetson Orin NX. Real-time verdict measured (`design/performance/orin_inference_optimization_design.md`):
single-stream at the quality tier is ~9× short of real-time (213 tok/s needed); offline auditions
fine (~6.5 min/song). The lens is **LEARNABILITY**: generalisation is won when the *encoding* lets a
bounded (~TC⁰) transformer cheaply predict the next token — minimise causal-state + horizon, prefer
induction-head copy over implicit counters, order by the driver causal DAG. The event model is the
direct product of this lens; the lever is tokenizer-side. Hub:
`design/references/learnability_token_ordering_theory.md`.

## Tests + runner

- **Framework**: `./run_tests.sh` (black, pytest, pylint, pyright, coverage ≥77).
- **xpt**: `pytest tests` at image build (`docker.yml`, push to main + PRs). Host CLI (no torch):
  `PYTHONPATH=. python3 -m preframr_experiments.run <spec> --root <work> [--tkvocab N ...]`.
  One spec module per A/B under `specs/`; runner stages data → parse → tokenize → train per
  (arm, seed) in `docker run` of `spec.image` (via `_docker_run`, which now sets `RUST_MIN_STACK`).
  `nohup … & disown` for long runs.
- **Macro passes = empty in practice.** The `Arm(macro_flags=…)` machinery survives registry-driven,
  but the event encoding is unconditional, so `full_macros`-vs-`baseline` is degenerate. The levers
  are BPE vocab/merges + embedding/conditioning, not flags.
- **Spec-dependent tokenization** (pre_run_hook mutating dumps): `PREFRAMR_DATASET_CACHE_DISABLE=1`.
- **Content-tier audit (decisive gate).** Per arm-seed, in the xpt image:
  `audit_checkpoint_per_class --ckpt … --work-dir <seed0> --out audit_per_class.json` — supports
  **`--device cpu`** (run audits without GPU contention with a live training job). Then host-side:
  `python3 -m preframr_experiments.audit.content_tier_report --results-root <dir>`. Per-tier
  `content_over_structural` is meaningful on event runs; the **by-op spotlight (op45) is moot under
  events** (no FREQ_TRAJ) — per-tier read only. Readers indexed in `audit/README.md`.

## Conventions

- **Code = frozen baked image by default.** Runs use baked `preframr/`; rebake to pick up edits.
  Bind-mount is opt-in (`--bind-src` / `$PREFRAMR_BIND_SRC=1`) and runs un-gated code — ask first.
  The baked `anarkiwi/preframr:0.2.27` is event-model-current (no bind / cache-disable needed).
- **Background runs**: `nohup`+`disown`; don't poll, use `ScheduleWakeup` or a tracked wait.
- **Comments**: no narration / dev-local paths / PR numbers; `tests/test_lint.py` rejects narrative
  `#` and >5-line docstrings (gen2 enforces the same gate).
- **NFS hygiene**: **fogbank IS the `/scratch` NFS server**; defroster mounts it `hard`, so heavy
  fogbank-local load overlapping a defroster parse can saturate `nfsd` → defroster D-state hang →
  reboot. Cap fogbank pools; canary defroster with `stat -f /scratch`. No lingering `tail -f` on
  workdir files; stop `preframr_tb` before deleting tb_logs subtrees.
- **Arm ordering**: target arm first, baseline last (runner is seed-major).
- **Renaming a transform** silently disables it in stale specs — grep specs on any rename.
- **Design docs** live in `design/`, indexed by axis in `design/README.md`; ship → `design/landed/`,
  rejection → `data/refuted/<exp>.md`.

### Wallclock anchors (event model, re-anchored)
parse ~1–2 min (cached) · `encode(verify=True)` over 856 dumps **~33 min** (the new bottleneck;
self-verify doubles work by design) · canonical body train ~1.5 min/epoch · early-stop bounded.

## Refuted alternatives

Registry: `data/refuted/<exp>.md`. Model-side interventions concentrated at the same ~0.13 eval_a
content ceiling (since lifted by tokenizer-side representation): `per_tier_heads_*`,
`mask_structural_loss`, `cluster_conditional_content_head`, `content_diffusion`,
`contrastive_infonce_auxiliary`, `motif_pass`, `weighted_token_loss`, `voice_trajectory`,
`set_to_diff`, and earlier nulls. Tokens-side rejections in gen2 STATUS (see Packages).

## Resolved log (compact; full detail in git log + design/landed/ + data/refuted/)

- **2026-06-12 (later)** — **release cascade + canonical 14M relaunch.** Stopped the `_v1`
  `generalize --tkvocab 2048` run; released **tokens 0.48.0** (~36% faster warm parse + dead-wood
  removal; tag `v0.48.0`) and **audio 0.5.9** (SID API-reference docs/tests; tag `v0.5.9`) to PyPI;
  rebaked **preframr 0.2.27** (floors `tokens>=0.48.0` / `audio>=0.5.9`; cuda build gate green on
  fogbank; pulled to defroster; `:latest`→0.2.27) + xpt `ARG BASE=0.2.27`. De-risked the tokens
  dead-wood removal: all 27 framework-imported symbols resolve against the 0.48.0 wheel (the
  `transforms_*_bit_exact` are submodule files, not package attrs — no framework code change).
  Relaunched the canonical 14M run into `_v2` (re-tokenized on 0.48.0).
- **2026-06-12** — **first event-model training runs.** `generalize` launched; flushed out the
  UnigramTrainer SIGSEGV (recursive `Rc::drop` stack overflow on long sentences) → fixed with
  `RUST_MIN_STACK` in `_docker_run`; confirmed at full scale. First content-tier numbers (atoms-only
  baseline) above the old 0.13 ceiling + generalize across composers. Event model SHIPPED earlier
  same day: built `event_gate.py` (train→generate→decode green), released tokens 0.47.0 / audio 0.5.8
  / preframr 0.2.26 (tag `v0.2.26`), xpt on `main`.
- **2026-06-11** — v3 canonical contract (`canonical_writes` oracle; NOTE_ON envelope lifecycle;
  recorded gate-edge sides; typed nibbles, BE varints, KEYFRAME). ADSR fully characterized as a
  24-test reference in preframr-audio. Wire format changed twice (cache-busting).
- **2026-06-08** — event-model (Option B) transition staged; 30 dead specs deleted; event-aware tier split.
- **earlier** — generator pipeline / instrument collapse / byte-exactness / arch exoneration. See git log.
