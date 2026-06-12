# Operational notes for agents (preframr-xpt)

This repo is the **experiment surface**: docker-driven runner + spec registry +
audits + design docs + tier data + refuted registry. Framework, libraries, corpus
live elsewhere (sibling repos under `/scratch/anarkiwi/`).

## State (2026-06-12)

The **event/tracker model is SHIPPED and end-to-end GREEN**: tokens 0.47.0 / audio
0.5.8 (PyPI), preframr 0.2.26 (Docker Hub `:0.2.26`+`:latest`, tag `v0.2.26`), xpt on
`main`. The `memorize` build-gate runs train→generate→decode via event-native
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
on event tokens** (scientific, not operational). First runs of the `generalize` spec are
executing. Findings so far:

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
- **First content-tier numbers (atoms-only `tkvocab=0` baseline, epoch ~28, still training):**
  eval_a content **0.33**, eval_b_daglish **0.35**, eval_b_follin **0.27**, content/structural
  0.74–0.90. Above the old **~0.13 eval_a content ceiling**, and held-out composers track
  in-distribution (no generalization collapse). Caveats: early (run unfinished), this is the
  **no-dictionary floor**, single-seed, 24-block sample, and the content-tier *definition* under
  events (value-digit atoms = op0) differs from the old substrate — so 0.33-vs-0.13 is
  directional, not apples-to-apples.

### NEXT (in order)
1. **Re-run `generalize` with a realizable `tkvocab`** (e.g. 2048) now that `RUST_MIN_STACK` is
   plumbed → get the **dictionary-run** content-tier numbers to compare against the atoms-only
   baseline. (This is the real BPE-vocab lever; the baseline is just the vocab=0 anchor.)
2. Re-point `learnability_triage` at the event stream at **seq_len 8192** (prodlike static read;
   mini 4096 mode-collapses — plumbing only), then write + run the canonical event-model spec.
   Levers: BPE merge count / vocab size, typed-nibble embedding treatment, KEYFRAME conditioning.
   NOT a macro-pass A/B (no flag surface exists). Gate on per-tier `content_over_structural` +
   per-op acc over event KINDs + `eval_b_*` held-out composers.

Carry-over: **all-tier val_acc is CONFOUNDED** across tokenizations — the content-tier read is the
verdict. Within-tune `--mode window` triage credits trivial redundancy — not the verdict.
Surviving runnable specs: `generalize` + `memorize` (build-gate, encoding-agnostic).

## Packages

- **`preframr` 0.2.26** — framework (train/inference/model/args/parse). Docker image
  `anarkiwi/preframr` (no PyPI). **Release = merge to `main`** (`release.yml` fires on main-push
  AND `v*` tags, `push:true` → `:VERSION`+`:latest`); also `git tag -a vX.Y.Z` each release.
  Floors `preframr-tokens>=0.47.0` (only `requirements.txt`). Tier instrumentation is event-aware
  via tokens-side `events/dataset.events_alphabet()` (value-digit atoms→content, markers→structural).
- **`preframr-tokens` 0.47.0** (PyPI; canonical repo `/scratch/anarkiwi/preframr-tokens`,
  gen2 merged in) — torch-free parser/tokenizer. Event model per the banner. Measured + rejected
  (don't re-propose): §8.4 joint freq/note DP, §2.7 mixed-radix ORDER-DT, DT-in-ticks, POLY degree
  cap, mid-note R-only NOTE_ON fold.
- **`preframr-audio` 0.5.8** (PyPI) — SID rendering + chip-semantics canonical reference
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
  The baked `anarkiwi/preframr:0.2.26` is event-model-current (no bind / cache-disable needed).
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
