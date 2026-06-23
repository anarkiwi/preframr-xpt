# Operational notes for agents (preframr-xpt)

Experiment surface: docker runner + specs + audits + design docs + refuted registry. Framework,
libraries, corpus live in sibling repos under `/scratch/anarkiwi/`.

## HARD RULE #0 — nothing is "irreducible" (read before any encoding work)

A SID trace is the output of a tiny deterministic playroutine. **You may NOT conclude a run is
"irreducible" / "random" / "incompressible" / "store the data" / "minimal recurrence = its length".**
That has been wrong every time — the **transposition trap recurred 4×**: per-note `micro` = the note
table; arp = semitones; the period-32 Monty bass = a 5-note pattern × 434 repeats; "pw table data" =
sweeps. Every "distinct body" was a few instruments rendered at many pitches; the error is judging a run
IN ISOLATION by ONE model class, blind to notes + repetition. **Before leaving any run as opaque, run the
3-step falsification protocol (mandatory, report it):** (1) decompose the period's VALUES into pitches —
few distinct = NOTES; (2) check if the cycle recurs elsewhere — yes = backward-reference (phrase-LZ);
(3) trace it to the driver source. Only after all three is it per-tune note DATA — and that still repeats.
There is no "irreducible" category. (Memory: `no-irreducible-runs`.)

**Two proofs pin this.** (a) *A Mind Is Born* — 256 bytes of arbitrary 6502 (no note table, no patterns):
the whole 8,193-frame song encodes byte-exact at **0.42 token/frame** by recovering its generators
(period-32 accumulator melody, +1-every-128 filter ramp, drum lane); LZMA of the 205k raw values floors
far above. If hand-written synthesis collapses to 0.42, nothing a human composed in a tool is a wall.
(b) *JCH NewPlayer* — "718 distinct pitches" is one `$60` slide command integrated:
`freq = note_table[arp] + porta_acc + vibrato_acc`. "Irreducible" hides in a TRACER capture gap as often
as a codec gap. Recover the generator; never compress the output.

**A wall always turns out to be one of:** a general compressor used as oracle (LZMA can't represent an
accumulator ramp — the floor is the PLAYER, hundreds of bytes); a framing/denominator bug
(Master_Composer "4.24" was 234 frames vs the 2280 raster denom → 0.525 byte-exact); or a non-canonical
cover that defeats REPEAT (92%-repeated phrases mis-segmented per occurrence). Check those before writing
"entropy".

ENFORCEMENT (mechanical — prose doesn't propagate; a subagent still called 7/12 tunes a "wall" this
session): `tools/codec_gate.py` defines "done" = residual-0 ∧ < 1 token/frame ∧ ~0 literal-floor (a FAIL
on token/frame = UNRECOVERED STRUCTURE, never a wall); the `codec_guard.sh` SubagentStop hook rejects a
finish that left scratch or claimed "irreducible/entropy/near-digi" without the falsification keywords;
bake this rule into every codec-subagent prompt.

## GOAL + current state — one generic recovery for the whole tracker zoo

**The gate: residual-0 (byte-exact) AND < 1 token/frame AND < 5s CPU per song**, by recovering the
PROGRAM, never compressing the trace. Achieved on one driver; being generalised to all of HVSC.

**Representation — one common tracker IR.** Every non-digi tune lowers into the SAME shape: a shared
pitch-invariant instrument pool + per-voice note rows `(note, instr, command, dur)` + an orderlist with
backward REPEAT/TRANSPOSE. Pitch = the absolute 12-TET A440 grid (one cross-driver alphabet). Recovered
byte-exact from the distill artifact; the output-fit generator COVER is the additive fallback for
genuinely algorithmic tunes (A Mind Is Born → 0.42, njit cover ~4.2s). Shipped: structure recovery
(JCH 6825→1225, **0.89 token/frame byte-exact**; 14/20 corpus drivers < 1).

**The zoo collapses to a handful — the 645-tracker RE corpus proved it.** `deplayroutine` RE'd every
unique HVSC tracker into `/scratch/preframr/re-trackers` (645 drivers + siddump oracles); the survey
(`re-trackers/SURVEY.md`) collapses them to: **1 freq model** (the integrator `note_table[arp] + porta +
vibrato`, 86% of HVSC) · **3 pointer-table packings × a relocation flag** (interleaved-stride-2 77% /
split / packed) · **4 row-grammar dialects** · **~3 orderlist conventions** · **2 instrument idioms × ~5
strides** · **2 tempo primitives** · **+ ~75 digi outliers, carved out**. Concentration: **top-6 drivers
= 59% of HVSC, top-30 = 82%, top-100 = 93%.** Not 645 techniques — a handful.

**The recovery — ONE validation-gated pipeline** (`re-trackers/GENERIC_RECOVERY_DESIGN.md`), njit-first,
no per-driver branches: digi carve-out → relocation resolve → IDXR-driven table discovery →
orderlist+pattern decode (the 4 dialects via one parameterised skeleton) → instrument table → integrator
+ accumulator-fit → IR. Every variant (packing, reloc delta, dialect, stride) is chosen by the
**byte-exact round-trip** as the universal selector. The expensive structure-discovery is OFFLOADED into
the C++ tracer: `preframr-sidtrace` emits IDXS/PWLK/RELO/SDAC/DIGI/TMPO (the resolved orderlist→pattern
pointer walk, the relocation delta, the scaled-index fix, accumulator addends, the digi signature) so the
recovery stays O(bytes), not O(image²). On `preframr-sidtrace` origin/main (`f1561df`), byte-exact
verified. Recovery in flight, flipping the last 6 xfail drivers (Digitalizer / RoMuzak / TFX / 20CC /
MoN-FC / DMC — each a detection or grammar gap the RE diagnosed, never a wall).

**Gates:** `preframr-tokens` `tests/test_corpus_budget.py` (20-driver parametrized), `test_amib_generic_budget.py`
(anti-wall < 1), `test_recover_timing.py` (< 5s CPU), `tools/codec_gate.py`. **End goal:** every non-digi
HVSC tracker byte-exact + < 1 token/frame, then re-encode the corpus + train. Design: `re-trackers/`
(`SURVEY.md`, `GENERIC_RECOVERY_DESIGN.md`) + `design/encoding/{generic_generator_state_machine.md,
sidtrace_accumulator_capture.md}`.

## Hard rules (operator) — do not violate

- **residual = 0 is the GATE.** The op-set is PROVEN complete, so an un-recovered run or non-zero residual
  = a MISSING PARAMETER of a known generator, never a new op. Explain it against the bus / driver source;
  never reclassify, average, or fall back to a delta-run; never proceed past it — **STOP and diagnose.**
- **< 1 token/frame is co-equal with residual = 0.** Byte-exact-but-dense is a failure (it stores
  playback, not the program). Recover the generative structure: steps + pitch-invariant instruments +
  backward repetition. Both gates, always.
- **Recover the GENERATOR, don't curve-fit the OUTPUT.** Read the RAM state-variable's per-frame update
  from the bus (driver source + the exception explainer); never RLE / compress the delta stream.
- **Model-facing form = INLINE STREAMING.** Any prefix is a valid continuable song; no preamble, no frozen
  tables. Reuse is backward-looking only (the inline orderlist / induction), never a forward declaration.
  Codebook / DEF→REF as a forward declaration is refuted.

## Project goal + learnability lens

Train a SID model that **generalises** — predicts unseen continuations from arbitrary mid-song prompts,
across composers (primary `val_acc`), ideally across engines. Ultimate goal: generation from diverse
prompts (short MIDI/keyboard phrase → arranged SID tune; `design/generation/`). Envelope: **train** single
RTX 4090 24GB (specs needing >~50M body to show Δ are out-of-envelope — refute in design, don't A/B);
**predict** Jetson Orin NX (offline auditions only). Lens = **LEARNABILITY**: minimise causal-state +
horizon, prefer induction-head copy over implicit counters, order by the driver causal DAG. The < 1
token/frame goal is this lens made concrete — the sparsest stream is the one closest to the generating
program. Hub: `design/references/learnability_token_ordering_theory.md`.

## Packages

- **`preframr`** — framework (train/inference/model/args/parse), Docker `anarkiwi/preframr` (no PyPI).
  Release = merge to `main` (`release.yml` + `v*` tag → `:VERSION`+`:latest`). **Ported to the BACC
  sid-only codec** (`recover_from_sid`, no `.dump.parquet`): parse consumes a `(.sid, subtune)`
  manifest (`--manifest`/`--sid-root`/`--songlengths`); the image bundles the `preframr-sidtrace`
  binary (`SIDTRACE_BIN`).
- **`preframr-tokens`** (PyPI; `/scratch/anarkiwi/preframr/preframr-tokens`) — torch-free
  step/tracker BACC codec. Public sid-only path: `recover_from_sid` (driver=`generic`) →
  `program_to_ids`/`measure`/`ids_to_program` (all dispatch the generic serializer). VOCAB 34, PAD 34.
- **`preframr-audio`** (PyPI) — SID render + chip-semantics reference tests (ADSR / write-liveness /
  release-position); answer envelope questions from these, not ad-hoc probes.
- **`preframr-experiments`** (this repo; PYTHONPATH, no PyPI) — runner + specs + `corpus/` (census +
  tracker-stratified selection) + the 3 decisive `audit/` tools (content-tier, copy-novel,
  free-running-gap).
- **HVSC tracker catalog** (`/scratch/anarkiwi/cbm/hvsc-tracker-catalog/data/results.csv`) — SIDId
  player id for every `.sid` (the ground-truth stratification axis).

**Run all non-GPU work (build, parse, audit, pytest, lint, decompiler) on `fogbank`** (72 cores; keep
defroster for training). Release/build/cache authority: `design/references/release_build_cache.md`.

## Tests + runner

- Framework: `./run_tests.sh` (black, pytest, pylint, pyright, coverage ≥77).
- xpt host CLI (no torch): `PYTHONPATH=. python3 -m preframr_experiments.run <spec> --root <work>`.
  One spec per A/B in `specs/`; runner stages `.sid` + writes a manifest → parse (sid-only recovery →
  `.blocks.npy`) → train per (arm,seed) in `docker run`. Default context = 4096 tokens (whole-song-in-
  context). Spec-dependent parse needs `PREFRAMR_DATASET_CACHE_DISABLE=1`.
- **Corpus = census-driven, tracker-stratified, residual-zero-gated** (`preframr_experiments/corpus/`):
  `census.py` over full HVSC → `select.py` pins tiers (smoke/mini/canonical/frontier) of `.sid`
  entries ≤ 4096 tokens. See `data/README.md`. (The old composer/k-means dump tiers are gone.)
- Decisive gate = `content_tier_report` (+ `copy_novel_audit`, `free_running_gap_audit`) over the
  trained checkpoint's blocks.

## Conventions

- Code = frozen baked image by default; rebake to pick up edits. Bind-mount opt-in
  (`--bind-src`/`$PREFRAMR_BIND_SRC=1`, runs un-gated code — ask first).
- Background runs: `nohup`+`disown`; add progress logging + incremental checkpoints. **Don't
  `pkill -f <script>` from a shell whose own command line contains `<script>`** (self-kill — kill by PID).
- **NFS hygiene**: fogbank IS the `/scratch` NFS server; defroster mounts it `hard`, so heavy fogbank
  load overlapping a defroster parse → D-state hang → reboot. Cap pools; watch `ulimit -n` (1024);
  canary `stat -f /scratch`.
- Comments: no narration / dev paths / PR numbers; lint rejects narrative `#` + >5-line docstrings.
- Design docs in `design/` (indexed `design/README.md`); ship → `design/landed/`, rejection →
  `data/refuted/<exp>.md`.

## Refuted (don't re-propose)

Registry `data/refuted/<exp>.md`. **Model-side interventions** all hit the same ~0.13 eval_a ceiling
and NULLED on free-running generation: `per_tier_heads`, `mask_structural_loss`, `content_diffusion`,
`contrastive_infonce`, `motif_pass`, `weighted_token_loss`, `voice_trajectory`, `set_to_diff`, Tier-3
augmentation, DAgger obj-1/obj-2, melody/timbre factorization, instrument DEF→REF. **Encoding shapes
refuted:** GoatTracker-as-target (one-driver cage), any invented op-set + residual escape lane,
frozen-table/codebook DEF→REF, frame/event-codec density compression (BPE / boundary dictionary) as the
context lever. Root cause of the old model-side null: the frame/event codec **signal-fits a dense trace**
— the step/tracker codec is the representation-level fix that reached < 1 token/frame.
