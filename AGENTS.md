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

**Two proofs pin this.** (a) *A Mind Is Born* — 256 bytes of arbitrary 6502 (no note TABLE, no pattern
bank, but real notes + instruments): recovered byte-exact as notes + instrument GENERATORS (period-32
accumulator melody, the +2/period-128 PW sweep, a filter ramp, the 32-frame drum) in the flat vocab — a
prototype reconstructs all 25 registers residual-0 from ~3.6k atoms (**~0.56 token/tick**); LZMA of the
205k raw values floors far above. The per-frame output-fit cover is REMOVED — A Mind is music, not a dump.
If hand-written synthesis collapses to notes+instruments, nothing a human composed in a tool is a wall.
(b) *JCH NewPlayer* — "718 distinct pitches" is one `$60` slide command integrated:
`freq = note_table[arp] + porta_acc + vibrato_acc`. "Irreducible" hides in a TRACER capture gap as often
as a codec gap. Recover the generator; never compress the output.

**A wall always turns out to be one of:** a general compressor used as oracle (LZMA can't represent an
accumulator ramp — the floor is the PLAYER, hundreds of bytes); a framing/denominator bug
(Master_Composer "4.24" was 234 frames vs the 2280 raster denom → 0.525 byte-exact); or a non-canonical
cover that defeats REPEAT (92%-repeated phrases mis-segmented per occurrence). Check those before writing
"entropy".

ENFORCEMENT (mechanical — prose doesn't propagate; a subagent still called 7/12 tunes a "wall" this
session): `tools/codec_gate.py` defines "done" = SID-output equivalence (residual-0 mod don't-care) ∧ the
anti-Goodhart structural constraints C1–C8 (per-LANE efficiency, O(1) generators, no LZ, derived-tick
denominator, render-from-tokens-alone, note falsification, byte-atom cap — `FLAT_VOCAB_MIGRATION.md`).
`token/tick` is a REPORTED metric, not a gate (it was hit pathologically — a per-frame dump relabeled);
an alert at > ~1 = UNRECOVERED STRUCTURE, never a wall. The `codec_guard.sh` SubagentStop hook rejects a
finish that left scratch or claimed "irreducible/entropy/near-digi" without the falsification keywords;
bake this rule into every codec-subagent prompt.

## GOAL + current state — one generic recovery for the whole tracker zoo

**The gate: SID-output equivalence (residual-0 byte-exact, modulo each backend's declared don't-care mask)
AND the anti-Goodhart structural constraints C1–C8 AND < 5s CPU per song**, by recovering the PROGRAM,
never compressing the trace. `< 1 token/tick` is DEMOTED to a reported metric — a CONSEQUENCE of the
constraints, not a target (the prior codec hit it pathologically). No perceptual / rendered-audio gate
(unreliable); the gate is the 25-register output. Achieved on one driver; being generalised to all of HVSC.

**Representation — one common tracker IR in a FLAT, learnability-first token vocab (v2).** Every non-digi
tune lowers into the SAME shape, all TYPED ATOMS, self-delimiting, INLINE (define-at-first-use + `REF` by
stable id; any prefix is a valid song; **no SEC_* preamble, no LZ, no numeric back-offsets**), context
target **8192 tokens**:
- **Pitch = global NOTE ⊕ per-tune TUNING ⊕ bounded DETUNE** (three orthogonal things). NOTE = the GLOBAL
  canonical A440 12-TET grid index (same token across every song → melody is learnable corpus-wide).
  TUNING = how NOTE resolves to the exact Fn, ONE per tune (most trackers a single constant offset; a code-
  tune like A_Mind a small byte-granular table). DETUNE = a small BOUNDED (≤±50c) learnable expression
  param (per-voice texture / just-intonation), never folded into NOTE, never unbounded.
- **Instruments = parametric GENERATORS** (`GEN_*` = HOLD/RAMP/QUAD/VIBRATO/ARP/TABLEWALK; **PW & filter
  ARE `GEN_RAMP`**, never stored output), defined once, referenced by notes.
- **Note rows** `(NOTE, INSTR_REF, DUR, effect*)` on a ROW/PATTERN grid; **time is the row grid** (durations
  implicit / small), **no `nframes`, no wide 16-bit fields, no escapes** (C8 numeric policy). Initial state
  defaults to ZERO (no boot dump); a non-zero start (continuation) is a sparse `SEED`.
- **OPTIMIZE pass** after recovery (lossless, SID-output-preserving) canonicalises to minimum causal-state
  (sustain-lift, generator absorption, content-addressed phrase `REF`).
The output-fit per-frame cover is REMOVED: a table-less tune (A_Mind_Is_Born) is recovered as
notes + instrument generators in the SAME vocab.

**Status (design LOCKED + prototype-validated; production migration IN FLIGHT).** The full v2 spec is
`preframr-tokens/.../FLAT_VOCAB_MIGRATION.md` (phases, C1–C8 gate, §2f numeric policy) +
`re-trackers/RETRACKERS_COMPAT_REVIEW.md`. A prototype encoder validated the design on **24 distinct
trackers** (the 6 top + 6 non-tracker + 12 more — DMC/GoatTracker/JCH/FC/Soundmonitor/Music_Assembler/
A_Mind/Laxity/Hubbard/Galway/TFX/20CC/SidWizard/RoMuzak/Huelsbeck/Daglish/…): **24/24 encode < 1 tok/tick**
(0.003–0.83) under global-NOTE + per-tune-TUNING (clustering ~+35c / ~0c, a few −10..−26c) + bounded DETUNE
(±50c) + `GEN_RAMP/VIBRATO/ARP` + inline `PATTERN`/`REF`, no wide values. The production migration (strip
v1 → flat-v2 inline → generic flat → render-from-tokens → OPTIMIZE) is being executed by a `sid-codec`
subagent (branch + PR, merge-on-green). Until it lands, the shipped code is the interim section-layout flat
GoatTracker codec (Grid_Runner residual-0, 9,480 tokens).

**The zoo collapses to a handful — the 645-tracker RE corpus proved it.** `deplayroutine` RE'd every
unique HVSC tracker into `/scratch/preframr/re-trackers` (645 drivers + siddump oracles); the survey
(`re-trackers/SURVEY.md`) collapses them to: **1 freq model** (the integrator `note_table[arp] + porta +
vibrato`, 86% of HVSC) · **3 pointer-table packings × a relocation flag** (interleaved-stride-2 77% /
split / packed) · **4 row-grammar dialects** · **~3 orderlist conventions** · **2 instrument idioms × ~5
strides** · **2 tempo primitives** · **+ ~75 digi outliers, EXCLUDED ENTIRELY** (detected → dropped,
`DigiExcluded`; no representation, no sub-frame primitive, no gate). Concentration: **top-6 drivers
= 59% of HVSC, top-30 = 82%, top-100 = 93%.** Not 645 techniques — a handful.

**The recovery — ONE validation-gated pipeline** (`re-trackers/GENERIC_RECOVERY_DESIGN.md`), njit-first,
no per-driver branches: digi carve-out → relocation resolve → IDXR-driven table discovery →
orderlist+pattern decode (the 4 dialects via one parameterised skeleton) → instrument table → integrator
+ accumulator-fit → IR. Every variant (packing, reloc delta, dialect, stride) is chosen by the
**byte-exact round-trip** as the universal selector. The expensive structure-discovery is OFFLOADED into
the C++ tracer: `preframr-sidtrace` emits IDXS/PWLK/RELO/SDAC/DIGI/TMPO (the resolved orderlist→pattern
pointer walk, the relocation delta, the scaled-index fix, accumulator addends, the digi signature) so the
recovery stays O(bytes), not O(image²). On `preframr-sidtrace` origin/main (`f1561df`), byte-exact
verified. Status (this design): the existing recovery reaches **residual-0 (SID-output equiv) on all 6 top
drivers — DMC / GoatTracker / Music_Assembler / FutureComposer / JCH_NewPlayer / Soundmonitor — AND 6
non-tracker tunes (incl. A Mind Is Born)** (12/12, validated; `re-trackers/RETRACKERS_COMPAT_REVIEW.md`).
The remaining work is the FLAT serializer for the generic path (notes + abstract instruments + `GEN_*` +
orderlist + tempo) per `preframr-tokens/.../FLAT_VOCAB_MIGRATION.md` — recovery is the inherited foundation,
flat serialization is the new target.

**Gates:** `tools/codec_gate.py` = SID-output equivalence ∧ C1–C8 (per-lane efficiency, O(1) generators,
no-LZ, derived-tick denominator, render-from-tokens-alone, note falsification, byte-atom cap);
`test_recover_timing.py` (< 5s CPU). `token/tick` is reported, alerted at > ~1, never a pass/fail. **End
goal:** every non-digi HVSC tracker SID-output-equivalent in the flat vocab, then re-encode the corpus +
train. Design: `preframr-tokens/.../FLAT_VOCAB_MIGRATION.md`, `re-trackers/`
(`SURVEY.md`, `GENERIC_RECOVERY_DESIGN.md`, `RETRACKERS_COMPAT_REVIEW.md`) +
`design/encoding/{generic_generator_state_machine.md, sidtrace_accumulator_capture.md}`.

## Hard rules (operator) — do not violate

- **residual = 0 is the GATE.** The op-set is PROVEN complete, so an un-recovered run or non-zero residual
  = a MISSING PARAMETER of a known generator, never a new op. Explain it against the bus / driver source;
  never reclassify, average, or fall back to a delta-run; never proceed past it — **STOP and diagnose.**
- **Gate on STRUCTURE; `token/tick` is a reported metric (anti-Goodhart).** Byte-exact-but-dense is a
  failure, but `< 1 token/tick` is NOT the test — it was hit pathologically (compressed output / a
  per-frame dump relabeled as "score"). The gate is the structural co-constraints C1–C8: per-LANE
  efficiency (not aggregate); O(1) generators (constant params in ticks, ≤K exceptions, NO raw-sequence
  escape); NO LZ (repetition only content-addressed); tick-count denominator DERIVED per tune (NOT assumed
  PAL/NTSC — single/multi-speed/CIA-timer); render from TOKENS ALONE (no `_state` anchor); note
  falsification (onsets ≪ ticks); byte-atom fraction cap. Under these, `< 1` falls out honestly.
- **Recover the GENERATOR, don't curve-fit the OUTPUT.** Read the RAM state-variable's per-frame update
  from the bus (driver source + the exception explainer); never RLE / compress the delta stream. A
  "generator" whose params scale with ticks is a disguised dump (C2).
- **Model-facing form = INLINE STREAMING (tenet STANDS).** Any prefix is a valid continuable song; no
  preamble / no frozen tables; reuse is backward-only via the inline orderlist / induction; codebook /
  DEF→REF forward declaration stays REFUTED. The flat vocab is laid out INLINE: **define-at-first-use +
  reference-by-stable-id** — when the orderlist first reaches pattern P, emit P's rows inline; later
  occurrences emit `REF P`. `REF P` is a NAME (content-addressed), NOT a numeric back-offset (so C3/no-LZ
  holds) and its def is always to the left (so prefix-validity holds; `REF P` after P's rows IS the
  induction-head copy). Motivation: context = **8192 tokens** and songs overflow it (Grid_Runner flat =
  9,480) — a section preamble would put the orderlist past the window and strand the back half; inline,
  the first 8192 tokens are valid music from the start. The shipped GoatTracker flat codec is
  section-layout and must be **re-laid-out inline** (migration task, not a redesign).
- **Initial state defaults to ZERO; don't store the all-zero boot.** The decoder assumes all 25 SID
  registers = 0 at tick 0; a fresh tune emits NO boot/init section (the old `boot[25]`/`boot1[25]` dump is
  removed — t=0 literal floor + a render-alignment crutch that render-from-tokens moots). A non-zero start
  (a continuation / window entry — voices mid-note, accumulators mid-phase) is an OPTIONAL sparse
  `SEED (reg val)*` of only the non-zero regs; a fresh tune's volume/filter/ADSR inits are first program
  events of their lanes, not seed entries. SEED is never a 25-reg re-dump (boot by another name).
- **OPTIMIZE pass after recovery (learnability canonicalization).** Pipeline is `recover → OPTIMIZE →
  serialize → gate`. The pass is a **lossless, SID-output-preserving rewrite whose objective is MINIMUM
  CAUSAL-STATE** (the learnability lens), never minimum tokens — it may only make the stream MORE
  structural: lift distracting local repetition to the generator that produced it (N re-struck identical
  rows → one row + duration / a held note; a local wiggle → `GEN_VIBRATO`; a local ramp → `GEN_RAMP`);
  canonicalize equivalent encodings to ONE spelling; factor genuine recurring phrases to content-addressed
  `REF` ONLY when it lowers causal-state (never a numeric copy). It MUST re-pass the gate (residual-0 +
  C1–C8) — so it cannot introduce a literal floor, an LZ offset, or a dangling ref. Distracting local
  repetition (a held note spelled as 4 rows) is removed; genuine structure (a recurring chorus) is kept.

## Project goal + learnability lens

Train a SID model that **generalises** — predicts unseen continuations from arbitrary mid-song prompts,
across composers (primary `val_acc`), ideally across engines. Ultimate goal: generation from diverse
prompts (short MIDI/keyboard phrase → arranged SID tune; `design/generation/`). Envelope: **train** single
RTX 4090 24GB (specs needing >~50M body to show Δ are out-of-envelope — refute in design, don't A/B);
**predict** Jetson Orin NX (offline auditions only). Lens = **LEARNABILITY**: typed-atom alphabet (no
place-value / no LZ offsets), minimise causal-state + horizon, prefer induction-head copy over implicit
counters, order by the driver causal DAG. The flat vocab + the C1–C8 structural gate IS this lens made
concrete — a stream that is genuine notes+instruments (not a compressed dump) is the one closest to the
generating program; sparsity is a consequence, not the target. Hub:
`design/references/learnability_token_ordering_theory.md`.

## Packages

- **`preframr`** — framework (train/inference/model/args/parse), Docker `anarkiwi/preframr` (no PyPI).
  Release = merge to `main` (`release.yml` + `v*` tag → `:VERSION`+`:latest`). **Ported to the BACC
  sid-only codec** (`recover_from_sid`, no `.dump.parquet`): parse consumes a `(.sid, subtune)`
  manifest (`--manifest`/`--sid-root`/`--songlengths`); the image bundles the `preframr-sidtrace`
  binary (`SIDTRACE_BIN`).
- **`preframr-tokens`** (PyPI; `/scratch/anarkiwi/preframr/preframr-tokens`) — torch-free
  step/tracker BACC codec. Public sid-only path: `recover_from_sid` (driver=`generic`) →
  `program_to_ids`/`measure`/`ids_to_program`. **FLAT learnability-first vocab** (`flat_serialize.py`):
  typed atoms + `GEN_*` + no LZ. GoatTracker dispatches the flat codec (shipped); generic flat in flight.
  VOCAB 576, PAD 576 after the Phase-1 widening (currently 544 GT-only; one bump with the v1 strip).
- **`preframr-audio`** (PyPI) — SID render + chip-semantics reference tests (ADSR / write-liveness /
  release-position); answer envelope questions from these, not ad-hoc probes. NOTE: not a codec gate —
  the gate is register-level SID-output equivalence, not rendered-audio / perceptual comparison.
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
  `.blocks.npy`) → train per (arm,seed) in `docker run`. Default context = **8192 tokens** (the target
  window; songs over 8192 — e.g. Grid_Runner flat = 9,480 — rely on the inline layout's prefix-validity,
  so any 8192-token window/prefix is a valid song). Spec-dependent parse needs
  `PREFRAMR_DATASET_CACHE_DISABLE=1`.
- **Corpus = census-driven, tracker-stratified, residual-zero-gated** (`preframr_experiments/corpus/`):
  `census.py` over full HVSC → `select.py` pins tiers (smoke/mini/canonical/frontier) of `.sid`
  entries ≤ 8192 tokens. See `data/README.md`. (The old composer/k-means dump tiers are gone.)
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
— the step/tracker codec (now the flat notes+instruments vocab) is the representation-level fix: a sparse,
structural stream, not a compressed dense trace.
