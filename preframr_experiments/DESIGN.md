# Experimental harness re-design

Goal: turn the existing memorize / generalize / multi-composer scripts
into a single reusable experiment runner so each macro opportunity in
TODO §1 (long-form PLAY_INSTRUMENT, MODE_VOL_REG burst, cross-frame
TRANSPOSE, PWM/FilterSweep palette, fuzzy fingerprint, LoopPass
N-step, global instrument-id space) can be A/B-decided on the SAME
stable data, with the SAME training compute budget, against
COMPARABLE metrics.

Status: design proposal. No code under this design is implemented yet.

---

## What's wrong today

Three issues block reusing the current harnesses for arm-level decisions:

1. **Data is generated, not pinned.** `pick_train_eval_all_goto80.py`
   and `pick_multi_composer.py` enumerate HVSC at run time, so the
   selection drifts every time the corpus mirror moves. Two arms run
   weeks apart aren't necessarily on the same data.

2. **One harness = one config.** `run_*_int_test.sh` bake their flag
   choices in. A/B'ing `--instrument-pass` (the only A/B done at
   scale so far) was wired by adding a single `EXTRA_CARGS` env hook;
   sweeping a parameter like `--instrument-window ∈ {4, 8, 16, 32}`
   means four hand-edited copies of the script.

3. **Metrics are inconsistent.** memorize reads `min_acc` from
   predict's stdout; generalize reads `val_acc` via `check_generalize`
   from TB events; multi-composer adds an encoding-side audit
   (`cross_composer_encoding_audit`). No single artefact summarises an
   arm's behaviour for cross-arm comparison.

## Proposed shape

Three layers, each replaceable independently:

```
preframr_experiments/
    data/                                 # pinned data lists (NEW)
        smoke.list                        # 4-8 SIDs, fast iteration
        canonical/
            train.list                    # ~150-200 SIDs, 5 composers
            eval-A.list                   # in-distribution holdouts
            eval-B-daglish.list           # cross-composer
            eval-B-follin.list
            HVSC_VERSION                  # corpus version pin
    experiments/                          # NEW
        base.py                           # ExperimentSpec + runner
        run.py                            # CLI: run.py <tier> <exp>
        instrument_window.py              # one file per macro opp
        vol_flip_ab.py
        transpose_xframe.py
        palette_pwm.py
        fuzzy_fingerprint.py
        loop_lookahead.py
        global_instr_ids.py
        report.py                         # cross-arm diff renderer
    run_memorize_int_test.sh              # stays; thin wrapper
    run_generalize_int_test.sh            # stays; thin wrapper
    run_multi_composer_int_test.sh        # stays; thin wrapper
```

The three existing `run_*_int_test.sh` scripts become thin wrappers
that invoke the experiment runner with a fixed spec. New
arm-comparison work goes through `experiments/run.py` directly; the
old scripts remain as single-arm entrypoints for the build gate.

---

## 1. Pinned data tiers

### Smoke (`data/smoke.list`)

4-8 SIDs chosen to exercise every macro class the encoder produces.
End-to-end wallclock target **< 10 min per arm** on stock host with a
warm dump cache.

Coverage rubric — at least one SID for each:

| Class                         | Why                                          |
|-------------------------------|----------------------------------------------|
| BACK_REF / PATTERN_REPLAY     | LoopPass + transposed/fuzzy variants         |
| GATE_REPLAY palette           | GateMacroPass; bundle replay                 |
| PLAY_INSTRUMENT body          | InstrumentProgramPass; envelope replay       |
| INTERVAL mirror               | IntervalPass; cross-voice freq link          |
| PWM / FilterSweep burst       | PwmPass / FilterSweepPass                    |
| Long DELAY frames             | Variable-IRQ idle handling                   |
| PAL + NTSC                    | irq mapping divergence                       |

Candidate set (Goto80 + a couple of others, all already cached):

```
MUSICIANS/G/Goto80/Truth.sid           # short, dense, voice-rotated
MUSICIANS/G/Goto80/Acid_10000.sid      # GATE_REPLAY-rich
MUSICIANS/G/Goto80/Skybox.sid          # InstrumentProgramPass
MUSICIANS/G/Goto80/CBM_85.sid          # PATTERN_REPLAY-rich
MUSICIANS/H/Hubbard_Rob/Commando.sid   # long ADSR (long-form PI)
MUSICIANS/G/Galway_Martin/Yie_Ar_Kung-Fu.sid   # arpeggio (xframe TRANSPOSE)
MUSICIANS/T/Tel_Jeroen/Cybernoid_2.sid # NTSC bias
```

Final list pinned to specific SID file SHA256 (HVSC versions filename
collisions are rare but possible) and committed to the repo.

### Canonical (`data/canonical/`)

Stable superset of the current MC pick (5 train composers × 30
in-distribution + 2 Eval-B composers × 16). Pinned at one HVSC
version, regenerated via `pick_multi_composer.py --pin-output` only
at intentional re-curation events.

Wallclock target **60-120 min per arm** (drives the multi-arm
combinatorial tier).

`HVSC_VERSION` file records the corpus version used to generate the
list — current convention is the upstream release tag (e.g. `81`).
A re-pin commit changes both the lists and the version pin together,
documenting which corpus update prompted it.

---

## 2. Experiment spec

Declarative Python module per experiment. Minimal example:

```python
# experiments/instrument_window.py
from experiments.base import ExperimentSpec, Arm

spec = ExperimentSpec(
    name="instrument_window",
    doc=(
        "Sweep --instrument-window over {4, 8, 16, 32}. Tests the "
        "long-form PLAY_INSTRUMENT hypothesis: Hubbard / Galway-style "
        "16-32-frame envelopes are truncated at the default 8."
    ),
    tier="canonical",
    arms=[
        Arm(label="iw04", extra_cargs="--instrument-window 4"),
        Arm(label="iw08", extra_cargs="--instrument-window 8"),  # baseline
        Arm(label="iw16", extra_cargs="--instrument-window 16"),
        Arm(label="iw32", extra_cargs="--instrument-window 32"),
    ],
    metrics=[
        "alphabet_size",
        "play_instrument_count",          # extra: PI-op share of vocab
        "encoded_tokens_per_song",
        "val_loss_best",
        "val_acc_at_best_loss",
        "eval_b_encodability",
    ],
    seeds=3,                              # 3 for canonical, 1 for smoke
)
```

`Arm` carries:
- `label` — short identifier used for output dirs and report columns
- `extra_cargs` — flag string forwarded to parse + tokenize + train + audit
- (optional) `training_overrides` — dict of train-only flag changes
  (e.g. `learning_rate`, `max_epochs`). For ablations that affect
  the encoder this is empty; for ones that change train compute it
  documents the divergence.

`ExperimentSpec` carries:
- `name`, `doc` — docstring renders into the report.
- `tier` — `smoke` or `canonical`. Selects data list + default seeds.
- `arms` — list of arms.
- `metrics` — superset of metrics to capture for this experiment.
  All arms compute the same set so the report is rectangular.
- `seeds` — passed as `--seed` to train; metrics are reported as
  `mean ± std` across seeds when > 1.
- (optional) `pre_run_hook` — for experiments that need extra
  preparation (e.g. write a custom flag file) before parse.
- (optional) `derived_metrics` — closures `(arm_artefacts) -> float`
  for experiment-specific extras (e.g. `play_instrument_count` reads
  `tokens.csv` and counts rows where `op == PLAY_INSTRUMENT_OP`).

The spec is the SINGLE place that defines what "this experiment
means". Everything downstream (data, runner, metrics, report) reads
from it.

---

## 3. Runner

```bash
# Smoke pass for fast iteration (single seed, smoke data).
preframr_experiments/run.py smoke instrument_window

# Canonical pass for the decision (3 seeds, canonical data, ~6 hr).
preframr_experiments/run.py canonical instrument_window
```

Per-arm output layout:

```
${ROOT}/results/instrument_window/
    iw04/
        seed0/
            tokens.csv                    # alphabet at this arm
            df-map.csv
            tb_logs/                      # full TB events
            metrics.json                  # arm-derived metrics
            train.log
        seed1/
        seed2/
    iw08/...
    iw16/...
    iw32/...
    report.md                             # cross-arm diff
    report.json                           # machine-readable
```

`report.md` renders one row per arm with metric columns, plus a "delta
vs baseline" column when one arm is labelled `baseline`. JSON sidecar
is for downstream consumers (e.g. plotting, regression detection).

### Resumability

Each arm × seed reads cached parse + tokenize artefacts when present,
so re-running with `--resume` only re-runs failed (arm, seed) pairs.
Caches keyed on the arm's `extra_cargs` string + seed; flag changes
invalidate downstream artefacts automatically.

### Parallelism

Smoke tier runs arms sequentially by default (each arm is < 10 min;
parallelism complicates GPU contention). Canonical can opt into
arm-level parallelism via `--max-parallel-arms N` when the host has
multiple GPUs, with arm-level locking on `${ROOT}/locks/`.

---

## 4. Metric set

Common to every experiment:

| Metric                       | Source                                  |
|------------------------------|-----------------------------------------|
| `alphabet_size`              | `wc -l tokens.csv`                      |
| `encoded_tokens_per_song`    | `df-map.csv` n-tokens column / row     |
| `train_loss_final`           | TB events                               |
| `val_loss_best`              | TB events, min over epochs              |
| `val_acc_at_best_loss`       | TB events at best-val_loss epoch        |
| `epochs_to_best_val_loss`    | TB events                               |
| `wallclock_train_min`        | runner timing                           |
| `eval_b_encodability`        | cross_composer_encoding_audit summary  |
| `palette_growth_rate`        | parse log: gate / instrument palettes  |

Experiment-specific extras (declared by the spec):

- `play_instrument_count` — `tokens.csv` rows with `op = PLAY_INSTRUMENT_OP`
- `pattern_replay_count` — same, op = `PATTERN_REPLAY_OP`
- `back_ref_max_dist` — max BR-DIST val seen
- `instrument_window_used` — Phase-1-captured program lengths' P95
- `validator_rejection_rate` — count of validator failures during preload
- (any additional float a derived-metric closure can compute)

The runner refuses to start an experiment whose declared metrics don't
all have a registered extractor — fails-loud on typo'd metric names.

---

## 5. KPI policy

When arms disagree across metrics, decisions resolve in this order:

1. **`val_acc_at_best_loss` is the primary KPI.** Top-1 token-prediction
   accuracy at the checkpoint that minimised val_loss; mirrors what
   `ModelCheckpoint(monitor='val_loss')` selects, so it tracks the
   deployed-checkpoint quality directly.
2. **`val_loss_best` is secondary** — a tie-breaker when val_acc Δ is
   within seed σ, and a convergence-speed proxy via
   `epochs_to_best_val_loss`. NOT a winner metric on its own.
3. **`alphabet_size` and `encoded_tokens_per_song` are constraints,
   not goals.** A val_acc win doesn't justify alphabet growth that
   breaks the Jetson predict path (see `AGENTS.md`). Material
   trade-offs go in the experiment's report.
4. **Wallclock is a tie-breaker for production-default decisions
   only.** Don't trade quality for minutes per training run.

### Why val_acc, not val_loss

Cross-entropy and top-1 accuracy can move opposite directions: a
finer-grained alphabet (e.g. `--cents 25`) lowers loss because wrong
predictions are "closer" in distribution but harder to be exactly
right on, so accuracy degrades. The 2026-05-10 `cents_sweep` mini run
showed c25 vs c50 at Δval_loss = -1.08 (~3σ) but Δval_acc = -0.0036
(~4σ against c25). Predicting the exact next super-token is what
the model does at inference; "close in distribution" isn't a thing
the renderer can consume.

### Mini-tier σ (from `mini_baseline_seeds`, n=5, 2026-05-10)

| metric | mean | σ | ~2σ Δ-credibility threshold |
|---|---|---|---|
| `val_loss_best` | 14.38 | 0.330 | 0.66 |
| `val_acc_at_best_loss` | 0.0519 | 0.0009 | 0.002 |
| `epochs_to_best_val_loss` | 80.00 | 0.000 | n/a (clipped) |

**Stale**: this table was calibrated at body=small (~1.8M body
params). `mini_capacity_diag` (2026-05-10) found the mini-tier
ceiling capacity-bound, and the mini-tier default is now
``mini_train_args(body="large")``. Awaiting refreshed σ from a
`mini_baseline_seeds` re-run at body=large; this row stays here
until that lands. Canonical-tier σ TBD — first canonical run with
`seeds≥3` writes its own table here.

### Decision template (use in report.md)

> Primary KPI Δval_acc = X (vs σ=...) → win/null/loss.
> Secondary Δval_loss = Y (vs σ=...) → agrees / disagrees.
> Constraint Δalphabet_size = Z → within / outside budget.
> Recommendation: …

---

## 6. Migration plan

Phase 1 — pin and infrastructure. No new experiments.

  1. Materialise `data/smoke.list` and `data/canonical/{train,eval-A,
     eval-B-*}.list` from the current pickers; commit. Add
     `data/canonical/HVSC_VERSION` recording the corpus version.
  2. Implement `experiments/base.py` with `ExperimentSpec` /
     `Arm` / runner / metric registry.
  3. Implement `experiments/report.py` (markdown + JSON).
  4. Add a regression test against a tiny fixture exercising the
     spec/runner contract end-to-end.

Phase 2 — re-express existing harnesses as specs.

  5. `experiments/memorize.py` (smoke tier, 1 arm, gate = MIN_ACC).
     `run_memorize_int_test.sh` becomes a 5-line wrapper that calls
     `run.py smoke memorize`. Existing build-gate behaviour preserved.
  6. `experiments/generalize.py` (canonical tier, 1 arm, gate =
     `MIN_VAL_ACC` once the floor lands per TODO §2). Same wrapper
     pattern.
  7. `experiments/multi_composer.py` (canonical tier, 1 arm, current
     `--no-instrument-pass` config).

Phase 3 — convert the macro-opportunity decisions to arm-level specs.

  8. `experiments/instrument_pass_ab.py` (already half-done via
     `EXTRA_CARGS`).
  9. `experiments/instrument_window.py` — sweep {4, 8, 16, 32}.
 10. `experiments/cents_sweep.py` — sweep `--cents` ∈ {25, 50, 100}.
 11. `experiments/global_instr_ids.py` — once the global-id encoder
     pass exists, A/B against block-local.
 12. `experiments/vol_flip_ab.py`, `experiments/transpose_xframe.py`,
     `experiments/palette_pwm.py`, `experiments/fuzzy_fingerprint.py`,
     `experiments/loop_lookahead.py` — each gated on its encoder
     change landing first; A/B at smoke tier first, canonical when
     the smoke result is promising.

---

## Open questions

1. **HVSC pinning.** The list files reference paths under
   `MUSICIANS/G/Goto80/...` etc. that change occasionally between
   HVSC releases. Options:
   - (a) Pin the corpus version in `HVSC_VERSION` and rely on the
     operator to mirror that version. Simplest; needs a public
     mirror snapshot per pinned version.
   - (b) Cache the actual dump.parquet files alongside the lists in
     the repo (LFS or similar). Largest reproducibility guarantee
     but ~MB-per-SID bloat.
   - (c) Pin SHA-256 of each SID file inline in the list. Falls back
     to `wget` from `hvsc.c64.org` per file if local mirror is stale.
   Recommendation: (a) initially; revisit if cross-host
   reproducibility becomes a pain point.

2. **Compute budget per arm.** Two options:
   - Fixed-epoch (`--max-epochs N`) — comparable across arms but
     wastes compute on already-converged ones.
   - EarlyStopping on `val_loss` (current generalize default) —
     efficient but introduces a hidden axis of variance (one arm
     stops at epoch 50, another at epoch 200).
   Recommendation: EarlyStopping, with `epochs_to_best_val_loss`
   reported as a metric so the variance is visible. A fixed-epoch
   override exists for arms that need it.

3. **Multi-seed for canonical.** 3 seeds gives ~half a standard
   error around each metric on the canonical-tier corpus, which is
   enough to call most macro-opp differences. 5+ adds confidence
   but triples wallclock. Make `seeds` per-spec configurable;
   default to 3 for canonical, 1 for smoke.

4. **Encoder vs trainer-only flags.** Some flags (`--instrument-pass`,
   `--cents`) bake into parsed parquets; others (`--learning-rate`,
   `--label-smoothing`) only affect train. The runner needs to flow
   the right subset to each stage. The current `EXTRA_CARGS` hook
   forwards everything; that works but causes the parser argparse to
   reject train-only flags. The new runner should split flags by
   stage (encoder / tokenize / train / predict / audit) at spec
   parse time, fail-loud on unknown flags.

5. **Combinatorial × multi-arm.** A 4-arm × 3-seed canonical run is
   12 train runs at ~1 hr each = ~12 hr wallclock. Acceptable for
   overnight; not for fast iteration. The smoke tier covers fast
   iteration; canonical-tier results are batched (run once per
   committed encoder change, not per code edit).

6. **Refuted-arm registry.** When an experiment refutes an arm
   (e.g. `--no-instrument-pass` won the Arm-B comparison), record
   the refutation in `data/refuted/<exp_name>.md` with the report.
   Future agents see "this was tried, here's why we don't do it"
   without re-spending compute. Mirror's the existing
   `untracked/{palette_merge,head_row_class,adsr_equivalence}_design.md`
   pattern but in-repo and tied to the experiment artefact.

---

## What this enables

- Each macro opportunity in TODO §1 has a 1:1 experiment spec; the
  decision lives in `report.md` of that experiment, not a session log.
- Smoke tier regressions catch encoder bugs in < 10 min before the
  canonical tier eats compute.
- Cross-experiment metrics are comparable because every spec uses
  the same data + the same metric extractors.
- Refuted-arm registry stops the "we tried this six months ago and
  forgot the result" failure mode the existing untracked/ design
  docs partially address.
- The build gate (`run_memorize_int_test.sh` /
  `run_generalize_int_test.sh`) is preserved as a single-arm spec, so
  CI behaviour doesn't change while the new abstraction lands.

What this does NOT change:

- The encoder pass classes themselves. Each arm still calls
  existing `parse.py` / `train.py` / `predict.py` with different
  flags. No code refactor inside `preframr/macros/`.
- Per-composer eval (`eval_per_composer.py`) — orthogonal; runs as
  a derived-metric closure on canonical-tier artefacts.

---

## Recommendation

Start at Phase 1 + 2 (data pinning + base abstraction + re-express
the existing 3 harnesses). That lands the infrastructure without
needing any encoder change to land first, and the existing build
gate stays green throughout.

Phase 3 (per-macro-opp spec) follows whichever encoder change ships
first; `instrument_window` is the cheapest target since it's a
single-flag sweep on existing code.
