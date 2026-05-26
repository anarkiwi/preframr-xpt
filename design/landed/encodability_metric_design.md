# Encodability-rate metric for `global_instr_ids` Phase A

## Why this doc

The Phase A aggregation design's §Validation L6 lists "encodability
metric extractor" as a deliverable; the parent design and AGENTS.md
both call out per-cluster Eval-B encodability as the cross-composer
KPI the A/B is buying. Spec
``integration_tests/experiments/global_instr_ids_phase_a.py`` ships
without this metric to avoid touching ``preframr/core/`` during the
in-flight run; this doc pins the impl so it can land immediately
after the A/B closes.

The audit script
``integration_tests/profile/audit_engine_fp_palette_eval_encodability.py``
already computes the rate post-hoc against the canonical artifact +
parsed dumps. The metric integration is the same data, plumbed
through ``metrics.py`` so it appears in the spec's `report.md`
alongside ``val_acc_at_best_loss`` etc.

## Two impl shapes

### Option 1 -- df.attrs side-channel (parse-time)

``InstrumentProgramPass`` emits two counters on ``df.attrs`` when
``--engine-fp-palette`` is on:

  - ``instrument_pass_captures_total``: number of candidate captures
    Phase 1 collected.
  - ``instrument_pass_literal_fallbacks``: count of captures whose
    program was absent from the canonical palette and fell through
    to LITERAL.

``metrics.py`` reads these from each arm's parsed dumps, aggregates
per-cluster (and per-Eval-B-subset for the cross-composer fan-out),
emits ``encodability_rate`` + ``encodability_rate_eval_b_<subset>``.

**Pros:** Single-source-of-truth at parse time; no re-parse cost at
metric time.
**Cons:** Touches ``preframr/core/macros/passes.py``
(training-affecting -- needs the post-A/B edit window). Adds attrs
overhead; small (two ints) but every dump pays.

### Option 2 -- post-hoc audit-script reuse

Wire ``audit_engine_fp_palette_eval_encodability.py`` as a derived
metric in ``ExperimentSpec.derived_metrics``: the runner calls it
after train completes; output JSON is parsed by ``metrics.py``.

**Pros:** No ``preframr/core/`` edit. Audit script + metric stay
one source of truth -- if you re-run the audit standalone, results
match the spec's report.
**Cons:** Re-parses eval dumps once per arm (~30s × n_eval_dumps);
adds modest wallclock per arm. Doesn't carry the per-dump
literal-fallback count (only the eval-side encodability rate), so
*train-side* encodability stays unmeasured -- but the cross-
composer KPI is the only one that matters for the design decision.

## Recommendation

**Option 2**, post-A/B. Encodability rate IS the eval-side metric;
re-using the standalone audit keeps the impl tight, lands without a
core edit window, and the wallclock cost (one parse pass over
eval-A + eval-B-*) is dominated by the existing train wallclock.

If a later round wants train-side encodability (e.g. to debug a
puzzling val_loss regression where eval encodability is high), add
Option 1 as a strict superset; the two are compatible.

## Concrete impl plan (Option 2)

### Step 1 -- expose the audit script as a callable

Move the audit's main loop into a Python function (the CLI stays as
a thin wrapper):

```python
# In integration_tests/profile/audit_engine_fp_palette_eval_encodability.py

def compute_encodability_summary(
    tier: str,
    corpus_base: Path,
    repo_root: Path,
    palette_artifact: Path | None = None,
) -> dict[str, Any]:
    """Returns the same payload that ``main`` writes to JSON. Caller
    serialises if needed.
    """
```

The CLI's ``main`` calls this then dumps to ``--out``.

### Step 2 -- register as a derived metric

In ``integration_tests/experiments/metrics.py``:

```python
def _encodability_rate_overall(arm_artefacts):
    """Derived metric: overall eval encodability against the
    canonical palette. Lives here so the spec can declare it without
    pulling the audit module's import into base.py.
    """
    from integration_tests.profile.audit_engine_fp_palette_eval_encodability import (
        compute_encodability_summary,
    )
    summary = compute_encodability_summary(
        tier=arm_artefacts.tier,
        corpus_base=arm_artefacts.src_root,
        repo_root=arm_artefacts.repo_root,
    )
    return summary["overall"]["encodability_rate"]


def _encodability_rate_eval_b(subset: str):
    def _extractor(arm_artefacts):
        ...  # filter summary["subsets"] for subset; return rate
    return _extractor
```

Spec opts in:

```python
spec = ExperimentSpec(
    ...
    metrics=[
        "alphabet_size", ..., "encodability_rate_overall",
        "encodability_rate_eval_b_daglish",
        "encodability_rate_eval_b_follin",
    ],
    derived_metrics={
        "encodability_rate_overall": _encodability_rate_overall,
        ...
    },
)
```

### Step 3 -- cache the audit per-arm

The audit's output depends only on (canonical artifact, eval lists,
parser code). It doesn't depend on the arm's training run -- so two
arms' encodability rates against the same canonical artifact are
identical, and we can compute it once per spec rather than per arm.

Cache key: hash of (canonical artifact mtime, parser revision).
Cache value: the JSON payload. Cache lives at
``<results_dir>/encodability_audit.json``.

### Step 4 -- report column

``integration_tests/experiments/report.py`` already handles
arbitrary metric columns; the new ones flow through automatically.

## Acceptance

- Spec runs end-to-end with the three new metric columns populated.
- Audit's CLI invocation produces the same JSON the spec writes
  (consistency between standalone and orchestrator paths).
- Build-gate test ``tests/test_experiment_spec.py`` covers the
  metric registration via the audit-cache mock.

## Effort

~1 hr work + ~0 wallclock (audit re-uses parsed parquets).

## Order of operations

1. Phase A A/B finishes (in-flight).
2. Fold A/B verdict into AGENTS.md.
3. Land Option 2 (this doc).
4. Re-render the Phase A report.md with encodability columns
   populated.
5. If the A/B was HOLD (1σ ≤ Δval_acc < 3σ), use the encodability
   columns to inform the canonical-tier re-test decision.

## Why not now

The impl touches ``integration_tests/experiments/metrics.py`` which
is bind-mounted into every (arm, seed) container per
``AGENTS.md §Mid-run code edits``. Editing it during a live run
invalidates the A/B mid-flight. Defer until the run closes.
