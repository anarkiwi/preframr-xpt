# auto_early_abort — design note

Cloud-rental prereq. Replaces the human-on-`pkill` mid-run abort with a
spec-declared `decision_rule` callable that the runner evaluates after
each (arm, seed) completes; on falsification the runner writes a
refutation stub and exits, releasing the GPU.

Sibling docs: `resume_design.md`, `max_parallel_arms_design.md`. All
three target `experiments/base.py` + `run.py`; **base.py edits are
blocked while `loop_lookahead_prodlike` is in flight** (mid-run
`integration_tests/experiments/` changes shift runner semantics between
(arm, seed) pairs and silently invalidate the A/B per AGENTS.md
§Mid-run code edits). This is design-only; implementation lands after
the prodlike run completes.

## Motivation

Today the prodlike mid-run abort is operator-driven (AGENTS.md
§Mid-run abort): after the first la1+la3 seed pair (~12-22 hr in),
inspect `report.md`; if Δ ≤ 0 or Δ < ⅛ × mini Δ, run `pkill -f
loop_lookahead_prodlike` and fold a refutation note. Costs:

- **Latency.** The decision needs a human in the loop. Cloud-rental
  $/hr makes this expensive — a 6-hr human-response delay on an
  8×A100 H100 rental is meaningful real money.
- **Inconsistency.** "Δ < ⅛ × mini Δ" lives in the spec docstring
  (`loop_lookahead_prodlike.py`), not the runner. Every spec
  re-states the rule in prose, so two specs with semantically
  identical rules can express them differently.
- **Audit trail.** A `pkill` leaves no first-class artefact tying
  the abort to the falsifying metric. The refutation note is
  human-authored after the fact.

`auto_early_abort` moves the rule into the spec as a callable, the
runner evaluates it after each (arm, seed) completes, and an
abort writes a machine-readable refutation alongside the report.

## Contract

```python
@dataclasses.dataclass
class EarlyAbortDecision:
    """Result of a decision_rule evaluation.

    abort: if True, the runner skips remaining (arm, seed) pairs and
        writes a refutation stub.
    reason: short human-readable explanation (logged + persisted in
        the refutation stub). Must be set when abort=True.
    falsifier: optional dict of metric -> value that triggered the
        abort. Used by the refutation stub for downstream forensics.
    """
    abort: bool
    reason: str = ""
    falsifier: Optional[dict] = None


# Added to ExperimentSpec:
decision_rule: Optional[
    Callable[[ExperimentSpec, dict[str, list[dict]]], EarlyAbortDecision]
] = None
```

Where `results` is the same `dict[str, list[dict]]` (arm_label → list
of per-seed metric dicts) that the runner already maintains in
`run.py:111` and passes to `render(spec, results, results_dir)`. The
rule receives the *cumulative* results dict — including the (arm,
seed) just completed — and returns `EarlyAbortDecision`.

The rule MUST be a pure function of `(spec, results)`. It MUST NOT
read TB events, parse logs, or anything outside `results`; any signal
needed for the decision must be a declared metric. This keeps the
falsification reproducible from `report.json` after the fact.

## Runner integration

`run.py` after each `(arm, seed)` metric extraction:

```python
metrics = compute_metrics(spec, artefacts)
results[arm.label].append(metrics)

if spec.decision_rule is not None:
    decision = spec.decision_rule(spec, results)
    if decision.abort:
        logger.warning(
            "auto_early_abort: arm=%s seed=%d triggered abort: %s",
            arm.label, seed, decision.reason,
        )
        _write_refutation_stub(spec, results_dir, decision, results)
        # Render whatever the partial table looks like, then exit.
        md_path = render(spec, results, results_dir)
        logger.info("partial report: %s", md_path)
        return 0
```

Exit code 0 (not 1): an early abort is the spec working as designed,
not a runner failure. Distinguish at the orchestrator log level by
the `auto_early_abort:` line + the refutation stub's presence.

## Refutation stub

`<results_dir>/refutation.json`:

```json
{
  "spec": "loop_lookahead_prodlike",
  "aborted_after": {"arm": "la3", "seed": 0},
  "reason": "Δval_acc=+0.0009 < ⅛ × mini Δ=+0.0012 (capacity-attenuation)",
  "falsifier": {
    "la1_val_acc": [0.0612, null, null],
    "la3_val_acc": [0.0621, null, null],
    "delta": 0.0009,
    "threshold": 0.0012
  },
  "rule_name": "capacity_attenuation_eighth",
  "results": { ... full results dict at abort time ... }
}
```

A sibling `<results_dir>/refutation.md` is rendered by `report.py`
with the same shape as the regular `report.md` but flagged with a
"REFUTED" header and the truncated arm × seed matrix.

The runner also writes a pointer file at
`data/refuted/<spec_name>.md` summarising the refutation in the
shared registry (per the §Refuted-arm registry framework follow-up).

## Standard rule library

Each rule is a free function in
`integration_tests/experiments/decision_rules.py` (new module). Specs
import + parameterise; the spec stays declarative.

### `capacity_attenuation`

The prodlike pattern. Refutes when the prodlike Δ falls below a
fraction of the mini Δ on the primary KPI.

```python
def capacity_attenuation(
    *, primary_metric: str, baseline_arm: str,
    treatment_arm: str, mini_delta: float, fraction: float = 0.125,
) -> Callable[..., EarlyAbortDecision]:
    """Returns a decision_rule. Fires when both arms have ≥1 seed
    and the per-arm seed-mean Δ on ``primary_metric`` is below
    ``fraction × mini_delta``.

    For loop_lookahead_prodlike: fraction=0.125 (⅛ × mini Δ).
    For a stricter capacity guard: fraction=0.25 (¼ × mini Δ, the
    AGENTS.md text rule).
    """
```

### `null_after_n_seeds`

Refutes when n seeds completed on both arms and the seed-mean Δ is
within ±k×σ on the primary KPI.

```python
def null_after_n_seeds(
    *, primary_metric: str, baseline_arm: str, treatment_arm: str,
    n: int = 2, sigma_k: float = 1.0, sigma_estimate: float,
) -> Callable[..., EarlyAbortDecision]:
    """Refutes the null hypothesis can't be rejected after n seeds.
    sigma_estimate from DESIGN.md §5 (mini: 0.0009 on val_acc)."""
```

### `regression_floor`

Refutes when any seed reports a primary-KPI value below an absolute
floor. Useful for `GENERALIZE_MIN_VAL_ACC` once that lands per
Pipeline coverage holes.

```python
def regression_floor(
    *, primary_metric: str, floor: float,
) -> Callable[..., EarlyAbortDecision]:
    """Refutes if any (arm, seed) reports primary_metric < floor.
    Run-as-canary; one bad seed is enough."""
```

### Composition

Specs can `any_of(rule_a, rule_b)` / `all_of(...)` to compose. The
runner evaluates the rule once per (arm, seed); composition is the
spec's concern.

## Worked example: `loop_lookahead_prodlike`

Current spec docstring rule:

> Δ ≤ 0 or Δ < ⅛ × mini Δ (~+0.0012, extreme capacity attenuation)

After landing:

```python
from integration_tests.experiments.decision_rules import (
    capacity_attenuation, regression_floor, any_of,
)

spec = ExperimentSpec(
    name="loop_lookahead_prodlike",
    # ... existing fields ...
    decision_rule=any_of(
        regression_floor(primary_metric="val_acc_at_best_loss", floor=0.0),
        capacity_attenuation(
            primary_metric="val_acc_at_best_loss",
            baseline_arm="la1", treatment_arm="la3",
            mini_delta=0.0094,  # from 2026-05-11 mini run
            fraction=0.125,
        ),
    ),
)
```

`regression_floor` catches Δ ≤ 0 after the first la3 seed.
`capacity_attenuation` catches the ⅛-rule.

The rule waits until both arms have ≥1 completed seed (a single arm
completed isn't enough to compute Δ). The runner serialises (la1
seed0 → la1 seed1 → la1 seed2 → la3 seed0 ...) by default, so the
first abort opportunity is after **la3 seed0** — ~12-22 hr in,
matching the AGENTS.md text rule. With `--max-parallel-arms ≥2` (see
sibling design), arms can interleave and the abort can fire earlier.

## Edge cases

- **Spec without `decision_rule`.** Runner behaves identically to
  today; no opt-in cost.
- **Rule raises.** Treated as a runner bug, not an abort signal.
  Logged at ERROR, the (arm, seed) is folded into `results`, and the
  loop continues. The rule must be defensive against missing metrics
  (return `EarlyAbortDecision(abort=False)` if data isn't ready).
- **All seeds for one arm fail before the other arm starts.** Some
  rules need both arms; they return `abort=False` until both have
  data. `regression_floor` is per-(arm, seed) so it fires earlier.
- **Re-run with `--resume`.** A re-run reads existing
  `metrics.json` per (arm, seed) into `results`; the rule fires on
  the *resumed* cumulative state. A re-run that completes a
  previously-aborted run is therefore a deliberate override (the
  operator chose to override the auto-abort by re-running) and the
  refutation stub is overwritten. Document as expected behaviour.
- **predict_gate interaction.** `predict_gate` runs after metric
  extraction (today). Ordering: `decision_rule` runs *before*
  `predict_gate` — an aborted spec doesn't run the gate. This
  matches the intent (the gate is per-arm sanity; the rule is the
  experiment-level decision).

## Out of scope

- **Cross-experiment refutation propagation** (e.g. an aborted mini
  spec auto-disables the prodlike re-test). Spec coupling lives in
  the human workflow, not the runner. The refuted-arm registry
  surfaces the signal; specs are still launched explicitly.
- **Dynamic rule changes mid-run.** Rules are frozen at spec import
  time. Editing the rule mid-run requires aborting + re-running with
  `--resume` (and is unsafe per the mid-run-edit rule anyway).
- **Auto-resume after abort.** An aborted run stays aborted until
  the operator opts in. Auto-resume would defeat the purpose.

## Effort

- `decision_rules.py` module with the 3 standard rules + composition:
  **~0.5 day**.
- `EarlyAbortDecision` dataclass + `decision_rule` field on
  `ExperimentSpec` + runner integration in `run.py`: **~0.5 day**.
- Refutation stub renderer in `report.py` + `data/refuted/<spec>.md`
  pointer write: **~0.5 day**.
- Unit tests against a fixture spec (synthetic metrics, assert rule
  fires / doesn't fire at expected (arm, seed)): **~0.5 day**.

Total: **~2 days**. Lands after `loop_lookahead_prodlike` completes.

## Order of operations

1. Land this design (reviewer pass).
2. Implement `decision_rules.py` standalone with unit tests
   (no `base.py` edit yet — module is import-only).
3. Land `EarlyAbortDecision` + `decision_rule` field + runner hook
   in `base.py` / `run.py` (single commit).
4. Land refutation stub renderer + registry pointer.
5. Wire `loop_lookahead_prodlike` to `decision_rule=any_of(...)` as
   the first user. Re-run a known-falsifying short experiment (e.g.
   a smoke spec with a metric guaranteed to fail the floor) to
   verify end-to-end.

## Validation

- **Unit:** synthetic `results` dicts → expected `EarlyAbortDecision`
  for each standard rule. Cover: zero seeds, partial arms, both
  arms complete, threshold edge cases.
- **Integration:** a fixture spec with `decision_rule=regression_floor`
  and a deliberately-failing arm. Runner aborts after the failing
  (arm, seed), writes `refutation.json` + `refutation.md`,
  registry pointer at `data/refuted/<spec>.md`. Exit code 0.
- **Regression:** existing specs without `decision_rule` produce
  byte-identical reports vs pre-landing. (Run `memorize` +
  `generalize` smoke before/after.)
