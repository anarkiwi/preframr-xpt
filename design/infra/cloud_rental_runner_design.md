# Cloud-rental parallel runner — design (resume + early-abort + max-parallel-arms)

**Status:** Deferred — cloud-rental prereq. Design-only; not built. (Unblocked: the
old `loop_lookahead_prodlike` base.py-edit freeze is over; gate is now just rental.)

> **Staging note (BACC reality):** the dump-staging path below (`stage_dumps` copying
> `.dump.parquet`) is superseded by the symlink-farm + RO-mount in
> [`runner_iteration_efficiency_design.md`](runner_iteration_efficiency_design.md), and the shipped
> recovery path is **sid-only** (`recover_from_sid` via `preframr-sidtrace` — no `.dump.parquet` copy).
> The resume/early-abort/max-parallel-arms design stands; treat the `stage_dumps` rows as illustrative.

These three features are one program for running A/B specs on a rented multi-GPU box
(e.g. 8×A100/H100). All three edit `preframr_experiments/base.py` + `run.py`:

- **§1 `--resume`** — reuse on-disk parse/tokenize artefacts so a partial run recovers
  without re-burning the ~30-min staging cost per (arm, seed).
- **§2 auto early-abort** — a spec-declared `decision_rule` the runner evaluates after
  each (arm, seed), replacing the human-on-`pkill` mid-run abort; writes a refutation
  stub and releases the GPU on falsification.
- **§3 `--max-parallel-arms N`** — run multiple (arm, seed) pairs concurrently across
  GPUs (≈6–11 hr vs 36–66 hr sequential for 2 arms × 3 seeds).

**Landing order: §1 → §2 → §3.** Parallel arms depend on the resume lock (§1) and the
decision-rule path (§2); `--max-parallel-arms` lands last.

Each part keeps its own Motivation / Method / Effort / Order-of-operations below.

---

## 1. `--resume`

### Motivation

Today's re-run cost (prodlike, per arm, per seed):

| Stage | Wallclock | Caches that exist now |
|---|---|---|
| stage_dumps (copy .dump.parquet) | ~1-3 min | none |
| parse (parse.py → .parsed.parquet) | ~5-20 min | none |
| tokenize (stftokenize.py → dataset.csv.zst, tokens.csv, tkmodel.json) | ~15-30 min | none |
| train (train.py → tb_logs, checkpoints) | ~5-10 hr | Lightning ckpt resume exists but isn't wired |

A run that fails late (e.g. OOM in train) discards parse + tokenize
artefacts and re-pays the ~30-min staging cost. For prodlike
(2 arms × 3 seeds = 6 runs), that's ~3 hr of avoidable burn per
re-launch. For a cloud rental at $20+/hr, it's the difference
between one productive arm and zero.

The original design (`DESIGN.md` §Resumability) calls for caches
keyed on `arm.extra_cargs + seed`, but the runner doesn't implement
it. This doc specifies the cache key and the reuse semantics
precisely.

### Cache layout

Per-(arm, seed) work-dir already contains every artefact we'd want
to reuse:

```
results/<spec>/<arm>/seed<N>/
    train/         # staged dumps (input, deterministic from data tier)
    eval_a/, eval_b_*/
    dataset.csv.zst, tokens.csv, df-map.csv, tkmodel.json   # tokenize output
    logs/parse.log, logs/tokenize.log, logs/train.log
    tb_logs/        # train output
    metrics.json   # post-run extraction
    _wallclock.json
    _resume.json   # NEW: per-stage completion + key
```

`_resume.json` is the manifest:

```json
{
  "spec_name": "loop_lookahead_prodlike",
  "arm_label": "la1",
  "seed": 0,
  "stages": {
    "stage_dumps": {"completed": true, "key": "<key>"},
    "parse":       {"completed": true, "key": "<key>"},
    "tokenize":    {"completed": true, "key": "<key>"},
    "train":       {"completed": false, "key": "<key>"}
  }
}
```

`completed: true` means the stage's output files exist and the
manifest was written *after* the docker container returned rc=0
(atomic write via `tmp + os.rename`). A crashed stage leaves
`completed: false` (or no entry).

### Cache keys

Each stage's key is a stable hash of the inputs that affect its
output. Key collision → reuse; key mismatch → re-run.

#### `stage_dumps`

Key = `sha256(sorted(rel_paths) + src_root + tier_subdir_layout)`.
The output is the set of `<subdir>/*.dump.parquet` files in the
work-dir. Reuse if key matches AND every expected file exists
non-empty.

Subtle: `stage_dumps` skips missing dumps with a warning
(`base.py:stage_dumps:317-326`). A resume should preserve that
skip-list — re-run if the *expected* count differs from the
*on-disk* count even when the key matches.

#### `parse`

Key = `sha256(stage_dumps_key + arm.extra_cargs + preframr_version)`.

`preframr_version` is captured from `preframr/__init__.py:__version__`
(if present) or a SHA of `preframr/macros/` (`git rev-parse
HEAD:preframr/macros`). The macros tree is the parse-affecting
surface; including its sha invalidates parse output when *any*
encoder-affecting code changes.

This is the load-bearing key for the mid-run-edit safety property:
if the operator (or a teammate) edits a macro pass between two
(arm, seed) pairs, the key changes, the resumed run re-parses, and
the A/B stays valid. Without this, `--resume` could silently mix
parse output from two different code states across seeds.

Output: `train/*.parsed.parquet`, `eval_*/`*.parsed.parquet`,
`logs/parse.log`. Reuse if key matches AND `logs/parse.log` ends
with the parse-OK marker.

#### `tokenize`

Key = `sha256(parse_key + arm.extra_cargs + spec.seq_len +
spec.tkvocab + spec.min_song_tokens + spec.block_stride +
spec.max_perm)`.

Output: `dataset.csv.zst`, `tokens.csv`, `df-map.csv`,
`tkmodel.json`, `logs/tokenize.log`. Reuse if all five files exist
+ tokenize-OK marker.

#### `train`

Key = `sha256(tokenize_key + spec.effective_train_args() +
arm.training_overrides + seed)`.

Output: `tb_logs/`, train.py checkpoint(s), `logs/train.log`.

**Train resume is special.** The other stages are all-or-nothing
(rerun if the key differs, otherwise reuse the whole output). Train
is incremental — Lightning's `ModelCheckpoint` already supports
resume from a `last.ckpt`. Two modes:

- **Cache hit, output complete.** `train.log` ends with the
  early-stop / max-epoch marker + `metrics.json` is current. Skip.
- **Cache hit, output partial.** `tb_logs/` exists but training
  didn't reach the stop condition. Pass `--resume-from-checkpoint
  <work_dir>/<last_ckpt>` to `train.py`; pick up at the next epoch.
  Requires `train.py` to honour the flag (verify before landing).
- **Cache miss.** Wipe `tb_logs/` + checkpoints; full re-train.

### CLI

```
python3 -m preframr_experiments.run <spec> --resume [--resume-from <stage>]
```

- `--resume` (default off): on per-(arm, seed), evaluate cache keys
  and reuse any matching stage.
- `--resume-from {stage_dumps,parse,tokenize,train}`: force-restart
  from a specific stage, bypassing the key check. Useful when the
  operator knows an upstream cache is poisoned (e.g. dump cache
  rebuilt) but doesn't want to clear the work-dir manually.
- Without `--resume`: today's behaviour. `rmtree` and rerun.

The `--only-arm` flag (existing) composes naturally: resume + only-arm
re-runs one arm with its cached stages reused.

### Cache invalidation rules

- **Spec change.** If the spec module's mtime is newer than
  `_resume.json` (and the new key differs), invalidate.
- **`preframr/macros/` change.** Parse key includes the macros sha →
  invalidation is automatic.
- **`preframr_experiments/base.py` change.** Not covered
  by any stage key. The operator is responsible for not editing
  base.py while a resumable run is paused. (The mid-run-edit rule
  already forbids it; `--resume` doesn't change the rule.)
- **Data list change.** `stage_dumps` key includes the rel-path
  list → invalidation is automatic.
- **HVSC corpus version change.** Out of band; the runner already
  reads `HVSC_VERSION` and we should add a check that
  `_resume.json` matches the current value (refuse to resume on
  mismatch). Folds into the HVSC pin enforcement framework item.

### NFS silly-rename interaction

AGENTS.md §Re-launch protocol calls out the failure mode:
`rmtree(work_dir)` raises `ENOTEMPTY` because NFS silly-renames
`tb_logs/` files when Lightning has them open. The current
recommendation is `rm -rf` before re-launching.

`--resume` makes this worse if not handled: the resume path
*should* preserve `tb_logs/` (train resume), so we can't `rmtree`
defensively. Mitigations:

1. **Detect open handles.** Before resuming train, attempt to
   acquire a flock on `tb_logs/_resume.lock`. If held, another
   instance is running this (arm, seed) — abort with a clear error
   rather than corrupting checkpoints.
2. **Atomic cache-state writes.** `_resume.json` is written via
   `tmp + os.rename` so a partial write can't poison the next
   resume.
3. **Stale lock recovery.** A flock that's been held > N hours with
   no orchestrator process owning it (PID checked against
   `/proc`) is considered stale; the runner clears it with a
   logged warning. Avoid the "lock from a crashed run blocks
   forever" failure mode.

### Edge cases

- **Partial dataset.csv.zst.** Tokenize streams to a single file; a
  crash mid-write leaves a truncated zstd stream. Resume must
  validate the file with a one-shot `zstd -t` (or equivalent
  Python) before declaring tokenize complete. Mark
  `completed: false` if the validation fails.
- **Mixed-key arms in one spec.** With `--max-parallel-arms ≥ 2`
  (the related part), arms run concurrently. Resume keys are
  per-(arm, seed), so cross-arm interference is structurally
  impossible. Document the property.
- **Spec docstring change.** Doesn't invalidate any stage (parse /
  tokenize / train don't read the docstring). The report rebuilds
  from `metrics.json` so a docstring tweak between resumes
  changes only the report header.
- **Decision-rule change** (§2 (auto early-abort)).
  The rule runs *after* metric extraction; resume preserves
  `metrics.json`, so a changed rule re-evaluates on resume. This
  is intentional — operators may relax a rule and re-run.

### Out of scope

- **Cross-host cache sharing.** Each host has its own work-dir; we
  don't try to ship caches across machines. Cloud-rental hosts
  start cold by design.
- **Dump-cache resume.** `/scratch/preframr/training-dumps/` is
  already content-addressed (per-SID), so `stage_dumps` is the
  cheapest stage. No need to cache it inside the work-dir beyond
  the existing copy.
- **Compaction of completed work-dirs.** A finished spec leaves
  ~MBs per (arm, seed) on disk. Cleanup is a manual operator step
  (delete `results/<spec>/`); the runner doesn't auto-prune.

### Effort

- `_resume.json` schema + write-on-success in each stage:
  **~0.5 day**.
- Cache-key computation (each stage): **~0.5 day**.
- `--resume` + `--resume-from` CLI flags + `run_arm` branch points:
  **~0.5 day**.
- Train-resume `--resume-from-checkpoint` wiring (assumes
  `train.py` supports it; verify first; if not, **~1 day** added):
  **~0.5 day**.
- NFS flock + stale-lock recovery: **~0.5 day**.
- Tests: synthetic spec with deliberate-failure injection at each
  stage; assert resume reaches completion on retry: **~1 day**.

Total: **~3-4 days**. First of the three parts to land (see the landing order at
the top); gated on cloud-rental need.

### Order of operations

1. Land this design (reviewer pass).
2. Verify `train.py` supports `--resume-from-checkpoint`; add if
   missing (separate landing, gated on prodlike completion since
   it touches `preframr/`).
3. Land `_resume.json` schema + writes per stage; no read path yet
   (so existing behaviour is unchanged).
4. Land cache-key computation + resume read path; `--resume` flag
   defaulted off.
5. Tests + NFS lock handling.
6. Flip `--resume` on opportunistically; document in AGENTS.md
   §Re-launch protocol.

### Validation

- **Unit:** cache-key stability (same inputs → same key; one input
  change → different key). Cover macros-sha invalidation explicitly.
- **Integration:** synthetic 2-arm 2-seed spec; kill mid-train on
  seed 1; relaunch with `--resume`; assert parse/tokenize reused,
  train resumed from last ckpt, final metrics within seed σ of a
  cold-run baseline.
- **Regression:** without `--resume`, behaviour is byte-identical
  to today. (Run smoke specs before/after, diff report.md.)

## 2. auto early-abort

### Motivation

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

### Contract

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

### Runner integration

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

### Refutation stub

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

### Standard rule library

Each rule is a free function in
`preframr_experiments/decision_rules.py` (new module). Specs
import + parameterise; the spec stays declarative.

#### `capacity_attenuation`

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

#### `null_after_n_seeds`

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

#### `regression_floor`

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

#### Composition

Specs can `any_of(rule_a, rule_b)` / `all_of(...)` to compose. The
runner evaluates the rule once per (arm, seed); composition is the
spec's concern.

### Worked example: `loop_lookahead_prodlike`

Current spec docstring rule:

> Δ ≤ 0 or Δ < ⅛ × mini Δ (~+0.0012, extreme capacity attenuation)

After landing:

```python
from preframr_experiments.decision_rules import (
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
the related part), arms can interleave and the abort can fire earlier.

### Edge cases

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

### Out of scope

- **Cross-experiment refutation propagation** (e.g. an aborted mini
  spec auto-disables the prodlike re-test). Spec coupling lives in
  the human workflow, not the runner. The refuted-arm registry
  surfaces the signal; specs are still launched explicitly.
- **Dynamic rule changes mid-run.** Rules are frozen at spec import
  time. Editing the rule mid-run requires aborting + re-running with
  `--resume` (and is unsafe per the mid-run-edit rule anyway).
- **Auto-resume after abort.** An aborted run stays aborted until
  the operator opts in. Auto-resume would defeat the purpose.

### Effort

- `decision_rules.py` module with the 3 standard rules + composition:
  **~0.5 day**.
- `EarlyAbortDecision` dataclass + `decision_rule` field on
  `ExperimentSpec` + runner integration in `run.py`: **~0.5 day**.
- Refutation stub renderer in `report.py` + `data/refuted/<spec>.md`
  pointer write: **~0.5 day**.
- Unit tests against a fixture spec (synthetic metrics, assert rule
  fires / doesn't fire at expected (arm, seed)): **~0.5 day**.

Total: **~2 days**. Lands second (after §1 `--resume`); gated on cloud-rental need.

### Order of operations

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

### Validation

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

## 3. `--max-parallel-arms`

### Motivation

Sequential constraint today is structural — every `run_arm` call
runs three docker containers (parse, tokenize, train) that each
assume exclusive use of the host's resources. Train sets
`--gpus=all` (`base.py:_docker_run:377-378`), so two concurrent
train containers would contend for the single GPU.

Cloud-rental amortises one-time costs (image push, dataset stage,
auth) across multiple arms; the per-spec wallclock gain only
materialises if those arms actually run in parallel. The
multi-GPU rental decision table in AGENTS.md gates rental on
"`--max-parallel-arms N` landed" precisely because rental at
sequential cadence offers no value over local.

### Contract

```
python3 -m preframr_experiments.run <spec> --max-parallel-arms N
```

`N` is the maximum number of concurrent (arm, seed) runs. `N=1`
(default) reproduces today's behaviour. `N>1` requires:

- ≥ `N` GPUs visible (`nvidia-smi -L` count).
- ≥ `N × peak_train_memory` host RAM (the train container caps at
  32g; for prodlike body, 2×32g = 64g — within an A100 box, tight
  on a 4090 host).
- No `predict_gate` declared by the spec, OR a `predict_gate` that
  serialises explicitly. (Predict gates today access the host GPU
  via `--gpus=all` for inference; concurrent gates would contend.
  Punt: refuse `--max-parallel-arms > 1` if any
  `predict_gate is not None`.)

The runner partitions the host's GPUs across slots and pins each
concurrent run to its slot via `CUDA_VISIBLE_DEVICES`. Slot count =
`min(N, gpu_count)`.

### Implementation

#### Slot allocation

```python
@dataclasses.dataclass
class _Slot:
    slot_id: int            # 0..N-1
    cuda_devices: list[int] # GPU indices assigned to this slot
    lock_path: Path         # flock target
```

`N` slots are constructed at runner startup; `cuda_devices` is a
disjoint partition of the host's GPU set (e.g. on an 8-GPU box
with `N=4`, slots [0,1], [2,3], [4,5], [6,7]; with `N=8`, slots
[0]..[7]).

A `(arm, seed)` claims a slot via `fcntl.flock` on
`<root>/locks/slot_<id>.lock`. Slots are reusable — a run releases
its slot via flock unlock when `run_arm` returns. Lock files are
created lazily and persist for the lifetime of the runner.

#### Concurrency primitive

`concurrent.futures.ThreadPoolExecutor(max_workers=N)` driving the
`(arm, seed)` cross product. Each worker:

1. Acquires a slot (blocks if all N are held).
2. Calls `run_arm` with the slot's CUDA device list passed in.
3. Releases the slot.
4. Posts metrics + decision-rule evaluation to the main thread.

The decision-rule path (§2 (auto early-abort)) must be
thread-safe: the `results` dict is accessed from multiple workers
on completion. Wrap mutation in a single `threading.Lock`; the
decision rule reads a snapshot copy of `results` so it doesn't
race with concurrent appends.

#### GPU pinning in docker

`_docker_run` currently passes `--gpus=all`. Extend to accept
`cuda_devices: list[int]`:

```python
if cuda_devices:
    cmd += [f"--gpus=device={','.join(map(str, cuda_devices))}"]
elif gpus:
    cmd += ["--gpus=all"]
```

`CUDA_VISIBLE_DEVICES` is set inside the container by the docker
GPU runtime; train code path needs no change.

#### Log demuxing

Today, `_docker_run` streams stdout+stderr to a per-stage log file
under the (arm, seed) work-dir. With parallel arms, the
orchestrator log (the top-level `<spec>.log`) interleaves lines
from concurrent workers, making it unreadable.

Mitigation: prefix every orchestrator-log line with
`[arm=<label>/seed=<N>/slot=<id>]`. The Python logger's
`extra={...}` mechanism + a custom formatter handles this; workers
attach the prefix when they construct the logger they pass into
`run_arm`. Per-stage log files (`logs/parse.log` etc.) are already
per-(arm, seed) so no demux needed there.

#### Stdout / report rendering

Today the runner prints `report.md` to stdout at the end. With
parallel arms, the *final* render happens after the executor
joins, so behaviour is unchanged. The intermediate state (between
the first and last completion) is observable via
`<results_dir>/report.md.partial` written after each (arm, seed)
completes — useful for tail-watching during long runs.

### Multi-GPU per arm (DDP)

`--max-parallel-arms` partitions the GPU set across arms.
Orthogonally, **a single arm could use multiple GPUs via DDP**.
The two compose: with 8 GPUs, 2 arms × 4 GPUs each, or 4 arms × 2
GPUs each, or 8 arms × 1 GPU each.

DDP wiring lives in `train.py` (out of scope here; sibling
`profile/ddp_scaling.py` benchmarks the scaling efficiency). This
design's contribution is the slot-level partitioning that *allows*
DDP arms to coexist with single-GPU arms in one spec.

When DDP scaling lands (`train.py` supports
`--gpus N`), the spec declares per-arm GPU count via a new field:

```python
Arm(label="la3", extra_cargs="--loop-lookahead 3", gpus_per_arm=2)
```

Default `gpus_per_arm=1`. The runner checks
`sum(arm.gpus_per_arm for arm in spec.arms) ≤ N × gpu_count_per_slot`
at startup.

### Edge cases

- **GPU OOM under contention.** Two concurrent arms each use ~18 GB
  on a 24 GB GPU — impossible. Slot partitioning to distinct GPUs
  is the only safe mode; `--max-parallel-arms N` requires ≥ N
  distinct GPUs visible. Refuse to start otherwise.
- **Concurrent dump-cache writes.** Multiple workers stage dumps
  into different work-dirs; no shared writes. The dump *source*
  (`/scratch/preframr/training-dumps/`) is read-only during a
  run. Safe.
- **Concurrent `_docker_run` with the same image.** Docker handles
  this fine. Container names need to be unique — `name=` is
  optional and not set today; if set in future, prefix with slot id.
- **`--only-arm` interaction.** Limits the arm set; the executor
  just sees fewer tasks. No special handling.
- **Decision-rule fires mid-run.** With concurrent arms, the first
  decision_rule(abort=True) needs to cancel in-flight tasks. The
  ThreadPoolExecutor doesn't support task cancellation cleanly
  once a docker container is running; the runner SIGINTs the
  per-worker docker process(es), waits for them to exit, then
  writes the refutation stub. Out-of-flight (queued but not
  started) tasks are dropped via `executor.shutdown(wait=False)`.
- **`--resume` interaction.** Cache-key checks are per-(arm, seed)
  and stateless — concurrent workers can each evaluate
  independently. No locking needed beyond slot acquisition.

### Locking semantics

Three lock scopes:

1. **Slot lock** (`<root>/locks/slot_<id>.lock`): which worker owns
   which GPU partition. Held for the duration of one (arm, seed).
2. **Spec lock** (`<root>/results/<spec>/.runner.lock`): at most
   one runner process drives a given spec at a time. Prevents two
   operators stepping on each other. Held for the duration of the
   runner.
3. **Resume lock** (`<work_dir>/tb_logs/_resume.lock`, per sibling
   resume design): which worker is actively training a given (arm,
   seed). Held for the duration of `run_arm`.

All three are `fcntl.flock` (advisory, POSIX). Stale locks are
detected by reading the lock file's PID payload + checking
`/proc/<pid>` — if the PID is gone, the lock is cleared with a
logged warning.

### Out of scope

- **Per-arm GPU type heterogeneity.** Assume all slots have
  identical GPUs (the cloud-rental case). Mixed-GPU is a future
  problem.
- **Auto-tuning `N`.** The operator picks `N`. We don't try to
  auto-detect optimal parallelism — the wallclock × $/hr tradeoff
  is workload-specific.
- **Distributed runner across hosts.** One host, one runner. Cloud
  rental is single-box.

### Effort

- Slot allocator + `concurrent.futures` integration in `run.py`:
  **~1 day**.
- `_docker_run` `cuda_devices` parameter + caller plumbing in
  `run_arm`: **~0.5 day**.
- Log-prefix formatter + per-worker logger: **~0.5 day**.
- Spec lock + stale-lock recovery: **~0.5 day**.
- Decision-rule mid-run cancellation (sibling `auto_early_abort`
  integration): **~0.5 day**.
- Tests: synthetic spec with N>1, assert wallclock ≈ sequential/N
  (within scheduler noise), correctness identical: **~1 day**.

Total: **~4 days**. Lands last — after §1 (`--resume`) and §2 (auto early-abort);
the slot allocator integrates with both. Gated on cloud-rental need.

### Order of operations

1. Land this design (reviewer pass).
2. Land §1 (`--resume`) + §2 (auto early-abort) first; they're
   independent of parallelism but parallelism integrates with both.
3. Land slot allocator + executor in `run.py`; default `N=1` so
   existing specs are unaffected.
4. Land `cuda_devices` plumbing in `_docker_run`.
5. Land log-prefix formatter.
6. Add `--max-parallel-arms` CLI flag; refuse `N>1` on
   single-GPU hosts.
7. First user: re-run a known mini spec with `N=2` on a 2-GPU
   test host (if available locally) or on a cloud rental as a
   smoke before committing to a full prodlike batch.

### Validation

- **Unit:** slot allocator returns disjoint GPU partitions for a
  range of (N, gpu_count) inputs; refuses impossible configurations.
- **Integration:** 2-arm 2-seed synthetic spec with N=2 on a
  2-GPU host. Assert wallclock < 0.6× of N=1; metrics
  byte-identical (same code paths, same seeds).
- **Regression:** N=1 produces byte-identical reports vs
  pre-landing. Run mini specs before/after.
- **Stress:** 4-arm × 3-seed spec with N=4 on an 8-GPU host;
  assert no slot collisions, all 12 (arm, seed) complete, log
  prefixes correctly demuxed.
