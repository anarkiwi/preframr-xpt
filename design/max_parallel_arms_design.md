# --max-parallel-arms — design note

Cloud-rental prereq. Today the runner serialises every (arm, seed)
through one GPU (`run.py`). `--max-parallel-arms N` lets
multiple (arm, seed) pairs run concurrently when the host has >1
GPU. The intended use is cloud-rental on an 8×A100/H100 box:
2 arms × 3 seeds = 6 concurrent runs, ~6-11 hr wallclock instead of
36-66 hr sequential.

Sibling docs: `auto_early_abort_design.md`, `resume_design.md`. All
three target `experiments/base.py` + `run.py`; **base.py edits are
blocked while `loop_lookahead_prodlike` is in flight** per
AGENTS.md §Mid-run code edits. This is design-only; implementation
lands post-prodlike.

## Motivation

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

## Contract

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

## Implementation

### Slot allocation

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

### Concurrency primitive

`concurrent.futures.ThreadPoolExecutor(max_workers=N)` driving the
`(arm, seed)` cross product. Each worker:

1. Acquires a slot (blocks if all N are held).
2. Calls `run_arm` with the slot's CUDA device list passed in.
3. Releases the slot.
4. Posts metrics + decision-rule evaluation to the main thread.

The decision-rule path (sibling `auto_early_abort` design) must be
thread-safe: the `results` dict is accessed from multiple workers
on completion. Wrap mutation in a single `threading.Lock`; the
decision rule reads a snapshot copy of `results` so it doesn't
race with concurrent appends.

### GPU pinning in docker

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

### Log demuxing

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

### Stdout / report rendering

Today the runner prints `report.md` to stdout at the end. With
parallel arms, the *final* render happens after the executor
joins, so behaviour is unchanged. The intermediate state (between
the first and last completion) is observable via
`<results_dir>/report.md.partial` written after each (arm, seed)
completes — useful for tail-watching during long runs.

## Multi-GPU per arm (DDP)

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

## Edge cases

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

## Locking semantics

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

## Out of scope

- **Per-arm GPU type heterogeneity.** Assume all slots have
  identical GPUs (the cloud-rental case). Mixed-GPU is a future
  problem.
- **Auto-tuning `N`.** The operator picks `N`. We don't try to
  auto-detect optimal parallelism — the wallclock × $/hr tradeoff
  is workload-specific.
- **Distributed runner across hosts.** One host, one runner. Cloud
  rental is single-box.

## Effort

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

Total: **~4 days**. Lands after `loop_lookahead_prodlike` AND
sibling `auto_early_abort` + `resume` designs are landed (the
slot allocator integrates with both).

## Order of operations

1. Land this design (reviewer pass).
2. Land sibling `auto_early_abort` + `resume` first; they're
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

## Validation

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
