# --resume — design note

Cloud-rental prereq. Today the runner blows away each (arm, seed)
work-dir on re-run (`shutil.rmtree(work_dir)` in
`base.py:run_arm` (line 786)); cached parse + tokenize artefacts are
discarded. `--resume` reuses what's on disk so a partial run can
recover without re-burning parse (~5-20 min) and tokenize (~15-30
min) per (arm, seed).

Sibling docs: `auto_early_abort_design.md`,
`max_parallel_arms_design.md`. All three target
`preframr_experiments/base.py` + `run.py`; **base.py edits are blocked while
`loop_lookahead_prodlike` is in flight** per AGENTS.md §Mid-run code
edits. This is design-only; implementation lands post-prodlike.

## Motivation

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

## Cache layout

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

## Cache keys

Each stage's key is a stable hash of the inputs that affect its
output. Key collision → reuse; key mismatch → re-run.

### `stage_dumps`

Key = `sha256(sorted(rel_paths) + src_root + tier_subdir_layout)`.
The output is the set of `<subdir>/*.dump.parquet` files in the
work-dir. Reuse if key matches AND every expected file exists
non-empty.

Subtle: `stage_dumps` skips missing dumps with a warning
(`base.py:stage_dumps:317-326`). A resume should preserve that
skip-list — re-run if the *expected* count differs from the
*on-disk* count even when the key matches.

### `parse`

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

### `tokenize`

Key = `sha256(parse_key + arm.extra_cargs + spec.seq_len +
spec.tkvocab + spec.min_song_tokens + spec.block_stride +
spec.max_perm)`.

Output: `dataset.csv.zst`, `tokens.csv`, `df-map.csv`,
`tkmodel.json`, `logs/tokenize.log`. Reuse if all five files exist
+ tokenize-OK marker.

### `train`

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

## CLI

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

## Cache invalidation rules

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

## NFS silly-rename interaction

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

## Edge cases

- **Partial dataset.csv.zst.** Tokenize streams to a single file; a
  crash mid-write leaves a truncated zstd stream. Resume must
  validate the file with a one-shot `zstd -t` (or equivalent
  Python) before declaring tokenize complete. Mark
  `completed: false` if the validation fails.
- **Mixed-key arms in one spec.** With `--max-parallel-arms ≥ 2`
  (sibling design), arms run concurrently. Resume keys are
  per-(arm, seed), so cross-arm interference is structurally
  impossible. Document the property.
- **Spec docstring change.** Doesn't invalidate any stage (parse /
  tokenize / train don't read the docstring). The report rebuilds
  from `metrics.json` so a docstring tweak between resumes
  changes only the report header.
- **Decision-rule change** (sibling `auto_early_abort` design).
  The rule runs *after* metric extraction; resume preserves
  `metrics.json`, so a changed rule re-evaluates on resume. This
  is intentional — operators may relax a rule and re-run.

## Out of scope

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

## Effort

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

Total: **~3-4 days**. Lands after `loop_lookahead_prodlike`
completes.

## Order of operations

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

## Validation

- **Unit:** cache-key stability (same inputs → same key; one input
  change → different key). Cover macros-sha invalidation explicitly.
- **Integration:** synthetic 2-arm 2-seed spec; kill mid-train on
  seed 1; relaunch with `--resume`; assert parse/tokenize reused,
  train resumed from last ckpt, final metrics within seed σ of a
  cold-run baseline.
- **Regression:** without `--resume`, behaviour is byte-identical
  to today. (Run smoke specs before/after, diff report.md.)
