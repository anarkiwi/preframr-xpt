# Experiment-runner iteration-efficiency wins

**Status:** #1 + #2 LANDED on `preframr-xpt:main` (`_dataset_affecting_cargs` +
`_dataset_cache_key(…, extra_cargs)`; symlink-farm staging + read-only `/dumps`
mount + dump-free cache; 8 new `test_dataset_cache` cases, 62-test suite green).
#3 pending. Findings from reading `preframr-xpt`
(`base.py` runner core, `run.py` loop). Targets per-run *fixed overhead* and the
disabled cache — the things that slow the parse→tokenize→train cycle independent
of GPU train time. `--resume` / `auto_early_abort` / `--max-parallel-arms`
already have deferred designs (`resume_design.md`, `auto_early_abort_design.md`,
`max_parallel_arms_design.md`); this doc is the **not-yet-captured** set.

## Findings (ranked by value/effort)

### 1. Dataset cache is force-disabled for exactly the experiments being iterated

`_dataset_cache_key` (`base.py:323`) hashes `pipeline_spec`, `seq_len`, `tkvocab`,
`min_song_tokens`, `block_stride`, `max_perm`, `tier`, `data_layout` — but **not
`arm.extra_cargs`**. Of the 14 specs using `extra_cargs`, most pass train-only
flags (`--per-tier-heads`, `--mask-structural-tier-loss`) where omission is
correct and lets two arms *share* one parse+tokenize. But the macro specs
(`full_macros_prodlike`: `--vibrato-env-pass`, `--freq-run-pass`, …) pass
**parse/tokenize-affecting** flags, so omitting them is a correctness bug — two
arms would collide on one cached dataset. The workaround is
`PREFRAMR_DATASET_CACHE_DISABLE=1`, which throws away the ~25 min parse+tokenize
reuse across seeds and retries — for precisely the tokenizer-rework experiments
in flight now (`OSCILLATE_REWORK`).

**Fix (LANDED):** `_dataset_cache_key` now keys on the parse/tokenize-affecting
slice of `arm.extra_cargs`. `_dataset_affecting_cargs` strips a denylist
(`_TRAIN_ONLY_CARG_FLAGS`) of flags verified to be read only under
`preframr/train/model/` (per-tier-heads / mos-k / entropy-lambda / cluster /
diffusion / mask-structural / infonce), keeping their value tokens out too.
**Correct-by-construction:** anything *not* on the denylist counts toward the
key, so a new parse/tokenize flag can never collide two arms — at worst an
unrecognised train-only flag costs one redundant parse+tokenize. Macro arms now
get distinct keys (cache stays ON, no collision); train-only arms still share
(per-tier-heads keeps its cross-arm win). Macro specs can drop
`PREFRAMR_DATASET_CACHE_DISABLE=1`: seed 2..N and every retry skip parse+tokenize
(~25 min/reuse at prodlike; the bulk of 12–20 min at mini ×(seeds−1)).

### 2. Cache hit re-copies 2.7 GB of raw dumps it doesn't need

`_try_dataset_cache_hit` (`base.py:349`) `shutil.copytree`s every staged data
subdir from the cache into the work_dir. Measured: one prodlike cache entry is
**2.7 GB / ~4,800 files** (train 2.4 GB / 4,437 dumps). The raw `.dump.parquet`
**content is not read at train time** — `RegLogParser.parse` reads the cached
`.parsed.parquet` sibling (`reglogparser.py`: `glob(name.replace(DUMP_SUFFIX,
PARSED_SUFFIX))`); the raw dumps only supply the glob filenames + are parse
inputs. So on a cache hit the runner copies 2.7 GB over NFS per (arm, seed) for
data that is already at `src_root` and whose content is never opened. On a miss
the same tree is copied twice more (stage src→work, populate work→cache).

**Fix — LANDED, RUNNER-ONLY (symlink farm + read-only dump mount).** Verified the
decoupling that looked necessary is *not*: `parse_runner.write_df` derives its
output path by pure string op — `base_name = name.replace(".dump.parquet","")`,
`pq_name = base_name + f".{i}.parquet"` — on the glob result, with no
`realpath`/`resolve` anywhere in the parse path (only `__file__`). So if the
parse **input** is a *symlink* at a work_dir path, the parsed **output** lands in
work_dir (a real file), while the symlink's *content* resolves through a
read-only mount. No main-repo / preframr-tokens change; isolated to
`preframr-xpt` and independent of the in-flight API rework.

Scope (all in `base.py` + `test_dataset_cache`):

1. `_docker_run`: mount the dump cache read-only on parse/tokenize/train —
   `-v {src_root}:/dumps:ro` (extend `extra_volumes` to carry a `:ro` mode).
2. `stage_dumps`: create **symlinks** instead of `shutil.copy` — for each `rel`,
   `os.symlink(f"/dumps/{rel}", bucket/<name>)` (a container-valid target; broken
   on host, which is fine — only followed in-container). Keep the host
   `src_root/rel` existence check for the missing-dump warning + the `n>0` guard.
   Instant; eliminates the 2.4 GB stage copy.
3. `run_arm`: call `stage_dumps` **unconditionally** (symlinks are cheap and the
   eval/`--reglogs` globs need the `*.dump.parquet` names to exist on a hit too);
   keep only parse/tokenize/populate behind `not cache_hit`.
4. `_populate_dataset_cache`: copy subdirs with
   `ignore=shutil.ignore_patterns("*.dump.parquet")` so the cache stores only
   outputs (`.{i}.parquet`, `.blocks.npy`, `.meta.parquet`, `.palettes.json`,
   `.uni.zst`) — never the dump symlinks. Shrinks each entry ~10×.
5. `_try_dataset_cache_hit`: left unchanged — staging runs *after* the hit
   restore (hit-first ordering), so the cached subdirs copytree into not-yet-
   existing dirs and `dirs_exist_ok` is unnecessary.
6. `_chown_bind_root_to_runner` + `_ensure_work_root_writable`: add `-h` to
   `chown -R` so it does not dereference the dump symlinks into the read-only
   mount (which would error per-file). (`_ensure_work_root_writable` already
   `lstat`s, and runner-created symlinks are runner-owned, so it stays a no-op.)
   This wrinkle disappears entirely if #3 (run containers as the runner uid)
   lands, removing the chown step.

Net: no 2.4 GB stage copy, no 2.7 GB per-hit copytree, ~10× smaller cache
entries. Risk: broken-on-host symlinks — `_robust_rmtree` unlinks them without
following (fine); nothing on the host reads dump content (parse runs in the
container). One integration check: assert a populated cache dir contains no
`*.dump.parquet`, and that a hit + symlink-stage yields working parse-sibling
lookups.

### 3. A throwaway chown container after every docker step

`_chown_bind_root_to_runner` (`base.py:456`) runs a `docker run … chown -R`
over the whole bind root after **each** parse / tokenize / train. At prodlike the
post-train tree is the staged dumps + parsed parquets + tb_logs + checkpoints —
a full `chown -R` NFS walk in a fresh container, 3× per run, plus
`_ensure_work_root_writable`'s `rglob("*")` over the same tree at preflight.

**Fix (low–med):** run the parse/tokenize/train containers with
`--user $(id -u):$(id -g)` so artefacts are created as the runner uid and no
chown is needed (verify the image tolerates non-root; ML images often assume
root for `pip`/cache dirs — fall back to chowning only the small artefact set if
not). Compounds with #2: if dumps are a read-only mount, the writable tree to
chown is just the artefacts, not 2.4 GB of dumps.

## Already-designed (deferred) — reference, not re-proposed

- `--resume` (`resume_design.md`): skip completed (arm, seed). The dataset cache
  covers parse+tokenize; a `_wallclock.json`-marker resume would also skip a
  finished train on crash/retry. Cheap subset worth pulling forward.
- `auto_early_abort` (`auto_early_abort_design.md`): kill an arm when a
  `decision_rule` falsifies mid-train (several refuted arms ran to completion).
- `--max-parallel-arms` (`max_parallel_arms_design.md`): single-GPU can't
  parallelize train; parse/tokenize (CPU, on `fogbank`) could overlap, but the
  dataset cache already removes most repeat parse/tokenize, so low marginal value
  until multi-GPU.

## Recommendation

Land #1 first — smallest change, highest immediate value (re-enables the cache
for the in-flight tokenizer-rework A/Bs), and it's isolated to `preframr-xpt`
(`base.py` + a `test_dataset_cache` case), untouched by the preframr-tokens API
rework. #2 + #3 are the per-run fixed-overhead reductions; #2 needs a small
main-repo `parse.py` output-dir decoupling, so sequence it with other parse
changes.

## References

- `preframr-xpt:preframr_experiments/base.py` (`_dataset_cache_key`,
  `_try_dataset_cache_hit`, `stage_dumps`, `_chown_bind_root_to_runner`,
  `run_arm`); `run.py` loop.
- AGENTS.md "Runner / infra fragility"; `full_macros_prodlike` cache-disable note.
- Deferred: `resume_design.md`, `auto_early_abort_design.md`,
  `max_parallel_arms_design.md`.
