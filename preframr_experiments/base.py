"""Experiment spec + runner core."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
PREFLIGHT_DIR = PACKAGE_DIR / "preflight"

_PREFRAMR_SRC_ENV = "PREFRAMR_SRC_DIR"
_PREFRAMR_SRC_DEFAULT = Path("/scratch/anarkiwi/preframr/preframr")
_PREFRAMR_BIND_SRC_ENV = "PREFRAMR_BIND_SRC"
_PREFRAMR_TOKENS_SRC = Path("/scratch/anarkiwi/gen2-preframr-tokens/preframr_tokens")
_PREFRAMR_TOKENS_DST = "/root/.local/lib/python3.12/site-packages/preframr_tokens"


def _preframr_src_dir() -> Path:
    """Path on the host to the main repo's ``preframr/`` source tree. Bind-mounted over the baked image only when ``_src_bind_enabled()``. Override with ``$PREFRAMR_SRC_DIR``."""
    raw = os.environ.get(_PREFRAMR_SRC_ENV)
    return Path(raw) if raw else _PREFRAMR_SRC_DEFAULT


def _src_bind_enabled() -> bool:
    """Whether to bind-mount the working-tree ``preframr/`` over the baked image code. OFF by default: runs use the frozen, tested baked image, so a long run can't be perturbed by edits and the working tree stays free to change. Set ``$PREFRAMR_BIND_SRC=1`` (run.py ``--bind-src``) to opt in for iterating without a rebake."""
    return os.environ.get(_PREFRAMR_BIND_SRC_ENV, "") not in ("", "0")


_DATASET_CACHE_SUBDIR = "dataset_cache"
_DATASET_CACHE_MARKER = ".complete"
_DATASET_CACHE_FILES = (
    "dataset.csv.zst",
    "tokens.csv",
    "tkmodel.json",
    "df-map.csv",
    "df-map_reg_widths.json",
)
_DATASET_CACHE_DISABLE_ENV = "PREFRAMR_DATASET_CACHE_DISABLE"


_MINI_BODY_FLAGS = {
    "small": "--layers 4 --heads 6 --kv-heads 3 --embed 192 --intermediate 512",
    "large": "--layers 6 --heads 8 --kv-heads 4 --embed 288 --intermediate 768",
}


def mini_train_args(body: str = "large") -> str:
    """Return the canonical mini-tier train arg string for the given
    body size. ``body`` in {"small", "large"}; "large" is the default
    (see _MINI_BODY_FLAGS docstring)."""
    if body not in _MINI_BODY_FLAGS:
        raise ValueError(
            f"mini_train_args: body={body!r} not in {sorted(_MINI_BODY_FLAGS)}"
        )
    return (
        "--model=llama3_2 "
        "--shuffle 1.0 --accumulate-grad-batches 8 --batch-size 4 "
        "--learning-rate 3e-4 --weight-decay 0.01 "
        f"{_MINI_BODY_FLAGS[body]} "
        "--attn-dropout 0.1 --max-epochs 160 "
        "--early-stop-patience 6 --early-stop-min-delta 0.01 "
        "--val-check-every 2"
    )


def micro_mini_train_args(body: str = "small") -> str:
    """Stage-C rapid-triage config: ~5 min/arm. small body, 30 epochs, gate ON."""
    if body not in _MINI_BODY_FLAGS:
        raise ValueError(
            f"micro_mini_train_args: body={body!r} not in {sorted(_MINI_BODY_FLAGS)}"
        )
    return (
        "--model=llama3_2 "
        "--shuffle 1.0 --accumulate-grad-batches 4 --batch-size 4 "
        "--learning-rate 3e-4 --weight-decay 0.01 "
        f"{_MINI_BODY_FLAGS[body]} "
        "--attn-dropout 0.1 --max-epochs 30 "
        "--val-check-every 1 "
        "--generalization-gate"
    )


@dataclasses.dataclass(frozen=True)
class Arm:
    """One training configuration within an experiment."""

    label: str
    extra_cargs: str = ""
    macro_flags: tuple[str, ...] = ()
    macro_config: str = ""
    training_overrides: Optional[dict] = None
    baseline: bool = False


@dataclasses.dataclass
class ExperimentSpec:
    """Declarative spec for an experiment."""

    name: str
    doc: str
    tier: str
    arms: list[Arm]
    metrics: list[str]
    seeds: Optional[int] = None
    train_args: str = ""
    tier_train_overrides: Optional[dict] = None
    predict_gate: Optional[Callable] = None
    derived_metrics: Optional[dict] = None
    pre_run_hook: Optional[Callable] = None
    seq_len: int = 8192
    block_stride: Optional[int] = None
    tkvocab: int = 131072
    min_song_tokens: int = 128
    max_perm: int = 1
    image: str = "anarkiwi/preframr"
    hvsc_root: str = "/scratch/preframr/hvsc"

    _VALID_TIERS = (
        "smoke",
        "micro_mini",
        "mini",
        "canonical",
        "prodlike",
        "prodlike_4x",
    )

    def __post_init__(self):
        if self.tier not in self._VALID_TIERS:
            raise ValueError(f"unknown tier {self.tier!r}")
        if not self.arms:
            raise ValueError(f"experiment {self.name}: no arms")
        labels = [a.label for a in self.arms]
        if len(set(labels)) != len(labels):
            raise ValueError(f"experiment {self.name}: duplicate arm labels")
        if self.seeds is None:
            self.seeds = 1 if self.tier in ("smoke", "micro_mini", "mini") else 3
        if self.block_stride is None:
            self.block_stride = self.seq_len // 4
        baselines = [a for a in self.arms if a.baseline]
        if len(baselines) > 1:
            raise ValueError(
                f"experiment {self.name}: at most one arm may set baseline=True"
            )
        if self.tier_train_overrides:
            unknown = [
                k for k in self.tier_train_overrides if k not in self._VALID_TIERS
            ]
            if unknown:
                raise ValueError(
                    f"experiment {self.name}: tier_train_overrides has unknown "
                    f"tier(s) {unknown}; valid tiers are {self._VALID_TIERS}"
                )

    def effective_train_args(self) -> str:
        """Resolve train_args for the current tier. ``tier_train_overrides``,
        when present and keyed on the spec's tier, fully replaces
        ``train_args``; otherwise the base ``train_args`` is used."""
        if self.tier_train_overrides and self.tier in self.tier_train_overrides:
            return self.tier_train_overrides[self.tier]
        return self.train_args


@dataclasses.dataclass
class ArmArtefacts:
    """Filesystem locations produced by one arm × seed run. Passed to
    metric extractors and to the predict gate."""

    arm: Arm
    seed: int
    work_dir: Path
    tokens_csv: Path
    df_map_csv: Path
    tb_logs: Path
    train_log: Path
    parse_log: Path
    tokenize_log: Path
    metrics_json: Path


def read_list(path: Path) -> list[str]:
    """Parse a pinned .list file. One HVSC-relative path per line;
    ``# ...`` comments stripped; blank lines ignored."""
    paths = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if line:
                paths.append(line)
    return paths


def smoke_paths() -> list[str]:
    return read_list(DATA_DIR / "smoke.list")


def canonical_paths() -> dict[str, list[str]]:
    base = DATA_DIR / "canonical"
    return {
        "train": read_list(base / "train.list"),
        "eval-A": read_list(base / "eval-A.list"),
        "eval-B-daglish": read_list(base / "eval-B-daglish.list"),
        "eval-B-follin": read_list(base / "eval-B-follin.list"),
    }


def mini_paths() -> dict[str, list[str]]:
    """Stratified subset of canonical -- 30 train + 6 eval-A per
    composer, 8 entries per Eval-B composer. Same HVSC version pin.
    Use when canonical wallclock is the bottleneck on a first-pass
    arm-level decision and the ~150-SID train scale is enough signal
    to gate proceeding to canonical."""
    base = DATA_DIR / "mini"
    return {
        "train": read_list(base / "train.list"),
        "eval-A": read_list(base / "eval-A.list"),
        "eval-B-daglish": read_list(base / "eval-B-daglish.list"),
        "eval-B-follin": read_list(base / "eval-B-follin.list"),
    }


def micro_mini_paths() -> dict[str, list[str]]:
    """Stage-C tier: deterministic 16-SID train slice + small eval cut from mini."""
    paths = mini_paths()
    return {
        "train": paths["train"][:16],
        "eval-A": paths["eval-A"][:8],
        "eval-B-daglish": paths["eval-B-daglish"][:4],
        "eval-B-follin": paths["eval-B-follin"][:4],
    }


def prodlike_paths() -> dict[str, list[str]]:
    """Near-production tier: ~4.4K train across 25 composers + 385
    Eval-A + every committed ``eval-B-<family>.list`` in the prodlike
    data dir. HVSC v84.
    """
    base = DATA_DIR / "prodlike"
    out: dict[str, list[str]] = {
        "train": read_list(base / "train.list"),
        "eval-A": read_list(base / "eval-A.list"),
    }
    for p in sorted(base.glob("eval-B-*.list")):
        out[p.stem] = read_list(p)
    return out


def prodlike_4x_paths() -> dict[str, list[str]]:
    """Scaled prodlike tier: ~17.7K train across 50 composers (top-N
    admitted from the corpus structural index, eval-B holdouts
    excluded). Eval-A and Eval-B lists are symlinked from the prodlike
    data dir so they're held out identically.
    """
    base = DATA_DIR / "prodlike_4x"
    out: dict[str, list[str]] = {
        "train": read_list(base / "train.list"),
        "eval-A": read_list(base / "eval-A.list"),
    }
    for p in sorted(base.glob("eval-B-*.list")):
        out[p.stem] = read_list(p)
    return out


_PRODLIKE_BODY_FLAGS = (
    "--layers 16 --heads 12 --kv-heads 4 --embed 768 --intermediate 2048"
)


def prodlike_train_args() -> str:
    """Canonical prodlike-tier train arg string. Mirrors
    ``mini_train_args`` / canonical _TRAIN_ARGS in shape so specs
    pick this up via one call.
    """
    return (
        "--model=llama3_2 "
        "--shuffle 1.0 --accumulate-grad-batches 16 --batch-size 2 "
        "--learning-rate 2e-4 --weight-decay 0.01 "
        f"{_PRODLIKE_BODY_FLAGS} "
        "--attn-dropout 0.1 --max-epochs 60 "
        "--early-stop-patience 10 --early-stop-min-delta 0.005 "
        "--val-check-every 1"
    )


def _stage_bucket(rel: str) -> str:
    """Return the per-composer subdir name for a HVSC-relative path."""
    parent = Path(rel).parent.name
    return parent or "_root_"


def stage_dumps(
    rels: list[str],
    src_root: Path,
    dst_dir: Path,
    logger: logging.Logger,
    link_root: Optional[str] = None,
) -> int:
    """Stage each .dump.parquet from ``src_root/rel`` into a per-composer
    subdir under ``dst_dir`` (``dst_dir/<composer>/<basename>``).

    When ``link_root`` is set, the staged entry is a **symlink** to
    ``f"{link_root}/{rel}"`` -- a path valid inside the container where
    ``src_root`` is bind-mounted read-only at ``link_root`` (broken on the host,
    which is fine: only ever followed in-container). The parser derives its
    output paths by string from the input name, so the parsed sidecars land in
    the writable ``dst_dir`` next to the symlink. When ``link_root`` is None the
    content is copied (needed when a ``pre_run_hook`` mutates the staged dumps).
    Skips files missing at ``src_root`` (logs a warning); returns the count
    actually staged.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    staged = 0
    missing = 0
    for rel in rels:
        src = src_root / rel
        if not src.exists():
            missing += 1
            continue
        bucket = dst_dir / _stage_bucket(rel)
        bucket.mkdir(parents=True, exist_ok=True)
        dest = bucket / src.name
        if link_root is not None:
            if not dest.is_symlink() and not dest.exists():
                os.symlink(f"{link_root}/{rel}", dest)
        else:
            shutil.copy(src, dest)
        staged += 1
    if missing:
        logger.warning(
            "dump cache missing %u/%u dumps under %s; experiment runs on the "
            "%u that were found",
            missing,
            len(rels),
            src_root,
            staged,
        )
    return staged


_TRAIN_ONLY_CARG_FLAGS = {
    "--per-tier-heads": False,
    "--per-tier-content-mos-k": True,
    "--per-tier-mos-entropy-lambda": True,
    "--content-cluster-head": False,
    "--content-cluster-c": True,
    "--content-cluster-index": True,
    "--content-diffusion": False,
    "--content-diffusion-t": True,
    "--mask-structural-tier-loss": False,
    "--infonce-content-loss-weight": True,
    "--infonce-distractors": True,
    "--onset-loss-weight": True,
}


def _dataset_affecting_cargs(extra_cargs: str) -> list[str]:
    """The parse/tokenize-affecting tokens of ``extra_cargs``, with known
    train/model-only flags (and their detached values) stripped. Conservative:
    any flag NOT in ``_TRAIN_ONLY_CARG_FLAGS`` is treated as dataset-affecting,
    so a new parse/tokenize flag can never collide two arms onto one cache entry
    -- at worst an unrecognised train-only flag costs a redundant parse+tokenize.
    The denylist holds only flags verified to be read solely under
    ``preframr/train/model/`` (+ predict)."""
    tokens = shlex.split(extra_cargs or "")
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        takes_value = _TRAIN_ONLY_CARG_FLAGS.get(tok.split("=", 1)[0])
        if takes_value is None:
            out.append(tok)
            i += 1
            continue
        i += 2 if (takes_value and "=" not in tok) else 1
    return out


def _macro_cli(arm: "Arm") -> str:
    """Render an arm's ``macro_flags`` / ``macro_config`` into the ``--macro-flags`` /
    ``--macro-config`` CLI passed to parse + tokenize + train. Empty for an arm with no
    macro passes (the atomic baseline)."""
    parts = []
    if arm.macro_flags:
        parts.append(f"--macro-flags {','.join(arm.macro_flags)}")
    if arm.macro_config:
        parts.append(f"--macro-config {arm.macro_config}")
    return " ".join(parts)


@functools.lru_cache(maxsize=None)
def _image_tokens_version(image: str) -> str:
    """preframr-tokens version baked into ``image`` (queried once per image). Folded into the dataset cache key so a tokenizer upgrade invalidates stale parse/tokenize artefacts instead of silently reusing them. The key is otherwise version-blind: pre-this-fix, a 0.17->0.18 bump kept the same key and served stale tokenization."""
    out = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            image,
            "python3",
            "-c",
            "import importlib.metadata as m; print(m.version('preframr-tokens'))",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return out.stdout.strip()


def _dataset_cache_key(
    spec: "ExperimentSpec",
    data_layout: dict[str, list[str]],
    extra_cargs: str = "",
    macro_flags: tuple[str, ...] = (),
    macro_config: str = "",
) -> str:
    """Hash the inputs that determine parse + tokenize output. Stable across runs as long as the arm's macro_flags/macro_config + tier-pinned data + corpus-shape args + the parse/tokenize-affecting slice of the arm's extra_cargs + the image's preframr-tokens version don't change."""
    payload = {
        "macro_flags": sorted(macro_flags),
        "macro_config": macro_config,
        "seq_len": spec.seq_len,
        "tkvocab": spec.tkvocab,
        "min_song_tokens": spec.min_song_tokens,
        "block_stride": spec.block_stride,
        "max_perm": spec.max_perm,
        "tier": spec.tier,
        "data_layout": {k: sorted(v) for k, v in data_layout.items()},
        "dataset_cargs": _dataset_affecting_cargs(extra_cargs),
        "tokens_version": _image_tokens_version(spec.image),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _dataset_cache_dir(src_root: Path, cache_key: str) -> Path:
    return src_root / _DATASET_CACHE_SUBDIR / cache_key


def _dataset_cache_disabled() -> bool:
    return os.environ.get(_DATASET_CACHE_DISABLE_ENV, "") not in ("", "0")


def _try_dataset_cache_hit(
    cache_dir: Path,
    work_dir: Path,
    data_layout: dict[str, list[str]],
    logger: logging.Logger,
) -> bool:
    """Copy cached parse + tokenize artefacts into work_dir; return True if a complete cache was found and copied. Skips any cache without the marker file (= populated atomically)."""
    if not (cache_dir / _DATASET_CACHE_MARKER).exists():
        return False
    logger.info("dataset cache HIT %s -> reusing artefacts", cache_dir)
    for name in _DATASET_CACHE_FILES:
        src = cache_dir / name
        if src.exists():
            shutil.copy2(src, work_dir / name)
    for subdir in data_layout.keys():
        src = cache_dir / subdir
        if src.exists():
            shutil.copytree(src, work_dir / subdir)
    return True


def _populate_dataset_cache(
    cache_dir: Path,
    work_dir: Path,
    data_layout: dict[str, list[str]],
    logger: logging.Logger,
) -> None:
    """Mirror parse + tokenize artefacts from work_dir into cache_dir via tmp + rename (so a concurrent reader never sees a partial cache). No-op if the cache is already populated."""
    if (cache_dir / _DATASET_CACHE_MARKER).exists():
        return
    cache_root = cache_dir.parent
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp = cache_root / f".{cache_dir.name}.tmp-{os.getpid()}"
    if tmp.exists():
        try:
            shutil.rmtree(tmp)
        except OSError:
            return
    try:
        tmp.mkdir(parents=True)
        for name in _DATASET_CACHE_FILES:
            src = work_dir / name
            if src.exists():
                shutil.copy2(src, tmp / name)
        for subdir in data_layout.keys():
            src = work_dir / subdir
            if src.exists():
                shutil.copytree(
                    src,
                    tmp / subdir,
                    ignore=shutil.ignore_patterns("*.dump.parquet"),
                )
        (tmp / _DATASET_CACHE_MARKER).write_text("")
        try:
            tmp.rename(cache_dir)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            return
    except OSError as exc:
        logger.warning("dataset cache populate skipped: %s", exc)
        shutil.rmtree(tmp, ignore_errors=True)
        return
    logger.info("dataset cache POPULATED %s", cache_dir)


def _docker_run(
    image: str,
    args: list[str],
    bind_root: Path,
    log_path: Path,
    extra_volumes: Optional[list[tuple[Path, str]]] = None,
    gpus: bool = False,
    memory: str = "32g",
    name: Optional[str] = None,
    role: str = "train",
) -> int:
    """Single ``docker run --rm`` invocation. Streams stdout+stderr to ``log_path``; returns the container exit code. By default the container runs the baked image code. When ``_src_bind_enabled()``, ``role`` selects which side of the working-tree preframr/ tree is bind-mounted over the bake: ``"train"`` mounts ``preframr/train/`` only; ``"inference"`` mounts ``preframr/inference/`` only. See ``design/landed/train_inference_split_design.md``."""
    cmd = [
        "docker",
        "run",
        "--rm",
        f"--memory={memory}",
        f"--memory-swap={memory}",
        "--ulimit",
        "nofile=1048576:1048576",
        "-v",
        f"{bind_root}:/scratch/preframr",
    ]
    if _src_bind_enabled():
        repo_preframr_pkg = _preframr_src_dir()
        if role == "inference":
            cmd += ["-v", f"{repo_preframr_pkg / 'inference'}:/preframr/inference"]
        else:
            cmd += ["-v", f"{repo_preframr_pkg / 'train'}:/preframr/train"]
        cmd += ["-v", f"{repo_preframr_pkg / 'args.py'}:/preframr/args.py"]
        if _PREFRAMR_TOKENS_SRC.exists():
            cmd += ["-v", f"{_PREFRAMR_TOKENS_SRC}:{_PREFRAMR_TOKENS_DST}"]
    for host_path, container_path in extra_volumes or []:
        cmd += ["-v", f"{host_path}:{container_path}"]
    if gpus:
        cmd += [
            "--gpus=all",
            "--env",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        ]
    if name:
        cmd += ["--name", name]
    cmd += [image]
    cmd += args
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    rc = proc.returncode
    _chown_bind_root_to_runner(image, bind_root, log_path)
    return rc


def _chown_bind_root_to_runner(image: str, bind_root: Path, log_path: Path) -> None:
    """Walk the bind root inside a throwaway root container and
    chown everything back to the runner UID:GID. Suppresses errors
    (the next rmtree will surface any real permission problem).
    The chown is purely about freeing the runner to delete / re-run.
    """
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{bind_root}:/work",
        image,
        "chown",
        "-Rh",
        uid_gid,
        "/work",
    ]
    with open(log_path, "ab") as f:
        f.write(b"\n[_chown_bind_root_to_runner]\n")
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)


_RMTREE_RETRY_ERRNOS = (16, 39)


def _stop_preframr_tb() -> None:
    """Stop the host TB sidecar so it releases NFS file handles. Idempotent: no-op if the container isn't running, and silently no-op on hosts (like the test container) where ``docker`` isn't on PATH."""
    try:
        subprocess.run(
            ["docker", "stop", "preframr_tb"], capture_output=True, check=False
        )
    except OSError:
        return


def _robust_rmtree(path: Path, attempts: int = 5, base_delay: float = 0.2) -> None:
    """``shutil.rmtree`` with backoff on Errno 39 (silly-renamed `.nfs*` from host TB's events-file scan) and Errno 16 (unlink race before the rename). On the first failure we stop ``preframr_tb`` to release its handles; the caller in ``run_arm`` respawns TB after the rmtree succeeds. Non-retry errnos raise immediately so real bugs surface fast."""
    last_err: Optional[OSError] = None
    bounced_tb = False
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError as e:
            last_err = e
            if e.errno not in _RMTREE_RETRY_ERRNOS:
                raise
            if not bounced_tb:
                _stop_preframr_tb()
                bounced_tb = True
            if i < attempts - 1:
                time.sleep(base_delay * (2**i))
    assert last_err is not None
    raise last_err


def _gpu_available() -> bool:
    return (
        subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, check=False
        ).returncode
        == 0
    )


def _hvsc_version_check(spec: ExperimentSpec, logger: logging.Logger) -> None:
    """Assert the HVSC checkout's version matches the tier's pinned
    ``HVSC_VERSION`` file. Silently skips when the HVSC tree isn't on
    disk (typical for hosts that NFS-mount from elsewhere or unit
    tests under a tmpdir) or when no pin file exists for the spec's
    tier (smoke / synthetic tiers).
    """
    from preframr_experiments.hvsc_version_check import (
        HvscVersionMismatch,
        _read_tier_pin,
        assert_hvsc_version,
    )

    hvsc_root = Path(getattr(spec, "hvsc_root", "/scratch/preframr/hvsc"))
    if not hvsc_root.exists():
        logger.info("preflight: HVSC tree not at %s; skipping version check", hvsc_root)
        return
    try:
        expected = _read_tier_pin(DATA_DIR, spec.tier)
    except HvscVersionMismatch as e:
        raise RuntimeError(f"preflight: HVSC pin unreadable: {e}") from None
    if expected is None:
        logger.info(
            "preflight: no HVSC_VERSION pin for tier %s; skipping check", spec.tier
        )
        return
    try:
        assert_hvsc_version(hvsc_root, expected, logger=logger)
    except HvscVersionMismatch as e:
        raise RuntimeError(f"preflight: {e}") from None


def _ensure_work_root_writable(
    work_root: Path, image: str, logger: logging.Logger
) -> None:
    """Auto-chown cross-uid leftovers (root-owned tb_logs/ckpts) so rmtree won't fail."""
    if not work_root.exists():
        return
    my_uid = os.getuid()
    foreign = []
    for path in work_root.rglob("*"):
        try:
            st = path.lstat()
        except OSError:
            continue
        if st.st_uid != my_uid:
            foreign.append(path)
            if len(foreign) >= 8:
                break
    if not foreign:
        return
    logger.info(
        "preflight: %d cross-uid path(s) under %s (sample: %s); chowning via %s",
        len(foreign),
        work_root,
        foreign[0],
        image,
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{work_root.resolve()}:/wd",
        image,
        "chown",
        "-Rh",
        f"{my_uid}:{os.getgid()}",
        "/wd",
    ]
    rc = subprocess.run(cmd, capture_output=True, check=False)
    if rc.returncode != 0:
        raise RuntimeError(
            f"preflight: failed to chown {work_root} via {image} "
            f"(rc={rc.returncode}): {rc.stderr.decode(errors='replace')[:400]}"
        )


def _restart_tensorboard(
    work_root: Path, image: str, logger: logging.Logger, tb_scope: str = "spec"
) -> None:
    """Restart the host TB container with the requested mount scope. ``tb_scope='spec'`` mounts ``work_root`` (this experiment's results subdir only); ``tb_scope='root'`` mounts ``work_root.parent`` (full experiments root, legacy behaviour for overnight batches)."""
    if tb_scope == "root":
        results_root = work_root.parent.resolve()
    else:
        results_root = work_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["docker", "rm", "-f", "preframr_tb"], capture_output=True, check=False
    )
    rc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            "preframr_tb",
            "-p",
            "6006:6006",
            "-v",
            f"{results_root}:/tb_logs",
            image,
            "tensorboard",
            "--bind_all",
            "--logdir",
            "/tb_logs",
            "--reload_multifile=true",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 0:
        logger.warning(
            "preflight: tb restart failed (rc=%d): %s",
            rc.returncode,
            rc.stderr.strip(),
        )
    else:
        logger.info(
            "preflight: tb restarted on %s -> http://localhost:6006", results_root
        )


def preflight_check(
    spec: ExperimentSpec,
    work_root: Path,
    logger: logging.Logger,
    tb_scope: str = "spec",
) -> None:
    """Run the train-path smoke in docker before any arm starts."""
    _hvsc_version_check(spec, logger)
    work_root.mkdir(parents=True, exist_ok=True)
    _ensure_work_root_writable(work_root, spec.image, logger)
    _restart_tensorboard(work_root, spec.image, logger, tb_scope=tb_scope)
    log_path = work_root / "preflight.log"
    smoke_path = PREFLIGHT_DIR / "train_preflight_smoke.py"
    if not smoke_path.exists():
        raise RuntimeError(
            f"preflight smoke script missing: {smoke_path}; bundled with "
            "preframr_experiments.preflight package"
        )
    cmd = [
        "docker",
        "run",
        "--rm",
        "--memory=4g",
        "-v",
        f"{PREFLIGHT_DIR.resolve()}:/preflight",
    ]
    if _src_bind_enabled():
        cmd += ["-v", f"{_preframr_src_dir().resolve()}:/preframr"]
    if _gpu_available():
        cmd += ["--gpus=all"]
    cmd += [spec.image, "python3", "/preflight/train_preflight_smoke.py"]
    logger.info("preflight: launching smoke in %s (log: %s)", spec.image, log_path)
    with open(log_path, "wb") as f:
        proc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, check=False, timeout=300
        )
    rc = proc.returncode
    if rc != 0:
        tail = log_path.read_text(errors="replace").splitlines()
        fail_lines = [ln for ln in tail if ln.startswith("FAIL ")][-3:]
        raise RuntimeError(
            f"preflight smoke failed (rc={rc}); see {log_path}. "
            f"Last failures: {' | '.join(fail_lines) if fail_lines else '(no FAIL line)'}"
        )
    logger.info("preflight: OK")


def run_arm(
    spec: ExperimentSpec,
    arm: Arm,
    seed: int,
    work_dir: Path,
    data_layout: dict[str, list[str]],
    src_root: Path,
    logger: logging.Logger,
) -> ArmArtefacts:
    """Stage data + run parse / tokenize / train for one (arm, seed)."""
    logger.info(
        "experiment=%s arm=%s seed=%d work_dir=%s",
        spec.name,
        arm.label,
        seed,
        work_dir,
    )
    if work_dir.exists():
        _robust_rmtree(work_dir)
        _restart_tensorboard(work_dir.parent.parent, spec.image, logger)
    work_dir.mkdir(parents=True)
    log_dir = work_dir / "logs"
    log_dir.mkdir()

    cache_disabled = _dataset_cache_disabled()
    cache_key = _dataset_cache_key(
        spec, data_layout, arm.extra_cargs, arm.macro_flags, arm.macro_config
    )
    cache_dir = _dataset_cache_dir(src_root, cache_key)
    parse_log = log_dir / "parse.log"
    tokenize_log = log_dir / "tokenize.log"
    cache_hit = False
    if not cache_disabled:
        cache_hit = _try_dataset_cache_hit(cache_dir, work_dir, data_layout, logger)

    macro_cli = _macro_cli(arm)

    link_root = "/dumps" if spec.pre_run_hook is None else None
    dump_volumes = (
        [(src_root.resolve(), "/dumps:ro")] if link_root is not None else None
    )
    for subdir, rels in data_layout.items():
        dst = work_dir / subdir
        n = stage_dumps(rels, src_root, dst, logger, link_root=link_root)
        if n == 0:
            raise RuntimeError(
                f"arm {arm.label}/seed{seed}: zero dumps staged into {dst}; "
                f"is the dump cache at {src_root} populated?"
            )

    if spec.pre_run_hook is not None:
        spec.pre_run_hook(arm, work_dir)

    cargs = (
        f"--no-require-pq --seq-len {spec.seq_len} "
        f"--tkvocab {spec.tkvocab} "
        f"--df-map-csv /scratch/preframr/df-map.csv "
        f"--no-max-autotune "
        f"--min-song-tokens {spec.min_song_tokens} "
        f"--block-stride {spec.block_stride} "
        f"--max-perm {spec.max_perm} "
        f"{macro_cli} "
        f"{arm.extra_cargs}".strip()
    )

    train_glob = "/scratch/preframr/train/*/*.dump.parquet"
    eval_subdirs = [k for k in data_layout.keys() if k != "train"]
    has_eval = bool(eval_subdirs)
    eval_glob = build_eval_reglogs_arg(Path("/scratch/preframr"), data_layout)

    parse_globs = train_glob
    if has_eval:
        eval_globs = ",".join(
            f"/scratch/preframr/{subdir}/*/*.dump.parquet" for subdir in eval_subdirs
        )
        parse_globs = f"{train_glob},{eval_globs}"

    if not cache_hit:
        parse_args_list = [
            "/preframr/parse.py",
            "--no-require-pq",
            "--max-files",
            "999999",
        ]
        if macro_cli:
            parse_args_list += shlex.split(macro_cli)
        parse_args_list += shlex.split(arm.extra_cargs)
        parse_args_list += ["--reglogs", parse_globs]
        rc = _docker_run(
            spec.image,
            parse_args_list,
            bind_root=work_dir,
            log_path=parse_log,
            memory="32g",
            extra_volumes=dump_volumes,
        )
        if rc != 0:
            raise RuntimeError(f"parse failed (rc={rc}); see {parse_log}")

    tokenize_args = [
        "/preframr/stftokenize.py",
        *shlex.split(cargs),
        "--dataset-csv",
        "/scratch/preframr/dataset.csv.zst",
        "--token-csv",
        "/scratch/preframr/tokens.csv",
        "--reglogs",
        train_glob,
    ]
    if has_eval:
        tokenize_args += ["--eval-reglogs", eval_glob]
    if not cache_hit:
        rc = _docker_run(
            spec.image,
            tokenize_args,
            bind_root=work_dir,
            log_path=tokenize_log,
            memory="16g",
            extra_volumes=dump_volumes,
        )
        if rc != 0:
            raise RuntimeError(f"tokenize failed (rc={rc}); see {tokenize_log}")
        if not cache_disabled:
            _populate_dataset_cache(cache_dir, work_dir, data_layout, logger)

    train_log = log_dir / "train.log"
    train_overrides_str = ""
    if arm.training_overrides:
        train_overrides_str = " ".join(
            f"--{k.replace('_','-')} {v}" for k, v in arm.training_overrides.items()
        )
    train_args = [
        "/preframr/train/trainer.py",
        *shlex.split(cargs),
        *shlex.split(spec.effective_train_args()),
        *shlex.split(train_overrides_str),
        "--reglogs",
        train_glob,
        "--dataset-csv",
        "/scratch/preframr/dataset.csv.zst",
        "--token-csv",
        "/scratch/preframr/tokens.csv",
    ]
    if has_eval:
        train_args += ["--eval-reglogs", eval_glob]
    train_t0 = time.monotonic()
    rc = _docker_run(
        spec.image,
        train_args,
        bind_root=work_dir,
        log_path=train_log,
        memory="32g",
        gpus=_gpu_available(),
        extra_volumes=dump_volumes,
    )
    train_elapsed_s = time.monotonic() - train_t0
    if rc != 0:
        raise RuntimeError(f"train failed (rc={rc}); see {train_log}")

    artefacts = ArmArtefacts(
        arm=arm,
        seed=seed,
        work_dir=work_dir,
        tokens_csv=work_dir / "tokens.csv",
        df_map_csv=work_dir / "df-map.csv",
        tb_logs=work_dir / "tb_logs",
        train_log=train_log,
        parse_log=parse_log,
        tokenize_log=tokenize_log,
        metrics_json=work_dir / "metrics.json",
    )
    (work_dir / "_wallclock.json").write_text(
        json.dumps({"train_elapsed_s": train_elapsed_s})
    )
    return artefacts


def resolve_data_layout(spec: ExperimentSpec) -> dict[str, list[str]]:
    """Map tier -> {subdir: [HVSC-rel-paths]}. The runner stages each
    subdir under its own work-dir directory so the existing parse /
    tokenize / train flag conventions (``--reglogs``,
    ``--eval-reglogs``) keep working unchanged.
    """
    if spec.tier == "smoke":
        return {"train": smoke_paths()}
    if spec.tier == "micro_mini":
        paths = micro_mini_paths()
        return {
            "train": paths["train"],
            "eval": paths["eval-A"] + paths["eval-B-daglish"] + paths["eval-B-follin"],
        }
    if spec.tier == "mini":
        paths = mini_paths()
        return {
            "train": paths["train"],
            "eval": paths["eval-A"] + paths["eval-B-daglish"] + paths["eval-B-follin"],
        }
    if spec.tier == "prodlike":
        paths = prodlike_paths()
    elif spec.tier == "prodlike_4x":
        paths = prodlike_4x_paths()
    else:
        paths = canonical_paths()
    layout: dict[str, list[str]] = {
        "train": paths["train"],
        "eval_a": paths["eval-A"],
    }
    for key in paths:
        if key.startswith("eval-B-"):
            family = key[len("eval-B-") :]
            layout[f"eval_b_{family}"] = paths[key]
    return layout


_TRAIN_SUBDIR = "train"
_LEGACY_EVAL_SUBDIR = "eval"


def build_eval_reglogs_arg(work_dir: Path, data_layout: dict) -> str:
    """Render the ``--eval-reglogs`` flag value from the staged data
    layout. Single eval subdir (legacy mini path) returns a bare
    glob. Multiple eval subdirs return the ``name=glob;name=glob``
    multi-subset form. No eval subdirs return ``""``.
    """
    eval_subdirs = [k for k in data_layout.keys() if k != _TRAIN_SUBDIR]
    if not eval_subdirs:
        return ""
    if eval_subdirs == [_LEGACY_EVAL_SUBDIR]:
        return f"{work_dir / _LEGACY_EVAL_SUBDIR}/*/*.dump.parquet"
    parts = []
    for subdir in eval_subdirs:
        parts.append(f"{subdir}={work_dir / subdir}/*/*.dump.parquet")
    return ";".join(parts)
