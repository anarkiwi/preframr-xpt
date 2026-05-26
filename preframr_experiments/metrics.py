"""Metric extractors registry."""

from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
from typing import Callable

from preframr_experiments.base import ArmArtefacts


def _alphabet_size(art: ArmArtefacts) -> float:
    """Row count of tokens.csv (atomic alphabet size, including the
    PAD row at idx 0)."""
    if not art.tokens_csv.exists():
        return float("nan")
    with open(art.tokens_csv) as f:
        reader = csv.reader(f)
        next(reader, None)
        return float(sum(1 for _ in reader))


_LONG_TAIL_MAX = 10


def _atom_counts_by_family(art: ArmArtefacts):
    """Yield ((op, reg, subreg), count) per real atom (count>0) from
    tokens.csv. Empty when tokens.csv is absent."""
    if not art.tokens_csv.exists():
        return
    with open(art.tokens_csv) as f:
        for row in csv.DictReader(f):
            c = int(row["count"])
            if c > 0:
                yield (row["op"], row["reg"], row["subreg"]), c


def _longtail_frac(art: ArmArtefacts) -> float:
    """Fraction of the atom alphabet (count>0) occurring < _LONG_TAIL_MAX
    times in training: the data-sparsity signal behind the content
    ceiling. FREQ_TRAJ collapsed it ~38% -> ~11%."""
    counts = [c for _, c in _atom_counts_by_family(art)]
    if not counts:
        return float("nan")
    return sum(c < _LONG_TAIL_MAX for c in counts) / len(counts)


def _worst_family_longtail_frac(art: ArmArtefacts) -> float:
    """Max long-tail fraction over (op, reg, subreg) families with >=50
    atoms: localises sparsity to a register family instead of averaging
    it away."""
    fam: dict = {}
    for key, c in _atom_counts_by_family(art):
        fam.setdefault(key, []).append(c)
    big = [v for v in fam.values() if len(v) >= 50]
    if not big:
        return float("nan")
    return max(sum(c < _LONG_TAIL_MAX for c in v) / len(v) for v in big)


def _encoded_tokens_per_song(art: ArmArtefacts) -> float:
    """Mean non-pad super-token count per training song. Read from
    ``train/*.blocks.npy`` (each file = one (composer, song, rotation),
    shape ``(n_blocks, seq_len+1)``, pad token = 0). Lower = better
    compression; the encoder's primary goal.
    """
    pat = str(art.work_dir / "train" / "**" / "*.blocks.npy")
    files = glob.glob(pat, recursive=True)
    if not files:
        return float("nan")
    script = (
        "import glob, json, numpy as np\n"
        "files = sorted(glob.glob('/work/train/**/*.blocks.npy', recursive=True))\n"
        "totals = [int(np.count_nonzero(np.load(f))) for f in files]\n"
        "print(json.dumps({'mean': sum(totals)/len(totals), 'n': len(totals)}))\n"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{art.work_dir}:/work:ro",
        "anarkiwi/preframr",
        "python3",
        "-c",
        script,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=120)
        return float(json.loads(out)["mean"])
    except (subprocess.SubprocessError, ValueError, KeyError):
        return float("nan")


_TB_EXTRACT_SCRIPT = (
    "import glob, json, os\n"
    "from tensorboard.backend.event_processing.event_accumulator "
    "import EventAccumulator\n"
    "pat = os.path.join('/tb', '**', 'events.out.tfevents.*')\n"
    "candidates = sorted(glob.glob(pat, recursive=True), key=os.path.getmtime)\n"
    "if not candidates:\n"
    "    print('{}')\n"
    "    raise SystemExit\n"
    "acc = EventAccumulator(candidates[-1], size_guidance={'scalars': 0})\n"
    "acc.Reload()\n"
    "out = {tag: [(e.step, e.value) for e in acc.Scalars(tag)] "
    "for tag in acc.Tags().get('scalars', [])}\n"
    "print(json.dumps(out))\n"
)


def _read_tb_scalars(tb_logs):
    """Read TB scalar series via the preframr docker image (host
    typically doesn't ship tensorboard). Each call spawns one
    ``docker run --rm`` -- ~5 s overhead, paid once per metric per
    arm at end-of-run, so the absolute cost is negligible."""
    if not os.path.isdir(tb_logs):
        return {}
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{tb_logs}:/tb:ro",
        "anarkiwi/preframr",
        "python3",
        "-c",
        _TB_EXTRACT_SCRIPT,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=120)
    except subprocess.SubprocessError:
        return {}
    return json.loads(out) if out.strip() else {}


def _val_loss_best(art: ArmArtefacts) -> float:
    if not art.tb_logs.exists():
        return float("nan")
    s = _read_tb_scalars(art.tb_logs)
    series = s.get("val_loss", [])
    if not series:
        return float("nan")
    return float(min(v for _, v in series))


def _val_acc_at_best_loss(art: ArmArtefacts) -> float:
    """val_acc at the epoch that minimised val_loss. Mirrors
    ``check_generalize.best_val_acc_at_min_loss``; reports the
    val_acc the ``ModelCheckpoint(monitor='val_loss')`` would pick."""
    if not art.tb_logs.exists():
        return float("nan")
    s = _read_tb_scalars(art.tb_logs)
    val_loss = s.get("val_loss", [])
    val_acc = s.get("val_acc", [])
    if not val_loss or not val_acc:
        return float("nan")
    best_step = min(val_loss, key=lambda sv: sv[1])[0]
    matches = [v for step, v in val_acc if step == best_step]
    return float(matches[0]) if matches else float("nan")


def _epochs_to_best_val_loss(art: ArmArtefacts) -> float:
    """Index (1-based) of the best-val_loss epoch. Visible-axis
    proxy for the convergence speed of an arm; the canonical-tier
    EarlyStopping default makes this variable across arms."""
    if not art.tb_logs.exists():
        return float("nan")
    s = _read_tb_scalars(art.tb_logs)
    val_loss = s.get("val_loss", [])
    if not val_loss:
        return float("nan")
    return float(1 + min(range(len(val_loss)), key=lambda i: val_loss[i][1]))


def _train_loss_final(art: ArmArtefacts) -> float:
    if not art.tb_logs.exists():
        return float("nan")
    s = _read_tb_scalars(art.tb_logs)
    series = s.get("train_loss_epoch") or s.get("train_loss") or []
    if not series:
        return float("nan")
    return float(series[-1][1])


def _wallclock_train_min(art: ArmArtefacts) -> float:
    """Train-stage wallclock recorded by the runner's _wallclock.json
    sidecar. Captures the cost the user paid for this arm."""
    sidecar = art.work_dir / "_wallclock.json"
    if not sidecar.exists():
        return float("nan")
    data = json.loads(sidecar.read_text())
    return float(data.get("train_elapsed_s", float("nan")) / 60.0)


def _val_loss_best_subset(subset: str):
    """Factory: per-subset ``val_loss_best`` extractor reading
    ``val_loss/<subset>`` from TB scalars. Multi-subset runs only --
    legacy single-subset runs only emit the un-suffixed ``val_loss``.
    """

    def _extract(art: ArmArtefacts) -> float:
        if not art.tb_logs.exists():
            return float("nan")
        s = _read_tb_scalars(art.tb_logs)
        series = s.get(f"val_loss/{subset}", [])
        if not series:
            return float("nan")
        return float(min(v for _, v in series))

    return _extract


def _val_acc_at_best_loss_subset(subset: str):
    """Factory: per-subset ``val_acc_at_best_loss`` extractor. Picks
    the val_acc value at the step that minimised the per-subset
    ``val_loss/<subset>`` series. Useful for cross-composer transfer
    reporting on prodlike-tier runs.
    """

    def _extract(art: ArmArtefacts) -> float:
        if not art.tb_logs.exists():
            return float("nan")
        s = _read_tb_scalars(art.tb_logs)
        val_loss = s.get(f"val_loss/{subset}", [])
        val_acc = s.get(f"val_acc/{subset}", [])
        if not val_loss or not val_acc:
            return float("nan")
        best_step = min(val_loss, key=lambda sv: sv[1])[0]
        matches = [v for step, v in val_acc if step == best_step]
        return float(matches[0]) if matches else float("nan")

    return _extract


def _val_loss_macro_best(art: ArmArtefacts) -> float:
    """Best epoch's equal-weight cross-subset ``val_loss_macro``.
    Reported only by multi-subset runs."""
    if not art.tb_logs.exists():
        return float("nan")
    s = _read_tb_scalars(art.tb_logs)
    series = s.get("val_loss_macro", [])
    if not series:
        return float("nan")
    return float(min(v for _, v in series))


def _val_acc_macro_at_best_loss(art: ArmArtefacts) -> float:
    """Equal-weight cross-subset ``val_acc_macro`` at the epoch that
    minimised the aggregate ``val_loss``. Multi-subset runs only."""
    if not art.tb_logs.exists():
        return float("nan")
    s = _read_tb_scalars(art.tb_logs)
    val_loss = s.get("val_loss", [])
    val_acc_macro = s.get("val_acc_macro", [])
    if not val_loss or not val_acc_macro:
        return float("nan")
    best_step = min(val_loss, key=lambda sv: sv[1])[0]
    matches = [v for step, v in val_acc_macro if step == best_step]
    return float(matches[0]) if matches else float("nan")


_EVAL_SUBSETS = (
    "eval_a",
    "eval_b_daglish",
    "eval_b_follin",
    "eval_b_crisps",
    "eval_b_mibri",
    "eval_b_marquis",
    "eval_b_dobek",
    "eval_b_winterberg",
    "eval_b_wilson",
)


METRIC_REGISTRY: dict[str, Callable[[ArmArtefacts], float]] = {
    "alphabet_size": _alphabet_size,
    "longtail_frac": _longtail_frac,
    "worst_family_longtail_frac": _worst_family_longtail_frac,
    "encoded_tokens_per_song": _encoded_tokens_per_song,
    "val_loss_best": _val_loss_best,
    "val_acc_at_best_loss": _val_acc_at_best_loss,
    "val_loss_macro_best": _val_loss_macro_best,
    "val_acc_macro_at_best_loss": _val_acc_macro_at_best_loss,
    "epochs_to_best_val_loss": _epochs_to_best_val_loss,
    "train_loss_final": _train_loss_final,
    "wallclock_train_min": _wallclock_train_min,
}
for _subset in _EVAL_SUBSETS:
    METRIC_REGISTRY[f"val_loss_{_subset}_best"] = _val_loss_best_subset(_subset)
    METRIC_REGISTRY[f"val_acc_{_subset}_at_best_loss"] = _val_acc_at_best_loss_subset(
        _subset
    )


def compute_metrics(spec, artefacts: ArmArtefacts) -> dict:
    """Apply each declared metric extractor; merge in spec-derived
    metrics. Result is dumped to artefacts.metrics_json.
    """
    out = {}
    derived = spec.derived_metrics or {}
    for name in spec.metrics:
        if name in derived:
            extractor = derived[name]
        else:
            extractor = METRIC_REGISTRY[name]
        try:
            out[name] = extractor(artefacts)
        except Exception as exc:  # pylint: disable=broad-except
            out[name] = float("nan")
            out[f"_{name}_error"] = str(exc)
    artefacts.metrics_json.write_text(json.dumps(out, indent=2))
    return out


def validate_metric_names(spec) -> None:
    """Fail-loud if a metric name in the spec doesn't resolve."""
    derived = set((spec.derived_metrics or {}).keys())
    unknown = [
        name
        for name in spec.metrics
        if name not in METRIC_REGISTRY and name not in derived
    ]
    if unknown:
        raise ValueError(
            f"experiment {spec.name}: unknown metric name(s) {unknown}; "
            f"register an extractor in metrics.py or add to "
            f"spec.derived_metrics"
        )
