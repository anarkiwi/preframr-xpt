"""Cross-arm report rendering."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


def _fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:.0f}"
        if abs(v) >= 10:
            return f"{v:.2f}"
        return f"{v:.4f}"
    return str(v)


def _aggregate(seed_values: list[float]):
    """Mean +/- std across seeds; falls back to the single value for
    seeds=1. Returns (mean, std) where std is None for n < 2."""
    if not seed_values:
        return float("nan"), None
    nums = [v for v in seed_values if not (isinstance(v, float) and math.isnan(v))]
    if not nums:
        return float("nan"), None
    if len(nums) == 1:
        return float(nums[0]), None
    return float(statistics.mean(nums)), float(statistics.stdev(nums))


def _delta(arm_value, baseline_value):
    """Signed delta vs baseline, or n/a when either side is missing."""
    if (
        arm_value is None
        or baseline_value is None
        or (isinstance(arm_value, float) and math.isnan(arm_value))
        or (isinstance(baseline_value, float) and math.isnan(baseline_value))
    ):
        return None
    return arm_value - baseline_value


def render(spec, results: dict[str, list[dict]], out_dir: Path) -> Path:
    """Write report.md + report.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = {}
    for arm in spec.arms:
        per_seed = results.get(arm.label, [])
        agg[arm.label] = {
            m: _aggregate([s.get(m, float("nan")) for s in per_seed])
            for m in spec.metrics
        }

    baseline_arm = next((a for a in spec.arms if a.baseline), None)

    lines = []
    lines.append(f"# Experiment: {spec.name}")
    lines.append("")
    lines.append(spec.doc.strip())
    lines.append("")
    lines.append(f"- **tier**: {spec.tier}")
    lines.append(f"- **seeds per arm**: {spec.seeds}")
    lines.append(f"- **arms**: {len(spec.arms)}")
    if baseline_arm is not None:
        lines.append(f"- **baseline**: `{baseline_arm.label}`")
    lines.append("")
    header = ["arm"] + list(spec.metrics)
    if baseline_arm is not None:
        header += [f"Δ {m}" for m in spec.metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for arm in spec.arms:
        row = [f"`{arm.label}`"]
        for m in spec.metrics:
            mean, std = agg[arm.label][m]
            cell = _fmt(mean)
            if std is not None:
                cell += f" ± {_fmt(std)}"
            row.append(cell)
        if baseline_arm is not None:
            for m in spec.metrics:
                arm_mean, _ = agg[arm.label][m]
                base_mean, _ = agg[baseline_arm.label][m]
                delta = _delta(arm_mean, base_mean)
                row.append(_fmt(delta) if delta is not None else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    overridden = [a for a in spec.arms if a.training_overrides]
    if overridden:
        lines.append("### Training overrides")
        lines.append("")
        lines.append(
            "These arms diverge from the spec's ``train_args`` baseline; "
            "their metrics aren't directly comparable to non-overridden "
            "arms. Document the divergence in the spec's docstring."
        )
        lines.append("")
        for a in overridden:
            lines.append(f"- `{a.label}`: `{a.training_overrides}`")
        lines.append("")

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines))

    json_path = out_dir / "report.json"
    json_path.write_text(
        json.dumps(
            {
                "name": spec.name,
                "tier": spec.tier,
                "seeds": spec.seeds,
                "arms": [
                    {
                        "label": a.label,
                        "extra_cargs": a.extra_cargs,
                        "training_overrides": a.training_overrides,
                        "baseline": a.baseline,
                        "metrics": {
                            m: {
                                "mean": agg[a.label][m][0],
                                "std": agg[a.label][m][1],
                                "per_seed": [
                                    s.get(m, float("nan"))
                                    for s in results.get(a.label, [])
                                ],
                            }
                            for m in spec.metrics
                        },
                    }
                    for a in spec.arms
                ],
            },
            indent=2,
            default=lambda v: None if isinstance(v, float) and math.isnan(v) else v,
        )
    )
    return md_path
