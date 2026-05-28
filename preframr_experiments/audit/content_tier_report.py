#!/usr/bin/env python3
"""Reusable content-tier read for representation A/Bs. Consumes the per-arm-seed
``audit_per_class.json`` (emitted by ``audit_checkpoint_per_class``) plus the seed's
``tokens.csv`` (token id -> op), and reports the decisive gate: per-tier content vs
structural accuracy (pooled across seeds + per-seed mean/std, with target-minus-baseline
deltas), per-family ``eval_b_*`` stratification, and a by-op breakdown WITHIN the content
tier (spotlight op, default FREQ_TRAJ op 45). Replaces the bespoke, hardcoded
``/scratch/tmp/{freqtraj_distribution,parse_per_class}.py``. Pure stdlib (no torch), so
it runs on the host and is unit-tested.

Arms and seeds are auto-discovered under ``--results-root`` (each arm dir holds
``seed*/audit_per_class.json`` + ``seed*/tokens.csv``). ``--baseline-arm`` names the arm
to subtract; if omitted, the alphabetically-last arm or an arm literally named
``baseline``/``unanchored`` is used (override when ambiguous)."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional

FREQ_TRAJ_OP = 45
_BASELINE_HINTS = ("baseline", "unanchored", "full", "control", "off")


def id_to_op(tokens_csv: Path) -> dict[int, int]:
    """Token id (row index in tokens.csv) -> op."""
    out: dict[int, int] = {}
    with open(tokens_csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            out[i] = int(row["op"])
    return out


def pooled_acc(pairs) -> Optional[float]:
    """(hits, n) pairs -> pooled hits/sum(n); None when no positions."""
    total_h = sum(h for h, n in pairs)
    total_n = sum(n for h, n in pairs)
    return (total_h / total_n) if total_n else None


def mean_std(xs) -> tuple[Optional[float], float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, 0.0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def by_op_content(per_class: dict, id_op: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Within the content tier, group (hits, n) by op."""
    acc: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for sid, d in per_class.items():
        if d.get("tier") != "content":
            continue
        op = id_op.get(int(sid))
        if op is None:
            continue
        acc[op][0] += int(d["hits"])
        acc[op][1] += int(d["n"])
    return {op: (h, n) for op, (h, n) in acc.items()}


def id_to_op_subreg(tokens_csv: Path) -> dict[int, tuple[int, int, int]]:
    """Token id -> (op, reg, subreg)."""
    out: dict[int, tuple[int, int, int]] = {}
    with open(tokens_csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            out[i] = (int(row["op"]), int(row["reg"]), int(row["subreg"]))
    return out


_FT_ONSET_SUBREGS = (1, 2)
_FT_DELTA_SUBREGS = (6,)
_FREQ_TRAJ_REGS_FROZEN = frozenset((0, 7, 14))
_NUDGE_PITCH_SUBREGS = frozenset((2, 3))
FREQ_TRAJ_OP_CONST = 45
FREQ_ONSET_OP_CONST = 48
FREQ_NUDGE_OP_CONST = 47


def ft_subreg_bucket(subreg: int) -> str:
    """FREQ_TRAJ subreg -> melodic role: V0 onset (1,2) vs DELTA shape (6) vs header.
    Subreg-only legacy bucket; the unified-melody read uses ``melodic_onset_bucket``."""
    if subreg in _FT_ONSET_SUBREGS:
        return "V0 onset"
    if subreg in _FT_DELTA_SUBREGS:
        return "DELTA shape"
    return "other header"


def melodic_onset_bucket(op: int, reg: int, subreg: int):
    """Op-aware melodic-content bucket on FREQ regs (0/7/14): groups op45 trajectory V0 +
    op48 FREQ_ONSET + op47 FREQ_NUDGE pitch (HI/LO) as the unified V0 onset, op45 subreg 6
    as DELTA shape. Non-freq / non-melodic rows return None."""
    if reg not in _FREQ_TRAJ_REGS_FROZEN:
        return None
    if op == FREQ_ONSET_OP_CONST:
        return "V0 onset"
    if op == FREQ_TRAJ_OP_CONST and subreg in _FT_ONSET_SUBREGS:
        return "V0 onset"
    if op == FREQ_NUDGE_OP_CONST and subreg in _NUDGE_PITCH_SUBREGS:
        return "V0 onset"
    if op == FREQ_TRAJ_OP_CONST and subreg in _FT_DELTA_SUBREGS:
        return "DELTA shape"
    if op == FREQ_TRAJ_OP_CONST:
        return "other header"
    return None


_FT_BUCKETS = ("V0 onset", "DELTA shape", "other header")


def onset_breakdown(
    results_root: Path, op: int = FREQ_TRAJ_OP, baseline_arm: Optional[str] = None
) -> dict:
    """Per arm, pooled (hits, n) for ``op`` split by FREQ_TRAJ subreg role (V0 onset /
    DELTA shape / header), on each arm's spotlight subset across seeds. The V0-onset acc
    is the decisive melodic-learnability number (vs the shape-dominated aggregate op acc).
    """
    arms: dict[str, dict] = {}
    for arm_dir in discover_arms(results_root):
        pooled: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for sd in seed_dirs(arm_dir):
            d = json.load(open(sd / "audit_per_class.json"))
            subs = d.get("subsets", {})
            if not subs:
                continue
            id_os = id_to_op_subreg(sd / "tokens.csv")
            sp = max(subs, key=lambda k: _content_n(subs[k]))
            for sid, cls in subs[sp]["per_class"].items():
                meta = id_os.get(int(sid))
                if not meta:
                    continue
                m_op, m_reg, m_subreg = meta
                name = melodic_onset_bucket(m_op, m_reg, m_subreg)
                if name is None:
                    continue
                pooled[name][0] += int(cls["hits"])
                pooled[name][1] += int(cls["n"])
        arms[arm_dir.name] = {b: (h, n) for b, (h, n) in pooled.items()}
    return {"op": op, "baseline": pick_baseline(list(arms), baseline_arm), "arms": arms}


def format_onset(report: dict) -> str:
    op, base, arms = report["op"], report["baseline"], report["arms"]
    order = [a for a in arms if a != base] + ([base] if base in arms else [])
    lines = [
        f"=== op{op} acc by subreg role (V0 onset = the melody), pooled across seeds ==="
    ]
    for a in order:
        parts = []
        for b in _FT_BUCKETS:
            hn = arms[a].get(b)
            parts.append(
                f"{b} {_fmt(pooled_acc([hn]) if hn else None)} (n={hn[1] if hn else 0})"
            )
        lines.append(f"  {a:12} " + "  ".join(parts))
    return "\n".join(lines)


def _content_n(subset_obj: dict) -> int:
    c = subset_obj.get("per_tier", {}).get("content", {})
    return int(c.get("n", 0)) if isinstance(c, dict) else 0


def seed_dirs(arm_dir: Path) -> list[Path]:
    return sorted(
        p for p in arm_dir.glob("seed*") if (p / "audit_per_class.json").exists()
    )


def discover_arms(results_root: Path) -> list[Path]:
    return sorted(p for p in results_root.iterdir() if p.is_dir() and seed_dirs(p))


def aggregate_arm(arm_dir: Path) -> dict:
    """Aggregate one arm across its seeds. Spotlight subset = the one with the most
    content positions; by-op pooling and content_over_structural are read there."""
    subset_content: dict[str, list] = defaultdict(list)
    subset_struct: dict[str, list] = defaultdict(list)
    cos: list[float] = []
    op_pool: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    spotlight = None
    seeds = seed_dirs(arm_dir)
    for sd in seeds:
        d = json.load(open(sd / "audit_per_class.json"))
        subs = d.get("subsets", {})
        if not subs:
            continue
        id_op = id_to_op(sd / "tokens.csv")
        spotlight = max(subs, key=lambda k: _content_n(subs[k]))
        for name, obj in subs.items():
            pt = obj.get("per_tier", {})
            c, s = pt.get("content"), pt.get("structural")
            if isinstance(c, dict):
                subset_content[name].append(c.get("acc"))
            if isinstance(s, dict):
                subset_struct[name].append(s.get("acc"))
            if name == spotlight and "content_over_structural" in obj:
                cos.append(obj["content_over_structural"])
        for op, (h, n) in by_op_content(subs[spotlight]["per_class"], id_op).items():
            op_pool[op][0] += h
            op_pool[op][1] += n
    return {
        "arm": arm_dir.name,
        "n_seeds": len(seeds),
        "spotlight_subset": spotlight,
        "subset_content": {k: list(v) for k, v in subset_content.items()},
        "subset_struct": {k: list(v) for k, v in subset_struct.items()},
        "cos": cos,
        "op_pool": {op: (h, n) for op, (h, n) in op_pool.items()},
    }


def pick_baseline(arm_names: list[str], explicit: Optional[str]) -> Optional[str]:
    if explicit:
        if explicit not in arm_names:
            raise ValueError(f"--baseline-arm {explicit!r} not in arms {arm_names}")
        return explicit
    for hint in _BASELINE_HINTS:
        for name in arm_names:
            if name == hint:
                return name
    return sorted(arm_names)[-1] if arm_names else None


def compare(results_root: Path, baseline_arm: Optional[str] = None) -> dict:
    arms = {a.name: aggregate_arm(a) for a in discover_arms(results_root)}
    if not arms:
        raise FileNotFoundError(
            f"no arm dirs with seed*/audit_per_class.json under {results_root}"
        )
    base = pick_baseline(list(arms), baseline_arm)
    return {"results_root": str(results_root), "baseline": base, "arms": arms}


def _fmt(x: Optional[float], width: int = 0, prec: int = 3) -> str:
    s = "n/a" if x is None else f"{x:.{prec}f}"
    return s.rjust(width) if width else s


def format_text(comparison: dict, spotlight_op: int = FREQ_TRAJ_OP) -> str:
    arms = comparison["arms"]
    base = comparison["baseline"]
    order = [a for a in arms if a != base] + ([base] if base in arms else [])
    base_rep = arms.get(base)
    lines = [f"results_root: {comparison['results_root']}", f"baseline arm: {base}", ""]

    spotlight = next((arms[a]["spotlight_subset"] for a in order), None)
    lines.append(f"=== per-tier content acc (spotlight subset = {spotlight}) ===")
    for a in order:
        rep = arms[a]
        cm, cs = mean_std(rep["subset_content"].get(spotlight, []))
        sm, _ = mean_std(rep["subset_struct"].get(spotlight, []))
        com, _ = mean_std(rep["cos"])
        per_seed = ", ".join(_fmt(x) for x in rep["subset_content"].get(spotlight, []))
        delta = ""
        if base_rep is not None and a != base:
            bm, _ = mean_std(base_rep["subset_content"].get(spotlight, []))
            if cm is not None and bm is not None:
                delta = f"  Δ {cm - bm:+.3f}"
        lines.append(
            f"  {a:14} content {_fmt(cm)} ± {cs:.3f}{delta}  "
            f"struct {_fmt(sm)}  c/s {_fmt(com)}  (seeds: {per_seed})"
        )

    lines.append("")
    lines.append(
        f"=== by-op content acc, spotlight op {spotlight_op} (pooled across seeds) ==="
    )
    for a in order:
        pool = arms[a]["op_pool"]
        content_overall = pooled_acc(list(pool.values()))
        sp = pool.get(spotlight_op)
        sp_acc = pooled_acc([sp]) if sp else None
        sp_share = (sp[1] / sum(n for _, n in pool.values())) if (sp and pool) else 0.0
        top = sorted(pool.items(), key=lambda kv: -kv[1][1])[:6]
        top_str = ", ".join(
            f"op{op}:{pooled_acc([(h,n)]):.3f}(n={n})" for op, (h, n) in top
        )
        lines.append(
            f"  {a:14} content-overall {_fmt(content_overall)}  "
            f"op{spotlight_op} {_fmt(sp_acc)} ({sp_share*100:.1f}% of content pos)"
        )
        lines.append(f"      top content ops by n: {top_str}")

    fam_names = sorted(
        {
            name[len("eval_b_") :]
            for a in arms
            for name in arms[a]["subset_content"]
            if name.startswith("eval_b_")
        }
    )
    if fam_names:
        lines.append("")
        lines.append("=== eval_b per-family content acc ===")
        header = "  " + "family".ljust(14) + "".join(a.rjust(13) for a in order)
        lines.append(header)
        for fam in fam_names:
            row = "  " + fam.ljust(14)
            for a in order:
                m, _ = mean_std(arms[a]["subset_content"].get(f"eval_b_{fam}", []))
                row += _fmt(m, 13)
            lines.append(row)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Experiment results dir holding <arm>/seed*/audit_per_class.json + tokens.csv.",
    )
    ap.add_argument(
        "--baseline-arm",
        default=None,
        help="Arm to subtract (default: an arm named baseline/unanchored/etc., else last).",
    )
    ap.add_argument(
        "--op",
        type=int,
        default=FREQ_TRAJ_OP,
        help="Spotlight op (default 45 = FREQ_TRAJ).",
    )
    ap.add_argument(
        "--onset",
        action="store_true",
        help="Also split --op accuracy by subreg role (V0 onset vs DELTA shape) -- the "
        "decisive melodic-learnability read for FREQ_TRAJ.",
    )
    ap.add_argument(
        "--json", action="store_true", help="Emit raw comparison JSON instead of text."
    )
    cli = ap.parse_args()
    comparison = compare(cli.results_root, cli.baseline_arm)
    if cli.json:
        print(json.dumps(comparison, indent=2, default=str))
    else:
        print(format_text(comparison, cli.op))
        if cli.onset:
            print()
            print(
                format_onset(
                    onset_breakdown(cli.results_root, cli.op, cli.baseline_arm)
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
