"""Unit tests for the reusable content-tier reader: id->op mapping, by-op content
pooling, baseline auto-pick, and cross-seed/cross-arm aggregation on synthetic
audit_per_class.json + tokens.csv fixtures (no torch, no real checkpoints)."""

import csv
import json
from pathlib import Path

from preframr_experiments.audit import content_tier_report as ctr

_TOKENS = [
    {"op": 45, "reg": 0, "subreg": 0, "val": 100},
    {"op": 45, "reg": 0, "subreg": 0, "val": 101},
    {"op": 10, "reg": 1, "subreg": 0, "val": 21},
    {"op": 99, "reg": 2, "subreg": 0, "val": 0},
    {"op": 0, "reg": -1, "subreg": -1, "val": 0},
]


def _write_tokens(path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["op", "reg", "subreg", "val", "count", "n"])
        w.writeheader()
        for row in _TOKENS:
            w.writerow({**row, "count": 0, "n": 0})


def _per_class(op45_hits, set_hits):
    return {
        "0": {"n": 100, "hits": op45_hits, "acc": op45_hits / 100, "tier": "content"},
        "1": {"n": 100, "hits": op45_hits, "acc": op45_hits / 100, "tier": "content"},
        "2": {"n": 200, "hits": set_hits, "acc": set_hits / 200, "tier": "content"},
        "3": {"n": 300, "hits": 200, "acc": 200 / 300, "tier": "structural"},
    }


def _subset(op45_hits, set_hits):
    pc = _per_class(op45_hits, set_hits)
    c_n = 100 + 100 + 200
    c_h = 2 * op45_hits + set_hits
    return {
        "per_class": pc,
        "per_tier": {
            "content": {"n": c_n, "hits": c_h, "acc": c_h / c_n},
            "structural": {"n": 300, "hits": 200, "acc": 200 / 300},
        },
        "content_over_structural": (c_h / c_n) / (200 / 300),
        "n_positions": c_n + 300,
    }


def _famsubset(acc, n=50):
    return {
        "per_class": {},
        "per_tier": {"content": {"n": n, "hits": int(acc * n), "acc": acc}},
        "content_over_structural": acc,
        "n_positions": n,
    }


def _write_arm(root: Path, arm: str, op45_hits, set_hits, famx, n_seeds=2) -> None:
    for s in range(n_seeds):
        sd = root / arm / f"seed{s}"
        sd.mkdir(parents=True)
        _write_tokens(sd / "tokens.csv")
        doc = {
            "ckpt": f"{arm}/seed{s}",
            "subsets": {
                "eval_a": _subset(op45_hits, set_hits),
                "eval_b_famx": _famsubset(famx),
            },
        }
        (sd / "audit_per_class.json").write_text(json.dumps(doc))


def test_id_to_op(tmp_path):
    _write_tokens(tmp_path / "tokens.csv")
    id_op = ctr.id_to_op(tmp_path / "tokens.csv")
    assert id_op == {0: 45, 1: 45, 2: 10, 3: 99, 4: 0}


def test_by_op_content_groups_and_excludes_non_content():
    pc = _per_class(op45_hits=30, set_hits=100)
    id_op = {0: 45, 1: 45, 2: 10, 3: 99}
    grouped = ctr.by_op_content(pc, id_op)
    assert grouped[45] == (60, 200)
    assert grouped[10] == (100, 200)
    assert 99 not in grouped


def test_pooled_acc_and_mean_std():
    assert ctr.pooled_acc([(60, 200), (100, 200)]) == 160 / 400
    assert ctr.pooled_acc([(0, 0)]) is None
    m, s = ctr.mean_std([0.4, 0.4])
    assert m == 0.4 and s == 0.0
    assert ctr.mean_std([None]) == (None, 0.0)


def test_pick_baseline():
    assert ctr.pick_baseline(["anchored", "unanchored"], None) == "unanchored"
    assert ctr.pick_baseline(["a", "z"], None) == "z"
    assert ctr.pick_baseline(["anchored", "unanchored"], "anchored") == "anchored"


def test_compare_and_deltas(tmp_path):
    _write_arm(tmp_path, "anchored", op45_hits=30, set_hits=100, famx=0.30)
    _write_arm(tmp_path, "unanchored", op45_hits=5, set_hits=100, famx=0.20)
    comp = ctr.compare(tmp_path)
    assert comp["baseline"] == "unanchored"

    anc, base = comp["arms"]["anchored"], comp["arms"]["unanchored"]
    assert anc["spotlight_subset"] == "eval_a"
    am, _ = ctr.mean_std(anc["subset_content"]["eval_a"])
    bm, _ = ctr.mean_std(base["subset_content"]["eval_a"])
    assert am == (160 / 400) and bm == (110 / 400)
    assert round(am - bm, 4) == 0.125

    assert ctr.pooled_acc([anc["op_pool"][45]]) == 60 / 200
    assert ctr.pooled_acc([base["op_pool"][45]]) == 10 / 200
    assert anc["op_pool"][10] == (200, 400)


def test_ft_subreg_bucket():
    assert ctr.ft_subreg_bucket(1) == "V0 onset"
    assert ctr.ft_subreg_bucket(2) == "V0 onset"
    assert ctr.ft_subreg_bucket(6) == "DELTA shape"
    assert ctr.ft_subreg_bucket(0) == "other header"


def test_melodic_onset_bucket_unifies_op45_op48_op47_on_freq_regs():
    assert ctr.melodic_onset_bucket(45, 0, 1) == "V0 onset"
    assert ctr.melodic_onset_bucket(45, 7, 2) == "V0 onset"
    assert ctr.melodic_onset_bucket(48, 0, -1) == "V0 onset"
    assert ctr.melodic_onset_bucket(48, 14, -1) == "V0 onset"
    assert ctr.melodic_onset_bucket(47, 0, 2) == "V0 onset"
    assert ctr.melodic_onset_bucket(47, 0, 3) == "V0 onset"
    assert ctr.melodic_onset_bucket(45, 0, 6) == "DELTA shape"
    assert ctr.melodic_onset_bucket(45, 0, 0) == "other header"
    assert ctr.melodic_onset_bucket(45, 2, 1) is None
    assert ctr.melodic_onset_bucket(48, 2, -1) is None
    assert ctr.melodic_onset_bucket(0, 0, -1) is None
    assert ctr.melodic_onset_bucket(47, 0, 0) is None


def test_onset_breakdown(tmp_path):
    toks = [(45, 0, 1, 0), (45, 0, 2, 0), (45, 0, 6, 0), (0, 1, -1, 0)]
    arm = tmp_path / "interval" / "seed0"
    arm.mkdir(parents=True)
    with open(arm / "tokens.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["op", "reg", "subreg", "val", "count", "n"])
        w.writeheader()
        for op, reg, sr, val in toks:
            w.writerow(
                {"op": op, "reg": reg, "subreg": sr, "val": val, "count": 0, "n": 0}
            )
    doc = {
        "subsets": {
            "eval": {
                "per_class": {
                    "0": {"n": 100, "hits": 40, "acc": 0.4, "tier": "content"},
                    "1": {"n": 100, "hits": 40, "acc": 0.4, "tier": "content"},
                    "2": {"n": 200, "hits": 10, "acc": 0.05, "tier": "content"},
                    "3": {"n": 50, "hits": 25, "acc": 0.5, "tier": "content"},
                },
                "per_tier": {"content": {"n": 450, "hits": 115, "acc": 0.255}},
            }
        }
    }
    (arm / "audit_per_class.json").write_text(json.dumps(doc))
    rep = ctr.onset_breakdown(tmp_path, op=45)
    arms = rep["arms"]["interval"]
    assert arms["V0 onset"] == (80, 200)
    assert arms["DELTA shape"] == (10, 200)
    assert "other header" not in arms
    assert ctr.pooled_acc([arms["V0 onset"]]) == 80 / 200


def test_format_text_renders_spotlight_and_family(tmp_path):
    _write_arm(tmp_path, "anchored", op45_hits=30, set_hits=100, famx=0.30)
    _write_arm(tmp_path, "unanchored", op45_hits=5, set_hits=100, famx=0.20)
    text = ctr.format_text(ctr.compare(tmp_path), spotlight_op=45)
    assert "spotlight subset = eval_a" in text
    assert "op45" in text
    assert "famx" in text
    assert "Δ +0.125" in text
