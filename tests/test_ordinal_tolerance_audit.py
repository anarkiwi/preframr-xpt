"""Unit tests for the tolerance-band bucketing: wrong_op / wrong_family / same-family
classification and cumulative within-tolerance accuracy on a synthetic vocab (no torch).
"""

import csv

from preframr_experiments.audit import ordinal_tolerance_audit as ota

#       id: op  reg subreg val
_VOCAB = [
    (45, 0, 0, 100),  # 0 FREQ_TRAJ family (0,0)
    (45, 0, 0, 102),  # 1 same family, +2 bins
    (45, 1, 0, 100),  # 2 FREQ_TRAJ different family
    (10, 1, 0, 21),  # 3 SET (other op)
]
_N_ATOMS = len(_VOCAB)  # ids >= 4 are merged pieces


def _cols():
    op = [r[0] for r in _VOCAB]
    reg = [r[1] for r in _VOCAB]
    subreg = [r[2] for r in _VOCAB]
    val = [r[3] for r in _VOCAB]
    return op, reg, subreg, val


def test_buckets_classify_and_tolerance():
    op, reg, subreg, val = _cols()
    #            gt: 0  0  0  0  0  3
    #          pred: 0  1  2  3  4  9   (last two: merge id / irrelevant, gt=3 skipped)
    gts = [0, 0, 0, 0, 0, 3]
    preds = [0, 1, 2, 3, 4, 9]
    res = ota.tolerance_buckets(
        preds, gts, op, reg, subreg, val, spotlight_op=45, n_atoms=_N_ATOMS
    )
    assert res["total"] == 5  # gt=3 (SET) excluded
    assert res["wrong_op"] == 2  # pred id 3 (SET) and id 4 (merge >= n_atoms)
    assert res["wrong_family"] == 1  # pred id 2 (reg differs)
    assert res["same_family"] == 2  # preds 0 (|Δ|0) and 1 (|Δ|2)
    assert res["dvals"] == [0, 2]
    wt = res["within_tol"]
    assert wt[0] == 1 / 5  # only the exact match
    assert wt[1] == 1 / 5  # |Δ|=2 not yet within 1
    assert wt[2] == 2 / 5  # both same-family preds within 2
    assert wt[16] == 2 / 5


def test_no_spotlight_positions_returns_empty_tolerance():
    op, reg, subreg, val = _cols()
    res = ota.tolerance_buckets([3], [3], op, reg, subreg, val, spotlight_op=45)
    assert res["total"] == 0
    assert res["within_tol"] == {}


def test_load_vocab(tmp_path):
    p = tmp_path / "tokens.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["op", "reg", "subreg", "val", "count", "n"])
        w.writeheader()
        for o, r, s, v in _VOCAB:
            w.writerow({"op": o, "reg": r, "subreg": s, "val": v, "count": 0, "n": 0})
    op, reg, subreg, val = ota.load_vocab(p)
    assert op == [45, 45, 45, 10]
    assert reg == [0, 0, 1, 1]
    assert val == [100, 102, 100, 21]
