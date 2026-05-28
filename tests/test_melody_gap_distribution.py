"""Unit tests for melody_gap_distribution: melody_positions (boolean LUT indexing),
gap_stats (k-back diff percentiles), within_context (fraction inside attention window),
pool_gap_diffs (per-block, no cross-block gaps)."""

import numpy as np

from preframr_experiments.audit import melody_gap_distribution as mgd


def _melody_lut(vocab, melody_uids):
    lut = np.zeros(vocab, dtype=bool)
    for u in melody_uids:
        lut[u] = True
    return lut


def test_melody_positions_basic():
    lut = _melody_lut(10, {3, 7})
    seq = np.array([0, 3, 1, 7, 3, 5, 7], dtype=np.int64)
    out = mgd.melody_positions(seq, lut)
    assert out.tolist() == [1, 3, 4, 6]


def test_melody_positions_no_melody():
    lut = _melody_lut(10, set())
    seq = np.array([0, 1, 2, 3], dtype=np.int64)
    assert mgd.melody_positions(seq, lut).size == 0


def test_gap_stats_k1_and_k2():
    positions = np.array([0, 10, 30, 60, 100], dtype=np.int64)
    res = mgd.gap_stats(positions, kbacks=(1, 2))
    assert res["n_positions"] == 5
    g1 = res["gaps"][1]
    assert g1["n"] == 4
    assert g1["median"] == 25.0
    assert g1["max"] == 40
    g2 = res["gaps"][2]
    assert g2["n"] == 3
    assert g2["median"] == 50.0
    assert g2["max"] == 70


def test_gap_stats_insufficient_positions():
    res = mgd.gap_stats(np.array([5], dtype=np.int64), kbacks=(1, 2))
    assert res["gaps"][1] is None
    assert res["gaps"][2] is None


def test_within_context_k2():
    positions = np.array([0, 100, 200, 1500, 2000, 2100], dtype=np.int64)
    wc = mgd.within_context(positions, ctx_len=1024, k=2)
    assert wc["n"] == 4
    assert wc["frac_in_ctx"] == 0.5
    assert wc["frac_in_half_ctx"] == 0.25


def test_pool_gap_diffs_no_cross_block():
    b1 = np.array([0, 100, 200], dtype=np.int64)
    b2 = np.array([0, 50], dtype=np.int64)
    b3 = np.array([10], dtype=np.int64)
    pooled = mgd.pool_gap_diffs([b1, b2, b3], k=1)
    assert sorted(pooled.tolist()) == [50, 100, 100]


def test_summarise_diffs_empty_is_none():
    assert mgd.summarise_diffs(np.empty(0, dtype=np.int64)) is None


def test_summarise_diffs_basic():
    diffs = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    res = mgd.summarise_diffs(diffs)
    assert res["n"] == 5
    assert res["median"] == 30.0
    assert res["max"] == 50
    assert 30.0 <= res["pct"][75] <= 50.0
