"""Unit tests for the melody predictability oracle: onset extraction from op45 rows, and
the n-gram / conditional-entropy / copy-from-history baselines on synthetic sequences
(periodic = predictable; random-uniform = aleatoric). No pandas, no torch."""

import random

from preframr_experiments.audit import melody_predictability as mp


def test_melody_lines_from_rows_pairs_v0_hi_lo():
    # one voice (reg 0), two trajectories: V0 = 0x0140=320 then 0x00FF=255
    rows = [
        (0, 0, 0),  # FLAGS (ignored)
        (0, 1, 0x01),  # V0_HI
        (0, 2, 0x40),  # V0_LO
        (0, 6, 99),  # DELTA shape sample (ignored)
        (0, 0, 1),  # FLAGS
        (0, 1, 0x00),  # V0_HI
        (0, 2, 0xFF),  # V0_LO
        (2, 1, 7),  # PW reg (not a freq reg, ignored)
        (2, 2, 7),
    ]
    lines = mp.melody_lines_from_rows(rows, regs=(0, 7, 14))
    assert lines == [[320, 255]]


def test_periodic_is_predictable():
    seqs = [[60, 62, 64] * 40]  # period-3 arp
    assert mp.ngram_accuracy(seqs, 2) > 0.99
    assert mp.copy_oracle(seqs, 2)[0] > 0.95
    h2, eff = mp.conditional_entropy(seqs, 2)
    assert h2 < 0.05  # near-zero conditional entropy
    assert eff < 1.1


def test_random_is_aleatoric():
    rng = random.Random(0)
    # 8 symbols, dense sampling -> unbiased plug-in entropy (sparse contexts would
    # spuriously look predictable via finite-sample downward bias)
    seqs = [[rng.randrange(8) for _ in range(6000)]]
    h0, _ = mp.conditional_entropy(seqs, 0)
    h1, _ = mp.conditional_entropy(seqs, 1)
    assert h0 > 2.9  # ~log2(8)=3
    assert h1 > 0.9 * h0  # 1-gram context does not reduce entropy -> aleatoric
    assert mp.ngram_accuracy(seqs, 1) < 0.2


def test_marginal_floor_and_copy_coverage():
    seqs = [[5, 5, 5, 5, 7]]
    floor, n = mp.marginal_floor(seqs)
    assert n == 5 and abs(floor - 4 / 5) < 1e-9
    _acc, cov = mp.copy_oracle(seqs, 1)
    # contexts at positions 1..4: (5)->5,(5)->5,(5)->7 with 5 seen before each time
    assert cov > 0.0


def test_analyze_shape():
    res = mp.analyze([[1, 2, 3, 1, 2, 3, 1, 2, 3]])
    for key in (
        "marginal_floor",
        "ngram_acc",
        "cond_entropy_bits",
        "copy_oracle",
        "n_onsets",
    ):
        assert key in res
    assert set(res["ngram_acc"]) == {1, 2, 3}
