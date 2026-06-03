"""Unit tests for the pure-stdlib learnability metrics (no torch, no tokenizer). Each metric
is probed where it is well-sampled: a periodic stream is low-entropy-rate / high-copy; a
well-sampled small-alphabet i.i.d. stream has history-independent entropy and ~0 MI; copy-rate
separates all-distinct (0) from periodic (~1); MI peaks at the generating period.
"""

import random

from preframr_experiments.audit import learnability_triage as lt


def _iid(n, k, seed=1234):
    r = random.Random(seed)
    return [[r.randrange(k) for _ in range(n)]]


def test_periodic_is_predictable():
    seqs = [[0, 1, 2, 3] * 200]
    assert lt.entropy_rate(seqs, 1) < 0.05
    assert lt.induction_copy_rate(seqs) > 0.97
    assert lt.first_occurrence_rate(seqs) < 0.02


def test_iid_history_does_not_help():
    # well-sampled: 4 symbols, 8000 draws -> 16 bigrams each ~500 samples
    seqs = _iid(8000, 4)
    h0 = lt.entropy_rate(seqs, 0)
    h1 = lt.entropy_rate(seqs, 1)
    assert 1.8 < h0 < 2.05
    assert abs(h1 - h0) < 0.15
    assert lt.mutual_information_lag(seqs, 1) < 0.05


def test_copy_rate_discriminates():
    distinct = [list(range(5000))]  # no bigram ever repeats
    periodic = [[0, 1, 2, 3] * 200]  # every bigram repeats
    assert lt.induction_copy_rate(distinct) < 0.01
    assert lt.induction_copy_rate(periodic) > 0.97


def test_mi_peaks_at_period():
    seqs = [[0, 1, 2, 3, 4] * 400]  # x_t determined by x_{t-5}
    assert lt.mutual_information_lag(seqs, 5) > lt.mutual_information_lag(seqs, 4)
    assert lt.mutual_information_lag(seqs, 5) > 2.0


def test_entropy_rate_drops_with_memory():
    seqs = [[0, 1, 0, 2, 1, 1, 2, 0, 2, 2, 1, 0] * 300]
    assert lt.entropy_rate(seqs, 3) <= lt.entropy_rate(seqs, 0) + 1e-9


def test_config_label_parsing():
    assert lt._config_label("codebook") == "codebook"
    assert lt._config_label("B=skeleton_pass+stamp_pass") == "B"


def test_codebook_skeleton_differ_by_exactly_the_codebooks():
    # skeleton = codebook - DEF_REF_CODEBOOKS, so the codebook arm minus skeleton must be
    # exactly the DEF->REF codebook passes present in the arm (clean isolation for the A/B/C).
    codebook = set(lt._BASE + lt._CODEBOOK)
    skeleton = codebook - set(lt._DEF_REF_CODEBOOKS)
    removed = codebook - skeleton
    assert removed == (codebook & set(lt._DEF_REF_CODEBOOKS))
    assert {"stamp_pass", "wavetable_pass", "patch_pass"} <= removed
    # the substrate + parametric sweeps stay on both sides
    assert {"skeleton_pass", "sweep_pass"} <= skeleton


def test_summarize_shapes():
    s = lt.summarize([[0, 1, 2, 3] * 50], frames=100, kmax=4, maxlag=8)
    assert len(s["h_k_per_token"]) == 5
    assert len(s["h_k_per_frame"]) == 5
    assert len(s["mi_lag"]) == 8
    assert s["tokens"] == 200
    assert s["tokens_per_frame"] == 2.0
