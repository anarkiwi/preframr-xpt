"""Tests for the calibration probe (pure measurement)."""

from __future__ import annotations

import unittest

from preframr_experiments.audit.calibration_audit import ece, summarize


class TestECE(unittest.TestCase):
    def test_perfectly_calibrated(self):
        # confidence 0.9, accuracy 0.9 in the same bin -> ECE 0
        self.assertAlmostEqual(ece([0.9] * 10, [1] * 9 + [0], n_bins=10), 0.0)

    def test_overconfident(self):
        # confidence 0.95, accuracy 0 -> ECE ~0.95
        self.assertAlmostEqual(ece([0.95] * 10, [0] * 10, n_bins=10), 0.95)

    def test_empty(self):
        self.assertIsNone(ece([], []))

    def test_confidence_one_clamps_to_last_bin(self):
        self.assertAlmostEqual(ece([1.0] * 4, [1, 1, 1, 1], n_bins=10), 0.0)


class TestSummarize(unittest.TestCase):
    def test_underconfident_gap(self):
        s = summarize([0.9, 0.9, 0.9, 0.9], [1, 1, 1, 1], [0.1, 0.1, 0.1, 0.1])
        self.assertEqual(s["accuracy"], 1.0)
        self.assertAlmostEqual(s["overconfidence"], -0.1)  # conf<acc -> underconfident
        self.assertAlmostEqual(s["mean_entropy_nats"], 0.1)

    def test_empty(self):
        self.assertEqual(summarize([], [], []), {"n": 0})


if __name__ == "__main__":
    unittest.main()
