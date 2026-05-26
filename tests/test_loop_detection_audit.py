"""Tests for the greedy-decode loop-detection auditor."""

from __future__ import annotations

import unittest

from preframr_experiments.audit.loop_detection_audit import detect_tail_cycle


class TestDetectTailCycle(unittest.TestCase):
    def test_random_no_cycle(self):
        tokens = list(range(200))
        self.assertIsNone(detect_tail_cycle(tokens))

    def test_constant_token_period_one(self):
        tokens = [7] * 200
        v = detect_tail_cycle(tokens)
        self.assertIsNotNone(v)
        self.assertEqual(v["period"], 1)

    def test_period_two(self):
        tokens = [3, 5] * 100
        v = detect_tail_cycle(tokens)
        self.assertIsNotNone(v)
        self.assertEqual(v["period"], 2)

    def test_period_seven(self):
        tokens = list(range(50)) + [1, 2, 3, 4, 5, 6, 7] * 30
        v = detect_tail_cycle(tokens)
        self.assertIsNotNone(v)
        self.assertEqual(v["period"], 7)

    def test_partial_cycle_rejected(self):
        tokens = list(range(50)) + [1, 2, 3, 4, 5, 6, 7] * 30 + [9, 9, 9, 9, 9]
        v = detect_tail_cycle(tokens, tail_window=128)
        self.assertIsNone(v)

    def test_short_input_returns_none(self):
        self.assertIsNone(detect_tail_cycle([1, 2, 3]))

    def test_min_repeats_threshold(self):
        tokens = list(range(50)) + [1, 2] * 2 + list(range(50))
        self.assertIsNone(detect_tail_cycle(tokens))


if __name__ == "__main__":
    unittest.main()
