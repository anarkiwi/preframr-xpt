"""Tests for the effective-context probe (pure measurement)."""

from __future__ import annotations

import unittest

from preframr_experiments.audit.effective_context_audit import (
    _probe_positions,
    context_curve,
    saturation_k,
    summarize,
)


class TestCurve(unittest.TestCase):
    def test_curve_sorted(self):
        c = context_curve({256: [9, 10], 16: [8, 10], 64: [9, 10]})
        self.assertEqual([p["k"] for p in c], [16, 64, 256])
        self.assertEqual(c[0]["acc"], 0.8)

    def test_saturation_picks_smallest_within_tol(self):
        c = context_curve({16: [80, 100], 64: [90, 100], 256: [91, 100]})
        # max ~0.91; tol 0.02 -> 0.64 (0.80) is below 0.89, 0.90@64 qualifies
        self.assertEqual(saturation_k(c, tol=0.02), 64)

    def test_saturation_flat_is_smallest(self):
        c = context_curve({16: [9, 10], 64: [9, 10]})
        self.assertEqual(saturation_k(c, tol=0.02), 16)

    def test_saturation_empty(self):
        self.assertIsNone(saturation_k([], tol=0.02))

    def test_summarize_gain(self):
        s = summarize({16: [8, 10], 256: [9, 10]}, tol=0.02)
        self.assertAlmostEqual(s["long_context_gain"], 0.1)
        self.assertEqual(s["saturation_k"], 256)


class TestProbePositions(unittest.TestCase):
    def test_none_when_block_too_short(self):
        self.assertEqual(_probe_positions(10, 16, 5), [])

    def test_all_when_under_cap(self):
        self.assertEqual(_probe_positions(20, 16, 0), [16, 17, 18, 19])

    def test_even_spacing_capped(self):
        pos = _probe_positions(100, 16, 5)
        self.assertEqual(len(pos), 5)
        self.assertEqual(pos[0], 16)
        self.assertEqual(pos[-1], 99)
        self.assertEqual(pos, sorted(pos))


if __name__ == "__main__":
    unittest.main()
