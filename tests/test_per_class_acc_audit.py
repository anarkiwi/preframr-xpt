"""Tests for the per-class / per-tier accuracy auditor."""

from __future__ import annotations

import unittest

from preframr_experiments.audit.per_class_acc_audit import audit


class TestAudit(unittest.TestCase):
    def setUp(self):
        self.tier_map = {1: "structural", 2: "structural", 10: "content", 11: "content"}

    def test_perfect_match(self):
        result = audit([1, 2, 10, 11], [1, 2, 10, 11], self.tier_map)
        self.assertEqual(result["per_tier"]["structural"]["acc"], 1.0)
        self.assertEqual(result["per_tier"]["content"]["acc"], 1.0)
        self.assertEqual(result["content_over_structural"], 1.0)

    def test_apush_signature(self):
        gt = [1, 1, 1, 1, 2, 2, 10, 10, 11, 11]
        pred = [1, 1, 1, 1, 2, 2, 10, 99, 99, 99]
        r = audit(pred, gt, self.tier_map)
        self.assertEqual(r["per_tier"]["structural"]["acc"], 1.0)
        self.assertEqual(r["per_tier"]["content"]["acc"], 0.25)
        self.assertAlmostEqual(r["content_over_structural"], 0.25)

    def test_unknown_token_class(self):
        r = audit([5], [5], {1: "structural"})
        self.assertEqual(r["per_tier"]["_unknown"]["n"], 1)

    def test_empty_inputs(self):
        r = audit([], [], self.tier_map)
        self.assertEqual(r["n_positions"], 0)
        self.assertEqual(r["content_over_structural"], 0.0)

    def test_mismatched_lengths_uses_min(self):
        r = audit([1, 2, 3, 4], [1, 2], self.tier_map)
        self.assertEqual(r["n_positions"], 2)


if __name__ == "__main__":
    unittest.main()
