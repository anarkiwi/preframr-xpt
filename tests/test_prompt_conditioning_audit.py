"""Tests for the prompt-conditioning auditor."""

from __future__ import annotations

import unittest

from preframr_experiments.audit.prompt_conditioning_audit import (
    audit_group,
    jaccard,
    ngram_overlap,
)


class TestPrimitives(unittest.TestCase):
    def test_jaccard_identical(self):
        self.assertEqual(jaccard([1, 2, 3], [1, 2, 3]), 1.0)

    def test_jaccard_disjoint(self):
        self.assertEqual(jaccard([1, 2], [3, 4]), 0.0)

    def test_ngram_overlap_identical(self):
        self.assertEqual(ngram_overlap([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], n=2), 1.0)

    def test_ngram_overlap_too_short(self):
        self.assertEqual(ngram_overlap([1], [1], n=4), 0.0)


class TestGroupAudit(unittest.TestCase):
    def test_identical_outputs_mean_jaccard_one(self):
        same = [[1, 2, 3, 4, 5]] * 4
        g = audit_group(same)
        self.assertEqual(g["mean_jaccard"], 1.0)

    def test_diverse_outputs_mean_jaccard_zero(self):
        diverse = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        g = audit_group(diverse)
        self.assertEqual(g["mean_jaccard"], 0.0)

    def test_single_stream_no_pairs(self):
        g = audit_group([[1, 2, 3]])
        self.assertEqual(g["n_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
