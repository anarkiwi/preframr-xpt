"""Tests for the memorization n-gram audit (pure index + scoring)."""

from __future__ import annotations

import unittest

from preframr_experiments.audit.memorization_audit import (
    longest_match_bucket,
    ngram_sets,
    novel_fraction,
    score_query,
)


class TestNgram(unittest.TestCase):
    def setUp(self):
        self.sets = ngram_sets([[1, 2, 3, 4, 5, 6, 7, 8]], ns=[2, 4])

    def test_novel_fraction_bigram(self):
        # (1,2)(2,3)(3,4) seen; (4,9)(9,9) novel
        self.assertAlmostEqual(novel_fraction([1, 2, 3, 4, 9, 9], self.sets, 2), 2 / 5)

    def test_novel_fraction_4gram(self):
        # (1,2,3,4) seen; (2,3,4,9)(3,4,9,9) novel
        self.assertAlmostEqual(novel_fraction([1, 2, 3, 4, 9, 9], self.sets, 4), 2 / 3)

    def test_short_query_returns_none(self):
        self.assertIsNone(novel_fraction([1], self.sets, 2))

    def test_fully_novel(self):
        self.assertEqual(novel_fraction([20, 21, 22], self.sets, 2), 1.0)

    def test_longest_match_bucket(self):
        self.assertEqual(longest_match_bucket([1, 2, 3, 4, 9, 9], self.sets, [2, 4]), 4)
        self.assertEqual(longest_match_bucket([2, 3, 9, 9], self.sets, [2, 4]), 2)
        self.assertEqual(longest_match_bucket([20, 21], self.sets, [2, 4]), 0)


class TestScoreQuery(unittest.TestCase):
    def test_shape(self):
        sets = ngram_sets([[1, 2, 3, 4, 5]], ns=[2, 3])
        r = score_query([1, 2, 3, 9], sets, [2, 3])
        self.assertEqual(r["query_len"], 4)
        self.assertIn("2", r["novel_fraction"])
        self.assertEqual(r["longest_match_bucket"], 3)
        self.assertIn("novel_frac_n8", r["headline"])  # present even if None


if __name__ == "__main__":
    unittest.main()
