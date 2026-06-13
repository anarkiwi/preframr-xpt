"""Tests for the copy-vs-novel accuracy decomposition (pure measurement)."""

from __future__ import annotations

import unittest

from preframr_experiments.audit.copy_novel_audit import (
    copy_novel_hits,
    copyable_positions,
    finalize,
    merge_hits,
)


class TestCopyable(unittest.TestCase):
    def test_repeated_bigram_is_copyable(self):
        # the second "5->6" completes a bigram already seen at positions 0-1
        self.assertEqual(
            copyable_positions([5, 6, 5, 6, 7]),
            {1: False, 2: False, 3: True, 4: False},
        )

    def test_pad_breaks_bigrams(self):
        out = copyable_positions([5, 6, 0, 6, 7])
        self.assertNotIn(2, out)  # block[2] is pad -> no event
        self.assertNotIn(3, out)  # block[3] target with pad predecessor -> skipped

    def test_all_novel_when_no_repeat(self):
        self.assertEqual(
            copyable_positions([1, 2, 3, 4]), {1: False, 2: False, 3: False}
        )


class TestHitsAndFinalize(unittest.TestCase):
    def test_split_all_tier(self):
        block = [5, 6, 5, 6, 7]
        tf_pred = [6, 5, 6, 9]  # predicts block[1..4]; only block[4] missed
        out = finalize(copy_novel_hits(block, tf_pred))
        self.assertEqual(out["copyable"]["all"]["acc"], 1.0)
        self.assertAlmostEqual(out["novel"]["all"]["acc"], 2 / 3)
        self.assertAlmostEqual(out["copy_minus_novel"]["all"], 1 / 3)
        self.assertEqual(out["headline"]["novel_source"], "all")

    def test_per_tier_headline_prefers_content(self):
        block = [5, 6, 5, 6, 7]
        tf_pred = [6, 5, 6, 9]
        tier_of = ["s", "s", "content", "content", "content"]  # indexed by position j
        out = finalize(copy_novel_hits(block, tf_pred, tier_of=tier_of))
        self.assertEqual(out["copyable"]["content"]["acc"], 1.0)  # j=3 hit
        self.assertEqual(out["novel"]["content"]["acc"], 0.5)  # j=2 hit, j=4 miss
        self.assertEqual(out["headline"]["novel_source"], "content")
        self.assertEqual(out["headline"]["copy_minus_novel"], 0.5)

    def test_merge_accumulates(self):
        acc = {}
        merge_hits(acc, copy_novel_hits([5, 6, 5, 6], [6, 5, 6]))
        merge_hits(acc, copy_novel_hits([5, 6, 5, 6], [6, 5, 6]))
        out = finalize(acc)
        self.assertEqual(out["copyable"]["all"]["n"], 2)  # j=3 in each block


if __name__ == "__main__":
    unittest.main()
