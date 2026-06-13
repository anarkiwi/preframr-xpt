"""Tests for the Tier-0 free-running pathology probe (pure measurement logic).

The torch CLI (load_model / Predictor forward + greedy decode) runs on a GPU host;
these cover the alignment, bucketing, and verdict math that CI can exercise without
a checkpoint or a GPU."""

from __future__ import annotations

import unittest

from preframr_experiments.audit.free_running_gap_audit import (
    _h_bucket,
    _scalars,
    aggregate_cache,
    build_result,
    cache_consistency,
    classify,
    finalize_curve,
    finalize_gap,
    gap_events,
    tf_start_events,
)


class TestHorizonBucket(unittest.TestCase):
    def test_isolates_h1_then_doubles(self):
        self.assertEqual(_h_bucket(1), (1, 1))
        self.assertEqual(_h_bucket(2), (2, 3))
        self.assertEqual(_h_bucket(3), (2, 3))
        self.assertEqual(_h_bucket(4), (4, 7))
        self.assertEqual(_h_bucket(7), (4, 7))
        self.assertEqual(_h_bucket(8), (8, 15))


class TestTfStartEvents(unittest.TestCase):
    def test_alignment_and_hits(self):
        # tf_pred[i] predicts block[i+1]
        block = [5, 6, 7, 8]
        tf_pred = [6, 9, 8]  # block[1]=6 hit, block[2]=7 miss, block[3]=8 hit
        self.assertEqual(tf_start_events(block, tf_pred), [(1, 1), (2, 0), (3, 1)])

    def test_pad_target_skipped(self):
        block = [5, 0, 7]
        tf_pred = [6, 7]
        self.assertEqual(tf_start_events(block, tf_pred), [(2, 1)])

    def test_uses_min_length(self):
        block = [1, 2, 3, 4, 5]
        tf_pred = [2, 3]  # shorter than block-1
        self.assertEqual([h for h, _ in tf_start_events(block, tf_pred)], [1, 2])


class TestGapEvents(unittest.TestCase):
    def test_alignment_same_predicted_token(self):
        block = [1, 2, 3, 4, 5, 6]
        tf_pred = [2, 3, 4, 9, 6]  # predicts block[1..5]; block[4]=5 missed
        fr_gen = [3, 9, 9, 6]  # predicts block[2..5]; block[3],block[4] missed
        out = gap_events(block, tf_pred, fr_gen, prompt_len=2)
        # (h, tf_hit, fr_hit, tier)
        self.assertEqual(
            out,
            [
                (1, 1, 1, None),
                (2, 1, 0, None),
                (3, 0, 0, None),
                (4, 1, 1, None),
            ],
        )

    def test_tier_passthrough(self):
        block = [1, 2, 3, 4]
        tf_pred = [2, 3, 4]
        fr_gen = [3, 4]
        tier_of = ["s", "s", "content", "structural"]
        out = gap_events(block, tf_pred, fr_gen, prompt_len=2, tier_of=tier_of)
        self.assertEqual(
            [(h, t) for h, _, _, t in out], [(1, "content"), (2, "structural")]
        )

    def test_stops_at_block_end(self):
        block = [1, 2, 3]
        tf_pred = [2, 3]
        fr_gen = [3, 0, 0, 0]  # longer than remaining truth
        out = gap_events(block, tf_pred, fr_gen, prompt_len=2)
        self.assertEqual(len(out), 1)


class TestFinalize(unittest.TestCase):
    def test_curve_fine_and_buckets(self):
        acc = {1: [2, 2], 2: [1, 2], 5: [0, 1]}
        c = finalize_curve(acc)
        self.assertEqual(
            [(r["h"], r["acc"], r["n"]) for r in c["fine"]],
            [(1, 1.0, 2), (2, 0.5, 2), (5, 0.0, 1)],
        )
        self.assertEqual(
            [(b["h_lo"], b["h_hi"], b["acc"]) for b in c["buckets"]],
            [(1, 1, 1.0), (2, 3, 0.5), (4, 7, 0.0)],
        )

    def test_gap_sign(self):
        tf = {1: [2, 2], 2: [1, 2]}
        fr = {1: [2, 2], 2: [0, 2]}
        g = finalize_gap(tf, fr)
        self.assertEqual([(r["h"], r["gap"]) for r in g["fine"]], [(1, 0.0), (2, 0.5)])
        self.assertEqual(g["buckets"][-1]["gap"], 0.5)


class TestClassify(unittest.TestCase):
    @staticmethod
    def _curve(near, far):
        return {
            "buckets": [
                {"h_lo": 1, "h_hi": 1, "acc": near, "n": 10},
                {"h_lo": 8, "h_hi": 15, "acc": far, "n": 10},
            ]
        }

    @staticmethod
    def _gap(near, far):
        return {
            "buckets": [
                {"h_lo": 1, "h_hi": 1, "gap": near, "n": 10},
                {"h_lo": 8, "h_hi": 15, "gap": far, "n": 10},
            ]
        }

    def test_healthy(self):
        v, _ = classify(self._curve(0.9, 0.9), self._gap(0.0, 0.02))
        self.assertEqual(v, "healthy")

    def test_exposure_bias(self):
        v, _ = classify(self._curve(0.85, 0.8), self._gap(0.05, 0.4))
        self.assertEqual(v, "exposure_bias")

    def test_short_context_takes_precedence(self):
        # teacher-forced itself decays -> diagnosed before exposure bias even with a big gap
        v, _ = classify(self._curve(0.9, 0.1), self._gap(0.1, 0.4))
        self.assertEqual(v, "short_context_or_bug")

    def test_inconclusive_midband(self):
        v, _ = classify(self._curve(0.8, 0.8), self._gap(0.08, 0.10))
        self.assertEqual(v, "inconclusive")

    def test_empty_inputs(self):
        v, reason = classify({"buckets": []}, {"buckets": []})
        self.assertEqual(v, "inconclusive")
        self.assertIn("insufficient", reason)


class TestBuildResult(unittest.TestCase):
    def test_verdict_prefers_content_tier(self):
        # all-tier gap is benign; content-tier gap is large -> verdict reads content
        a_acc = {
            1: [9, 10],
            8: [9, 10],
        }  # teacher-forced flat & high (not short-context)
        b_tf = {1: [10, 10], 8: [10, 10]}
        b_fr = {1: [10, 10], 8: [9, 10]}  # all-tier gap small
        tier_tf = {"content": {1: [10, 10], 8: [10, 10]}}
        tier_fr = {
            "content": {1: [10, 10], 8: [1, 10]}
        }  # content gap large at far horizon
        result = build_result(a_acc, b_tf, b_fr, tier_tf, tier_fr, {"n_blocks": 1})
        self.assertEqual(result["verdict"], "exposure_bias")
        self.assertEqual(result["scalars"]["verdict_source"], "content")
        self.assertAlmostEqual(result["scalars"]["gap_far"], 0.9)

    def test_scalars_shape(self):
        result = build_result({1: [1, 1]}, {1: [1, 1]}, {1: [1, 1]}, {}, {}, {})
        s = _scalars(result)
        self.assertIn("tf_acc_far", s)
        self.assertEqual(s["verdict_source"], "all")


class TestCacheConsistency(unittest.TestCase):
    def test_identical_is_consistent(self):
        cc = cache_consistency([1, 2, 3], [1, 2, 3])
        self.assertEqual(cc["consistent_frac"], 1.0)
        self.assertIsNone(cc["first_divergence"])
        self.assertEqual(cc["match"], 3)

    def test_divergence_located(self):
        cc = cache_consistency([1, 2, 3], [1, 9, 3])
        self.assertAlmostEqual(cc["consistent_frac"], 2 / 3)
        self.assertEqual(cc["first_divergence"], 1)

    def test_uses_min_length(self):
        cc = cache_consistency([1, 2, 3, 4], [1, 2])
        self.assertEqual(cc["n"], 2)
        self.assertEqual(cc["consistent_frac"], 1.0)

    def test_empty(self):
        self.assertEqual(cache_consistency([], [])["n"], 0)

    def test_aggregate_pools_blocks(self):
        a = aggregate_cache(
            [cache_consistency([1, 2, 3], [1, 2, 3]), cache_consistency([4, 5], [4, 9])]
        )
        self.assertEqual(a["n_blocks"], 2)
        self.assertAlmostEqual(a["consistent_frac"], 4 / 5)
        self.assertEqual(a["inconsistent_blocks"], 1)
        self.assertEqual(a["earliest_divergence"], 1)

    def test_aggregate_empty(self):
        a = aggregate_cache([])
        self.assertEqual(a["n_blocks"], 0)
        self.assertIsNone(a["consistent_frac"])


if __name__ == "__main__":
    unittest.main()
