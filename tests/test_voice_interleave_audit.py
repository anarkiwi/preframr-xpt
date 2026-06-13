"""Tests for the per-voice interleave probe (pure gaps/bucketing; grammar walk CI-only)."""

from __future__ import annotations

import unittest

from preframr_experiments.audit.voice_interleave_audit import (
    _gap_bucket,
    annotate_voices,
    finalize,
    interleave_hits,
    voice_gaps,
)

try:
    import numpy as np
    import preframr_tokens.events.stream as _s
    from preframr_tokens.events.constrained import EventStreamState

    _HAVE_TOKENS = True
except Exception:  # pylint: disable=broad-except
    _HAVE_TOKENS = False


class TestGaps(unittest.TestCase):
    def test_same_voice_distance(self):
        self.assertEqual(voice_gaps([0, 1, 0, 2, 0]), [None, None, 2, None, 2])

    def test_negative_voice_skipped(self):
        self.assertEqual(voice_gaps([0, -1, 0]), [None, None, 2])

    def test_gap_buckets(self):
        self.assertEqual(_gap_bucket(1), (1, 1))
        self.assertEqual(_gap_bucket(2), (2, 3))
        self.assertEqual(_gap_bucket(4), (4, 7))


class TestInterleaveHits(unittest.TestCase):
    def test_by_voice_and_gap(self):
        block = [5, 6, 7, 8, 9]
        voices = [0, 1, 0, 2, 0]
        gaps = voice_gaps(voices)
        tf_pred = [6, 7, 8, 1]  # predicts block[1..4]; block[4] missed
        out = finalize(interleave_hits(block, tf_pred, voices, gaps))
        self.assertEqual(out["by_voice"]["1"]["acc"], 1.0)
        self.assertEqual(out["by_voice"]["0"]["acc"], 0.5)  # j=2 hit, j=4 miss
        # both gapped positions (j=2,j=4) land in the (2,3) bucket: one hit, one miss
        self.assertEqual(out["by_gap"][0]["acc"], 0.5)

    def test_interleave_penalty(self):
        acc = {"voice": {0: [1, 2]}, "gap": {(1, 1): [9, 10], (8, 15): [1, 10]}}
        out = finalize(acc)
        self.assertAlmostEqual(out["interleave_penalty"], 0.8)  # near 0.9 - far 0.1


@unittest.skipUnless(_HAVE_TOKENS, "preframr_tokens not installed")
class TestAnnotateVoices(unittest.TestCase):
    def test_voice_tags_labeled(self):
        st = EventStreamState()
        toks = []
        for _ in range(300):
            valid = np.nonzero(st.valid_mask())[0]
            if not len(valid):
                break
            t = int(valid[0])  # deterministic grammar walk
            toks.append(t)
            st.push(t)
        voices = annotate_voices(toks)
        self.assertEqual(len(voices), len(toks))
        for v in voices:
            self.assertIn(v, (-1, 0, 1, 2, 3))
        for tok, v in zip(toks, voices):
            if _s.VOICE_BASE <= tok < _s.VOICE_BASE + 4:
                self.assertEqual(v, tok - _s.VOICE_BASE)


if __name__ == "__main__":
    unittest.main()
