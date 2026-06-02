"""Tests for the codebook coupling audit's pure block-walk core."""

from __future__ import annotations

import unittest

from preframr_experiments.audit.codebook_coupling import (
    STAMP_REF_OP,
    WAVETABLE_DEF_OP,
    WAVETABLE_REF_OP,
    coupling_from_blocks,
    transfer_summary,
)

# token_id -> (op, subreg, val)
_FILLER = 1
_WT_DEF7 = 100
_WT_REF7 = 101
_WT_REF9 = 102
_STAMP_REF = 200
ATOM = {
    _FILLER: (0, -1, 0),
    _WT_DEF7: (WAVETABLE_DEF_OP, 0, 7),
    _WT_REF7: (WAVETABLE_REF_OP, 0, 7),
    _WT_REF9: (WAVETABLE_REF_OP, 0, 9),
    _STAMP_REF: (STAMP_REF_OP, 1, 0),  # subreg 1 -> no id atom, op-level acc only
}

# DEF id7, then REF id7 (defined), a STAMP_REF, then REF id9 (never defined).
_GT = [_FILLER, _WT_DEF7, _FILLER, _WT_REF7, _STAMP_REF, _WT_REF9]
_PRED_PERFECT = _GT[1:]  # pred[j] predicts gt[j+1]


class TestCouplingCore(unittest.TestCase):
    def test_perfect_block(self):
        r = coupling_from_blocks({"eval_a": [(_GT, _PRED_PERFECT)]}, ATOM)["eval_a"]
        # Op-level REF selection accuracy: both WT refs + the stamp ref are exact.
        self.assertEqual(r["ref_op_acc"][str(WAVETABLE_REF_OP)], {"acc": 1.0, "n": 2})
        self.assertEqual(r["ref_op_acc"][str(STAMP_REF_OP)], {"acc": 1.0, "n": 1})
        # Only the REF whose DEF was seen earlier (id7, distance 2) gets a distance bucket.
        self.assertEqual(r["dist_acc"], {"<=8": {"acc": 1.0, "n": 1}})
        # Learned validity: predicted id7 was defined (valid), id9 was not (invalid) -> 0.5.
        self.assertEqual(r["learned_validity"], {"rate": 0.5, "n": 2})
        # Reuse: one DEF, two REFs of the WAVETABLE family.
        self.assertEqual(
            r["reuse"]["WAVETABLE"], {"defs": 1, "refs": 2, "refs_per_def": 2.0}
        )

    def test_wrong_ref_prediction_drops_acc(self):
        pred = list(_PRED_PERFECT)
        pred[2] = _FILLER  # mispredict the WT_REF7 at gt index 3
        r = coupling_from_blocks({"eval_a": [(_GT, pred)]}, ATOM)["eval_a"]
        self.assertEqual(r["ref_op_acc"][str(WAVETABLE_REF_OP)], {"acc": 0.5, "n": 2})
        # The mispredicted position's distance bucket reflects the miss.
        self.assertEqual(r["dist_acc"], {"<=8": {"acc": 0.0, "n": 1}})

    def test_transfer_summary_eval_a_vs_eval_b(self):
        pred_b = list(_PRED_PERFECT)
        pred_b[2] = _FILLER  # held-out composers mispredict the ref -> transfer drop
        per = coupling_from_blocks(
            {"eval_a": [(_GT, _PRED_PERFECT)], "eval_b_x": [(_GT, pred_b)]}, ATOM
        )
        t = transfer_summary(per)[str(WAVETABLE_REF_OP)]
        self.assertEqual(t["eval_a_acc"], 1.0)
        self.assertEqual(t["eval_b_acc"], 0.5)
        self.assertEqual(t["transfer_drop"], 0.5)

    def test_no_refs_is_empty(self):
        r = coupling_from_blocks({"eval_a": [([_FILLER, _FILLER], [_FILLER])]}, ATOM)
        self.assertEqual(r["eval_a"]["ref_op_acc"], {})
        self.assertIsNone(r["eval_a"]["learned_validity"])


if __name__ == "__main__":
    unittest.main()
