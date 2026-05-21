"""Tests for micro_mini tier scaffolding (Stage-C rapid triage)."""

from __future__ import annotations

import unittest

from preframr_experiments.base import (
    Arm,
    ExperimentSpec,
    micro_mini_paths,
    micro_mini_train_args,
    resolve_data_layout,
)


class TestMicroMiniTrainArgs(unittest.TestCase):
    def test_gate_flag_baked_in(self):
        args = micro_mini_train_args()
        self.assertIn("--generalization-gate", args)
        self.assertIn("--max-epochs 30", args)

    def test_small_body_by_default(self):
        args = micro_mini_train_args()
        self.assertIn("--layers 4", args)

    def test_large_body_supported(self):
        args = micro_mini_train_args(body="large")
        self.assertIn("--layers 6", args)

    def test_rejects_unknown_body(self):
        with self.assertRaises(ValueError):
            micro_mini_train_args(body="huge")


class TestMicroMiniPaths(unittest.TestCase):
    def test_train_capped_at_16(self):
        paths = micro_mini_paths()
        self.assertLessEqual(len(paths["train"]), 16)
        self.assertGreater(len(paths["train"]), 0)

    def test_eval_a_capped_at_8(self):
        paths = micro_mini_paths()
        self.assertLessEqual(len(paths["eval-A"]), 8)

    def test_eval_b_capped_at_4(self):
        paths = micro_mini_paths()
        self.assertLessEqual(len(paths["eval-B-daglish"]), 4)
        self.assertLessEqual(len(paths["eval-B-follin"]), 4)


class TestSpecResolution(unittest.TestCase):
    def test_micro_mini_tier_accepted(self):
        spec = ExperimentSpec(
            name="x",
            doc="t",
            tier="micro_mini",
            arms=[Arm(label="a", baseline=True)],
            metrics=["val_loss_best"],
            train_args=micro_mini_train_args(),
        )
        self.assertEqual(spec.tier, "micro_mini")
        self.assertEqual(spec.seeds, 1)

    def test_data_layout_for_micro_mini(self):
        spec = ExperimentSpec(
            name="x",
            doc="t",
            tier="micro_mini",
            arms=[Arm(label="a", baseline=True)],
            metrics=["val_loss_best"],
            train_args=micro_mini_train_args(),
        )
        layout = resolve_data_layout(spec)
        self.assertIn("train", layout)
        self.assertIn("eval", layout)
        self.assertLessEqual(len(layout["train"]), 16)


if __name__ == "__main__":
    unittest.main()
