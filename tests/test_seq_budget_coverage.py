"""Sanity checks for seq_budget_coverage."""

import unittest

import pandas as pd

from preframr_experiments.audit.seq_budget_coverage import compute_coverage


def _toy_manifest():
    return pd.DataFrame(
        [
            {
                "n_frames": 100,
                "n_atoms_raw": 200,
                "irq": 19656,
                "skipped_by_filter": False,
            },
            {
                "n_frames": 1000,
                "n_atoms_raw": 4000,
                "irq": 19656,
                "skipped_by_filter": False,
            },
            {
                "n_frames": 10000,
                "n_atoms_raw": 80000,
                "irq": 19656,
                "skipped_by_filter": False,
            },
            {
                "n_frames": 0,
                "n_atoms_raw": 0,
                "irq": 0,
                "skipped_by_filter": True,
            },
        ]
    )


class TestComputeCoverage(unittest.TestCase):
    def test_admitted_count(self):
        out = compute_coverage(_toy_manifest(), 1.8, 0.5, 2048, 8192)
        self.assertEqual(out["n_admitted"], 3)
        self.assertEqual(out["n_total"], 4)

    def test_density_seconds_monotonic_in_density(self):
        out = compute_coverage(_toy_manifest(), 1.8, 0.5, 2048, 8192)
        secs = [row["max_seconds"] for row in out["density_table"]]
        self.assertGreater(secs[0], secs[1])
        self.assertGreater(secs[1], secs[2])

    def test_prompt_scales_linearly(self):
        out_2k = compute_coverage(_toy_manifest(), 1.8, 0.5, 2048, 8192)
        out_4k = compute_coverage(_toy_manifest(), 1.8, 0.5, 4096, 8192)
        for r2, r4 in zip(out_2k["density_table"], out_4k["density_table"]):
            self.assertAlmostEqual(r4["prompt_seconds"], r2["prompt_seconds"] * 2)

    def test_higher_atoms_per_token_means_more_audio(self):
        baseline = compute_coverage(_toy_manifest(), 1.8, 0.5, 2048, 8192)
        compressed = compute_coverage(_toy_manifest(), 3.6, 0.5, 2048, 8192)
        for b, c in zip(baseline["density_table"], compressed["density_table"]):
            self.assertGreater(c["max_seconds"], b["max_seconds"])


if __name__ == "__main__":
    unittest.main()
