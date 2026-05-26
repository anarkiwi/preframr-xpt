"""Smoke test for the corpus-index aggregator."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preframr_experiments.audit.aggregate_corpus_index import (
    macro_potential,
    manifest_summary,
    reg_op_val_freq,
    voice_state_reuse,
)


def _encode_blob(tuples):
    arr = np.array(tuples, dtype=np.uint8)
    return arr.tobytes()


def _write_manifest(root: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "sid_path": "DEMOS/A/foo.dump.parquet",
                "composer": "A",
                "engine_fp_cluster": 1,
                "engine_fp_canonical": 1234,
                "irq": 19656,
                "n_frames": 8,
                "n_atoms_raw": 24,
                "n_voice_frames_active": 18,
                "max_writes_per_frame": 6,
                "max_ctrl_writes_per_voice_frame": 4,
                "max_vol_writes_per_frame": 1,
                "has_pcm_voice": False,
                "skipped_by_filter": False,
                "skip_reason": "",
            },
            {
                "sid_path": "DEMOS/A/bar.dump.parquet",
                "composer": "A",
                "engine_fp_cluster": 1,
                "engine_fp_canonical": 9876,
                "irq": 0,
                "n_frames": 0,
                "n_atoms_raw": 0,
                "n_voice_frames_active": 0,
                "max_writes_per_frame": 0,
                "max_ctrl_writes_per_voice_frame": 0,
                "max_vol_writes_per_frame": 0,
                "has_pcm_voice": False,
                "skipped_by_filter": True,
                "skip_reason": "no_irq",
            },
            {
                "sid_path": "DEMOS/B/baz.dump.parquet",
                "composer": "B",
                "engine_fp_cluster": 2,
                "engine_fp_canonical": 5555,
                "irq": 19656,
                "n_frames": 4,
                "n_atoms_raw": 10,
                "n_voice_frames_active": 6,
                "max_writes_per_frame": 3,
                "max_ctrl_writes_per_voice_frame": 1,
                "max_vol_writes_per_frame": 0,
                "has_pcm_voice": True,
                "skipped_by_filter": False,
                "skip_reason": "",
            },
        ]
    )
    df.to_parquet(root / "manifest.parquet", index=False)


def _write_per_frame(root: Path) -> None:
    cdir = root / "per_frame" / "cluster=1" / "composer=A"
    cdir.mkdir(parents=True, exist_ok=True)
    blob_a = _encode_blob([(0, 0, 0x40, 0x01), (0, 4, 0x41, 0)])
    blob_b = _encode_blob([(1, 2, 0xAA, 0x00), (255, 24, 0x0F, 0)])
    df = pd.DataFrame(
        {
            "intra_frame_op_hist": [blob_a, blob_b],
        }
    )
    df.to_parquet(cdir / "foo.parquet", index=False)


def _write_per_voice_frame(root: Path) -> None:
    cdir = root / "per_voice_frame" / "cluster=1" / "composer=A"
    cdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fi in range(6):
        for v in range(3):
            rows.append(
                {
                    "frame_idx": fi,
                    "voice": v,
                    "ctrl": 0x41 if v == 0 else 0,
                    "ad": 0,
                    "sr": 0,
                    "pwm": 0x800 + fi if v == 0 else 0,
                    "freq16": 0x1000 + (fi % 3) * 0x10 if v == 1 else 0,
                    "ctrl_writes_this_frame": 4 if (v == 0 and fi == 2) else 0,
                    "freq_writes_this_frame": 1 if v == 1 else 0,
                    "pwm_writes_this_frame": 1 if v == 0 else 0,
                    "step_from_prev_freq": 0,
                    "step_from_prev_pwm": 0,
                    "gate_transition": 0,
                    "voice_state_fp": 0xABCDEF if v == 0 else (0x111 + v),
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(cdir / "foo.parquet", index=False)


class TestAggregateCorpusIndex(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _write_manifest(self.root)
        _write_per_frame(self.root)
        _write_per_voice_frame(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_manifest_summary(self):
        df = manifest_summary(self.root)
        self.assertEqual(set(df["engine_fp_cluster"]), {1, 2})
        c1 = df[df["engine_fp_cluster"] == 1].iloc[0]
        self.assertEqual(c1["n_sids"], 2)
        self.assertEqual(c1["n_admitted"], 1)
        self.assertAlmostEqual(c1["admit_rate"], 0.5)
        c2 = df[df["engine_fp_cluster"] == 2].iloc[0]
        self.assertEqual(c2["has_pcm_voice_count"], 1)

    def test_reg_op_val_freq(self):
        df = reg_op_val_freq(self.root)
        self.assertGreater(len(df), 0)
        self.assertIn("count_global", df.columns)
        v0_freq = df[(df["voice"] == 0) & (df["sub"] == 0) & (df["val"] == 0x140)]
        self.assertEqual(int(v0_freq["count_global"].iloc[0]), 1)
        non_voice = df[(df["voice"] == 255) & (df["sub"] == 24)]
        self.assertEqual(int(non_voice["count_global"].iloc[0]), 1)

    def test_voice_state_reuse(self):
        df = voice_state_reuse(self.root)
        self.assertGreater(len(df), 0)
        top = df.sort_values("frame_count", ascending=False).iloc[0]
        self.assertEqual(int(top["voice_state_fp"]), 0xABCDEF)
        self.assertEqual(int(top["frame_count"]), 6)

    def test_macro_potential(self):
        df = macro_potential(self.root)
        names = set(df["macro_name"]) if len(df) else set()
        self.assertIn("pwm_preset_b2b", names)
        self.assertIn("last_write_wins_ctrl", names)
        lww = df[df["macro_name"] == "last_write_wins_ctrl"].iloc[0]
        self.assertEqual(int(lww["sites_count"]), 1)
        self.assertEqual(int(lww["est_atom_savings"]), 3)
        pwm = df[df["macro_name"] == "pwm_preset_b2b"].iloc[0]
        self.assertGreaterEqual(int(pwm["sites_count"]), 1)


if __name__ == "__main__":
    unittest.main()
