"""Unit tests for the parse + tokenize artefact cache in ``preframr_experiments.base``. The cache is what lets a retried prodlike run skip the ~25 min stftokenize step when nothing has changed."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from preframr_experiments.base import (
    ExperimentSpec,
    Arm,
    _DATASET_CACHE_MARKER,
    _dataset_cache_dir,
    _dataset_cache_key,
    _populate_dataset_cache,
    _try_dataset_cache_hit,
)


def _toy_spec(pipeline_spec=None, seq_len=128):
    return ExperimentSpec(
        name="cache_unit",
        doc="",
        tier="smoke",
        arms=[Arm(label="a")],
        metrics=[],
        seq_len=seq_len,
        pipeline_spec=pipeline_spec or {"transforms": []},
    )


def _populate_work_dir(work_dir: Path, data_layout):
    """Write the artefacts the cache is supposed to capture so we can round-trip them through populate -> hit."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "dataset.csv.zst").write_bytes(b"zst-bytes")
    (work_dir / "tokens.csv").write_text("op,reg\n")
    (work_dir / "tkmodel.json").write_text("{}")
    (work_dir / "df-map.csv").write_text("a,b\n")
    (work_dir / "df-map_reg_widths.json").write_text("{}")
    for subdir in data_layout.keys():
        sub = work_dir / subdir
        sub.mkdir()
        (sub / "Composer").mkdir()
        (sub / "Composer" / "Song.0.blocks.npy").write_bytes(b"npy-bytes")


class TestDatasetCacheKey(unittest.TestCase):
    def test_stable_under_layout_reordering(self):
        spec = _toy_spec()
        a = _dataset_cache_key(spec, {"train": ["x", "y"], "eval": ["e"]})
        b = _dataset_cache_key(spec, {"eval": ["e"], "train": ["y", "x"]})
        self.assertEqual(a, b)

    def test_changes_with_pipeline_spec(self):
        a = _dataset_cache_key(
            _toy_spec(pipeline_spec={"transforms": [{"name": "slope"}]}),
            {"train": ["x"]},
        )
        b = _dataset_cache_key(
            _toy_spec(pipeline_spec={"transforms": [{"name": "preset"}]}),
            {"train": ["x"]},
        )
        self.assertNotEqual(a, b)

    def test_changes_with_seq_len(self):
        a = _dataset_cache_key(_toy_spec(seq_len=128), {"train": ["x"]})
        b = _dataset_cache_key(_toy_spec(seq_len=256), {"train": ["x"]})
        self.assertNotEqual(a, b)


class TestDatasetCachePopulateAndHit(unittest.TestCase):
    def test_miss_when_no_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            spec = _toy_spec()
            key = _dataset_cache_key(spec, {"train": ["x"]})
            cache_dir = _dataset_cache_dir(tmp, key)
            work_dir = tmp / "work"
            work_dir.mkdir()
            hit = _try_dataset_cache_hit(
                cache_dir, work_dir, {"train": ["x"]}, logging.getLogger()
            )
            self.assertFalse(hit)

    def test_populate_then_hit_roundtrips_files_and_subdirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_layout = {"train": ["a"], "eval_a": ["b"]}
            spec = _toy_spec()
            key = _dataset_cache_key(spec, data_layout)
            cache_dir = _dataset_cache_dir(tmp, key)

            src_work = tmp / "src_work"
            _populate_work_dir(src_work, data_layout)
            _populate_dataset_cache(
                cache_dir, src_work, data_layout, logging.getLogger()
            )
            self.assertTrue((cache_dir / _DATASET_CACHE_MARKER).exists())

            dst_work = tmp / "dst_work"
            dst_work.mkdir()
            hit = _try_dataset_cache_hit(
                cache_dir, dst_work, data_layout, logging.getLogger()
            )
            self.assertTrue(hit)
            self.assertEqual((dst_work / "dataset.csv.zst").read_bytes(), b"zst-bytes")
            self.assertTrue(
                (dst_work / "train" / "Composer" / "Song.0.blocks.npy").exists()
            )
            self.assertTrue(
                (dst_work / "eval_a" / "Composer" / "Song.0.blocks.npy").exists()
            )

    def test_incomplete_cache_treated_as_miss(self):
        """If a populate crashed between file copy and marker write, the next reader must not consume the half-populated dir."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            spec = _toy_spec()
            key = _dataset_cache_key(spec, {"train": ["x"]})
            cache_dir = _dataset_cache_dir(tmp, key)
            cache_dir.mkdir(parents=True)
            (cache_dir / "dataset.csv.zst").write_bytes(b"partial")
            work_dir = tmp / "work"
            work_dir.mkdir()
            hit = _try_dataset_cache_hit(
                cache_dir, work_dir, {"train": ["x"]}, logging.getLogger()
            )
            self.assertFalse(hit)
            self.assertFalse((work_dir / "dataset.csv.zst").exists())

    def test_populate_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            spec = _toy_spec()
            key = _dataset_cache_key(spec, {"train": ["x"]})
            cache_dir = _dataset_cache_dir(tmp, key)
            work_dir = tmp / "work"
            _populate_work_dir(work_dir, {"train": ["x"]})
            _populate_dataset_cache(
                cache_dir, work_dir, {"train": ["x"]}, logging.getLogger()
            )
            mtime_first = (cache_dir / _DATASET_CACHE_MARKER).stat().st_mtime
            _populate_dataset_cache(
                cache_dir, work_dir, {"train": ["x"]}, logging.getLogger()
            )
            mtime_second = (cache_dir / _DATASET_CACHE_MARKER).stat().st_mtime
            self.assertEqual(mtime_first, mtime_second)


if __name__ == "__main__":
    unittest.main()
