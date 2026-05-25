"""Unit tests for the parse + tokenize artefact cache in ``preframr_experiments.base``. The cache is what lets a retried prodlike run skip the ~25 min stftokenize step when nothing has changed."""

from __future__ import annotations

import logging
import os
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
    stage_dumps,
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
            _toy_spec(pipeline_spec={"transforms": [{"name": "freq_trajectory"}]}),
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

    def test_parse_tokenize_cargs_change_key(self):
        """Macro flags affect parse/tokenize output, so they must bust the key
        (else two arms collide -- the bug that forced cache-disable)."""
        spec = _toy_spec()
        layout = {"train": ["x"]}
        base = _dataset_cache_key(spec, layout, "")
        macro = _dataset_cache_key(spec, layout, "--freq-nudge-pass --release-update-pass")
        self.assertNotEqual(base, macro)

    def test_train_only_cargs_do_not_change_key(self):
        """Train/model-only flags don't touch parse/tokenize, so arms that
        differ only in them must SHARE one cached dataset (preserve the
        cross-arm parse+tokenize reuse)."""
        spec = _toy_spec()
        layout = {"train": ["x"]}
        base = _dataset_cache_key(spec, layout, "")
        store_true = _dataset_cache_key(spec, layout, "--per-tier-heads")
        with_value = _dataset_cache_key(
            spec, layout, "--per-tier-heads --per-tier-content-mos-k 4"
        )
        eq_value = _dataset_cache_key(spec, layout, "--per-tier-content-mos-k=4")
        self.assertEqual(base, store_true)
        self.assertEqual(base, with_value)
        self.assertEqual(base, eq_value)

    def test_mixed_cargs_key_on_dataset_slice_only(self):
        """A macro flag + a train-only flag keys the same as the macro flag
        alone -- the train-only part is stripped, the macro part is kept."""
        spec = _toy_spec()
        layout = {"train": ["x"]}
        macro_only = _dataset_cache_key(spec, layout, "--freq-nudge-pass")
        mixed = _dataset_cache_key(
            spec, layout, "--per-tier-heads --freq-nudge-pass --per-tier-content-mos-k 4"
        )
        self.assertEqual(macro_only, mixed)

    def test_unknown_flag_treated_as_dataset_affecting(self):
        """Fail-safe: an unrecognised flag busts the key (never silently
        collides two arms)."""
        spec = _toy_spec()
        layout = {"train": ["x"]}
        base = _dataset_cache_key(spec, layout, "")
        unknown = _dataset_cache_key(spec, layout, "--some-new-pass")
        self.assertNotEqual(base, unknown)


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


class TestStageDumps(unittest.TestCase):
    def _make_src(self, tmp: Path) -> Path:
        src = tmp / "src"
        (src / "Composer").mkdir(parents=True)
        (src / "Composer" / "Song.1.dump.parquet").write_bytes(b"dump")
        return src

    def test_symlinks_to_link_root_without_copying(self):
        """link_root mode stages a container-valid symlink, not a content copy."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._make_src(tmp)
            dst = tmp / "work" / "train"
            n = stage_dumps(
                ["Composer/Song.1.dump.parquet"],
                src,
                dst,
                logging.getLogger(),
                link_root="/dumps",
            )
            self.assertEqual(n, 1)
            staged = dst / "Composer" / "Song.1.dump.parquet"
            self.assertTrue(staged.is_symlink())
            self.assertEqual(os.readlink(staged), "/dumps/Composer/Song.1.dump.parquet")

    def test_copies_when_no_link_root(self):
        """Default (pre_run_hook) mode copies real content so a hook can mutate."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._make_src(tmp)
            dst = tmp / "work" / "train"
            n = stage_dumps(
                ["Composer/Song.1.dump.parquet"], src, dst, logging.getLogger()
            )
            self.assertEqual(n, 1)
            staged = dst / "Composer" / "Song.1.dump.parquet"
            self.assertFalse(staged.is_symlink())
            self.assertEqual(staged.read_bytes(), b"dump")

    def test_missing_src_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._make_src(tmp)
            dst = tmp / "work" / "train"
            n = stage_dumps(
                ["Composer/Song.1.dump.parquet", "Composer/Gone.9.dump.parquet"],
                src,
                dst,
                logging.getLogger(),
                link_root="/dumps",
            )
            self.assertEqual(n, 1)
            self.assertFalse((dst / "Composer" / "Gone.9.dump.parquet").exists())


class TestCacheExcludesDumps(unittest.TestCase):
    def test_populate_skips_dump_files(self):
        """The cache stores parse/tokenize OUTPUTS only -- never the raw dumps
        (real or symlinked), which already live at src_root."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            work = tmp / "work"
            sub = work / "train" / "Composer"
            sub.mkdir(parents=True)
            (work / "dataset.csv.zst").write_bytes(b"z")
            (sub / "Song.dump.parquet").write_bytes(b"raw-input")
            (sub / "Song.0.parquet").write_bytes(b"parsed-output")
            (sub / "Song.0.blocks.npy").write_bytes(b"blocks")
            spec = _toy_spec()
            key = _dataset_cache_key(spec, {"train": ["x"]})
            cache_dir = _dataset_cache_dir(tmp, key)
            _populate_dataset_cache(
                cache_dir, work, {"train": ["x"]}, logging.getLogger()
            )
            cached = cache_dir / "train" / "Composer"
            self.assertTrue((cached / "Song.0.parquet").exists())
            self.assertTrue((cached / "Song.0.blocks.npy").exists())
            self.assertFalse((cached / "Song.dump.parquet").exists())


if __name__ == "__main__":
    unittest.main()
