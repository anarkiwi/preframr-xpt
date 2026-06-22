"""Unit tests for the BACC parse-artefact cache in ``preframr_experiments.base``. The cache stores each tune's recovered ``.blocks.npy`` so a retried run skips the py65 recovery when the corpus + codec version are unchanged."""

from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from preframr_experiments import base as xpt_base
from preframr_experiments.base import (
    ExperimentSpec,
    Arm,
    _DATASET_CACHE_MARKER,
    _dataset_cache_dir,
    _dataset_cache_key,
    _populate_dataset_cache,
    _try_dataset_cache_hit,
    stage_sids,
)


def _toy_spec(seq_len=128):
    return ExperimentSpec(
        name="cache_unit",
        doc="",
        tier="smoke",
        arms=[Arm(label="a")],
        metrics=[],
        seq_len=seq_len,
    )


def _populate_work_dir(work_dir: Path, data_layout):
    """Write the per-tune ``.blocks.npy`` the cache is supposed to capture so we can round-trip them through populate -> hit."""
    work_dir.mkdir(parents=True, exist_ok=True)
    for subdir in data_layout.keys():
        sub = work_dir / subdir
        sub.mkdir()
        (sub / "Composer").mkdir()
        (sub / "Composer" / "Song.1.blocks.npy").write_bytes(b"npy-bytes")


class _CacheKeyTestCase(unittest.TestCase):
    """Stub the per-image tokens-version docker query so cache-key tests stay hermetic (no docker)."""

    def setUp(self):
        p = mock.patch.object(
            xpt_base, "_image_tokens_version", return_value="test-0.0.0"
        )
        p.start()
        self.addCleanup(p.stop)


class TestDatasetCacheKey(_CacheKeyTestCase):
    def test_stable_under_layout_reordering(self):
        spec = _toy_spec()
        a = _dataset_cache_key(spec, {"train": ["x", "y"], "eval": ["e"]})
        b = _dataset_cache_key(spec, {"eval": ["e"], "train": ["y", "x"]})
        self.assertEqual(a, b)

    def test_changes_with_seq_len(self):
        a = _dataset_cache_key(_toy_spec(seq_len=128), {"train": ["x"]})
        b = _dataset_cache_key(_toy_spec(seq_len=256), {"train": ["x"]})
        self.assertNotEqual(a, b)

    def test_parse_tokenize_cargs_change_key(self):
        """Parse/tokenize-affecting extra_cargs must bust the key (else two arms
        collide -- the bug that forced cache-disable)."""
        spec = _toy_spec()
        layout = {"train": ["x"]}
        base = _dataset_cache_key(spec, layout, "")
        affecting = _dataset_cache_key(
            spec, layout, "--voice-id-on-marker --voice-order-on-marker"
        )
        self.assertNotEqual(base, affecting)

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
        """A dataset-affecting flag + a train-only flag keys the same as the
        dataset-affecting flag alone -- the train-only part is stripped."""
        spec = _toy_spec()
        layout = {"train": ["x"]}
        affecting_only = _dataset_cache_key(spec, layout, "--voice-id-on-marker")
        mixed = _dataset_cache_key(
            spec,
            layout,
            "--per-tier-heads --voice-id-on-marker --per-tier-content-mos-k 4",
        )
        self.assertEqual(affecting_only, mixed)

    def test_unknown_flag_treated_as_dataset_affecting(self):
        """Fail-safe: an unrecognised flag busts the key (never silently
        collides two arms)."""
        spec = _toy_spec()
        layout = {"train": ["x"]}
        base = _dataset_cache_key(spec, layout, "")
        unknown = _dataset_cache_key(spec, layout, "--some-new-pass")
        self.assertNotEqual(base, unknown)

    def test_changes_with_tokens_version(self):
        """A tokenizer upgrade busts the key so stale parse/tokenize artefacts
        from the prior version are never silently reused."""
        spec = _toy_spec()
        layout = {"train": ["x"]}
        with mock.patch.object(
            xpt_base, "_image_tokens_version", return_value="0.17.0"
        ):
            v17 = _dataset_cache_key(spec, layout)
        with mock.patch.object(
            xpt_base, "_image_tokens_version", return_value="0.18.0"
        ):
            v18 = _dataset_cache_key(spec, layout)
        self.assertNotEqual(v17, v18)


class TestDatasetCachePopulateAndHit(_CacheKeyTestCase):
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
            self.assertTrue(
                (dst_work / "train" / "Composer" / "Song.1.blocks.npy").exists()
            )
            self.assertTrue(
                (dst_work / "eval_a" / "Composer" / "Song.1.blocks.npy").exists()
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


class TestStageSids(unittest.TestCase):
    def _make_src(self, tmp: Path) -> Path:
        src = tmp / "src"
        (src / "Composer").mkdir(parents=True)
        (src / "Composer" / "Song.sid").write_bytes(b"PSID")
        return src

    def test_symlinks_to_link_root_without_copying(self):
        """link_root mode stages a container-valid symlink, not a content copy."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._make_src(tmp)
            work = tmp / "work"
            rows = stage_sids(
                ["Composer/Song.sid\t1"],
                src,
                "train",
                work,
                logging.getLogger(),
                link_root="/dumps",
            )
            self.assertEqual(rows, [("train/Composer/Song.sid", 1)])
            staged = work / "train" / "Composer" / "Song.sid"
            self.assertTrue(staged.is_symlink())
            self.assertEqual(os.readlink(staged), "/dumps/Composer/Song.sid")

    def test_copies_when_no_link_root(self):
        """Default (pre_run_hook) mode copies real content so a hook can mutate."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._make_src(tmp)
            work = tmp / "work"
            rows = stage_sids(
                ["Composer/Song.sid\t1"], src, "train", work, logging.getLogger()
            )
            self.assertEqual(rows, [("train/Composer/Song.sid", 1)])
            staged = work / "train" / "Composer" / "Song.sid"
            self.assertFalse(staged.is_symlink())
            self.assertEqual(staged.read_bytes(), b"PSID")

    def test_missing_src_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._make_src(tmp)
            work = tmp / "work"
            rows = stage_sids(
                ["Composer/Song.sid\t1", "Composer/Gone.sid\t1"],
                src,
                "train",
                work,
                logging.getLogger(),
                link_root="/dumps",
            )
            self.assertEqual(rows, [("train/Composer/Song.sid", 1)])
            self.assertFalse((work / "train" / "Composer" / "Gone.sid").exists())


class TestCacheExcludesInputs(_CacheKeyTestCase):
    def test_populate_skips_sid_files(self):
        """The cache stores the parse OUTPUT (.blocks.npy) only -- never the raw
        .sid (real or symlinked), which already lives at src_root."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            work = tmp / "work"
            sub = work / "train" / "Composer"
            sub.mkdir(parents=True)
            (sub / "Song.sid").write_bytes(b"raw-sid")
            (sub / "Song.1.blocks.npy").write_bytes(b"blocks")
            spec = _toy_spec()
            key = _dataset_cache_key(spec, {"train": ["x"]})
            cache_dir = _dataset_cache_dir(tmp, key)
            _populate_dataset_cache(
                cache_dir, work, {"train": ["x"]}, logging.getLogger()
            )
            cached = cache_dir / "train" / "Composer"
            self.assertTrue((cached / "Song.1.blocks.npy").exists())
            self.assertFalse((cached / "Song.sid").exists())


if __name__ == "__main__":
    unittest.main()
