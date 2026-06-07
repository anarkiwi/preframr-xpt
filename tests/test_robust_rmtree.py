"""Unit tests for ``_robust_rmtree`` in ``preframr_experiments.base``."""

from __future__ import annotations

import errno
import shutil
import time
import unittest
from pathlib import Path
from unittest import mock

from preframr_experiments.base import _robust_rmtree


class TestRobustRmtree(unittest.TestCase):
    def test_succeeds_on_normal_dir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "victim"
            target.mkdir()
            (target / "a.txt").write_text("hello")
            (target / "nested").mkdir()
            (target / "nested" / "b.txt").write_text("world")
            _robust_rmtree(target)
            self.assertFalse(target.exists())

    def test_retries_on_transient_errno_39(self):
        calls = []
        real_rmtree = shutil.rmtree
        n_to_fail = 2

        def flaky_rmtree(path, *args, **kwargs):
            calls.append(path)
            if len(calls) <= n_to_fail:
                raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))
            return real_rmtree(path, *args, **kwargs)

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "victim"
            target.mkdir()
            (target / "x").write_text("y")
            with mock.patch(
                "preframr_experiments.base.shutil.rmtree",
                side_effect=flaky_rmtree,
            ):
                t0 = time.monotonic()
                _robust_rmtree(target, attempts=5, base_delay=0.01)
                elapsed = time.monotonic() - t0
            self.assertEqual(len(calls), n_to_fail + 1)
            self.assertGreaterEqual(elapsed, 0.025)

    def test_raises_after_exhausting_retries(self):
        n_attempts = 3
        calls = []

        def always_fail(path, *_args, **_kwargs):
            calls.append(path)
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(path))

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "victim"
            target.mkdir()
            with mock.patch(
                "preframr_experiments.base.shutil.rmtree",
                side_effect=always_fail,
            ):
                with self.assertRaises(OSError) as cm:
                    _robust_rmtree(target, attempts=n_attempts, base_delay=0.001)
            self.assertEqual(cm.exception.errno, errno.ENOTEMPTY)
            self.assertEqual(len(calls), n_attempts)
            self.assertTrue(target.exists())

    def test_retries_on_errno_16_busy_and_bounces_tb(self):
        """Errno 16 (EBUSY) is what NFS hands us when unlink races a process with the file open; _robust_rmtree must retry AND stop preframr_tb so the handle is released."""
        import tempfile

        calls = []
        stop_calls = []
        n_to_fail = 2
        real_rmtree = shutil.rmtree

        def flaky_rmtree(path, *args, **kwargs):
            calls.append(path)
            if len(calls) <= n_to_fail:
                raise OSError(errno.EBUSY, "Device or resource busy", str(path))
            return real_rmtree(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "victim"
            target.mkdir()
            (target / "x").write_text("y")
            with (
                mock.patch(
                    "preframr_experiments.base.shutil.rmtree",
                    side_effect=flaky_rmtree,
                ),
                mock.patch(
                    "preframr_experiments.base._stop_preframr_tb",
                    side_effect=lambda: stop_calls.append(1),
                ),
            ):
                _robust_rmtree(target, attempts=5, base_delay=0.001)
        self.assertEqual(len(calls), n_to_fail + 1)
        self.assertEqual(len(stop_calls), 1)
        self.assertFalse(target.exists())

    def test_non_retry_errno_raised_immediately(self):
        """EACCES is not in {EBUSY, ENOTEMPTY}; raise on the first attempt rather than burning retries on a real bug (e.g. permission misconfig)."""
        import tempfile

        calls = []

        def always_fail(path, *_args, **_kwargs):
            calls.append(path)
            raise OSError(errno.EACCES, "Permission denied", str(path))

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "victim"
            target.mkdir()
            with mock.patch(
                "preframr_experiments.base.shutil.rmtree",
                side_effect=always_fail,
            ):
                with self.assertRaises(OSError) as cm:
                    _robust_rmtree(target, attempts=5, base_delay=0.001)
            self.assertEqual(cm.exception.errno, errno.EACCES)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
