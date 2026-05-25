"""The working-tree preframr/ bind-mount is opt-in (off by default), so runs use
the frozen baked image unless ``--bind-src`` / ``$PREFRAMR_BIND_SRC`` is set."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from preframr_experiments import base
from preframr_experiments.base import _PREFRAMR_BIND_SRC_ENV, _src_bind_enabled


class _FakeProc:
    returncode = 0


def _captured_docker_cmd(role: str) -> list[str]:
    """Run _docker_run with subprocess stubbed; return the main docker argv."""
    calls = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return _FakeProc()

    with tempfile.TemporaryDirectory() as tmp:
        bind_root = Path(tmp) / "work"
        bind_root.mkdir()
        log_path = Path(tmp) / "run.log"
        with mock.patch.object(base.subprocess, "run", _fake_run):
            base._docker_run(
                "anarkiwi/preframr",
                ["echo", "hi"],
                bind_root=bind_root,
                log_path=log_path,
                role=role,
            )
    return next(c for c in calls if "echo" in c)


class TestSrcBindGate(unittest.TestCase):
    def setUp(self):
        self._saved = {k: v for k, v in __import__("os").environ.items()}

    def tearDown(self):
        import os

        os.environ.clear()
        os.environ.update(self._saved)

    def test_disabled_by_default(self):
        import os

        os.environ.pop(_PREFRAMR_BIND_SRC_ENV, None)
        self.assertFalse(_src_bind_enabled())

    def test_enabled_by_env(self):
        import os

        os.environ[_PREFRAMR_BIND_SRC_ENV] = "1"
        self.assertTrue(_src_bind_enabled())
        os.environ[_PREFRAMR_BIND_SRC_ENV] = "0"
        self.assertFalse(_src_bind_enabled())

    def test_no_src_mount_by_default(self):
        import os

        os.environ.pop(_PREFRAMR_BIND_SRC_ENV, None)
        for role in ("train", "inference"):
            cmd = _captured_docker_cmd(role)
            joined = " ".join(cmd)
            self.assertNotIn("/preframr/train", joined)
            self.assertNotIn("/preframr/inference", joined)

    def test_src_mount_when_enabled(self):
        import os

        os.environ[_PREFRAMR_BIND_SRC_ENV] = "1"
        self.assertIn("/preframr/train", " ".join(_captured_docker_cmd("train")))
        self.assertIn(
            "/preframr/inference", " ".join(_captured_docker_cmd("inference"))
        )


if __name__ == "__main__":
    unittest.main()
