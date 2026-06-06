"""Standalone-run support for the staged tracker round-trip tests.

When this package eventually lands in ``preframr-tokens/tests/``, ``preframr_tokens``,
``pysidwizard`` and ``pydefmon`` will be importable (the latter two as test-only deps),
and this conftest is a no-op. Until then it lets the file run in place from the xpt tree
by pointing at the local source checkouts -- ONLY for modules not already importable, so
an explicit ``PYTHONPATH`` (e.g. a tokens worktree) always wins.

Override any default with the matching env var.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_DEFAULT_SRC = {
    "preframr_tokens": ("PREFRAMR_TOKENS_SRC", "/scratch/tmp/tokens-main-triage"),
    "pysidwizard": ("PYSIDWIZARD_SRC", "/scratch/anarkiwi/pysidwizard/src"),
    "pydefmon": ("PYDEFMON_SRC", "/scratch/anarkiwi/pydefmon"),
}


def _ensure_importable():
    for mod, (env, default) in _DEFAULT_SRC.items():
        if importlib.util.find_spec(mod) is not None:
            continue
        path = os.environ.get(env, default)
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_ensure_importable()
