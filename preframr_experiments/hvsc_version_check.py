#!/usr/bin/env python3
"""Read the HVSC release version from a checked-out HVSC tree."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

DEFAULT_HVSC_ROOT = Path("/scratch/preframr/hvsc")

_RELEASE_RE = re.compile(r"^\s*Release\s+(\d+)\s*$", re.MULTILINE)

_UPDATE_RE = re.compile(r"^Update(\d+)\.hvs$")


class HvscVersionMismatch(RuntimeError):
    """Raised when the HVSC tree's version differs from expectation,
    or when the version is unparseable."""


def read_hvsc_version(hvsc_root: Path) -> int:
    """Return the integer HVSC release version from
    ``<root>/DOCUMENTS/HVSC.txt``.
    """
    docs = hvsc_root / "DOCUMENTS" / "HVSC.txt"
    if not docs.exists():
        raise HvscVersionMismatch(f"missing {docs}; is {hvsc_root} an HVSC checkout?")
    text = docs.read_text(encoding="utf-8", errors="replace")
    match = _RELEASE_RE.search(text)
    if not match:
        head = "\n".join(text.splitlines()[:10])
        raise HvscVersionMismatch(
            f"{docs}: 'Release NN' header not found. First 10 lines:\n{head}"
        )
    return int(match.group(1))


def read_hvsc_version_via_updates(hvsc_root: Path) -> int | None:
    """Fallback signal: highest ``UpdateNN.hvs`` under DOCUMENTS/.
    Returns None if the directory is missing or empty. Informational
    only; not the primary source of truth.
    """
    docs_dir = hvsc_root / "DOCUMENTS"
    if not docs_dir.is_dir():
        return None
    max_n = -1
    for entry in docs_dir.iterdir():
        m = _UPDATE_RE.match(entry.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n if max_n >= 0 else None


def assert_hvsc_version(
    hvsc_root: Path,
    expected: int,
    logger: logging.Logger | None = None,
) -> None:
    """Read the HVSC tree's version and assert it equals ``expected``."""
    actual = read_hvsc_version(hvsc_root)
    if actual != expected:
        raise HvscVersionMismatch(
            f"HVSC version mismatch at {hvsc_root}: tree reports {actual}, "
            f"caller expected {expected}. Re-pin the tier (commit "
            f"`integration_tests/data/<tier>/HVSC_VERSION`) or update the "
            f"HVSC checkout to match."
        )
    fallback = read_hvsc_version_via_updates(hvsc_root)
    if fallback is not None and fallback != actual:
        if logger:
            logger.warning(
                "HVSC.txt header reports v%u but DOCUMENTS/UpdateNN.hvs max "
                "is v%u; the tree may be partially upgraded. Investigate "
                "before relying on the leak-audit baseline.",
                actual,
                fallback,
            )
    if logger:
        logger.info("HVSC version OK: %s reports v%u", hvsc_root, actual)


_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def _read_tier_pin(data_dir: Path | None, tier: str) -> int | None:
    """Read ``<data_dir>/<tier>/HVSC_VERSION`` and return the int it contains. Returns None if the file is missing. ``data_dir`` defaults to the bundled ``preframr_experiments/data`` dir."""
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    pin = data_dir / tier / "HVSC_VERSION"
    if not pin.exists():
        return None
    raw = pin.read_text().strip()
    try:
        return int(raw)
    except ValueError as e:
        raise HvscVersionMismatch(f"{pin}: contents {raw!r} not an int") from e


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hvsc-root",
        type=Path,
        default=DEFAULT_HVSC_ROOT,
        help="HVSC checkout root (default %(default)s)",
    )
    parser.add_argument(
        "--expected",
        type=int,
        default=None,
        help="Expected HVSC version (int). If set, assert; otherwise "
        "just print the detected version.",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default=None,
        choices=("smoke", "mini", "canonical", "prodlike"),
        help="If set (and --expected omitted), read the tier's pin from "
        "<data-dir>/<tier>/HVSC_VERSION and assert against it.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Data dir for --tier pin lookup (default: bundled data/).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("hvsc_version_check")

    expected = args.expected
    if expected is None and args.tier:
        expected = _read_tier_pin(args.data_dir, args.tier)
        if expected is None:
            logger.error(
                "no pin file under %s for tier %r",
                args.data_dir / args.tier,
                args.tier,
            )
            return 2

    try:
        actual = read_hvsc_version(args.hvsc_root)
    except HvscVersionMismatch as exc:
        logger.error("%s", exc)
        return 2

    if expected is None:
        print(actual)
        return 0

    try:
        assert_hvsc_version(args.hvsc_root, expected, logger=logger)
    except HvscVersionMismatch as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
