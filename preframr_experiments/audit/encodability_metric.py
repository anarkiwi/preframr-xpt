"""Encodability-rate metric helper for `global_instr_ids` Phase A."""

from __future__ import annotations

import functools
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("encodability_metric")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CORPUS_BASE = Path("/scratch/preframr/training-dumps")
_DOCKER_IMAGE = "anarkiwi/preframr"


def _docker_run_audit(tier: str, corpus_base: Path) -> dict[str, Any]:
    """Invoke the audit script in docker; return parsed JSON payload."""
    audit_json = (
        _REPO_ROOT
        / "integration_tests"
        / "data"
        / "audit"
        / f"encodability_for_metric_{tier}.json"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{_REPO_ROOT / 'preframr'}:/preframr",
        "-v",
        f"{_REPO_ROOT / 'integration_tests'}:/integration_tests",
        "-v",
        f"{corpus_base}:/training-dumps",
        _DOCKER_IMAGE,
        "python3",
        "-m",
        "integration_tests.profile.audit_engine_fp_palette_eval_encodability",
        "--tier",
        tier,
        "--corpus-base",
        "/training-dumps",
        "--out",
        f"/integration_tests/data/audit/encodability_for_metric_{tier}.json",
    ]
    logger.info("encodability audit (tier=%s) launching", tier)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "encodability audit failed (rc=%d):\n%s",
            result.returncode,
            result.stderr[-2000:],
        )
        return {}
    if not audit_json.exists():
        logger.error("encodability audit produced no JSON at %s", audit_json)
        return {}
    return json.loads(audit_json.read_text())


@functools.lru_cache(maxsize=4)
def _encodability_summary_for_tier(tier: str, corpus_base_str: str) -> dict[str, Any]:
    """Cached per (tier, corpus_base). Returns ``{}`` on failure --
    extractors then yield NaN per ``compute_metrics``'s broad-except
    contract.
    """
    return _docker_run_audit(tier, Path(corpus_base_str))


def _rate_for_subset(summary: dict[str, Any], subset: str | None) -> float:
    """Pull the rate field from the summary. ``subset=None`` returns
    the overall rate; otherwise picks the matching per-subset entry.
    """
    if not summary:
        return float("nan")
    if subset is None:
        try:
            return float(summary["overall"]["encodability_rate"])
        except (KeyError, TypeError):
            return float("nan")
    for entry in summary.get("subsets", []):
        if entry.get("subset") == subset:
            try:
                return float(entry["encodability_rate"])
            except (KeyError, TypeError):
                return float("nan")
    logger.warning("encodability subset %r not in summary", subset)
    return float("nan")


def make_encodability_extractor(
    tier: str,
    subset: str | None = None,
    corpus_base: Path | None = None,
) -> Callable[[Any], float]:
    """Build a ``(arm_artefacts) -> float`` extractor returning the
    encodability rate for ``tier`` and the named subset (or overall
    if ``subset`` is None). All arms of the same spec share one
    cached summary; the extractor signature only takes ``art`` to
    fit ``compute_metrics``'s contract.
    """
    base = corpus_base or _DEFAULT_CORPUS_BASE

    def _extract(_art: Any) -> float:
        summary = _encodability_summary_for_tier(tier, str(base))
        return _rate_for_subset(summary, subset)

    _extract.__name__ = (
        f"encodability_{tier}_overall"
        if subset is None
        else f"encodability_{tier}_{subset}"
    )
    return _extract
