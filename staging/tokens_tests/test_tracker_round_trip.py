"""Forward tracker round-trip: module -> register log -> tokenizer(generator) -> decode == player output.

This is the §7B Tier-1 / §7.2 cross-driver test the generator work order specified but the
landing PRs (#62-#68) shipped WITHOUT. It proves the deployed generator-MDL encoding losslessly
round-trips the register stream that real SID-Wizard and defMON players emit -- "lossless" in the
project's sense: ``decode(parse(log)) == log`` under the byte-exact oracle.

Equivalence is asserted the sanctioned way: parse the rendered log with ``parse_audit='raise'`` under
the deployed default (``full_macros`` = ``generator_pass`` + kept passes). The parser fires the
fidelity oracle after EVERY pass (EXACT_REGS byte-exact in input order, FREQ within cent tolerance on
audible frames, frame-aligned), so completing the parse == byte-exact / same output. We never hand-roll
a ``register_state`` diff -- see ``design/verification_and_audits.md`` "THE TRAP".

STAGED OUTSIDE preframr-tokens (see this dir's README). The REVERSE half of the loop
(log -> SWM -> log, the recompiler in ``design/log_to_swm_recompiler_design.md``) is not built; it is a
documented xfail placeholder below so this file describes the full intended loop.

Run in place (from the xpt tree), on fogbank in the xpt image, with a tokens-main checkout on PYTHONPATH::

    PYTHONPATH=/scratch/tmp/tokens-main-triage \
      python3 -m pytest staging/tokens_tests/test_tracker_round_trip.py -v

Fixtures: SWM from ``$PREFRAMR_SWM_DIR`` else the local SID-Wizard 1.94 example cache; defMON from
``$PREFRAMR_DEFMON_DIR`` else the ``pydefmon`` bundled ``build/fixtures``. Counts/length are env-tunable
(``TRACKER_RT_MAX_FIXTURES``, ``TRACKER_RT_NFRAMES``).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest

from tracker_render import render_to_parquet

NFRAMES = int(os.environ.get("TRACKER_RT_NFRAMES", "1500"))
MAX_FIXTURES = int(os.environ.get("TRACKER_RT_MAX_FIXTURES", "5"))


def _swm_fixtures():
    env = os.environ.get("PREFRAMR_SWM_DIR")
    cands = []
    if env and os.path.isdir(env):
        cands = sorted(glob.glob(os.path.join(env, "*.swm")))
    if not cands:
        cands = sorted(glob.glob("/scratch/tmp/swmflat/*.swm"))  # local SID-Wizard 1.94 example cache
    return cands[:MAX_FIXTURES]


def _defmon_fixtures():
    env = os.environ.get("PREFRAMR_DEFMON_DIR")
    if env and os.path.isdir(env):
        cands = sorted(glob.glob(os.path.join(env, "*.prg")))
    else:
        cands = []
        try:
            import pydefmon

            bundled = Path(pydefmon.__file__).resolve().parent.parent / "build" / "fixtures"
            cands = sorted(glob.glob(str(bundled / "*.prg")))
        except Exception:  # noqa: BLE001
            cands = []
    return cands[:MAX_FIXTURES]


def _assert_round_trip(kind, src_path, tmp_path):
    """Render -> parse under the deployed generator default with parse_audit='raise'. Reaching the
    end without an AssertionError IS the byte-exact / same-output guarantee."""
    pytest.importorskip("preframr_tokens")
    from preframr_tokens.reglogparser import RegLogParser
    from preframr_tokens.tokenizer_config import named_config

    name = Path(src_path).stem
    try:
        dump = render_to_parquet(kind, src_path, Path(tmp_path) / f"{name}.dump.parquet", NFRAMES)
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ in {"DefmonError", "SWMError", "SWMFormatError"}:
            pytest.skip(f"{kind} {name}: tracker player cannot load this fixture ({exc}) -- not a round-trip failure")
        raise

    args = named_config("full_macros", parse_audit="raise")
    parser = RegLogParser(args=args)
    blocks = 0
    try:
        for _ in parser.parse(str(dump), max_perm=1, require_pq=False, reparse=True):
            blocks += 1
    except AssertionError as exc:  # parse_audit fired -> a pass broke byte-exactness
        pytest.fail(f"{kind} {name}: generator round-trip NOT lossless -- {str(exc)[:300]}")
    assert blocks >= 1, f"{kind} {name}: parser yielded no token blocks (excluded/empty?)"


@pytest.mark.parametrize("src", _swm_fixtures() or [pytest.param(None, marks=pytest.mark.skip(reason="no SWM fixtures; set PREFRAMR_SWM_DIR"))])
def test_swm_forward_round_trip(src, tmp_path):
    pytest.importorskip("pysidwizard")
    _assert_round_trip("swm", src, tmp_path)


@pytest.mark.parametrize("src", _defmon_fixtures() or [pytest.param(None, marks=pytest.mark.skip(reason="no defMON fixtures; set PREFRAMR_DEFMON_DIR"))])
def test_defmon_forward_round_trip(src, tmp_path):
    pytest.importorskip("pydefmon")
    _assert_round_trip("defmon", src, tmp_path)


@pytest.mark.xfail(reason="log->SWM recompiler not built (design/log_to_swm_recompiler_design.md); the full SWM->log->SWM->log loop is the unbuilt reverse half", strict=True)
def test_reverse_recompile_swm_full_loop():
    """Placeholder for the FULL loop: render -> log -> recompile to SWM -> render again -> identical log.
    Requires the register-log -> SWM recompiler (designed, not implemented). Marked strict-xfail so it
    flips to a hard failure the moment the recompiler exists and this body is filled in."""
    pytest.importorskip("pysidwizard")
    raise NotImplementedError("register-log -> SWM recompiler not implemented")
