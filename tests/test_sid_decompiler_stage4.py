"""Stage 4 -- the decisive corpus scaling test on depacker-less Goto80 drivers
(design 6: "the decisive scaling test the design exists to pass").

The prior output-shaped catalog was LENGTH-PROPORTIONAL: Goto80's generic catalog
went 10.5k -> 34k tokens as the capture window grew. This stage proves the
recurrence model beats exactly that on real depacker-less drivers (JCH NewPlayer +
DefMon) -- there is NO depacker oracle here, so residual-zero on the byte-exact SID
register stream is the SOLE gate (design 2.7).

Properties under test (the numbers, not a vibe):

  1. residual-zero on the RECOVERED SUBSET over the FULL stream, from a SHORT fit
     window -- the fixed-size object derived from 512 frames reproduces the whole
     tune byte-exact on the axes it claims (design 1.4 + 2.7).
  2. the recovered {G_A} byte-size is FLAT vs the fit window (slope 0) -- the
     anti-10.5k->34k proof (design 1.4 + 2.8).
  3. whole-tune residual is nonzero and lives ENTIRELY in the surfaced axes
     (honesty: the gate is never faked to 0 -- design 5).

Fixtures are real Goto80 tracer artifacts (depacker-less):
  * goto80_jch    -- "Truth" (JCH NewPlayer, init=$1000/play=$1003, 50Hz raster)
  * goto80_defmon -- "Bokl0v" (DefMon, init=$0fd0/play=$0fe1, CIA 2x multispeed)

The harness that produced these (and the full 8-tune corpus sweep) is
``preframr_experiments.sid_decompiler.stage4_corpus_scaling``. This is a
MEASUREMENT test: it drives generalize.py as a black box; it does not modify any
recovery algorithm.
"""

import os

import numpy as np
import pytest

from preframr_experiments.sid_decompiler import sdst as S
from preframr_experiments.sid_decompiler.generalize import (
    recover_program_from,
    residual,
)

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "fixtures", "sid_decompiler")

FIT_WINDOWS = (512, 1024, 2048, 4096)


def _load(name):
    art = S.parse_sdst(os.path.join(FIX, f"{name}.distill.bin"))
    ref = np.load(os.path.join(FIX, f"{name}_ref.npz"))["ref"].astype(np.int64)
    return art, ref


@pytest.fixture(scope="module", params=["goto80_jch", "goto80_defmon"])
def tune(request):
    art, ref = _load(request.param)
    prog, ref_a, phase, info = recover_program_from(art, ref, window=512)
    return {"name": request.param, "art": art, "ref": ref_a, "prog": prog,
            "phase": phase, "info": info}


def test_recovered_subset_residual_zero_full_length(tune):
    """The SOLE gate (design 2.7): the recovered subset, derived from a SHORT 512
    window, renders byte-exact over the FULL stream. No depacker oracle exists for
    these drivers -- this byte-exact register stream IS the oracle."""
    prog, ref, phase = tune["prog"], tune["ref"], tune["phase"]
    rec = prog.recovered_regs()
    assert len(rec) >= 1, "no axis closed at all"
    rendered = prog.render(len(ref), phase)
    assert residual(rendered, ref, regs=rec) == 0, (
        f"{tune['name']}: recovered subset not byte-exact over full stream")


def test_recovered_size_flat_vs_fit_window(tune):
    """THE decisive metric (design 1.4): the recovered {G_A} byte-size does NOT grow
    as the fit window grows -- slope 0, in direct contrast to the prior 10.5k->34k
    token blowup. Same fixed artifact; only the number of frames the recurrences are
    fit from changes; the recovered object stays the same size."""
    art, ref = tune["art"], tune["ref"]
    windows = [w for w in FIT_WINDOWS if w < len(ref)] + [len(ref)]
    sizes = []
    for w in windows:
        prog, ref_a, phase, info = recover_program_from(art, ref, window=w)
        rec = prog.recovered_regs()
        rendered = prog.render(len(ref_a), phase)
        # every window must keep the recovered subset residual-zero
        assert residual(rendered, ref_a, regs=rec) == 0
        sizes.append(info["size_bytes"])
    assert max(sizes) - min(sizes) == 0, (
        f"{tune['name']}: recovered size NOT flat vs window: "
        f"{dict(zip(windows, sizes))}")
    assert max(sizes) < 256, "recovered object is not a small fixed-size {G_A}"


def test_whole_tune_residual_is_exactly_the_surfaced_axes(tune):
    """HONESTY (design 5): the whole-tune residual is nonzero (these complex
    sequencer drivers surface most axes) and ALL of it lives in the surfaced
    (unrecovered) columns -- the gate is never faked to 0 by admitting an axis that
    does not reproduce the stream."""
    prog, ref, phase = tune["prog"], tune["ref"], tune["phase"]
    rec, unrec = prog.recovered_regs(), prog.unrecovered_regs()
    rendered = prog.render(len(ref), phase)
    assert residual(rendered, ref, regs=rec) == 0
    whole = residual(rendered, ref)
    assert whole > 0, "expected nonzero whole-tune residual on a complex driver"
    assert residual(rendered, ref, regs=unrec) == whole


def test_artifact_is_bounded_not_streamed(tune):
    """The SDST artifact is O(code sites + state cells), not O(frames) (design 3):
    SIDDF/SDCU are a few dozen entries, STATESEQ a handful of cells x bounded
    samples -- not a per-frame stream. (Boundedness vs N is measured corpus-wide in
    stage4_corpus_scaling; here we assert the artifact shape is small + bounded.)"""
    art = tune["art"]
    assert len(art.siddf) <= 256
    assert len(art.sdcu) <= 256
    assert len(art.stateseq) <= 256
    for e in art.stateseq:
        assert len(e.samples) <= 1024  # bounded M samples/cell (design 3.2)
    for s in art.siddf:
        assert len(s.op_seq) <= 64 and len(s.leaves) <= 256


def test_defmon_has_no_state_cell_sections_REPORTED_LIMITATION():
    """REPORTED tracer limitation (not fixed here -- Stage 3b owns recovery edits):
    on the DefMon (CIA 2x-multispeed) driver the tracer emits ZERO SDCU/STATESEQ
    state-cell sections, while JCH (raster) emits them. The host therefore has only
    SIDDF (no per-state-cell UPDATE DAG, no inter-frame samples) for DefMon and
    closes fewer axes. This documents the gap so it is not mistaken for a host bug;
    the residual-zero gate stays honest (the unclosable axes are surfaced)."""
    dm, _ = _load("goto80_defmon")
    jch, _ = _load("goto80_jch")
    assert len(dm.sdcu) == 0 and len(dm.stateseq) == 0, (
        "DefMon limitation changed -- update the report")
    assert len(jch.sdcu) > 0 and len(jch.stateseq) > 0, (
        "JCH should carry state-cell sections")
