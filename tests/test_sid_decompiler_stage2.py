"""Stage 2 -- the host generalizer: derive {G_A} from the bounded tracer artifact
(SIDDF + STATESEQ + SNAP) and re-execute the RECURRENCES (NOT the seed image, NOT
the white-box player) to a byte-exact [N,25] SID register stream.

This is the decisive Stage-2 milestone (design 3.2 + 2.5): the seed-image
re-execution crutch that Stage 0 used (``AmindRecurrence._play`` running the
original 254-byte image on the 6502 core) is RETIRED. Here the host derives the
per-axis ``(S,U,O,K)`` recurrences from the artifact and renders those.

The gate (project HARD RULE #0 / design 2.7) is residual-zero. An axis is admitted
as 'recovered' ONLY if the fixed-size object derived from a SHORT window reproduces
the FULL gate stream byte-exact (verify_and_finalize); an axis that does not close
is honestly SURFACED as unrecovered (design 5) -- never faked, never patched, never
fell back to the seed image. So the residual-zero assertions below are over the
RECOVERED SUBSET, and the surfaced axes are asserted to be exactly the honest
holdouts (the fast SMC skydive cells for A Mind; the table-walk cursors for Grid).

Fixtures (committed, small): the SDST artifact (.distill.bin, a few KB) + the
compact [N,25] gate reference (.npz). The references are reconstructed from the
tracer's .sidwr.bin render-gate stream and stored compressed.
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


def _load(name):
    art = S.parse_sdst(os.path.join(FIX, f"{name}.distill.bin"))
    ref = np.load(os.path.join(FIX, f"{name}_ref.npz"))["ref"].astype(np.int64)
    return art, ref


@pytest.fixture(scope="module")
def amind():
    art, ref = _load("amind")
    prog, ref_a, phase, info = recover_program_from(art, ref, window=512)
    return {"art": art, "prog": prog, "ref": ref_a, "phase": phase, "info": info}


@pytest.fixture(scope="module")
def grid():
    art, ref = _load("grid")
    prog, ref_a, phase, info = recover_program_from(art, ref, window=512)
    return {"art": art, "prog": prog, "ref": ref_a, "phase": phase, "info": info}


# --------------------------------------------------------------------------- #
# A Mind Is Born -- the generative no-table exemplar.
# --------------------------------------------------------------------------- #
def test_amind_master_counter_recovered_from_artifact(amind):
    """The shared master counter (design 2.6) is recovered automatically from the
    STATESEQ samples as the +2 accumulator at zero-page $13 (the INC->carry chain)."""
    info = amind["info"]
    assert info["master_addr"] == 0x13
    assert info["master_step"] == 2


def test_amind_three_closed_forms_recovered_automatically(amind):
    """The explicit success criterion: the 3 closed-form axes come out of the
    generalizer automatically (pw1=C&0xFF on reg9; filter-hi=C>>8 on reg22; v2 ctrl
    counter-bit toggle on reg18) -- derived from the artifact, no white-box."""
    prog = amind["prog"]
    for reg in (9, 18, 22):
        assert prog.axes[reg].status == "recovered", (
            f"closed-form axis reg{reg} not recovered: {prog.axes[reg].reason}"
        )
    assert "C&0xFF" in prog.axes[9].form
    assert "C>>8" in prog.axes[22].form
    assert "toggle" in prog.axes[18].form


def test_amind_recovered_subset_residual_zero_full_length(amind):
    """Residual-zero over ALL 8,190 frames on the recovered axes, from the artifact
    alone -- NO LftBackend, NO seed-image re-execution. The recurrences are
    re-executed as recurrences (prog.render), and the masked stream is byte-exact."""
    prog, ref, phase = amind["prog"], amind["ref"], amind["phase"]
    rec = prog.recovered_regs()
    assert len(rec) >= 18, f"expected >=18 recovered axes, got {len(rec)}"
    rendered = prog.render(len(ref), phase)
    assert residual(rendered, ref, regs=rec) == 0


def test_amind_surfaced_axes_are_the_honest_holdouts(amind):
    """The axes that do NOT close are surfaced (design 5), not faked. They are the
    fast SMC skydive / accumulator cells: v0 freq-lo (reg1), v1 freq-hi (reg8), v1
    PW-hi (reg10) -- whose mid-call SMC update the bounded op_seq + call-start
    STATESEQ under-determine. Each carries a specific reason."""
    prog = amind["prog"]
    unrec = set(prog.unrecovered_regs())
    # the genuine fast-SMC holdouts must be among the surfaced axes
    assert {1, 8, 10} <= unrec, f"expected reg1/8/10 surfaced, got {sorted(unrec)}"
    for reg in unrec:
        assert prog.axes[reg].reason, f"reg{reg} surfaced without a reason"
        assert prog.axes[reg].render is None  # never a usable generator
        assert prog.axes[reg].status == "unrecovered"


def test_amind_length_independence(amind):
    """Derive the recurrences from a SHORT window (512) and render the FULL length
    (8,190) residual-zero from the SAME fixed-size object; the recovered size does
    not grow with render length (slope 0) -- design 1.4."""
    prog, ref, phase = amind["prog"], amind["ref"], amind["phase"]
    rec = prog.recovered_regs()
    sizes = []
    for n in (512, 1024, 2048, 4096, len(ref)):
        rendered = prog.render(n, phase)
        assert residual(rendered, ref[:n], regs=rec) == 0, f"residual nonzero at N={n}"
        sizes.append(prog.size_bytes())
    assert max(sizes) - min(sizes) == 0  # size flat vs render length
    assert prog.size_bytes() < 256  # fixed-size object, not length-proportional


def test_amind_no_seed_image_dependency():
    """The generalizer derives {G_A} from the artifact ONLY -- it must not import the
    Stage-0 seed-image interpreter (AmindRecurrence / cpu6502) or the white-box
    LftBackend. Guard the actual code (AST body, ignoring docstrings/comments): no
    such names are referenced, so the recovery cannot re-execute the seed image."""
    import ast
    import inspect

    from preframr_experiments.sid_decompiler import generalize

    tree = ast.parse(inspect.getsource(generalize))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for n in node.names:
                names.add(n.name.split(".")[-1])
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[-1])
    forbidden = {"AmindRecurrence", "LftBackend", "cpu6502", "CPU6502", "_play"}
    leaked = forbidden & names
    assert not leaked, f"Stage 2 must not reference the seed-image crutch: {leaked}"


def test_amind_section_table_axes_close_with_full_window():
    """The generalizer's REACH: given a window covering >=2 section cycles, the
    section-indexed-table axes (reg0 v0 freq-hi, reg11 v1 AD, reg17 v2 AD, reg24
    volume) ALSO close as fixed-size recurrences O=K_table[(C>>8)] (design 2.4 array
    K). Only the 3 fast SMC cells then remain surfaced. (Length-independence still
    holds per the 512-window test; this documents the reach, not a weaker gate.)"""
    art, ref = _load("amind")
    prog, ref_a, phase, info = recover_program_from(art, ref, window=len(ref))
    rec = set(prog.recovered_regs())
    assert {0, 11, 17, 24} <= rec, "section-table axes should close with a full window"
    assert residual(prog.render(len(ref_a), phase), ref_a, regs=tuple(rec)) == 0
    # only the fast SMC accumulators remain
    assert set(prog.unrecovered_regs()) == {1, 8, 10}


# --------------------------------------------------------------------------- #
# Grid_Runner -- the table-driven exemplar.
# --------------------------------------------------------------------------- #
def test_grid_recovered_subset_residual_zero(grid):
    """Whatever Grid axes close (the set-once constants) must render residual-zero
    over the captured frames from the artifact alone."""
    prog, ref, phase = grid["prog"], grid["ref"], grid["phase"]
    rec = prog.recovered_regs()
    assert len(rec) >= 1
    rendered = prog.render(len(ref), phase)
    assert residual(rendered, ref, regs=rec) == 0


def test_grid_freq_axes_lift_k_table_by_identity(grid):
    """The Grid freq axes are table-walks; their note-table K is liftable VERBATIM
    from SNAP at the SIDDF strided interval BY IDENTITY (design 2.4). The freq axes
    do not fully close (the cursor sequencer is under-determined by the bounded
    artifact -- honestly surfaced, design 5), but the K-table identity lift is the
    structure the artifact DOES support and is attached to the surfaced axis."""
    prog = grid["prog"]
    freq_regs = (0, 1, 7, 8, 14, 15)
    lifted = [r for r in freq_regs if prog.axes[r].table_k is not None]
    assert lifted, "no Grid freq axis lifted a K-table by identity from SNAP"
    for r in lifted:
        tk = prog.axes[r].table_k
        assert len(tk.bytes_) >= 1
        assert any(tk.bytes_)  # the lifted bytes are real SNAP content


def test_grid_freq_cursor_surfaced_not_faked(grid):
    """The Grid freq axes are surfaced as unrecovered (the cursor sequencer does not
    close as a fixed-size recurrence from the bounded artifact) -- never faked by
    replaying the output samples (that would be length-proportional, design 2.8)."""
    prog = grid["prog"]
    for r in (0, 7, 8):  # changing freq registers
        assert prog.axes[r].status == "unrecovered"
        assert prog.axes[r].render is None
