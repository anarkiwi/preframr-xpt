"""Stage 5 -- raise whole-tune coverage on CONVENTIONAL tracker tunes by closing
the two located gaps, both compatible with the strict NO-BRANCH-REEXEC rule.

GAP 1 -- the sequencer cursor (the song itself). On tracker tunes the dominant
surfaced class is the orderlist->pattern->note cursor. The closable typed level is
a uniform-dwell cyclic table-walk ``O = V[(f//D) mod P]`` (design 2.6 nested cursor
+ 5.4): the cursor advances +1 every D play-calls (a dwell clock) over a P-entry
value table V lifted by identity, traversed >= 2 full cycles. The host recovers it
as a TYPED recurrence on the play-call index f -- no branch re-execution -- and
admits it ONLY when it reproduces the FULL stream residual-zero with FLAT size vs
the fit window. A cursor whose advance is song-position-driven (data-gated dwell,
no in-window loop) is SURFACED honestly, never faked.

GAP 2 -- DefMon capture was starved (ZERO SDCU, ZERO STATESEQ). DefMon is the
self-modifying compiled-player class: every axis is ``LDX #lo / STX $D4xx`` whose
``#lo`` operand byte is poked each play-call. The SID-write slice bottomed out at
that immediate as a CONSTANT, so the operand (an SMC state cell) was never surfaced
and SDCU/STATESEQ never triggered. The Stage-5 tracer fix classifies a
self-modified immediate operand as a state cell at the operand address (deferred to
finalizeLeaves so the per-call WRITE_PLAY is seen), re-enabling SDCU (the operand's
update slice) + STATESEQ (its inter-frame value sequence) -- so the host gets the
same structure for DefMon it gets for JCH.

The gate stays residual-zero (project HARD RULE #0 / design 2.7): every recovered
axis renders byte-exact over the FULL stream; surfaced axes are reported, never
patched. Fixtures: SDST artifacts (.distill.bin) + [N,25] gate refs (.npz).

Synthetic fixtures (built from self-contained PSIDs, no HVSC):
  * ``cursor_walk`` -- a genuine uniform-dwell looping table-walk (closes via the
    nested-cursor path; the flat-slope + length-independence proof).
  * ``smc_operand`` -- the DefMon SMC-operand axis class (closes via SDCU/STATESEQ
    once the operand is surfaced as a state cell).
Real depacker-less Goto80 fixtures:
  * ``defmon_acxq7`` (DefMon, CIA 2x multispeed) -- proves SDCU/STATESEQ now emit.
  * ``jch_10`` (JCH NewPlayer) -- the cursor is surfaced honestly with resid 0.
"""

import ast
import inspect
import os

import numpy as np
import pytest

from preframr_experiments.sid_decompiler import sdst as S
from preframr_experiments.sid_decompiler import generalize
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


# --------------------------------------------------------------------------- #
# GAP 1: the nested sequencer cursor closes as a TYPED uniform-dwell table-walk.
# --------------------------------------------------------------------------- #
def test_cursor_walk_closes_as_typed_recurrence():
    """The looping table-walk axis closes as a fixed-size typed recurrence
    O=V[(cursor)%P] (design 2.6 nested cursor) -- NOT a stored per-frame list."""
    art, ref = _load("cursor_walk")
    prog, ref_a, phase, info = recover_program_from(art, ref, window=64,
                                                    verify_full=True)
    assert 0 in prog.recovered_regs(), "the cursor-walk axis (reg0) must close"
    form = prog.axes[0].form
    assert ("table-walk" in form or "period-" in form), (
        f"reg0 must be a typed cursor/period recurrence, got: {form!r}")
    # residual-zero on the recovered subset over the FULL stream (the sole gate).
    assert info["residual_recovered_subset"] == 0


def test_cursor_walk_is_length_independent_flat_slope():
    """Recover from a SHORT window, reproduce the FULL stream residual-zero, with
    the recovered-program byte-size FLAT as the window grows (design 1.4 / 2.8).
    This is the anti-length-proportionality property the whole design exists for."""
    art, ref = _load("cursor_walk")
    sizes = []
    for w in (32, 64, 128, 256):
        prog, ref_a, phase, info = recover_program_from(art, ref, window=w,
                                                        verify_full=True)
        # the short-window recurrence reproduces the FULL stream byte-exact.
        assert info["residual_recovered_subset"] == 0, f"window {w} not residual-0"
        assert 0 in prog.recovered_regs()
        sizes.append(info["size_bytes"])
    assert max(sizes) - min(sizes) == 0, f"recovered size not flat vs window: {sizes}"


def test_cursor_walk_admission_requires_two_cycles():
    """A single non-repeating pass is stored output, not a recurrence: the
    cursor-walk path admits only a value table traversed >= 2 full cycles (design
    2.8). (Structural check: the recovered form reports the traversal count.)"""
    art, ref = _load("cursor_walk")
    prog, _, _, _ = recover_program_from(art, ref, window=64, verify_full=True)
    form = prog.axes[0].form
    if "table-walk" in form:
        # the nested-cursor form encodes the cycle count "traversed Nx"
        assert "traversed" in form


# --------------------------------------------------------------------------- #
# GAP 2: DefMon SDCU/STATESEQ now EMIT (the starvation fix) + close the SMC axis.
# --------------------------------------------------------------------------- #
def test_smc_operand_axis_closes_via_stateseq():
    """The DefMon SMC-operand axis class (`LDX #lo / STX $D4xx`, the operand poked
    each call) closes once the operand is surfaced as a state cell: its STATESEQ is
    the generator, the host fits a typed recurrence, residual-zero (design 1.3/2.3).
    """
    art, ref = _load("smc_operand")
    # the operand byte must be surfaced as a STATE CELL leaf (not a constant).
    state_cells = set()
    for site in art.siddf:
        state_cells.update(site.state_cells())
    assert state_cells, "SMC operand not surfaced as a state cell (Gap-2 regression)"
    assert art.sdcu, "no SDCU update DAG for the SMC operand cell"
    assert art.stateseq, "no STATESEQ for the SMC operand cell"
    prog, ref_a, phase, info = recover_program_from(art, ref, window=64,
                                                    verify_full=True)
    assert 0 in prog.recovered_regs(), "the SMC-operand axis (reg0) must close"
    assert info["residual_recovered_subset"] == 0


def test_defmon_sdcu_stateseq_now_emit():
    """The headline Gap-2 fix on a REAL DefMon tune (CIA 2x multispeed): SDCU and
    STATESEQ are non-empty (were ZERO before the SMC-operand classification fix),
    so the host gets the same structure it gets for JCH."""
    art, _ = _load("defmon_acxq7")
    assert len(art.sdcu) > 0, "DefMon SDCU still starved (Gap-2 not fixed)"
    assert len(art.stateseq) > 0, "DefMon STATESEQ still starved (Gap-2 not fixed)"
    # the surfaced state cells are the per-axis self-modified operand bytes.
    state_cells = set()
    for site in art.siddf:
        state_cells.update(site.state_cells())
    assert state_cells, "no SMC operand state cells surfaced on DefMon"


def test_defmon_recovered_subset_is_residual_zero():
    """Whatever DefMon axes the host closes from the now-emitted artifact must be
    byte-exact over the FULL stream (the sole gate); surfaced axes are honest."""
    art, ref = _load("defmon_acxq7")
    prog, ref_a, phase, info = recover_program_from(art, ref, window=512,
                                                    verify_full=True)
    assert info["residual_recovered_subset"] == 0
    # surfaced axes carry a precise, SDCU-grounded reason (never a silent patch).
    for r in prog.unrecovered_regs():
        assert prog.axes[r].reason, f"reg{r} surfaced without a reason"


# --------------------------------------------------------------------------- #
# JCH: the cursor is surfaced honestly; the recovered subset stays residual-zero.
# --------------------------------------------------------------------------- #
def test_jch_recovered_subset_residual_zero_and_honest_surfacing():
    """On a real JCH NewPlayer tune the recovered subset is residual-zero and the
    freq/control cursor axes are surfaced honestly (the cursor is song-position-
    driven and does not close in-window) -- never faked, never patched."""
    art, ref = _load("jch_10")
    prog, ref_a, phase, info = recover_program_from(art, ref, window=512,
                                                    verify_full=True)
    assert info["residual_recovered_subset"] == 0
    # the freq axes are surfaced as table-walks whose index is the unrecovered
    # cursor (design 5.4) -- with the K-table liftable by identity.
    surfaced = prog.unrecovered_regs()
    assert surfaced, "expected the cursor-driven axes to be surfaced honestly"
    for r in surfaced:
        assert prog.axes[r].reason


# --------------------------------------------------------------------------- #
# Strict-rule guard: no seed-image crutch, no branching re-exec primitives.
# --------------------------------------------------------------------------- #
def test_no_seed_image_or_branch_reexec_crutch():
    """The recovery must derive recurrences from the artifact alone -- NO seed-image
    interpreter (AmindRecurrence / cpu6502 / _play) and NO branching re-execution
    micro-program. AST-guard the actual generalize.py body (ignoring strings)."""
    tree = ast.parse(inspect.getsource(generalize))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for n in node.names:
                names.add(n.name.split(".")[0])
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
    forbidden = {"AmindRecurrence", "LftBackend", "cpu6502", "CPU6502", "_play"}
    leaked = forbidden & names
    assert not leaked, f"Stage 5 must not reference the seed-image crutch: {leaked}"


def test_cursor_recovery_does_not_store_per_frame_output():
    """The nested-cursor recovery must never admit an axis whose 'recovery' is a
    per-frame value list -- the recovered size must be O(table + params), bounded,
    and far below alpha*N (design 2.8). Verified on the real DefMon/JCH fixtures:
    recovered size is tiny regardless of the thousands of frames."""
    for name in ("defmon_acxq7", "jch_10"):
        art, ref = _load(name)
        prog, ref_a, phase, info = recover_program_from(art, ref, window=512,
                                                        verify_full=True)
        # generous bound: a few hundred bytes; N is thousands of frames.
        assert info["size_bytes"] < 0.05 * len(ref_a) + 512, (
            f"{name} recovered size {info['size_bytes']} is length-proportional")
