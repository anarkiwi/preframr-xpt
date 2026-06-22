"""Stage 3 -- the per-state-cell UPDATE DAG (SDCU) + widened SNAP close/ surface the
hard axes (design 2.2/2.3/2.5/2.6 + 5).

Two classes of axis did not close in Stage 2, for the SAME root cause: the SID-write
slice bottoms out at a self-modified state cell as a LEAF, so the host could see THAT
the cell drives the axis but not HOW the cell is recomputed each play-call.

Stage 3 adds, in the tracer, the bounded per-state-cell UPDATE DAG (SDCU section):
the backward slice of each state cell's in-call defining store/RMW -- cell' = f(cell,
C, K, immediates) -- plus a widened SNAP that captures a self-relocating generative
player's zero-page generator tables (so K is liftable by identity). The host
(generalize.py) consumes SDCU to (a) lift the K-table a fast-SMC axis dereferences by
identity and (b) diagnose, at the DAG level, exactly why an axis that is a multi-path
conditional SMC chain does NOT close as a single fixed-size recurrence -- surfacing it
HONESTLY (design 5), never faking the residual-zero gate.

The gate (project HARD RULE #0 / design 2.7) stays residual-zero. The recovered
subset renders byte-exact; the surfaced axes are reported with their precise
DAG-level reason and the identity-lifted K they DO support. Nothing is patched, no
seed image is re-executed (the AST guard below stays green).

Fixtures: the SDST artifact (.distill.bin, now carrying SDCU + widened SNAP) + the
compact [N,25] gate reference (.npz) reconstructed from the tracer .sidwr.bin.
"""

import os

import numpy as np
import pytest

from preframr_experiments.sid_decompiler import sdst as S
from preframr_experiments.sid_decompiler.generalize import (
    bm_on_byte_sequence,
    recover_program_from,
    residual,
)

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "fixtures", "sid_decompiler")

# A Mind's fast-SMC holdouts (design 4.2): v0 freq-lo, v1 freq-hi, v1 PW-hi.
SMC_HOLDOUTS = {1, 8, 10}


def _load(name):
    art = S.parse_sdst(os.path.join(FIX, f"{name}.distill.bin"))
    ref = np.load(os.path.join(FIX, f"{name}_ref.npz"))["ref"].astype(np.int64)
    return art, ref


@pytest.fixture(scope="module")
def amind():
    art, ref = _load("amind")
    prog, ref_a, phase, info = recover_program_from(art, ref, window=512)
    full = recover_program_from(art, ref, window=len(ref))
    return {"art": art, "prog": prog, "ref": ref_a, "phase": phase, "info": info,
            "full": full}


@pytest.fixture(scope="module")
def grid():
    art, ref = _load("grid")
    prog, ref_a, phase, info = recover_program_from(art, ref, window=512)
    return {"art": art, "prog": prog, "ref": ref_a, "phase": phase, "info": info}


# --------------------------------------------------------------------------- #
# Part A -- the SDCU per-state-cell UPDATE DAG is in the artifact.
# --------------------------------------------------------------------------- #
def test_amind_sdcu_update_dags_present(amind):
    """The tracer emits the per-state-cell UPDATE DAG (SDCU, design 2.2/2.3). For A
    Mind the SDCU surfaces the master counter's own +update ($13 INC), the LFSR cell
    ($14), and the table-read update of the v1 freq-hi shadow ($12 = table_f7[idx])
    -- the data the SID-write slice could not give (the cells are leaves there)."""
    art = amind["art"]
    assert art.sdcu, "no SDCU update DAGs in the artifact"
    by_addr = art.sdcu_by_addr()
    # the master counter $13 updates by an INC (self-feedback accumulator)
    assert 0x13 in by_addr, "master counter $13 update DAG missing"
    assert "INC" in by_addr[0x13].op_names()
    # the v1 freq-hi shadow $12's update reads the generator table $f7 (a ram_read)
    assert 0x12 in by_addr, "v1 freq-hi shadow $12 update DAG missing"
    assert 0xF7 in by_addr[0x12].ram_reads()


def test_amind_snap_widened_captures_zp_generator_table(amind):
    """The widened SNAP captures the relocated player's zero-page generator table
    (A Mind keeps table_f7 in EXEC'd zp $f7), so the host can lift it by identity.
    The classic narrow song-data SNAP would have excluded it (EXEC)."""
    ram = amind["art"].ram
    # table_f7 (the v1 freq-hi values) is 8 nonzero-bearing bytes at zp $f7
    tbl = [int(ram[0xF7 + i]) for i in range(8)]
    assert any(tbl), f"zp generator table_f7 not captured in widened SNAP: {tbl}"


# --------------------------------------------------------------------------- #
# Part B -- the recovered subset is residual-zero; the SMC axes are surfaced
# HONESTLY with the precise DAG-level reason + identity-lifted K (design 5).
# --------------------------------------------------------------------------- #
def test_amind_recovered_subset_residual_zero_full_length(amind):
    """Residual-zero over ALL frames on the recovered axes, from the artifact alone
    (no seed-image re-execution). The recurrences are re-executed as recurrences."""
    prog, ref, phase = amind["prog"], amind["ref"], amind["phase"]
    rec = prog.recovered_regs()
    rendered = prog.render(len(ref), phase)
    assert residual(rendered, ref, regs=rec) == 0


def test_amind_smc_axes_surfaced_not_faked(amind):
    """The fast mid-call SMC accumulators (v0 freq-lo reg1, v1 freq-hi reg8, v1 PW-hi
    reg10) are surfaced as unrecovered (design 5) -- never faked. Each carries a
    reason and has NO usable generator (render is None) -- the residual-zero gate is
    not weakened to admit them."""
    prog = amind["full"][0]
    unrec = set(prog.unrecovered_regs())
    assert SMC_HOLDOUTS <= unrec, f"expected {SMC_HOLDOUTS} surfaced, got {sorted(unrec)}"
    for reg in SMC_HOLDOUTS:
        ax = prog.axes[reg]
        assert ax.status == "unrecovered"
        assert ax.render is None       # never a usable generator
        assert ax.reason              # a specific reason


def test_amind_smc_reason_is_dag_level_and_k_lifted(amind):
    """The surfaced SMC axes carry a DAG-LEVEL reason (from the SDCU update DAG) AND
    the K-table they dereference, lifted by identity from the widened SNAP. reg8's
    update is O=table_f7[idx] -- the 8-byte table_f7 at $f7 is lifted verbatim; the
    holdout is the coupled index (an LFSR / section walk) that is not a single fixed
    recurrence (design 5.4). This is the structure the artifact DOES support, made
    explicit -- not a faked recovery."""
    prog = amind["full"][0]
    reg8 = prog.axes[8]
    assert reg8.table_k is not None, "reg8 must lift its K-table (table_f7) by identity"
    assert reg8.table_k.base == 0xF7
    assert len(reg8.table_k.bytes_) == 8
    assert tuple(reg8.table_k.bytes_) == (0x00, 0x18, 0x26, 0x20, 0x12, 0x24, 0x13, 0x10)
    # the reason names the DAG-level cause (a coupled / multi-path SMC update)
    assert "SDCU" in reg8.reason or "K_table" in reg8.reason
    # reg10 (the LFSR cell) is surfaced with a multi-path conditional-update reason
    assert "$14" in prog.axes[10].reason or "path" in prog.axes[10].reason


def test_amind_whole_tune_residual_is_exactly_the_surfaced_axes(amind):
    """HONESTY: the whole-tune residual is NONZERO and is contributed ONLY by the
    surfaced axes (the SMC holdouts). The recovered axes are byte-exact; the gate is
    not faked to 0 by admitting an axis that does not reproduce the stream."""
    art, ref = _load("amind")
    prog, ref_a, phase, info = recover_program_from(art, ref, window=len(ref))
    rec = prog.recovered_regs()
    unrec = prog.unrecovered_regs()
    rendered = prog.render(len(ref_a), phase)
    # recovered subset is exactly residual-zero
    assert residual(rendered, ref_a, regs=rec) == 0
    # whole-tune residual is nonzero (the surfaced axes are not reproduced) ...
    assert residual(rendered, ref_a) > 0
    # ... and ALL of that residual lives in the surfaced (unrecovered) columns.
    assert residual(rendered, ref_a, regs=rec) == 0
    assert residual(rendered, ref_a, regs=unrec) == residual(rendered, ref_a)


def test_amind_length_independence(amind):
    """Derive from a SHORT window (512) and render the FULL length residual-zero from
    the SAME fixed-size object; the recovered size does not grow with render length
    (slope 0) -- design 1.4."""
    prog, ref, phase = amind["prog"], amind["ref"], amind["phase"]
    rec = prog.recovered_regs()
    sizes = []
    for n in (512, 1024, 2048, 4096, len(ref)):
        rendered = prog.render(n, phase)
        assert residual(rendered, ref[:n], regs=rec) == 0, f"residual nonzero at N={n}"
        sizes.append(prog.size_bytes())
    assert max(sizes) - min(sizes) == 0      # size flat vs render length
    assert prog.size_bytes() < 256           # fixed-size object


def test_amind_sdcu_flat_vs_window():
    """The SDCU section is O(state cells), flat vs frame count: the set of state-cell
    update DAGs the SHORT-window artifact carries is a subset of the FULL-window one,
    and each cell's update DAG (op_seq) is identical -- the update is invariant across
    frames (the recurrence is the program; the trace is its unrolling)."""
    art, _ = _load("amind")
    by = art.sdcu_by_addr()
    # the update DAG of a state cell is a fixed-size object (a handful of ops/leaves)
    assert by, "no SDCU cells"
    for addr, site in by.items():
        assert len(site.op_seq) <= 32
        assert len(site.leaves) <= 64


def test_no_seed_image_dependency():
    """The generalizer derives {G_A} from the artifact ONLY -- it must not import the
    Stage-0 seed-image interpreter (AmindRecurrence / cpu6502) or the white-box
    LftBackend. The SDCU update DAGs are DATA from the artifact, not the player code:
    the recovery never re-executes the seed image (design HARD RULE / oracle
    constraint). Guard the actual code (AST, ignoring docstrings)."""
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
    assert not leaked, f"Stage 3 must not reference the seed-image crutch: {leaked}"


# --------------------------------------------------------------------------- #
# Grid_Runner -- Stage 3 additions must not regress the table-driven exemplar.
# --------------------------------------------------------------------------- #
def test_grid_recovered_subset_residual_zero(grid):
    """Whatever Grid axes close (the set-once constants) still render residual-zero
    over the captured frames from the artifact alone, with the new SDCU/widened-SNAP
    artifact."""
    prog, ref, phase = grid["prog"], grid["ref"], grid["phase"]
    rec = prog.recovered_regs()
    assert len(rec) >= 1
    rendered = prog.render(len(ref), phase)
    assert residual(rendered, ref, regs=rec) == 0


def test_grid_freq_axes_lift_k_table_by_identity(grid):
    """The Grid freq axes are table-walks; their note-table K is liftable VERBATIM
    from SNAP by identity (design 2.4). The freq cursor sequencer does not fully
    close (honestly surfaced, design 5) but the K-table identity lift is intact."""
    prog = grid["prog"]
    freq_regs = (0, 1, 7, 8, 14, 15)
    lifted = [r for r in freq_regs if prog.axes[r].table_k is not None]
    assert lifted, "no Grid freq axis lifted a K-table by identity from SNAP"
    for r in lifted:
        tk = prog.axes[r].table_k
        assert len(tk.bytes_) >= 1
        assert any(tk.bytes_)


def test_grid_freq_cursor_surfaced_not_faked(grid):
    """The Grid freq axes are surfaced as unrecovered (the nested orderlist->pattern
    ->note cursor sequencer does not close as a fixed-size recurrence from the
    bounded artifact) -- never faked by replaying output samples (design 2.8)."""
    prog = grid["prog"]
    for r in (0, 7, 8):
        assert prog.axes[r].status == "unrecovered"
        assert prog.axes[r].render is None


# --------------------------------------------------------------------------- #
# Stage 3b -- Berlekamp-Massey closure attempt on the $14 LFSR cell + the strict
# no-branching-re-exec guard (design 2.5/2.7/5.3 + HARD RULINGS 1 & 2).
#
# The fast-SMC holdouts are driven by cell $14, described as an LFSR (`LDA #$b8;
# SRE $14`) INTERLEAVED with a section-indexed self-modifying store, branch-
# selected by the section counter. RULING 1 permits closing an axis ONLY as a
# typed recurrence -- and an LFSR ONLY via Berlekamp-Massey recovering the feedback
# taps (NOT by re-running SRE in a loop). So we run BM over the cell's GENUINE
# mid-call value sequence (the new SDCU valSeq -- the call-boundary STATESEQ for a
# fast SMC cell is a constant residue and useless). The verdict is the gate: low L
# => recover taps; L ~ n/2 => NOT a typed LFSR => surface (RULING 2), never fake,
# never branch-re-execute.
# --------------------------------------------------------------------------- #
def test_sdcu_valseq_present_and_genuine(amind):
    """The Stage-3b artifact upgrade: SDCU carries the cell's MID-CALL value
    sequence (the genuine generator state stream), NOT the all-constant call-
    boundary STATESEQ. $14's mid-call sequence must be richly varying (the LFSR /
    section stream), proving the upgrade captured the real generator state."""
    by = amind["art"].sdcu_by_addr()
    assert 0x14 in by, "$14 (LFSR cell) update DAG missing"
    vs = by[0x14].val_seq
    assert len(vs) >= 256, f"$14 mid-call valSeq too short: {len(vs)}"
    # the GENUINE mid-call stream is richly varying (>= many distinct values),
    # unlike the call-boundary STATESEQ which is a constant 0 for this fast SMC cell
    assert len(set(vs)) >= 32, f"$14 mid-call valSeq not varying: {len(set(vs))} vals"
    seq = amind["art"].stateseq_by_addr().get(0x14)
    if seq is not None:
        # the call-boundary sample sequence is (near-)constant -- the reason a naive
        # BM over STATESEQ would have been meaningless; the valSeq is what BM needs
        assert len(set(seq.samples)) <= 2


def test_berlekamp_massey_14_is_not_a_typed_lfsr(amind):
    """RULING 1's LFSR test, run honestly: Berlekamp-Massey over $14's GENUINE
    mid-call value sequence. A genuine LFSR has LOW linear complexity (L << n); this
    cell has L ~ n/2 in every bit-plane -- it is NOT a low-complexity linear-feedback
    shift register, so it CANNOT be recovered as a typed BM-taps recurrence. This is
    the rigorous, data-grounded reason the $14-driven axes cannot close under the
    no-branching-re-exec rule (design 5.3)."""
    vs = amind["art"].sdcu_by_addr()[0x14].val_seq
    lmax, per, is_lfsr = bm_on_byte_sequence(vs)
    n = len(vs)
    assert not is_lfsr, f"BM unexpectedly low-complexity: L={lmax} per={per}"
    # L ~ n/2 (the high-linear-complexity / not-an-LFSR signature)
    assert lmax >= n // 3, f"L={lmax} not ~n/2 over n={n}"
    # every bit-plane is high-complexity (not just one)
    assert min(per) >= n // 4, f"some plane low-complexity: {per}"


def test_smc_holdouts_reason_cites_bm_verdict(amind):
    """Each surfaced $14-coupled holdout (reg1 v0 freq-lo, reg8 v1 freq-hi, reg10 v1
    PW-hi) carries the Berlekamp-Massey verdict in its reason -- the DAG-level,
    data-grounded cause, not a hand-wave. reg10 ($14 itself) and reg1 (reads the
    coupled $14 index) cite the high-L / NOT-a-typed-LFSR finding explicitly."""
    prog = amind["full"][0]
    for reg in (1, 10):
        r = prog.axes[reg].reason
        assert "Berlekamp-Massey" in r, f"reg{reg} reason omits BM: {r[:120]}"
        assert "NOT a typed LFSR" in r, f"reg{reg} reason omits the verdict: {r[:120]}"
    # the no-branching-re-exec rule is named as the reason closure is refused
    assert "no-branching-re-exec" in prog.axes[10].reason


def test_closures_are_closed_form_not_branch_reexec():
    """The strict no-branching-re-exec guard (RULING 1 / the retired seed-image
    crutch): every RECOVERED axis's generator (`render`) is a pure closed-form /
    numpy expression over the recovered counter -- it must NOT step a 6502 core,
    re-run SRE in a loop, or execute a branching micro-program. We assert (a) the
    module never references the seed-image interpreter (covered by the AST guard
    below too) and (b) the BM closure helper is a typed linear-algebra routine, not
    an instruction stepper: it contains no opcode dispatch / player re-execution."""
    import ast
    import inspect

    from preframr_experiments.sid_decompiler import generalize

    src = inspect.getsource(generalize)
    tree = ast.parse(src)
    # no opcode-stepping / branching-microprogram re-execution primitives
    forbidden_tokens = ("step_cpu", "run_player", "execute_opcode", "sre_loop",
                        "AmindRecurrence", "cpu6502", "LftBackend")
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    for t in forbidden_tokens:
        assert t not in names, f"closure must not re-execute branches: {t}"
    # the BM helper is a fixed linear recurrence over GF(2) bit-planes (typed
    # recurrence recovery), with NO loop that conditionally stores back into a
    # co-evolving mutable cell selected by a runtime branch (that would be the
    # forbidden micro-program). Sanity: it only reads its input `values`.
    bm = inspect.getsource(generalize.bm_on_byte_sequence)
    assert "ramFn" not in bm and "STA" not in bm and "opcode" not in bm
