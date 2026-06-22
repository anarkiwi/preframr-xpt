"""Vendored reader for the sidtrace SDST artifact (``<prefix>.distill.bin``).

This is the *bounded tracer artifact* the Stage-2 host consumes. The format is
produced by ``preframr-sidtrace`` (``src/sidtrace.cpp`` ``emit_distill``) and is a
stable binary container; vendoring the reader keeps this repo self-contained (the
host derives ALL structure from this artifact + the SNAP RAM image, never from any
depacker or hand backend -- project HARD RULE / oracle constraint).

Sections (each a 4-byte tag): ACMP (RLE access-type map), SNAP (post-init RAM
snapshot), SIDW (PC-tagged write summary), IDXR (indexed-read VSA), SDDF (per-write
data-flow slice, design 3.1), STSQ (inter-frame state-cell sample sequence, design
3.2), END.

This reader is byte-compatible with ``preframr-sidtrace/tests/sidtrace_records.py``
(the canonical definition); see that file / ``emit_distill``'s doc comment for the
authoritative layout.
"""

import struct
from dataclasses import dataclass, field

import numpy as np

# Per-address access-type bits (mirror membus_trace.h AccBits).
ACC_EXEC_INIT = 1 << 0
ACC_READ_INIT = 1 << 1
ACC_WRITE_INIT = 1 << 2
ACC_EXEC_PLAY = 1 << 3
ACC_READ_PLAY = 1 << 4
ACC_WRITE_PLAY = 1 << 5

# Leaf kinds (mirror membus_trace.h LeafKind).
LK_IMMEDIATE = 0
LK_RAM_READ = 1
LK_STATE_CELL = 2
LK_EXOGENOUS = 3
LK_OUT_OF_WINDOW = 4
LEAF_KIND_NAME = {
    LK_IMMEDIATE: "immediate",
    LK_RAM_READ: "ram_read",
    LK_STATE_CELL: "state_cell",
    LK_EXOGENOUS: "exogenous",
    LK_OUT_OF_WINDOW: "out_of_window",
}

# ALU ops (mirror membus_trace.h AluOp).
ALU_NAME = {
    0: "NONE", 1: "ADC", 2: "SBC", 3: "AND", 4: "ORA", 5: "EOR",
    6: "ASL", 7: "LSR", 8: "ROL", 9: "ROR", 10: "INC", 11: "DEC", 12: "CMP",
}

_SIDW_ENTRY = struct.Struct("<HBBIB3x")
_IDXR_ENTRY = struct.Struct("<HHiBBHI")
_SIDDF_HEAD = struct.Struct("<HBBIBBBxHiBB")
_SIDDF_LEAF = struct.Struct("<BxHBx")
_STSQ_HEAD = struct.Struct("<HBxIIH")


@dataclass
class Leaf:
    kind: int
    addr: int
    value: int

    @property
    def kind_name(self):
        return LEAF_KIND_NAME[self.kind]


@dataclass
class SiddfSite:
    """One (PC, reg) per-write data-flow slice summary (design 3.1)."""

    pc: int
    reg: int
    count: int
    val_lo: int
    val_hi: int
    val_first: int
    has_stride: bool
    any_out_of_window: bool
    stride_base: int
    stride_step: int
    stride_idx_min: int
    stride_idx_max: int
    slice_pcs: list
    leaves: list  # list[Leaf]
    op_seq: list  # list[int] ALU ops
    val_seq: list = field(default_factory=list)  # SDCU mid-call value sequence

    def state_cells(self):
        return [l.addr for l in self.leaves if l.kind == LK_STATE_CELL]

    def immediates(self):
        return [l.value for l in self.leaves if l.kind == LK_IMMEDIATE]

    def ram_reads(self):
        return [l.addr for l in self.leaves if l.kind == LK_RAM_READ]

    def op_names(self):
        return [ALU_NAME[o] for o in self.op_seq]


@dataclass
class StateSeq:
    """One state cell's bounded inter-frame sample sequence (design 3.2)."""

    addr: int
    holds_to_end: bool
    wide: bool
    total_frames: int
    first_seen_frame: int
    samples: list


@dataclass
class Sdst:
    version: int
    init: int
    play: int
    load: int
    subtune: int
    nframes: int
    cycles_per_frame: int
    t0_cycle: int
    load_len: int
    acc: np.ndarray
    ram: np.ndarray  # post-init RAM snapshot (SNAP), 0 outside the song region
    sid_writes: list
    idx_reads: list
    siddf: list = field(default_factory=list)
    stateseq: list = field(default_factory=list)
    sdcu: list = field(default_factory=list)  # per-state-cell update DAG (design 2.2)

    def sdcu_by_addr(self):
        """Map state-cell addr -> SiddfSite (its update DAG). Highest-count wins."""
        out = {}
        for s in self.sdcu:
            if s.pc not in out or s.count > out[s.pc].count:
                out[s.pc] = s
        return out

    def siddf_by_reg(self):
        """Map reg -> SiddfSite. If multiple PCs write a reg, the highest-count
        site (the steady per-frame writer) wins."""
        out = {}
        for s in self.siddf:
            if s.reg not in out or s.count > out[s.reg].count:
                out[s.reg] = s
        return out

    def stateseq_by_addr(self):
        return {e.addr: e for e in self.stateseq}


def parse_sdst(path):
    with open(path, "rb") as handle:
        buf = handle.read()
    if buf[:4] != b"SDST":
        raise ValueError(f"not an SDST artifact: {buf[:4]!r}")
    off = 4
    version, _r = struct.unpack_from("<HH", buf, off)
    off += 4
    init, play, load, subtune, nframes, _r2 = struct.unpack_from("<6H", buf, off)
    off += 12
    (cpf,) = struct.unpack_from("<I", buf, off)
    off += 4
    (t0,) = struct.unpack_from("<q", buf, off)
    off += 8
    (load_len,) = struct.unpack_from("<I", buf, off)
    off += 4

    acc = np.zeros(65536, dtype=np.uint8)
    ram = np.zeros(65536, dtype=np.uint8)
    sid_writes, idx_reads, siddf, stateseq, sdcu = [], [], [], [], []
    while off < len(buf):
        tag = buf[off : off + 4]
        off += 4
        if tag == b"END\x00":
            break
        if tag in (b"ACMP", b"SNAP"):
            (nbytes,) = struct.unpack_from("<I", buf, off)
            off += 4
            body = buf[off : off + nbytes]
            off += nbytes
            if tag == b"ACMP":
                pos, addr = 0, 0
                while pos + 3 <= len(body) and addr < 65536:
                    run = body[pos] | (body[pos + 1] << 8)
                    acc[addr : addr + run] = body[pos + 2]
                    addr += run
                    pos += 3
            else:
                pos = 0
                while pos + 4 <= len(body):
                    a = body[pos] | (body[pos + 1] << 8)
                    ln = body[pos + 2] | (body[pos + 3] << 8)
                    pos += 4
                    ram[a : a + ln] = np.frombuffer(body[pos : pos + ln], np.uint8)
                    pos += ln
        elif tag == b"SIDW":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                pc, reg, _p, count, last_val = _SIDW_ENTRY.unpack_from(buf, off)
                off += _SIDW_ENTRY.size
                sid_writes.append((pc, reg, count, last_val))
        elif tag == b"IDXR":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                pc, base, stride, imin, imax, _p, count = _IDXR_ENTRY.unpack_from(
                    buf, off
                )
                off += _IDXR_ENTRY.size
                idx_reads.append((pc, base, stride, imin, imax, count))
        elif tag in (b"SDDF", b"SDCU"):
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            dest = siddf if tag == b"SDDF" else sdcu
            for _ in range(nent):
                (
                    pc, reg, flags, count, vlo, vhi, vfirst,
                    sbase, sstep, simin, simax,
                ) = _SIDDF_HEAD.unpack_from(buf, off)
                off += _SIDDF_HEAD.size
                (npcs,) = struct.unpack_from("<H", buf, off)
                off += 2
                pcs = list(struct.unpack_from(f"<{npcs}H", buf, off))
                off += 2 * npcs
                (nleaves,) = struct.unpack_from("<H", buf, off)
                off += 2
                leaves = []
                for _ in range(nleaves):
                    kind, addr, value = _SIDDF_LEAF.unpack_from(buf, off)
                    off += _SIDDF_LEAF.size
                    leaves.append(Leaf(kind, addr, value))
                (nops,) = struct.unpack_from("<H", buf, off)
                off += 2
                ops = list(struct.unpack_from(f"<{nops}B", buf, off)) if nops else []
                off += nops
                # SDCU mid-call value sequence (design 2.5/2.7): the genuine
                # generator state stream sampled at the cell's UPDATE site, which the
                # host feeds to Berlekamp-Massey for the LFSR-vs-not verdict. SDDF
                # writes nValSeq=0 (the field is present for a uniform entry shape).
                (nval,) = struct.unpack_from("<H", buf, off)
                off += 2
                vseq = (list(struct.unpack_from(f"<{nval}B", buf, off))
                        if nval else [])
                off += nval
                # For SDCU the `pc` field carries the state-cell ADDRESS (key).
                dest.append(
                    SiddfSite(
                        pc=pc, reg=reg, count=count, val_lo=vlo, val_hi=vhi,
                        val_first=vfirst, has_stride=bool(flags & 1),
                        any_out_of_window=bool(flags & 2), stride_base=sbase,
                        stride_step=sstep, stride_idx_min=simin, stride_idx_max=simax,
                        slice_pcs=pcs, leaves=leaves, op_seq=ops, val_seq=vseq,
                    )
                )
        elif tag == b"STSQ":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                addr, flags, total, first_seen, nsamp = _STSQ_HEAD.unpack_from(
                    buf, off
                )
                off += _STSQ_HEAD.size
                samples = list(struct.unpack_from(f"<{nsamp}B", buf, off))
                off += nsamp
                stateseq.append(
                    StateSeq(
                        addr=addr, holds_to_end=bool(flags & 1), wide=bool(flags & 2),
                        total_frames=total, first_seen_frame=first_seen,
                        samples=samples,
                    )
                )
        else:
            raise ValueError(f"unknown SDST section {tag!r}")

    return Sdst(
        version=version, init=init, play=play, load=load, subtune=subtune,
        nframes=nframes, cycles_per_frame=cpf, t0_cycle=t0, load_len=load_len,
        acc=acc, ram=ram, sid_writes=sid_writes, idx_reads=idx_reads,
        siddf=siddf, stateseq=stateseq, sdcu=sdcu,
    )
