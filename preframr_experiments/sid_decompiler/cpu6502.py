"""Minimal, self-contained 6502 core for the Stage-0 recurrence interpreter.

This is NOT py65 and NOT the project's white-box ``LftBackend``. It is a small,
auditable execution engine that re-executes a *fixed-size* recovered program (a
zero-page image + entry vector) one play-call at a time. It implements exactly
the legal + illegal (undocumented NMOS) opcodes the A Mind Is Born play handler
uses; any opcode outside that set raises, so the interpreter can never silently
diverge from the program it claims to recover.

The whole point of Stage 0 is to drive the recovered ``{G_A}`` recurrence model
(``stage0_amind``); this core is the substrate that re-executes the lifted
per-call transition over the fixed-size state. The state carried between calls is
the 256-byte zero page (the master counter ``$13``/``$20`` and the handful of
self-modified accumulator cells), which is what makes the recovery length
independent: it does not grow with the number of frames rendered.
"""

N, V, B, D, I, Z, C = 0x80, 0x40, 0x10, 0x08, 0x04, 0x02, 0x01


class CPU6502:
    """Flat 64K memory 6502 with the opcodes the lft play handler needs."""

    def __init__(self, mem):
        self.m = mem  # bytearray(0x10000)
        self.a = self.x = self.y = 0
        self.sp = 0xFF
        self.p = 0x24
        self.pc = 0
        self.sid = bytearray(0x20)  # 32-register SID shadow latched on D4xx writes

    # ---- memory ----
    def rb(self, addr):
        return self.m[addr & 0xFFFF]

    def wb(self, addr, val):
        addr &= 0xFFFF
        val &= 0xFF
        if 0xD400 <= addr < 0xD420:
            self.sid[addr - 0xD400] = val
        self.m[addr] = val

    def _fetch(self):
        op = self.m[self.pc]
        self.pc = (self.pc + 1) & 0xFFFF
        return op

    def _imm(self):
        v = self.m[self.pc]
        self.pc = (self.pc + 1) & 0xFFFF
        return v

    def _rel(self):
        off = self.m[self.pc]
        self.pc = (self.pc + 1) & 0xFFFF
        if off & 0x80:
            off -= 0x100
        return off

    # ---- flags ----
    def _nz(self, v):
        v &= 0xFF
        self.p &= ~(N | Z)
        if v == 0:
            self.p |= Z
        else:
            self.p |= v & N
        return v

    # ---- one instruction; returns False on the JMP $EA7E exit trap ----
    def step(self):
        pc0 = self.pc
        op = self._fetch()
        # JMP $EA7E (Kernal IRQ exit) is the play-call terminator.
        if op == 0x4C:
            lo = self._imm()
            hi = self._imm()
            self.pc = lo | (hi << 8)
            return self.pc != 0xEA7E
        h = self._OPS.get(op)
        if h is None:
            raise NotImplementedError(f"opcode {op:#04x} at {pc0:#06x}")
        h(self)
        return True

    # ---------------- opcode handlers ----------------
    # Legal subset
    def _lda_imm(self):
        self.a = self._nz(self._imm())

    def _lda_zp(self):
        self.a = self._nz(self.m[self._imm()])

    def _lda_zpx(self):
        self.a = self._nz(self.m[(self._imm() + self.x) & 0xFF])

    def _and_imm(self):
        self.a = self._nz(self.a & self._imm())

    def _and_abs(self):
        lo = self._imm()
        hi = self._imm()
        self.a = self._nz(self.a & self.m[(lo | (hi << 8)) & 0xFFFF])

    def _eor_imm(self):
        self.a = self._nz(self.a ^ self._imm())

    def _ora_zp(self):
        self.a = self._nz(self.a | self.m[self._imm()])

    def _sta_zp(self):
        self.wb(self._imm(), self.a)

    def _sta_zpx(self):
        self.wb((self._imm() + self.x) & 0xFF, self.a)

    def _sta_iy(self):
        zp = self._imm()
        base = self.m[zp] | (self.m[(zp + 1) & 0xFF] << 8)
        self.wb((base + self.y) & 0xFFFF, self.a)

    def _sta_abs(self):
        lo = self._imm()
        hi = self._imm()
        self.wb(lo | (hi << 8), self.a)

    def _stx_zp(self):
        self.wb(self._imm(), self.x)

    def _sty_zp(self):
        self.wb(self._imm(), self.y)

    def _ldx_imm(self):
        self.x = self._nz(self._imm())

    def _ldx_zpy(self):
        self.x = self._nz(self.m[(self._imm() + self.y) & 0xFF])

    def _ldy_imm(self):
        self.y = self._nz(self._imm())

    def _tax(self):
        self.x = self._nz(self.a)

    def _tay(self):
        self.y = self._nz(self.a)

    def _txa(self):
        self.a = self._nz(self.x)

    def _dex(self):
        self.x = self._nz((self.x - 1) & 0xFF)

    def _dey(self):
        self.y = self._nz((self.y - 1) & 0xFF)

    def _inc_zp(self):
        a = self._imm()
        self.m[a] = self._nz(self.m[a] + 1)

    def _dec_zp(self):
        a = self._imm()
        self.m[a] = self._nz(self.m[a] - 1)

    def _lsr_a(self):
        self.p &= ~C
        self.p |= self.a & 1
        self.a = self._nz(self.a >> 1)

    def _cmp_imm(self):
        m = self._imm()
        t = self.a - m
        self.p &= ~(C | N | Z)
        if t >= 0:
            self.p |= C
        t &= 0xFF
        if t == 0:
            self.p |= Z
        self.p |= t & N

    def _cpx_imm(self):
        m = self._imm()
        t = self.x - m
        self.p &= ~(C | N | Z)
        if t >= 0:
            self.p |= C
        t &= 0xFF
        if t == 0:
            self.p |= Z
        self.p |= t & N

    def _nop(self):
        pass

    def _branch(self, cond):
        off = self._rel()
        if cond:
            self.pc = (self.pc + off) & 0xFFFF

    def _bne(self):
        self._branch(not (self.p & Z))

    def _beq(self):
        self._branch(bool(self.p & Z))

    def _bcc(self):
        self._branch(not (self.p & C))

    def _bcs(self):
        self._branch(bool(self.p & C))

    def _bpl(self):
        self._branch(not (self.p & N))

    # Illegal / undocumented NMOS subset (semantics match the project MPU65ILL)
    def _lax_zp(self):  # LAX zpg: A = X = M
        v = self.m[self._imm()]
        self.a = self.x = self._nz(v)

    def _lax_zpy(self):  # LAX zpg,Y
        v = self.m[(self._imm() + self.y) & 0xFF]
        self.a = self.x = self._nz(v)

    def _lxa_imm(self):  # 0xAB LXA/ATX: A = X = imm (canonical impl)
        v = self._imm()
        self.a = self.x = self._nz(v)

    def _alr_imm(self):  # ALR/ASR: A = (A & imm) >> 1 ; C = bit0 before shift
        v = self.a & self._imm()
        self.p &= ~(C | N | Z)
        self.p |= v & 1
        self.a = self._nz(v >> 1)

    def _anc_imm(self):  # ANC: A &= imm ; C = bit7
        self.a = self._nz(self.a & self._imm())
        self.p &= ~C
        if self.a & 0x80:
            self.p |= C

    def _axs_imm(self):  # AXS/SBX: X = (A & X) - imm ; C like CMP
        m = self._imm()
        t = (self.a & self.x) - m
        self.p &= ~(C | N | Z)
        if t >= 0:
            self.p |= C
        self.x = self._nz(t & 0xFF)

    def _sre_zp(self):  # SRE/LSE: M >>= 1 (C=bit0) ; A ^= M
        a = self._imm()
        v = self.m[a]
        self.p &= ~C
        self.p |= v & 1
        v >>= 1
        self.m[a] = v
        self.a = self._nz(self.a ^ v)


CPU6502._OPS = {
    0x29: CPU6502._and_imm,
    0x2D: CPU6502._and_abs,
    0x49: CPU6502._eor_imm,
    0x05: CPU6502._ora_zp,
    0x4A: CPU6502._lsr_a,
    0x85: CPU6502._sta_zp,
    0x95: CPU6502._sta_zpx,
    0x91: CPU6502._sta_iy,
    0x8D: CPU6502._sta_abs,
    0x86: CPU6502._stx_zp,
    0x84: CPU6502._sty_zp,
    0xA9: CPU6502._lda_imm,
    0xA5: CPU6502._lda_zp,
    0xB5: CPU6502._lda_zpx,
    0xA2: CPU6502._ldx_imm,
    0xB6: CPU6502._ldx_zpy,
    0xA0: CPU6502._ldy_imm,
    0xAA: CPU6502._tax,
    0xA8: CPU6502._tay,
    0x8A: CPU6502._txa,
    0x88: CPU6502._dey,
    0xE6: CPU6502._inc_zp,
    0xC6: CPU6502._dec_zp,
    0xC9: CPU6502._cmp_imm,
    0xE0: CPU6502._cpx_imm,
    0xEA: CPU6502._nop,
    0xD0: CPU6502._bne,
    0xF0: CPU6502._beq,
    0x90: CPU6502._bcc,
    0xB0: CPU6502._bcs,
    0x10: CPU6502._bpl,
    # illegal / undocumented
    0xA7: CPU6502._lax_zp,
    0xB7: CPU6502._lax_zpy,
    0xAB: CPU6502._lxa_imm,
    0x4B: CPU6502._alr_imm,
    0x2B: CPU6502._anc_imm,
    0x0B: CPU6502._anc_imm,
    0xCB: CPU6502._axs_imm,
    0x47: CPU6502._sre_zp,
}
