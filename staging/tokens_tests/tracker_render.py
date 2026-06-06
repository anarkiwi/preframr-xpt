"""Render a tracker module (SID-Wizard SWM / defMON .prg) through its OWN bit-exact
player into a preframr register-write dump (the same ``clock,irq,chipno,reg,val``
parquet schema ``RegLogParser`` ingests).

This is the front half of the forward round-trip test
(``module -> register log -> tokenizer -> decode -> same output``). It deliberately
reuses each tracker's verified player (``pysidwizard.SWMPlayer`` /
``pydefmon.DefmonPlayer``) so the log IS the tracker's authentic output, then frames
it the way a real $D4xx logger would: each tick's writes share one ``irq``, ticks are
spaced a constant PAL frame period apart, empty ticks become timing gaps (the parser
turns those into DELAYs). Multispeed tunes are flattened to one tick == one frame:
the per-frame register-STATE sequence is preserved (only wall-clock tempo changes,
which is irrelevant to register-state losslessness), keeping every synthesized ``irq``
inside the parser's ``min_irq..max_irq`` admission window.
"""

from __future__ import annotations

D400 = 0xD400
MAX_REG = 24
PAL_FRAME_PERIOD = 19656  # cycles/frame; constant, inside the parser's 15000..25000 IRQ window


def _reg_off(reg: int) -> int:
    return reg - D400 if reg >= D400 else reg


def render_swm(path, nframes: int):
    """Per-tick list of (reg_offset, value) writes from the SID-Wizard player."""
    from pysidwizard import read_swm
    from pysidwizard.player import SWMPlayer

    player = SWMPlayer(read_swm(str(path)))
    return _drive(player, nframes)


def render_defmon(path, nframes: int):
    """Per-tick list of (reg_offset, value) writes from the defMON player."""
    from pydefmon import DefmonSong, DefmonPlayer

    player = DefmonPlayer(DefmonSong.from_file(str(path)))
    return _drive(player, nframes)


def _drive(player, nframes: int):
    frames = []
    for _ in range(nframes):
        writes = []
        for reg, val in player.play_frame():
            off = _reg_off(int(reg))
            if 0 <= off <= MAX_REG:
                writes.append((off, int(val) & 0xFF))
        frames.append(writes)
    return frames


def frames_to_dump_df(frames, period: int = PAL_FRAME_PERIOD):
    """Frame-major writes -> a dump DataFrame in the parser's ingest schema.

    Each tick with >=1 write shares one ``irq = (tick+1)*period``; ``clock`` is
    monotonic within and across ticks. Empty ticks are skipped, so the boundary
    ``irqdiff`` is a multiple of ``period`` -- exactly the DELAY shape the parser
    reconstructs. Intra-tick write ORDER is preserved (it is audible per the
    fidelity contract)."""
    import pandas as pd

    clocks, irqs, regs, vals = [], [], [], []
    for tick, writes in enumerate(frames):
        if not writes:
            continue
        irq = (tick + 1) * period
        for k, (reg, val) in enumerate(writes):
            clocks.append(irq + k + 1)
            irqs.append(irq)
            regs.append(reg)
            vals.append(val)
    return pd.DataFrame(
        {
            "clock": pd.array(clocks, dtype="UInt32"),
            "irq": pd.array(irqs, dtype="UInt32"),
            "chipno": pd.array([0] * len(clocks), dtype="UInt8"),
            "reg": pd.array(regs, dtype="UInt8"),
            "val": pd.array(vals, dtype="UInt8"),
        }
    )


def render_to_parquet(kind: str, src_path, out_path, nframes: int):
    """Render ``src_path`` (kind 'swm' | 'defmon') to a dump parquet at ``out_path``."""
    frames = render_swm(src_path, nframes) if kind == "swm" else render_defmon(src_path, nframes)
    df = frames_to_dump_df(frames)
    if len(df) == 0:
        raise RuntimeError(f"{src_path}: player produced no register writes")
    df.to_parquet(str(out_path))
    return out_path
