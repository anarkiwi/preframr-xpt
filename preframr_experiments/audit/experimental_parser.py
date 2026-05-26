"""Experiment-only parser subclasses (Phase C investigation)."""

from __future__ import annotations

import pandas as pd

from preframr_tokens import RegLogParser
from preframr_tokens.stfconstants import FC_LO_REG, PCM_BITS, VOICES, VOICE_REG_SIZE

GLOBAL_REGS_PRESERVE = frozenset({21, 22, 23, 24})


class FilterPreservingRegLogParser(RegLogParser):
    """Skip ``_squeeze_changes`` dedup on regs 21-24. Voice regs
    unchanged. Use this to measure the lower bound: how much filter
    detail survives only the squeeze step."""

    def _squeeze_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        prev = df.groupby("reg")["val"].shift()
        preserve_mask = df["reg"].astype("int64").isin(GLOBAL_REGS_PRESERVE)
        change_mask = prev.isna() | (prev != df["val"])
        keep = change_mask | preserve_mask
        return df.loc[keep, ["clock", "irq", "reg", "val"]].reset_index(drop=True)


class FilterUnquantizedRegLogParser(FilterPreservingRegLogParser):
    """Above + bypass ``_combine_reg``'s FILTER_BITS=5 mask and
    diffmax-based duplicate collapse for the FC_LO combined cutoff.
    Voice reg combines stay at production settings."""

    def _combine_regs(self, df: pd.DataFrame) -> pd.DataFrame:
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            for reg, bits in ((v_offset, 0), ((v_offset + 2), PCM_BITS)):
                df = self._combine_reg(df, reg=reg, bits=bits)
        df = self._combine_reg(df, FC_LO_REG, bits=0, diffmax=1)
        return df.sort_values("clock", kind="stable").reset_index(drop=True)
