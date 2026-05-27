"""Diagnostic PW+filter ablation for the freq-core learnability A/B.

From a RAW SID register dump, drop every pulse-width (PW) and filter write, keep
control(gate)/ADSR/frequency/volume, and inject a one-time PW=midpoint (50% duty)
per voice so pulse waveforms still sound. Operating on the raw (clock,irq,chipno,
reg,val) log lets the parser's irq-delta DELAY mechanism consolidate the now-empty
frames automatically, so total timing is preserved (verified: render duration
unchanged, frame count drops as empties fold into multi-frame DELAYs).

``ablate_raw_df`` is the library entry used by the A/B ``pre_run_hook``;
``ablate_staged_dumps`` rewrites a work_dir's staged dumps symlink-safely. Run as a
module with ``--dump ... --render`` to audition original vs ablated WAVs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

VOICES = 3
VOICE_SIZE = 7
PW_REGS = {v * VOICE_SIZE + 2 for v in range(VOICES)} | {
    v * VOICE_SIZE + 3 for v in range(VOICES)
}
FILTER_REGS = {21, 22, 23}  # FC_LO, FC_HI, RES/FILT routing; keep 24=MODE/VOL (volume)
DROP_REGS = PW_REGS | FILTER_REGS
_PW_HI = {v: v * VOICE_SIZE + 3 for v in range(VOICES)}
_PW_LO = {v: v * VOICE_SIZE + 2 for v in range(VOICES)}
_PW_MID_HI, _PW_MID_LO = 0x08, 0x00  # 12-bit PW = 0x800 = 2048 (50% duty)


def ablate_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop PW + filter writes; inject one PW=midpoint per (chip, voice) at the
    start. Surviving rows keep their clock/irq, so frame timing is unchanged."""
    kept = df[~df["reg"].isin(DROP_REGS)].copy()
    inject = []
    for chip in sorted(df["chipno"].unique()):
        cdf = df[df["chipno"] == chip]
        clock0, irq0 = int(cdf["clock"].min()), int(cdf["irq"].min())
        for v in range(VOICES):
            inject.append((clock0 - 2, irq0, chip, _PW_HI[v], _PW_MID_HI))
            inject.append((clock0 - 1, irq0, chip, _PW_LO[v], _PW_MID_LO))
    inj = pd.DataFrame(inject, columns=["clock", "irq", "chipno", "reg", "val"])
    out = pd.concat([inj, kept], ignore_index=True)
    out = out.sort_values(["chipno", "clock"], kind="stable").reset_index(drop=True)
    return out[list(df.columns)]


def ablate_staged_dumps(work_dir: Path) -> int:
    """Ablate every ``*.dump.parquet`` under work_dir/train + eval*/ in place,
    symlink-safely (a staged dump may be a symlink into a shared link_root --
    never write through it). Drops the stale ``.meta.parquet`` sibling so the
    parser regenerates the fingerprint from the ablated dump. Returns file count."""
    n = 0
    roots = [work_dir / "train"] + sorted(work_dir.glob("eval*"))
    for root in roots:
        if not root.exists():
            continue
        for dump in root.rglob("*.dump.parquet"):
            ablated = ablate_raw_df(pd.read_parquet(dump))
            if dump.is_symlink():
                dump.unlink()  # break the symlink; write a local real file
            ablated.to_parquet(dump)
            meta = dump.with_name(dump.name.replace(".dump.parquet", ".meta.parquet"))
            if meta.is_symlink() or meta.exists():
                meta.unlink()
            n += 1
    return n


def _render(dump: Path, outdir: Path, cents: int = 50) -> None:
    from preframr_audio.fidelity import render_df_to_wav
    from preframr_tokens import RegLogParser, read_initial_irq
    from preframr_tokens.tokenizer_config import named_config

    raw = pd.read_parquet(dump)
    abl = ablate_raw_df(raw)
    abl_path = outdir / dump.name  # keep .dump.parquet so write_meta derives a sibling
    abl.to_parquet(abl_path)
    print(
        f"raw {len(raw)} -> ablated {len(abl)} rows "
        f"(dropped {int(raw['reg'].isin(DROP_REGS).sum())} PW/filter; "
        f"injected {6 * raw['chipno'].nunique()} PW-midpoint)"
    )
    args = named_config("baseline")
    args.cents = cents
    for tag, src in (("orig", dump), ("ablated", abl_path)):
        df = next(
            RegLogParser(args=args).parse(
                str(src), max_perm=1, require_pq=False, reparse=True
            ),
            None,
        )
        if df is None or len(df) == 0:
            print(f"  {tag}: parse empty")
            continue
        irq = read_initial_irq(df)
        wav = outdir / f"{dump.stem}.{tag}.wav"
        n, _ = render_df_to_wav(df, irq, args, wav)
        print(f"  {tag}: {wav} ({n} samples, {int((df['reg'] == -128).sum())} frames)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("/scratch/tmp/ablation_audition"))
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--cents", type=int, default=50)
    cli = ap.parse_args()
    cli.outdir.mkdir(parents=True, exist_ok=True)
    if cli.render:
        _render(cli.dump, cli.outdir, cli.cents)
    else:
        ablate_raw_df(pd.read_parquet(cli.dump)).to_parquet(cli.outdir / cli.dump.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
