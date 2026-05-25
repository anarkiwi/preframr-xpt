"""Phase-0 viability audit for melody-transfer augmentation: splice donor/host
pairs, then validate the output parses (admission gate), renders audible
(rms>floor), and is not DC-pinned. Reports admission + plausibility rates over
the pairs. CPU-only."""

from __future__ import annotations

import argparse
import glob
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd

from preframr_experiments.audit.augment_melody_transfer import splice_dumps  # pylint: disable=import-error
from preframr.args import add_args
from preframr_audio.audio_driver import render_to_samples
from preframr_audio.sidwav import sidq
from preframr_tokens.reglogparser import (
    RegLogParser,
    prepare_df_for_audio,
    read_initial_irq,
)
from preframr_tokens.stfconstants import DUMP_SUFFIX

RMS_FLOOR = 50.0
DC_FRAC = 0.10


def _base_args():
    p = argparse.ArgumentParser()
    add_args(p)
    return p.parse_args([])


def _parse(args, path: str):
    rot = list(
        RegLogParser(args).parse(path, max_perm=1, require_pq=False, reparse=True)
    )
    return rot[0] if rot else None


def _render(df):
    irq = read_initial_irq(df)
    da, rw = prepare_df_for_audio(df, {}, irq, sidq(), strict=False)
    return render_to_samples(da, reg_widths=rw, irq=irq, cents=50)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/preframr/training-dumps")
    ap.add_argument("--tmp", default="/scratch/tmp/aug_probe")
    ap.add_argument(
        "--sample", type=int, default=5, help="dumps to draw; pairs = ordered combos"
    )
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--max-rows", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    args = _base_args()
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    dumps = glob.glob(f"{a.root}/**/*.dump.parquet", recursive=True)
    random.Random(a.seed).shuffle(dumps)
    dumps = dumps[: a.sample]
    pairs = list(itertools.permutations(dumps, 2))[: a.max_pairs]

    n = admitted = audible = clean = 0
    for donor_p, host_p in pairs:
        try:
            donor = pd.read_parquet(donor_p).iloc[: a.max_rows].copy()
            host = pd.read_parquet(host_p).iloc[: a.max_rows].copy()
            spliced = splice_dumps(donor, host)
        except Exception as exc:  # noqa: BLE001
            print(f"splice fail {Path(donor_p).name}->{Path(host_p).name}: {exc}")
            continue
        n += 1
        sp = tmp / ("splice" + DUMP_SUFFIX)
        spliced.to_parquet(sp)
        try:
            parsed = _parse(args, str(sp))
        except Exception as exc:  # noqa: BLE001
            print(f"  parse REJECT: {exc}")
            continue
        if parsed is None or parsed.empty:
            print("  parse REJECT: empty")
            continue
        admitted += 1
        try:
            s, _ = _render(parsed)
            s = np.asarray(s, dtype=np.float64)
            rms = float(np.sqrt(np.mean(s * s))) if s.size else 0.0
            if rms > RMS_FLOOR:
                audible += 1
                if abs(float(s.mean())) < DC_FRAC * rms:
                    clean += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  render fail: {exc}")

    print(
        f"\nmelody-transfer Phase-0 audit ({n} spliced pairs, max_rows={a.max_rows}):"
    )
    print(f"  parser-admitted : {admitted}/{n}")
    print(f"  audible (rms>floor): {audible}/{admitted if admitted else 0}")
    print(f"  not DC-pinned   : {clean}/{audible if audible else 0}")
    print("\nGate (design Phase 0): parser admits ~100%, audio non-silent. Gate-count")
    print(
        "vs donor + spectral-centroid checks are the phase-1 plausibility refinement."
    )


if __name__ == "__main__":
    main()
