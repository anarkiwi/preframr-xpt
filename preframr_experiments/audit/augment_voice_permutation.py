#!/usr/bin/env python3
"""Voice-permutation augmentation: emit N variants of each input dump.parquet with SID voice indices permuted. See integration_tests/design/melody_transfer_augmentation_design.md 'Voice permutation variant' for the why + the bit-routing details."""

from __future__ import annotations

import argparse
import glob
import sys
from itertools import permutations
from pathlib import Path

import pandas as pd

VOICE_REG_SIZE = 7
N_VOICES = 3
_VOICE_REG_HI = N_VOICES * VOICE_REG_SIZE
_RES_FILT_REG = 23
_RES_FILT_VOICE_BITS = (0x01, 0x02, 0x04)


def _all_non_identity_permutations() -> list[tuple[int, ...]]:
    identity = tuple(range(N_VOICES))
    return [p for p in permutations(range(N_VOICES)) if p != identity]


def _permute_reg(reg: int, perm: tuple[int, ...]) -> int:
    if reg < 0 or reg >= _VOICE_REG_HI:
        return reg
    voice = reg // VOICE_REG_SIZE
    offset = reg % VOICE_REG_SIZE
    return perm[voice] * VOICE_REG_SIZE + offset


def _permute_res_filt_val(val: int, perm: tuple[int, ...]) -> int:
    """Permute the FILT_VOICE_1/2/3 routing bits in RES_FILT. Bits 0/1/2 map to voices 0/1/2; remap them via perm."""
    high_bits = val & ~(0x01 | 0x02 | 0x04)
    voice_bits = 0
    for v in range(N_VOICES):
        if val & _RES_FILT_VOICE_BITS[v]:
            voice_bits |= _RES_FILT_VOICE_BITS[perm[v]]
    return high_bits | voice_bits


def permute_dump(df: pd.DataFrame, perm: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    out["reg"] = out["reg"].astype(int).apply(lambda r: _permute_reg(r, perm))
    res_filt_mask = out["reg"] == _RES_FILT_REG
    if res_filt_mask.any():
        out.loc[res_filt_mask, "val"] = (
            out.loc[res_filt_mask, "val"]
            .astype(int)
            .apply(lambda v: _permute_res_filt_val(v, perm))
        )
    return out


def _perm_suffix(perm: tuple[int, ...]) -> str:
    return "".join(str(v) for v in perm)


def _output_path(in_path: Path, perm: tuple[int, ...]) -> Path:
    base = in_path.name.removesuffix(".dump.parquet")
    return in_path.parent / f"{base}.perm{_perm_suffix(perm)}.dump.parquet"


def augment_one(in_path: Path, perms: list[tuple[int, ...]]) -> list[Path]:
    df = pd.read_parquet(in_path)
    out_paths: list[Path] = []
    for perm in perms:
        out_path = _output_path(in_path, perm)
        if out_path.exists():
            continue
        permuted = permute_dump(df, perm)
        permuted.to_parquet(out_path, index=False)
        out_paths.append(out_path)
    return out_paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--input-glob",
        required=True,
        help="glob pattern for input dump.parquet files (e.g. /work/train/*/*.dump.parquet).",
    )
    ap.add_argument(
        "--permutations",
        default="all",
        help="comma-separated 3-digit permutations (e.g. '120,201') or 'all' for all 5 non-identity.",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="emit permuted dumps next to the inputs (default: requires --out-dir).",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    cli = ap.parse_args()
    if not cli.in_place and cli.out_dir is None:
        ap.error("either --in-place or --out-dir must be set")

    if cli.permutations == "all":
        perms = _all_non_identity_permutations()
    else:
        perms = []
        for token in cli.permutations.split(","):
            t = token.strip()
            if len(t) != N_VOICES or not t.isdigit():
                ap.error(f"bad permutation token {t!r}; expected 3-digit string")
            p = tuple(int(c) for c in t)
            if sorted(p) != list(range(N_VOICES)):
                ap.error(
                    f"permutation {t!r} is not a valid bijection of 0..{N_VOICES-1}"
                )
            perms.append(p)

    inputs = [Path(p) for p in sorted(glob.glob(cli.input_glob))]
    if not inputs:
        print(f"no inputs matched {cli.input_glob}", file=sys.stderr)
        return 1
    print(
        f"augmenting {len(inputs)} dump(s) with {len(perms)} permutation(s) "
        f"-> {len(inputs) * len(perms)} variants"
    )
    n_emitted = 0
    for in_path in inputs:
        out_paths = augment_one(in_path, perms)
        n_emitted += len(out_paths)
    print(
        f"emitted {n_emitted} new variants ({len(inputs) * len(perms) - n_emitted} skipped, already exist)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
