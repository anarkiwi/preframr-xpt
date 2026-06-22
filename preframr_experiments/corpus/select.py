"""Build tracker-stratified, residual-zero-gated tier .list files from the
codec-coverage census (``census.py`` output).

Selection axes that the new codec makes possible:
- **gate** on ``residual_ok`` (byte-exact encodable) -- a tune the codec cannot
  reproduce never enters training;
- **whole-song-in-context**: primary tiers require ``n_tokens <= 4096`` (one
  context window, no mid-song windowing -- the Orin NX real-time lever);
- **stratify by ground-truth tracker** (backend family, else player id), capped
  per composer to limit memorisation;
- **held-out eval splits**: in-distribution holdout (eval-A) + entire held-out
  composers (eval-B-<composer>) for cross-composer generalization.

List entries are ``relpath\tsubtune`` (``.sid`` HVSC-relative + 1-based subtune).
``--report`` prints coverage stats without writing (run this first on a census).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_DIR / "data"
MAX_TOKENS = 4096

# One residual-zero fixture per backend family -- the end-to-end smoke tier.
SMOKE_FIXTURES = [
    ("MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid", 1),
    ("MUSICIANS/H/Hubbard_Rob/5_Title_Tunes.sid", 2),
    ("MUSICIANS/J/Jammer/Grid_Runner.sid", 1),
    ("MUSICIANS/L/Lft/A_Mind_Is_Born.sid", 1),
    ("MUSICIANS/A/Ass_It/Ode_to_Music.sid", 1),
]

# Per-tier shape. eval-B composers are held out ENTIRELY from train (cross-composer
# generalization); eval-A is an in-distribution holdout of unseen tunes from the
# train composers. ``per_family``/``cap_per_composer`` None means "take all".
TIERS = {
    "mini": dict(per_family=20, cap_per_composer=4, eval_a_per_family=3),
    "canonical": dict(per_family=200, cap_per_composer=30, eval_a_per_family=16),
    "frontier": dict(per_family=None, cap_per_composer=200, eval_a_per_family=16),
}


def load_census(path) -> pd.DataFrame:
    """Load census.parquet; coerce types and drop unparsed rows."""
    df = pd.read_parquet(path)
    df = df[df["residual_ok"] == True]  # noqa: E712 (pandas mask)
    df = df.dropna(subset=["n_tokens"])
    df["n_tokens"] = df["n_tokens"].astype(int)
    df["subtune"] = df["subtune"].astype(int)
    df["group"] = df["family"].fillna(df["player"]).fillna("unknown")
    df["composer"] = df["relpath"].map(composer_of)
    return df


def composer_of(relpath: str) -> str:
    """Composer/grouping key: the MUSICIANS composer dir, else top/parent dir."""
    parts = Path(relpath).parts
    if parts and parts[0] == "MUSICIANS" and len(parts) >= 4:
        return parts[2]
    return f"{parts[0]}/{parts[-2]}" if len(parts) >= 2 else parts[0]


def coverage_report(df_all: pd.DataFrame, max_tokens: int = MAX_TOKENS) -> None:
    """Print per-tracker residual-zero rate and the <= max_tokens fraction."""
    print(f"census rows: {len(df_all)}")
    ok = df_all[df_all["residual_ok"] == True]  # noqa: E712
    print(f"residual_ok: {len(ok)} ({len(ok) / max(len(df_all), 1):.1%})")
    fit = ok[ok["n_tokens"].fillna(1e18).astype(float) <= max_tokens]
    print(f"residual_ok AND <= {max_tokens} tokens: {len(fit)} "
          f"({len(fit) / max(len(ok), 1):.1%} of encodable)")
    grp = df_all.assign(
        _ok=df_all["residual_ok"] == True,  # noqa: E712
        _fit=(df_all["residual_ok"] == True)  # noqa: E712
        & (df_all["n_tokens"].fillna(1e18).astype(float) <= max_tokens),
    ).groupby(df_all["family"].fillna(df_all["player"]).fillna("unknown"))
    summary = grp.agg(total=("relpath", "size"), ok=("_ok", "sum"), fit=("_fit", "sum"))
    summary = summary.sort_values("total", ascending=False)
    print("\nper-tracker (top 25):")
    for name, r in summary.head(25).iterrows():
        print(f"  {name:28s} total={int(r.total):6d} ok={int(r.ok):6d} "
              f"fit4096={int(r.fit):6d}")


def _round_robin(df: pd.DataFrame, per_family, cap_per_composer) -> pd.DataFrame:
    """Pick rows balanced across composers within each family (deterministic):
    round-robin one tune per composer until per_family / cap reached."""
    chosen = []
    for _, fam in df.sort_values(["group", "composer", "relpath", "subtune"]).groupby(
        "group", sort=True
    ):
        by_comp = {c: list(g.index) for c, g in fam.groupby("composer", sort=True)}
        taken: list = []
        counts = {c: 0 for c in by_comp}
        progress = True
        while progress and (per_family is None or len(taken) < per_family):
            progress = False
            for comp in sorted(by_comp):
                if per_family is not None and len(taken) >= per_family:
                    break
                if counts[comp] < len(by_comp[comp]) and (
                    cap_per_composer is None or counts[comp] < cap_per_composer
                ):
                    taken.append(by_comp[comp][counts[comp]])
                    counts[comp] += 1
                    progress = True
        chosen.extend(taken)
    return df.loc[chosen]


def build_tier(df: pd.DataFrame, cfg: dict, heldout: list[str]) -> dict:
    """Return {split: [(relpath, subtune)]} for one tier."""
    eval_b = {c: df[df["composer"] == c] for c in heldout}
    pool = df[~df["composer"].isin(heldout)]
    train = _round_robin(pool, cfg["per_family"], cfg["cap_per_composer"])
    leftover = pool.drop(index=train.index)
    eval_a = _round_robin(leftover, cfg["eval_a_per_family"], cfg["eval_a_per_family"])
    splits = {
        "train": _rows(train),
        "eval-A": _rows(eval_a),
    }
    for comp, sub in eval_b.items():
        if len(sub):
            splits[f"eval-B-{comp}"] = _rows(sub)
    return splits


def _rows(df: pd.DataFrame) -> list[tuple[str, int]]:
    return list(
        df.sort_values(["relpath", "subtune"])[["relpath", "subtune"]].itertuples(
            index=False, name=None
        )
    )


def write_list(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for relpath, subtune in rows:
            handle.write(f"{relpath}\t{subtune}\n")


def write_tier(out_root: Path, name: str, splits: dict) -> None:
    tier_dir = out_root / name
    write_list(tier_dir / "train.list", splits.get("train", []))
    write_list(tier_dir / "eval-A.list", splits.get("eval-A", []))
    for split, rows in splits.items():
        if split.startswith("eval-B-"):
            write_list(tier_dir / f"{split}.list", rows)
    total = sum(len(v) for v in splits.values())
    print(f"  {name}: {total} tunes across {len(splits)} splits "
          f"(train={len(splits.get('train', []))}, eval-A={len(splits.get('eval-A', []))})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", required=True, help="census.parquet")
    parser.add_argument("--out-root", default=str(DATA_DIR))
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--report", action="store_true", help="print coverage, no write")
    parser.add_argument("--tier", action="append", help="tiers to build (default all)")
    parser.add_argument(
        "--heldout-composer", action="append", default=[],
        help="composer held out entirely as eval-B (repeatable)",
    )
    args = parser.parse_args()

    raw = pd.read_parquet(args.census)
    if args.report:
        coverage_report(raw, args.max_tokens)
        return 0

    df = load_census(args.census)
    df = df[df["n_tokens"] <= args.max_tokens]
    out_root = Path(args.out_root)

    write_list(out_root / "smoke.list", list(SMOKE_FIXTURES))
    print(f"  smoke: {len(SMOKE_FIXTURES)} fixtures")

    tiers = args.tier or list(TIERS)
    for name in tiers:
        splits = build_tier(df, TIERS[name], args.heldout_composer)
        write_tier(out_root, name, splits)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
