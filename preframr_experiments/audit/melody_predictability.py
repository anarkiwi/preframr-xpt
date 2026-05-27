#!/usr/bin/env python3
"""Non-neural predictability ceiling for the anchored melodic line -- the decisive
diagnostic for "is FREQ_TRAJ/melody unlearnable because it's aleatoric, or because the
model/encoding is failing?". Operates on the per-voice sequence of trajectory ONSET
pitches (the anchored base value V0 = (FT_SUBREG_V0_HI<<8)|FT_SUBREG_V0_LO, one per
FREQ_TRAJ trajectory, on the freq regs 0/7/14 -- NOT the per-frame DELTA shape samples,
NOT PW/filter). Reports, per the pooled melodic lines:

  - marginal floor      : accuracy of always predicting the most common pitch
  - n-gram accuracy     : Bayes-optimal next-pitch accuracy under a k-order Markov model
                          (in-sample ceiling) for k=1,2,3
  - conditional entropy : H(next | prev k) in bits + effective branching 2^H (compare to
                          the design's gate-anchored Hubbard bass ~2.68 bits)
  - copy-from-history   : within-song "predict what followed this k-context last time"

Read vs the neural model's op45 accuracy (mini ~0.002, prodlike 0.067):
  best(n-gram, copy) >> neural  -> melody IS predictable; model/encoding is failing (Branch B)
  marginal ~ n-gram ~ copy ~ neural, H ~ marginal H -> history doesn't help (Branch A, aleatoric)

The predictability functions are pure stdlib and unit-tested; the parquet loader imports
pandas lazily so the module stays host-importable."""

from __future__ import annotations

import argparse
import glob as _glob
import math
from collections import Counter, defaultdict

FREQ_TRAJ_OP = 45
FREQ_REGS = (0, 7, 14)
V0_HI_SUBREG = 1
V0_LO_SUBREG = 2


def melody_lines_from_rows(rows, regs=FREQ_REGS) -> list[list[int]]:
    """rows: iterable of (reg, subreg, val) for op45 atoms of ONE song, in stream order.
    Returns one onset-pitch line per reg: V0 = (hi<<8)|lo paired per trajectory."""
    hi: dict[int, list[int]] = {r: [] for r in regs}
    lo: dict[int, list[int]] = {r: [] for r in regs}
    regset = set(regs)
    for reg, subreg, val in rows:
        if reg not in regset:
            continue
        if subreg == V0_HI_SUBREG:
            hi[reg].append(int(val))
        elif subreg == V0_LO_SUBREG:
            lo[reg].append(int(val))
    lines = []
    for r in regs:
        n = min(len(hi[r]), len(lo[r]))
        line = [(hi[r][i] << 8) | lo[r][i] for i in range(n)]
        if len(line) >= 2:
            lines.append(line)
    return lines


def marginal_floor(seqs) -> tuple[float, int]:
    """Accuracy of always predicting the single most common symbol; (acc, n_positions)."""
    c: Counter = Counter()
    for s in seqs:
        c.update(s)
    total = sum(c.values())
    return (c.most_common(1)[0][1] / total, total) if total else (0.0, 0)


def _context_counts(seqs, k):
    ctx: dict[tuple, Counter] = defaultdict(Counter)
    for s in seqs:
        for i in range(k, len(s)):
            ctx[tuple(s[i - k : i])][s[i]] += 1
    return ctx


def ngram_accuracy(seqs, k) -> float:
    """Bayes-optimal in-sample accuracy of a k-order Markov predictor (argmax next | ctx)."""
    ctx = _context_counts(seqs, k)
    correct = total = 0
    for counter in ctx.values():
        n = sum(counter.values())
        correct += counter.most_common(1)[0][1]
        total += n
    return correct / total if total else 0.0


def conditional_entropy(seqs, k) -> tuple[float, float]:
    """H(next | prev k) in bits and effective branching 2^H."""
    ctx = _context_counts(seqs, k)
    total = sum(sum(c.values()) for c in ctx.values())
    if not total:
        return 0.0, 1.0
    h = 0.0
    for counter in ctx.values():
        n = sum(counter.values())
        ch = -sum((v / n) * math.log2(v / n) for v in counter.values())
        h += (n / total) * ch
    return h, 2.0**h


def copy_oracle(seqs, k) -> tuple[float, float]:
    """Within-song 'predict the symbol that followed this k-context last time'.
    Returns (accuracy over all predicted positions, coverage = fraction with a prior ctx).
    """
    correct = covered = total = 0
    for s in seqs:
        last_next: dict[tuple, int] = {}
        for i in range(k, len(s)):
            c = tuple(s[i - k : i])
            total += 1
            if c in last_next:
                covered += 1
                if last_next[c] == s[i]:
                    correct += 1
            last_next[c] = s[i]
    return (correct / total if total else 0.0, covered / total if total else 0.0)


def lines_from_parquets(pattern: str, regs=FREQ_REGS) -> list[list[int]]:
    """Read parsed-atom parquets (one per song) and return pooled melodic onset lines.
    Lazy pandas import so the module is host-importable."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    seqs: list[list[int]] = []
    for f in sorted(_glob.glob(pattern)):
        df = pd.read_parquet(f, columns=["op", "reg", "subreg", "val"])
        df = df[df["op"] == FREQ_TRAJ_OP]
        rows = zip(df["reg"].tolist(), df["subreg"].tolist(), df["val"].tolist())
        seqs.extend(melody_lines_from_rows(rows, regs))
    return seqs


def analyze(seqs) -> dict:
    floor, n_pos = marginal_floor(seqs)
    h0, _ = conditional_entropy(seqs, 0)
    return {
        "n_lines": len(seqs),
        "n_onsets": n_pos,
        "distinct_pitches": len({v for s in seqs for v in s}),
        "marginal_floor": floor,
        "marginal_entropy_bits": h0,
        "ngram_acc": {k: ngram_accuracy(seqs, k) for k in (1, 2, 3)},
        "cond_entropy_bits": {k: conditional_entropy(seqs, k)[0] for k in (1, 2)},
        "eff_branching": {k: conditional_entropy(seqs, k)[1] for k in (1, 2)},
        "copy_oracle": {k: copy_oracle(seqs, k) for k in (2, 3, 4)},
    }


def format_report(res: dict, neural_acc: float | None) -> str:
    lines = [
        f"melodic onset lines: {res['n_lines']}  onsets: {res['n_onsets']:,}  "
        f"distinct pitches: {res['distinct_pitches']}",
        f"marginal floor (predict mode): {res['marginal_floor']:.4f}  "
        f"marginal entropy: {res['marginal_entropy_bits']:.2f} bits "
        f"({2**res['marginal_entropy_bits']:.0f} eff. pitches)",
        "n-gram accuracy (Bayes-optimal, in-sample): "
        + "  ".join(f"k={k}:{a:.4f}" for k, a in res["ngram_acc"].items()),
        "conditional entropy: "
        + "  ".join(
            f"k={k}:{h:.2f}b/{res['eff_branching'][k]:.1f}x"
            for k, h in res["cond_entropy_bits"].items()
        ),
        "copy-from-history: "
        + "  ".join(
            f"k={k}:acc={a:.4f}(cov {c:.2f})"
            for k, (a, c) in res["copy_oracle"].items()
        ),
    ]
    if neural_acc is not None:
        best = max(
            max(res["ngram_acc"].values()),
            max(a for a, _ in res["copy_oracle"].values()),
        )
        lines.append(f"\nneural op45 acc (reference): {neural_acc:.4f}")
        lines.append(f"best non-neural baseline:    {best:.4f}")
        if best > max(2 * neural_acc, neural_acc + 0.05):
            lines.append(
                "=> BRANCH B: a non-neural baseline beats the model -> melody is "
                "predictable, the model/encoding is under-performing."
            )
        elif res["cond_entropy_bits"][2] > 0.85 * res["marginal_entropy_bits"]:
            lines.append(
                "=> BRANCH A (leaning): history barely reduces entropy -> melody is "
                "~aleatoric at the token level; pivot the metric (distributional/perceptual)."
            )
        else:
            lines.append(
                "=> AMBIGUOUS: history helps somewhat but no baseline clears the model; "
                "inspect per-arm + escalate."
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--parquets",
        required=True,
        help="Glob for parsed-atom parquets (one song each), e.g. '<arm>/seed0/train/*/*.0.parquet'.",
    )
    ap.add_argument(
        "--neural-acc", type=float, default=None, help="Model op45 acc for comparison."
    )
    ap.add_argument(
        "--regs",
        default=",".join(str(r) for r in FREQ_REGS),
        help="Comma-separated freq regs (default 0,7,14).",
    )
    cli = ap.parse_args()
    regs = tuple(int(x) for x in cli.regs.split(","))
    seqs = lines_from_parquets(cli.parquets, regs)
    print(f"parquets: {cli.parquets}")
    print(format_report(analyze(seqs), cli.neural_acc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
