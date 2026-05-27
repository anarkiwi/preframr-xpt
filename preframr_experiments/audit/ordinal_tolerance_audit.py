#!/usr/bin/env python3
"""Tolerance-band accuracy for a contiguous-bin op (default FREQ_TRAJ op45, whose val is
an ordinal frequency bin 0..256): is a low exact-match accuracy near-miss noise or a real
miss? For every ground-truth position of the spotlight op, the prediction is classified
as wrong_op (predicted a different op or a merged piece), wrong_family (same op, different
reg/subreg), or same-family (record |Δval|); cumulative within-tolerance accuracy over
ALL spotlight-op positions tells whether coarser/ordinal binning would recover credit.

The bucketing (`tolerance_buckets`, `load_vocab`) is pure stdlib and unit-tested; the
checkpoint forward (`forward_predictions`) imports torch lazily so this module stays
importable on the host. Promoted from `/scratch/tmp/audit_freqtraj_tolerance.py`."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

FREQ_TRAJ_OP = 45
_DEFAULT_TOL = (0, 1, 2, 4, 8, 16)


def load_vocab(tokens_csv: Path):
    """tokens.csv -> parallel lists (op, reg, subreg, val) indexed by token id (row)."""
    op, reg, subreg, val = [], [], [], []
    with open(tokens_csv) as f:
        for r in csv.DictReader(f):
            op.append(int(r["op"]))
            reg.append(int(r["reg"]))
            subreg.append(int(r["subreg"]))
            val.append(int(r["val"]))
    return op, reg, subreg, val


def tolerance_buckets(
    pred_ids,
    gt_ids,
    op,
    reg,
    subreg,
    val,
    spotlight_op=FREQ_TRAJ_OP,
    n_atoms=None,
    tolerances=_DEFAULT_TOL,
):
    """Classify each spotlight-op gt position; return counts + cumulative within-tolerance
    accuracy. Base atoms are ids < n_atoms (default len(op)); merged pieces (id >= n_atoms)
    can never be a single spotlight-op atom, so they count as wrong_op."""
    if n_atoms is None:
        n_atoms = len(op)
    total = wrong_op = wrong_family = 0
    dvals: list[int] = []
    for p, g in zip(pred_ids, gt_ids):
        if g >= n_atoms or op[g] != spotlight_op:
            continue
        total += 1
        if p >= n_atoms or op[p] != spotlight_op:
            wrong_op += 1
        elif reg[p] != reg[g] or subreg[p] != subreg[g]:
            wrong_family += 1
        else:
            dvals.append(abs(int(val[p]) - int(val[g])))
    within = (
        {k: sum(1 for d in dvals if d <= k) / total for k in tolerances}
        if total
        else {}
    )
    return {
        "total": total,
        "wrong_op": wrong_op,
        "wrong_family": wrong_family,
        "same_family": len(dvals),
        "within_tol": within,
        "dvals": dvals,
    }


def forward_predictions(ckpt_path: Path, work_dir: Path, subset: str, device: str):
    """Forward each eval block through the ckpt; yield (pred_ids, gt_ids) flat lists.
    Imports torch + Model lazily so the module is host-importable."""
    import copy  # pylint: disable=import-outside-toplevel

    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    from preframr.train.model import Model  # pylint: disable=import-outside-toplevel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt["hyper_parameters"]
    args = copy.deepcopy(hp["args"])
    args.compile = False
    model = Model(
        args,
        hp["n_vocab"],
        hp["tokens"],
        hp["tkmodel"],
        hp.get("metadata"),
        reg_widths=hp.get("reg_widths"),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(device)
    preds: list[int] = []
    gts: list[int] = []
    with torch.inference_mode():
        for f in sorted(glob.glob(str(work_dir / subset / "*" / "*.0.blocks.npy"))):
            arr = np.load(f)
            if arr.ndim == 1:
                arr = arr[None, :]
            for block in arr:
                x = torch.from_numpy(block[:-1]).long().unsqueeze(0).to(device)
                logits = model.model(x)
                if isinstance(logits, list):
                    pred = torch.cat([c.argmax(dim=-1) for c in logits], dim=1)
                else:
                    pred = logits.argmax(dim=-1)
                preds.extend(int(t) for t in pred.flatten().tolist())
                gts.extend(int(t) for t in block[1:].tolist())
    return preds, gts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--subset", default="eval_a")
    ap.add_argument("--op", type=int, default=FREQ_TRAJ_OP)
    ap.add_argument("--device", default="cuda")
    cli = ap.parse_args()

    op, reg, subreg, val = load_vocab(cli.work_dir / "tokens.csv")
    preds, gts = forward_predictions(cli.ckpt, cli.work_dir, cli.subset, cli.device)
    res = tolerance_buckets(preds, gts, op, reg, subreg, val, spotlight_op=cli.op)
    total = res["total"]
    print(f"op{cli.op} ground-truth positions: {total}")
    if not total:
        return 0
    print(
        f"  pred wrong op            : {res['wrong_op']} ({100*res['wrong_op']/total:.1f}%)"
    )
    print(
        f"  pred same op, wrong family: {res['wrong_family']} "
        f"({100*res['wrong_family']/total:.1f}%)"
    )
    print(
        f"  pred same family         : {res['same_family']} ({100*res['same_family']/total:.1f}%)"
    )
    print("  cumulative acc within |Δval| tolerance (of ALL gt positions):")
    for k, acc in res["within_tol"].items():
        print(f"      |Δval| <= {k:>2} : {acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
