"""Characterize op48 FREQ_ONSET: value distribution + register breakdown, and the
model's predicted-op at op48 ground-truth positions (op-level starvation vs value
error). op45 V0 interval atom is the learnable contrast."""

import argparse
import copy
from collections import Counter

import numpy as np
import torch

from preframr.train.model import Model
from preframr_experiments.audit.audit_checkpoint_per_class import _iter_eval_blocks

FREQ_LO = {0, 7, 14}
FREQ_HI = {1, 8, 15}
PW = {2, 3, 9, 10, 16, 17}
FILT = {21, 22}


def regname(r):
    if r in FREQ_LO:
        return "freqLO"
    if r in FREQ_HI:
        return "freqHI"
    if r in PW:
        return "PW"
    if r in FILT:
        return "filter"
    return f"reg{r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--device", default="cuda")
    cli = ap.parse_args()

    ckpt = torch.load(cli.ckpt, map_location=cli.device, weights_only=False)
    hp = ckpt["hyper_parameters"]
    args = copy.deepcopy(hp["args"])
    args.compile = False
    tokens = hp["tokens"]
    model = Model(
        args,
        hp["n_vocab"],
        tokens,
        hp["tkmodel"],
        hp.get("metadata"),
        reg_widths=hp.get("reg_widths"),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(cli.device)

    # tkvocab 0 -> uid == base atom row. Map uid -> (op,reg,subreg,val).
    n = model.n_vocab
    uid_op = np.full(n, -1)
    uid_reg = np.full(n, -1)
    uid_val = np.full(n, -1)
    for uid in range(min(n, len(tokens))):
        row = tokens.iloc[uid]
        uid_op[uid] = int(row["op"])
        uid_reg[uid] = int(row["reg"])
        uid_val[uid] = int(row["val"])

    gt_all, pred_all = [], []
    with torch.inference_mode():
        from pathlib import Path as _P

        for _name, block in _iter_eval_blocks(_P(cli.work_dir), 0):
            x = torch.from_numpy(block[:-1]).long().unsqueeze(0).to(cli.device)
            logits = model.model(x)
            pred = (
                (
                    torch.cat([c.argmax(-1) for c in logits], 1)
                    if isinstance(logits, list)
                    else logits.argmax(-1)
                )
                .flatten()
                .tolist()
            )
            gt_all.extend(int(t) for t in block[1:].tolist())
            pred_all.extend(int(t) for t in pred)
    gt = np.array(gt_all)
    pred = np.array(pred_all)
    gt_op = uid_op[gt]
    pred_op = uid_op[pred]

    def report(op, label):
        m = gt_op == op
        ngt = int(m.sum())
        print(f"\n=== {label} (op{op}) : {ngt} ground-truth positions ===")
        if not ngt:
            return
        # value/reg structure of the GT atoms
        vals = uid_val[gt[m]]
        regs = uid_reg[gt[m]]
        print(
            f"  distinct GT values: {len(set(vals.tolist()))}  | reg mix: "
            f"{dict(Counter(regname(r) for r in regs.tolist()).most_common())}"
        )
        print(f"  top GT values: {Counter(vals.tolist()).most_common(8)}")
        # what op does the model predict AT these positions?
        pop = Counter(int(o) for o in pred_op[m].tolist())
        print(f"  predicted-op histogram here: {pop.most_common(6)}")
        same_op = int((pred_op[m] == op).sum())
        print(
            f"  predicted op{op} at all (op-level recall): {same_op}/{ngt} = {same_op/ngt:.3f}"
        )
        if same_op:
            exact = int((pred[m] == gt[m]).sum())
            print(
                f"  exact-atom match (op+reg+subreg+val): {exact}/{ngt} = {exact/ngt:.3f}"
            )
        # model's overall appetite for emitting this op anywhere
        print(
            f"  model emits op{op} anywhere: {int((pred_op==op).sum())} times "
            f"(of {len(pred)} predictions)"
        )

    print(f"total scored positions: {len(gt)}")
    report(48, "FREQ_ONSET")
    report(45, "FREQ_TRAJ (incl V0 interval)")


if __name__ == "__main__":
    main()
