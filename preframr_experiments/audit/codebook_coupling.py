#!/usr/bin/env python3
"""Codebook coupling audit: does the inline define->reference->reuse MECHANISM learn,
and does it TRANSFER to held-out composers?

The STAMP / PATCH / WAVETABLE codebooks are *inline* -- the DEF is emitted in-stream
just before the REF that points into it -- so a REF id is a pointer into a locally
defined set, not an absolute label. ``gate/op_acc(STAMP_REF)`` alone conflates two
ways to score it: memorising per-tune id frequencies vs genuinely tracking the local
DEF. This audit separates them on a saved checkpoint's eval blocks:

1. Eval-family stratification (the transfer test). REF-op selection accuracy on
   ``eval_a`` (in-distribution composers) vs each ``eval_b_*`` (held-out composers).
   Because ids are defined in-context, a model that learned the *mechanism* references
   correctly on held-out composers (the DEF is in the prompt); a memoriser's eval_b
   REF acc collapses. eval_b REF acc is the transfer read.
2. DEF->REF distance stratification. Selection acc bucketed by how far back the
   matching DEF is. Flat-with-distance = real tracking; near-only = recency bias.
3. Learned validity (mask-off). Under teacher-forced eval (no constrained mask), the
   rate at which the PREDICTED REF id was already DEFINED earlier in the block --
   learned validity vs mask-guaranteed validity.
4. Reuse descriptors: REF/DEF counts and refs-per-def per family (how much reuse is
   even available to learn).

The block-walk core (``coupling_from_blocks``) is pure + unit-tested with synthetic
sequences; ``audit`` is the torch checkpoint driver (run in the xpt image), mirroring
``audit_checkpoint_per_class``'s forward pass but keeping block boundaries.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from pathlib import Path

# Codebook op ids (preframr_tokens.stfconstants). Imported lazily in the driver so the
# pure core stays import-light for tests; duplicated here as the audit's contract.
STAMP_DEF_OP = 56
STAMP_REF_OP = 59
PATCH_DEF_OP = 60
STAMP_REL_REF_OP = 63
WAVETABLE_DEF_OP = 65
WAVETABLE_REF_OP = 68
BACK_REF_OP = 15

# REF op -> the family whose DEF it points into ("BACK" = loop back-ref, def-less).
REF_OP_FAMILY = {
    STAMP_REF_OP: "STAMP",
    STAMP_REL_REF_OP: "STAMP",
    WAVETABLE_REF_OP: "WAVETABLE",
    BACK_REF_OP: "BACK",
}
DEF_OP_FAMILY = {
    STAMP_DEF_OP: "STAMP",
    PATCH_DEF_OP: "PATCH",
    WAVETABLE_DEF_OP: "WAVETABLE",
}
# (op, subreg) atoms that carry the codebook id in their ``val``. WAVETABLE id atom is
# subreg WT_REF_SUBREG_ID/WT_STEP id == 0 for both DEF and REF; STAMP/PATCH carry the
# id directly. Conservative: an op absent here gets op-level acc but no id coupling.
_WT_ID_SUBREG = 0
ID_ATOM = {
    (WAVETABLE_REF_OP, _WT_ID_SUBREG): "WAVETABLE",
    (WAVETABLE_DEF_OP, _WT_ID_SUBREG): "WAVETABLE",
}


def _dist_bucket(d: int) -> str:
    if d <= 8:
        return "<=8"
    if d <= 32:
        return "9-32"
    if d <= 128:
        return "33-128"
    return ">128"


def coupling_from_blocks(subset_blocks, atom_of_id):
    """Pure coupling core.

    ``subset_blocks``: {subset_name: [(gt_block, pred_block), ...]} where ``gt_block``
    is the token-id sequence and ``pred_block[j]`` is the model's argmax prediction of
    ``gt_block[j+1]`` (len(pred) == len(gt)-1; teacher-forced).
    ``atom_of_id``: token_id -> (op, subreg, val).

    Returns a nested dict: per subset -> {ref_op_acc, dist_acc, learned_validity, reuse}.
    """
    out: dict = {}
    for subset, blocks in subset_blocks.items():
        ref_hits: dict = {}  # op -> [n_correct, n]
        dist_hits: dict = {}  # bucket -> [n_correct, n]
        valid_pred = [0, 0]  # predicted-id-was-defined-earlier (n_yes, n_total)
        def_count: dict = {}  # family -> n DEF id-atoms
        ref_count: dict = {}  # family -> n REF id-atoms
        for gt_block, pred_block in blocks:
            defined: dict = {}  # family -> {id: last_def_pos}
            for i in range(1, len(gt_block)):
                gid = int(gt_block[i])
                op, subreg, val = atom_of_id.get(gid, (-1, -1, -1))
                fam = (op, subreg)
                # Track DEF id-atoms seen so far in this block.
                if op in DEF_OP_FAMILY and ID_ATOM.get(fam):
                    f = ID_ATOM[fam]
                    defined.setdefault(f, {})[int(val)] = i
                    def_count[f] = def_count.get(f, 0) + 1
                if op not in REF_OP_FAMILY:
                    continue
                # Op-level selection accuracy (exact-token == correct id at tkvocab=0).
                pred = int(pred_block[i - 1])
                correct = pred == gid
                h = ref_hits.setdefault(op, [0, 0])
                h[0] += correct
                h[1] += 1
                # Id-level coupling, where the REF id atom is resolvable.
                idf = ID_ATOM.get(fam)
                if idf is None:
                    continue
                ref_count[idf] = ref_count.get(idf, 0) + 1
                defset = defined.get(idf, {})
                rid = int(val)
                if rid in defset:
                    b = _dist_bucket(i - defset[rid])
                    db = dist_hits.setdefault(b, [0, 0])
                    db[0] += correct
                    db[1] += 1
                # Learned validity of the PREDICTED id (mask-off).
                p_op, p_sub, p_val = atom_of_id.get(pred, (-1, -1, -1))
                if ID_ATOM.get((p_op, p_sub)) == idf:
                    valid_pred[1] += 1
                    valid_pred[0] += int(int(p_val) in defset)
        out[subset] = {
            "ref_op_acc": {
                str(op): {"acc": h[0] / h[1], "n": h[1]}
                for op, h in sorted(ref_hits.items())
            },
            "dist_acc": {
                b: {"acc": v[0] / v[1], "n": v[1]} for b, v in sorted(dist_hits.items())
            },
            "learned_validity": (
                {"rate": valid_pred[0] / valid_pred[1], "n": valid_pred[1]}
                if valid_pred[1]
                else None
            ),
            "reuse": {
                f: {
                    "defs": def_count.get(f, 0),
                    "refs": ref_count.get(f, 0),
                    "refs_per_def": (
                        (ref_count.get(f, 0) / def_count[f])
                        if def_count.get(f)
                        else None
                    ),
                }
                for f in sorted(set(def_count) | set(ref_count))
            },
        }
    return out


def transfer_summary(per_subset: dict) -> dict:
    """Collapse per-subset ref_op_acc into an eval_a vs eval_b transfer read per REF op."""
    fams = {"eval_a": {}, "eval_b": {}}
    for subset, d in per_subset.items():
        bucket = (
            "eval_a"
            if subset == "eval_a"
            else ("eval_b" if subset.startswith("eval_b") else None)
        )
        if bucket is None:
            continue
        for op, v in d["ref_op_acc"].items():
            agg = fams[bucket].setdefault(op, [0.0, 0])
            agg[0] += v["acc"] * v["n"]
            agg[1] += v["n"]
    summary = {}
    for op in sorted(set(fams["eval_a"]) | set(fams["eval_b"])):
        a = fams["eval_a"].get(op)
        b = fams["eval_b"].get(op)
        acc_a = a[0] / a[1] if a and a[1] else None
        acc_b = b[0] / b[1] if b and b[1] else None
        summary[op] = {
            "eval_a_acc": acc_a,
            "eval_b_acc": acc_b,
            "transfer_drop": (
                (acc_a - acc_b) if (acc_a is not None and acc_b is not None) else None
            ),
            "n_a": a[1] if a else 0,
            "n_b": b[1] if b else 0,
        }
    return summary


def _derive_eval_b_composers() -> set:
    """Composer dir-names designated as eval-B holdouts across tiers (mini / canonical /
    prodlike). Used to recover the eval_a-vs-eval_b transfer split at mini tier, where the
    runner merges eval-A + eval-B into one ``eval/`` subdir (the composer survives in the
    staged path ``eval/<composer>/<tune>.blocks.npy``)."""
    from preframr_experiments.base import (
        canonical_paths,
        mini_paths,
        prodlike_paths,
    )

    out: set = set()
    for fn in (mini_paths, canonical_paths, prodlike_paths):
        try:
            layout = fn()
        except Exception:  # noqa: BLE001 -- a tier's lists may be absent
            continue
        for key, rels in layout.items():
            if key.startswith("eval-B-"):
                out |= {Path(rel).parent.name for rel in rels}
    return out


def _subset_label(subset_dir_name: str, composer: str, eval_b_composers: set) -> str:
    """Stratify a block into the transfer buckets. A designated eval-B composer is
    labelled ``eval_b_<composer>`` wherever it appears (recovers the split at mini);
    otherwise the native ``eval`` subdir becomes ``eval_a`` and an already-split
    ``eval_b_*`` / ``eval_a`` subdir is kept as-is (canonical / prodlike)."""
    if eval_b_composers and composer in eval_b_composers:
        return f"eval_b_{composer}"
    if subset_dir_name == "eval":
        return "eval_a"
    return subset_dir_name


def _iter_eval_blocks(work_dir: Path, max_blocks_per_subset: int):
    """Yield (subset_dir_name, composer, block_array) from each eval*/<composer>/*.0.blocks.npy."""
    import numpy as np

    subsets = sorted(p for p in work_dir.iterdir() if p.name.startswith("eval"))
    for subset_dir in subsets:
        files = sorted(glob.glob(str(subset_dir / "*" / "*.0.blocks.npy")))
        emitted = 0
        for f in files:
            composer = Path(f).parent.name
            arr = np.load(f)
            if arr.ndim == 1:
                arr = arr[None, :]
            for row in arr:
                yield subset_dir.name, composer, row
                emitted += 1
                if max_blocks_per_subset and emitted >= max_blocks_per_subset:
                    break
            if max_blocks_per_subset and emitted >= max_blocks_per_subset:
                break


def _atom_of_id_map(args, tkmodel, tokens, n_vocab: int) -> dict:
    """token_id -> (op, subreg, val) of the first base atom (tkvocab=0: identity)."""
    from preframr_tokens import RegTokenizer

    rt = RegTokenizer(args, tokens)
    if tkmodel is not None and not isinstance(tkmodel, str):
        tkmodel = tkmodel.to_str()
    rt.load(tkmodel, tokens)
    n_atoms = len(tokens) if tokens is not None else 0
    out: dict = {}
    for uid in range(n_vocab):
        base_ids = rt.decode([uid]) if rt.tkmodel else [uid]
        op, subreg, val = -1, -1, -1
        for bid in base_ids:
            bid = int(bid)
            if 0 <= bid < n_atoms:
                row = tokens.iloc[bid]
                op, subreg, val = int(row["op"]), int(row["subreg"]), int(row["val"])
                break
        out[uid] = (op, subreg, val)
    return out


def audit(
    ckpt_path: Path,
    work_dir: Path,
    max_blocks_per_subset: int,
    device: str,
    eval_b_composers: set = None,
) -> dict:
    """Load ckpt, forward eval blocks keeping block boundaries, compute coupling stats.
    ``eval_b_composers`` (default: derived holdout set) stratifies blocks into the
    eval_a-vs-eval_b transfer buckets by composer (needed at mini tier)."""
    import torch

    from preframr.train.model import Model

    if eval_b_composers is None:
        eval_b_composers = _derive_eval_b_composers()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    args = copy.deepcopy(hparams["args"])
    args.compile = False
    model = Model(
        args,
        hparams["n_vocab"],
        hparams["tokens"],
        hparams["tkmodel"],
        hparams.get("metadata"),
        reg_widths=hparams.get("reg_widths"),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model = model.to(device)
    atom_of_id = _atom_of_id_map(
        args, hparams["tkmodel"], hparams["tokens"], model.n_vocab
    )
    subset_blocks: dict = {}
    with torch.inference_mode():
        for name, composer, block in _iter_eval_blocks(work_dir, max_blocks_per_subset):
            label = _subset_label(name, composer, eval_b_composers)
            x = torch.from_numpy(block[:-1]).long().unsqueeze(0).to(device)
            logits = model.model(x)
            if isinstance(logits, list):
                pred = torch.cat([c.argmax(dim=-1) for c in logits], dim=1)
            else:
                pred = logits.argmax(dim=-1)
            pred_block = [int(t) for t in pred.flatten().tolist()]
            subset_blocks.setdefault(label, []).append(
                ([int(t) for t in block.tolist()], pred_block)
            )
    per_subset = coupling_from_blocks(subset_blocks, atom_of_id)
    return {
        "ckpt": str(ckpt_path),
        "eval_b_composers": sorted(eval_b_composers),
        "per_subset": per_subset,
        "transfer": transfer_summary(per_subset),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--max-blocks", type=int, default=0, help="Per subset; 0 = all.")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument(
        "--eval-b-composers",
        default=None,
        help="Comma-sep composer dir-names to treat as eval-B holdouts; default = derived.",
    )
    ap.add_argument("--out", type=Path, default=None)
    cli = ap.parse_args()
    eb = (
        {c.strip() for c in cli.eval_b_composers.split(",") if c.strip()}
        if cli.eval_b_composers
        else None
    )
    result = audit(
        cli.ckpt, cli.work_dir, cli.max_blocks, cli.device, eval_b_composers=eb
    )
    text = json.dumps(result, indent=2, default=str)
    if cli.out is not None:
        cli.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
