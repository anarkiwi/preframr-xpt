"""Channel-factorization probe (design/melody_channel_factorization.md): does multiplexing
ornament into the melody prediction position steal skeleton (note) predictability?

Two arms from ONE `extract_sid_melody --channels` extraction, same llama3_2 mini body / held-out
-by-dump split / interval metric as `melody_ladder`:

  interleaved   : train on the full skeleton+ornament stream (model must also predict ornament);
                  score held-out next-token accuracy on SKELETON positions only.
  skeleton_only : same sequences, ornament removed; score skeleton accuracy (reproduces the
                  ladder L2-interval anchor ~0.247).

Decision: interleaved << skeleton_only -> multiplexing steals melody predictability; the encoding
is not sufficient under the deployment condition and channel factorization is justified.
interleaved ~= skeleton_only -> multiplexing is not the lever; the data ceiling survives.

Usage:
  python3 -m preframr_experiments.audit.melody_channel_probe --data mini_channels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from preframr.train.model.bodies import get_llama3_2


def load(path, cap):
    seqs = json.load(open(path))["seqs"]
    out = []
    for s in seqs:
        if len(s["tokens"]) >= 12:
            out.append(
                (s["dump"], s["tokens"][:cap], [bool(b) for b in s["is_skel"][:cap]])
            )
    return out


def split(seqs, seed=0):
    dumps = sorted({d for d, _, _ in seqs})
    rng = np.random.default_rng(seed)
    held = set(rng.choice(dumps, max(1, len(dumps) // 5), replace=False))
    train = [s for s in seqs if s[0] not in held]
    test = [s for s in seqs if s[0] in held]
    return train, test


def skel_ngram_ceiling(train, test, k=2):
    """2-gram ceiling scored only where the predicted token is skeleton."""
    ctx = defaultdict(Counter)
    for _, t, _ in train:
        for j in range(k, len(t)):
            ctx[tuple(t[j - k : j])][t[j]] += 1
    model = {c: cnt.most_common(1)[0][0] for c, cnt in ctx.items()}
    h = n = 0
    for _, t, sk in test:
        for j in range(k, len(t)):
            if not sk[j]:
                continue
            n += 1
            h += model.get(tuple(t[j - k : j]), -999) == t[j]
    return h / max(n, 1)


def train_and_score(seqs, epochs, dev, seed=0):
    """Train next-token over all positions; return (2-gram skel ceiling, train skel acc,
    held-out skel acc). Accuracy is scored only where the TARGET token is skeleton."""
    torch.manual_seed(seed)
    train, test = split(seqs, seed)
    alpha = {}
    for _, t, _ in seqs:
        for tok in t:
            alpha.setdefault(tok, len(alpha))
    vocab = len(alpha)
    maxlen = max(len(t) for _, t, _ in seqs)

    def pack(arm):
        x = np.zeros((len(arm), maxlen), np.int64)
        pad = np.zeros((len(arm), maxlen), bool)
        skel = np.zeros((len(arm), maxlen), bool)
        for i, (_, t, sk) in enumerate(arm):
            ids = [alpha[tok] for tok in t]
            x[i, : len(ids)] = ids
            pad[i, : len(ids)] = True
            skel[i, : len(sk)] = sk
        return x, pad, skel

    xtr, mtr, str_ = pack(train)
    xte, mte, ste = pack(test)
    base = skel_ngram_ceiling(train, test, 2)
    a = argparse.Namespace(
        layers=4,
        heads=8,
        kv_heads=4,
        embed=192,
        max_seq_len=maxlen,
        attn_dropout=0.1,
        norm_eps=1e-5,
        rope_base=500000,
        rope_scale=1.0,
        tie_word_embeddings=False,
    )
    model = get_llama3_2(vocab, a).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    def fwd(x):
        o = model(x)
        return torch.cat(o, 1) if isinstance(o, list) else o

    rng = np.random.default_rng(1 + seed)
    for _ in range(epochs):
        model.train()
        for i in range(0, len(xtr), 16):
            b = rng.permutation(len(xtr))[i : i + 16]
            xb = torch.from_numpy(xtr[b]).to(dev)
            lg = fwd(xb)
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, vocab), xb[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def acc(x, pad, skel):
        # target position t (=index 1..) is scored when it is both real (pad) and skeleton.
        model.eval()
        h = n = 0
        with torch.inference_mode():
            for i in range(0, len(x), 16):
                xb = torch.from_numpy(x[i : i + 16]).to(dev)
                pr = fwd(xb).argmax(-1)[:, :-1]
                tg = xb[:, 1:]
                sel = torch.from_numpy((pad[i : i + 16, 1:] & skel[i : i + 16, 1:])).to(
                    dev
                )
                h += int(((pr == tg) & sel).sum())
                n += int(sel.sum())
        return h / max(n, 1)

    return base, acc(xtr, mtr, str_), acc(xte, mte, ste)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", type=Path, required=True, help="extract_sid_melody --channels json"
    )
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument(
        "--seeds", type=int, default=1, help="run seeds 0..N-1; report mean delta"
    )
    cli = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = load(cli.data, cli.cap)
    skel_only = [
        (d, [t for t, s in zip(t_, sk) if s], [True] * sum(sk)) for d, t_, sk in seqs
    ]

    print(f"{cli.data.name}: {len(seqs)} seqs")
    deltas = []
    for seed in range(cli.seeds):
        base_i, tr_i, ho_i = train_and_score(seqs, cli.epochs, dev, seed)
        base_s, tr_s, ho_s = train_and_score(skel_only, cli.epochs, dev, seed)
        deltas.append(ho_s - ho_i)
        print(f"  [seed {seed}]")
        print(
            f"    interleaved   2-gram(skel)={base_i:.3f}  train={tr_i:.3f}  HELDOUT(skel)={ho_i:.3f}"
        )
        print(
            f"    skeleton_only 2-gram      ={base_s:.3f}  train={tr_s:.3f}  HELDOUT      ={ho_s:.3f}"
        )
        print(
            f"    multiplexing delta (skeleton_only - interleaved) = {ho_s - ho_i:+.3f}"
        )
    if len(deltas) > 1:
        d = np.array(deltas)
        print(
            f"  MEAN delta = {d.mean():+.3f} (range {d.min():+.3f}..{d.max():+.3f}, n={len(d)})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
