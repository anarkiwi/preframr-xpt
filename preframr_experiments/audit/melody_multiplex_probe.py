"""Cross-voice multiplexing probe (design/melody_channel_factorization.md RESULTS pointed here):
does interleaving all 3 SID voices into ONE next-token stream — the deployment condition — drop
skeleton (note) predictability vs the single-voice de-multiplexed line?

One extraction (`extract_sid_melody --multiplex`), same llama3_2 mini body / held-out-by-dump split
as the single-voice channel probe. Train next-token on the frame-multiplexed 3-voice stream; score
held-out accuracy on SKELETON positions (pooled + per voice). Reference single-voice numbers
(`melody_channel_probe`): interleaved 0.336, skeleton_only 0.368. If multiplex skeleton acc falls
well below those, cross-voice multiplexing is the larger lever (and the deployed V0-onset≈0 driver),
consistent with the de-merge / voice-lane axis.

Usage (xpt image, GPU):
  python3 -m preframr_experiments.audit.melody_multiplex_probe --data /data/mini_multiplex.json \
      --cap 512 --seeds 3
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
    out = []
    for s in json.load(open(path))["seqs"]:
        if len(s["tokens"]) >= 12:
            out.append(
                (
                    s["dump"],
                    s["tokens"][:cap],
                    [bool(b) for b in s["is_skel"][:cap]],
                    s["voice"][:cap],
                )
            )
    return out


def split(seqs, seed=0):
    dumps = sorted({d for d, *_ in seqs})
    rng = np.random.default_rng(seed)
    held = set(rng.choice(dumps, max(1, len(dumps) // 5), replace=False))
    return [s for s in seqs if s[0] not in held], [s for s in seqs if s[0] in held]


def skel_ngram_ceiling(train, test, k=2):
    ctx = defaultdict(Counter)
    for _, t, _, _ in train:
        for j in range(k, len(t)):
            ctx[tuple(t[j - k : j])][t[j]] += 1
    model = {c: cnt.most_common(1)[0][0] for c, cnt in ctx.items()}
    h = n = 0
    for _, t, sk, _ in test:
        for j in range(k, len(t)):
            if not sk[j]:
                continue
            n += 1
            h += model.get(tuple(t[j - k : j]), -999) == t[j]
    return h / max(n, 1)


def train_and_score(seqs, epochs, dev, seed=0):
    torch.manual_seed(seed)
    train, test = split(seqs, seed)
    alpha = {}
    for _, t, _, _ in seqs:
        for tok in t:
            alpha.setdefault(tok, len(alpha))
    vocab = len(alpha)
    maxlen = max(len(t) for _, t, _, _ in seqs)

    def pack(arm):
        x = np.zeros((len(arm), maxlen), np.int64)
        pad = np.zeros((len(arm), maxlen), bool)
        skel = np.zeros((len(arm), maxlen), bool)
        vox = np.full((len(arm), maxlen), -1, np.int64)
        for i, (_, t, sk, vv) in enumerate(arm):
            x[i, : len(t)] = [alpha[tok] for tok in t]
            pad[i, : len(t)] = True
            skel[i, : len(sk)] = sk
            vox[i, : len(vv)] = vv
        return x, pad, skel, vox

    xtr, mtr, str_, _ = pack(train)
    xte, mte, ste, vte = pack(test)
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
    bs = 8
    for _ in range(epochs):
        model.train()
        for i in range(0, len(xtr), bs):
            b = rng.permutation(len(xtr))[i : i + bs]
            xb = torch.from_numpy(xtr[b]).to(dev)
            lg = fwd(xb)
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, vocab), xb[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    # held-out skeleton-position accuracy, pooled + per voice (target position j scored when
    # real & skeleton; voice taken from the target token).
    hit = np.zeros(3, np.int64)
    tot = np.zeros(3, np.int64)
    th = tt = 0
    with torch.inference_mode():
        for i in range(0, len(xte), bs):
            xb = torch.from_numpy(xte[i : i + bs]).to(dev)
            pr = fwd(xb).argmax(-1)[:, :-1].cpu().numpy()
            tg = xte[i : i + bs, 1:]
            sel = mte[i : i + bs, 1:] & ste[i : i + bs, 1:]
            vv = vte[i : i + bs, 1:]
            correct = (pr == tg) & sel
            th += int(correct.sum())
            tt += int(sel.sum())
            for v in range(3):
                vm = sel & (vv == v)
                hit[v] += int((correct & (vv == v)).sum())
                tot[v] += int(vm.sum())
    pooled = th / max(tt, 1)
    per_voice = [hit[v] / max(tot[v], 1) for v in range(3)]
    return base, pooled, per_voice


def render_demo(seqs, dumps_glob, out_dir, epochs, dev, seed):
    """Train a multiplex model and render a held-out 3-voice tune: the model's polyphonic
    continuation (prediction) + the ground-truth polyphony, for audition."""
    import glob as _glob

    from preframr_experiments.audit.melody_channel_render import (
        generate as mc_generate,
        render_multiplex_tokens,
        render_polyphony,
    )
    from preframr_experiments.audit.melody_channel_render import train as mc_train

    alpha = {}
    for _, t, _, _ in seqs:
        for tok in t:
            alpha.setdefault(tok, len(alpha))
    inv = {i: v for v, i in alpha.items()}
    vocab = len(alpha)
    maxlen = max(len(t) for _, t, _, _ in seqs)
    train_seqs, test_seqs = split(seqs, seed)
    model = mc_train(
        [(d, t, None) for d, t, _, _ in train_seqs],
        epochs,
        dev,
        seed,
        vocab,
        maxlen,
        alpha,
    )
    cand = [s for s in test_seqs if len(set(s[3])) == 3 and len(s[1]) >= 60]
    tune = max(cand, key=lambda s: len(s[1]))
    dump, toks = tune[0], tune[1][:400]
    prompt = [alpha[t] for t in toks[: max(2, len(toks) // 3)]]
    gen = mc_generate(model, prompt, len(toks), vocab, dev, temp=1.0, seed=seed)
    vals = [inv[i] for i in gen]
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    render_multiplex_tokens(vals, od / "channel_multiplex_pred.wav")
    dump_path = sorted(_glob.glob(dumps_glob))[dump]
    render_polyphony(dump_path, od / "channel_multiplex_gt.wav")
    print(f"  multiplex render from {Path(dump_path).name}: pred + gt WAVs in {od}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument(
        "--render-dir", default=None, help="if set, train+render a polyphony demo"
    )
    ap.add_argument(
        "--dumps", default=None, help="dump glob (matches the extraction), for render"
    )
    cli = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = load(cli.data, cli.cap)
    print(f"{cli.data.name}: {len(seqs)} seqs, cap={cli.cap}")
    pooled_all = []
    for seed in range(cli.seeds):
        base, pooled, pv = train_and_score(seqs, cli.epochs, dev, seed)
        pooled_all.append(pooled)
        print(
            f"  [seed {seed}] 2-gram(skel)={base:.3f}  HELDOUT skel pooled={pooled:.3f}  "
            f"per-voice=[{pv[0]:.3f} {pv[1]:.3f} {pv[2]:.3f}]"
        )
    if len(pooled_all) > 1:
        p = np.array(pooled_all)
        print(
            f"  MEAN pooled skel acc = {p.mean():.3f} (range {p.min():.3f}..{p.max():.3f})"
        )
    print(
        "  reference single-voice (melody_channel_probe): interleaved 0.336, skeleton 0.368"
    )
    if cli.render_dir and cli.dumps:
        render_demo(seqs, cli.dumps, cli.render_dir, cli.epochs, dev, 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
