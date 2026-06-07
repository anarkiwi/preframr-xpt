"""Does the CURRENT encoding carry a learnable rule to generalization?

Companion to ``framework_arch_test`` (which proved the llama3_2 body generalizes a
deterministic motif rule in a TOY vocab). This asks the encoding question: take the
SAME kind of deterministic, generalizable melody rule and express it in the **real
FREQ_TRAJ atom encoding** — FRAME tick + VOICE marker + op45 trajectory
(FLAGS, V0_HI, V0_LO, COUNT_HI, COUNT_LO, DELTA) — then test whether the model can
still generalize it. Synthetic data lets us remove the "is real melody multi-modal"
confound: the rule is deterministic, so a *sufficient* encoding must generalize it.

Two conditions, identical rule and pitches:
- FLAT:  each note = one onset token (op48-style), only FRAME+VOICE between notes.
- REAL:  each note = the full op45 trajectory; consecutive onset *values* (V0_LO)
         are split across a byte (V0_HI/V0_LO) and separated by ~7 structural tokens
         (FLAGS/COUNT/DELTA + FRAME/VOICE) — the exact structure suspected of
         obscuring melody.

Rule (mirrors framework_arch_test): within a motif, pitch[i+1] = (pitch[i]+1) % P.
N_SHARED motifs in train+val, N_HELD only in val; the rule is shared, so held-out
motif continuations are predictable by RULE, not memory. The melodic value lives in
V0_LO (pitch < 256 => V0_HI constant), so the model must predict the V0_LO successor
through the real structural surroundings.

Verdict (the encoding question):
- FLAT generalizes (held-out onset acc high) AND REAL generalizes  => encoding SUFFICIENT
  (the real-data melody failure is data/multi-modality, not the encoding).
- FLAT generalizes but REAL does NOT                              => encoding DEFICIENT
  (the trajectory structure/byte-split/separation obscures a learnable rule).

Run inside the xpt image:
  python3 -m preframr_experiments.audit.encoding_generalization_test
  python3 -m preframr_experiments.audit.encoding_generalization_test --quick
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from preframr.train.model.bodies import get_llama3_2

# real atom signatures (op, reg, subreg) observed in the current encoding; val filled per note
_FRAME = (0, -128, -1, 0)
_VOICE = (0, -126, -1, 0)


def _make_motifs(n_motifs, motif_len, P, rng):
    seeds = rng.choice(P, size=n_motifs, replace=False)
    return [[(int(s) + j) % P for j in range(motif_len)] for s in seeds]


def _tune_notes(motifs, allowed, motifs_per_tune, rng):
    notes, boundary = [], []
    for _ in range(motifs_per_tune):
        m = motifs[int(rng.choice(allowed))]
        for j, p in enumerate(m):
            notes.append(p)
            boundary.append(j == 0)  # first note of a motif is not rule-predictable
    return notes, boundary


def _emit(notes, boundary, condition):
    """Return (atoms, onset_mask, boundary_mask) for one tune. ``atoms`` are
    (op,reg,subreg,val) tuples; onset_mask marks the rule-bearing pitch token."""
    atoms, onset, bnd = [], [], []

    def push(tup, is_onset=False, is_bnd=False):
        atoms.append(tup)
        onset.append(is_onset)
        bnd.append(is_bnd)

    for p, b in zip(notes, boundary):
        push(_FRAME)
        push(_VOICE)
        if condition == "flat":
            push((48, 0, -1, p), is_onset=True, is_bnd=b)  # single onset token = pitch
        else:  # real op45 trajectory; pitch lives in V0_LO (p<256 => V0_HI const 0)
            push((45, 0, 0, 2))  # FLAGS (RUN, absolute)
            push((45, 0, 1, (p >> 8) & 0xFF))  # V0_HI (constant 0)
            push((45, 0, 2, p & 0xFF), is_onset=True, is_bnd=b)  # V0_LO = the pitch
            push((45, 0, 3, 0))  # COUNT_HI
            push((45, 0, 4, 1))  # COUNT_LO
            push((45, 0, 6, 0))  # DELTA (flat shape)
    return atoms, np.array(onset), np.array(bnd)


def build_corpus(condition, P, motif_len, n_shared, n_held, n_train, n_val, seed):
    rng = np.random.default_rng(seed)
    motifs = _make_motifs(n_shared + n_held, motif_len, P, rng)
    shared = np.arange(n_shared)
    all_m = np.arange(n_shared + n_held)
    tunes = []
    for allowed, n in ((shared, n_train), (all_m, n_val)):
        split = []
        for _ in range(n):
            notes, b = _tune_notes(motifs, allowed, motifs_per_tune=8, rng=rng)
            split.append(_emit(notes, b, condition))
        tunes.append(split)
    train, val = tunes
    # shared alphabet over BOTH splits
    alpha = {}
    for split in (train, val):
        for atoms, _, _ in split:
            for a in atoms:
                alpha.setdefault(a, len(alpha))

    def pack(split):
        L = max(len(a) for a, _, _ in split)
        x = np.zeros((len(split), L), dtype=np.int64)
        om = np.zeros((len(split), L), dtype=bool)
        bm = np.zeros((len(split), L), dtype=bool)
        for i, (atoms, onset, bnd) in enumerate(split):
            ids = [alpha[a] for a in atoms]
            x[i, : len(ids)] = ids
            om[i, : len(onset)] = onset
            bm[i, : len(bnd)] = bnd
        return x, om, bm

    return (*pack(train), *pack(val), len(alpha))


def build_model(vocab, seq_len, layers, heads, kv_heads, embed, device):
    args = argparse.Namespace(
        layers=layers,
        heads=heads,
        kv_heads=kv_heads,
        embed=embed,
        max_seq_len=seq_len,
        attn_dropout=0.1,
        norm_eps=1e-5,
        rope_base=500000,
        rope_scale=1.0,
        tie_word_embeddings=False,
    )
    return get_llama3_2(vocab, args).to(device)


def _forward(model, x):
    out = model(x)
    return torch.cat(out, dim=1) if isinstance(out, list) else out


def _onset_acc(model, x, om, bm, vocab, device, bs=32):
    model.eval()
    hits = tot = 0
    with torch.inference_mode():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i : i + bs]).to(device)
            pred = _forward(model, xb).argmax(-1)[:, :-1]
            tgt = xb[:, 1:]
            # predict an onset token that is INSIDE a motif (rule-predictable)
            mask = torch.from_numpy(om[i : i + bs, 1:] & ~bm[i : i + bs, 1:]).to(device)
            hits += int(((pred == tgt) & mask).sum())
            tot += int(mask.sum())
    return hits / max(tot, 1)


def run(condition, cfg, device):
    x, om, bm, vx, vom, vbm, vocab = build_corpus(
        condition,
        cfg["P"],
        cfg["motif_len"],
        cfg["n_shared"],
        cfg["n_held"],
        cfg["n_train"],
        cfg["n_val"],
        cfg["seed"],
    )
    seq_len = x.shape[1]
    model = build_model(
        vocab,
        seq_len,
        cfg["layers"],
        cfg["heads"],
        cfg["kv_heads"],
        cfg["embed"],
        device,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = np.random.default_rng(0)
    for _ in range(cfg["epochs"]):
        model.train()
        idx = rng.permutation(len(x))
        for i in range(0, len(idx), cfg["batch_size"]):
            b = idx[i : i + cfg["batch_size"]]
            xb = torch.from_numpy(x[b]).to(device)
            logits = _forward(model, xb)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab), xb[:, 1:].reshape(-1)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    tr = _onset_acc(model, x, om, bm, vocab, device)
    va = _onset_acc(model, vx, vom, vbm, vocab, device)
    return vocab, seq_len, tr, va


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    cli = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(
        P=64,
        motif_len=8,
        n_shared=6,
        n_held=2,
        n_train=1024,
        n_val=256,
        layers=6,
        heads=8,
        kv_heads=4,
        embed=288,
        epochs=20,
        batch_size=32,
        seed=0,
    )
    if cli.quick:
        cfg.update(
            P=16,
            motif_len=4,
            n_shared=3,
            n_held=1,
            n_train=128,
            n_val=64,
            layers=2,
            heads=4,
            kv_heads=2,
            embed=64,
            epochs=3,
        )
    print(f"device={device}")
    res = {}
    for cond in ("flat", "real"):
        vocab, seq_len, tr, va = run(cond, cfg, device)
        res[cond] = va
        print(
            f"[{cond:>4}] vocab={vocab} seq_len={seq_len} "
            f"train_onset_acc={tr:.3f} HELDOUT_onset_acc={va:.3f}"
        )
    thr = 0.0 if cli.quick else 0.80
    verdict = (
        "ENCODING SUFFICIENT (real generalizes the rule)"
        if res["real"] >= thr and res["flat"] >= thr
        else (
            "ENCODING DEFICIENT (flat generalizes, real does not)"
            if res["flat"] >= thr > res["real"]
            else "INCONCLUSIVE (flat itself did not generalize)"
        )
    )
    print(f"VERDICT: {verdict}")
    return 0 if cli.quick or res["real"] >= thr else 1


if __name__ == "__main__":
    sys.exit(main())
