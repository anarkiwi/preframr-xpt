"""Generalization test for the unified pitch encoding (design/unified_pitch_encoding.md).

One extraction (`unified_pitch --out`: per-note {skel semitone-interval, ornament descriptor}). Build
the per-note stream `[SKEL, DESC]`, train the llama3_2 mini body held-out-by-dump, and report — the
same way prior encoding-generalization tests did:
  * SKELETON melody: held-out next-skeleton-interval accuracy + cross-tune 2-gram ceiling (cf.
    melody_ladder: Bach 0.39, mini-hetero 0.225) — does the melody line generalize through this encoding?
  * ORNAMENT: emission rate (vs corpus) + JS(type) of generated vs held-out (does ornament generate?).
Then render a held-out tune: ground truth vs the model's free-run continuation, decoded through the
unified decoder (skeleton->LUT[note] + ornament synthesis) to WAV.

Usage (xpt image, GPU): python3 -m preframr_experiments.audit.unified_pitch_probe \
    --data /data/mini_unified.json --seeds 3 [--render-dir /data/audition]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from preframr_experiments.audit.melody_channel_render import generate as mc_generate
from preframr_experiments.audit.melody_channel_render import train as mc_train

SKEL_BASE, DESC_BASE, VIB_BASE = 1000, 5000, 8000
TYPES = ["PLAIN", "OCTAVE", "ARP", "SLIDE", "VIB", "RESID"]
VIB_LEVELS = [0, 1, 2]  # sub-semitone vibrato depth buckets (per-note VIB token)


def split(seqs, seed):
    dumps = sorted(s["dump"] for s in seqs)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(dumps, max(1, len(dumps) // 5), replace=False))
    return [s for s in seqs if s["dump"] not in held], [
        s for s in seqs if s["dump"] in held
    ]


def build_desc_vocab(train, topn=64):
    c = Counter(n["desc"] for s in train for n in s["notes"])
    keep = {d: i for i, (d, _) in enumerate(c.most_common(topn))}
    resid = len(keep)
    inv = {i: d for d, i in keep.items()}
    inv[resid] = "RESID"
    return keep, resid, inv


def stream(notes, vocab, resid):
    out = []
    for n in notes:
        out.append(SKEL_BASE + n["skel"])
        out.append(DESC_BASE + vocab.get(n["desc"], resid))
        out.append(VIB_BASE + int(n.get("vib", 0)))  # sub-semitone vibrato-depth token
    return out


def desc_type(s):
    t = s.split("|")[0]
    return t if t in TYPES else "RESID"


def js_bits(p, q):
    p = np.asarray(p, float) / max(np.sum(p), 1)
    q = np.asarray(q, float) / max(np.sum(q), 1)
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def skel_2gram(train, test):
    """Cross-tune 2-gram ceiling on the skeleton-interval sequence (skels only)."""
    ctx = defaultdict(Counter)
    for s in train:
        sk = [n["skel"] for n in s["notes"]]
        for j in range(2, len(sk)):
            ctx[(sk[j - 2], sk[j - 1])][sk[j]] += 1
    model = {c: cnt.most_common(1)[0][0] for c, cnt in ctx.items()}
    h = t = 0
    for s in test:
        sk = [n["skel"] for n in s["notes"]]
        for j in range(2, len(sk)):
            t += 1
            h += model.get((sk[j - 2], sk[j - 1]), 10**9) == sk[j]
    return h / max(t, 1)


def run(seqs, epochs, dev, seed, maxlen=512):
    train, test = split(seqs, seed)
    vocab, resid, inv = build_desc_vocab(train)
    alpha = {}
    streams = [(s["dump"], stream(s["notes"], vocab, resid)) for s in train + test]
    for _, st in streams:
        for t in st:
            alpha.setdefault(t, len(alpha))
    inv_alpha = {i: t for t, i in alpha.items()}
    td = {s["dump"] for s in train}
    arm = [(d, st[:maxlen], None) for d, st in streams if d in td]
    model = mc_train(arm, epochs, dev, seed, len(alpha), maxlen, alpha)

    # SKELETON held-out next-interval accuracy (score positions whose target is a SKEL token).
    sh = st_ = 0
    gen_types, actual_types, gen_vib, actual_vib = [], [], [], []
    for s in test:
        full = stream(s["notes"], vocab, resid)[:maxlen]
        if len(full) < 6:
            continue
        ids = [alpha[t] for t in full]
        xb = torch.tensor([ids], device=dev)
        with torch.inference_mode():
            o = model(xb)
            pr = (
                (torch.cat(o, 1) if isinstance(o, list) else o)
                .argmax(-1)[0, :-1]
                .cpu()
                .numpy()
            )
        tg = ids[1:]
        for j, p in enumerate(pr):
            raw = inv_alpha[tg[j]]
            if SKEL_BASE <= raw < DESC_BASE:  # target is a skeleton token
                st_ += 1
                sh += int(inv_alpha[int(p)] == raw)
        # ornament + vibrato emission/JS via free-run from a 1/3 prompt
        prompt = ids[: max(2, len(ids) // 3)]
        g = mc_generate(model, prompt, len(ids), len(alpha), dev, temp=1.0, seed=seed)
        for tid in (inv_alpha[i] for i in g):
            if DESC_BASE <= tid < VIB_BASE:
                gen_types.append(desc_type(inv.get(tid - DESC_BASE, "RESID")))
            elif tid >= VIB_BASE:
                gen_vib.append(tid - VIB_BASE)
        actual_types += [desc_type(n["desc"]) for n in s["notes"][: maxlen // 3]]
        actual_vib += [int(n.get("vib", 0)) for n in s["notes"][: maxlen // 3]]

    base = skel_2gram(train, test)
    skel_acc = sh / max(st_, 1)
    emit = 1 - Counter(gen_types).get("PLAIN", 0) / max(len(gen_types), 1)
    corpus_emit = 1 - Counter(actual_types).get("PLAIN", 0) / max(len(actual_types), 1)
    th = lambda ts: [Counter(ts).get(t, 0) for t in TYPES]
    vh = lambda vs: [Counter(vs).get(x, 0) for x in VIB_LEVELS]
    vib_emit = 1 - Counter(gen_vib).get(0, 0) / max(len(gen_vib), 1)
    vib_corpus = 1 - Counter(actual_vib).get(0, 0) / max(len(actual_vib), 1)
    vib_js = js_bits(vh(gen_vib), vh(actual_vib))
    return (
        base,
        skel_acc,
        emit,
        corpus_emit,
        js_bits(th(gen_types), th(actual_types)),
        vib_emit,
        vib_corpus,
        vib_js,
    )


def render_demo(seqs, out_dir, epochs, dev, seed=0, maxlen=512):
    import pandas as pd

    from preframr_audio.fidelity import render_df_to_wav
    from preframr_tokens.tokenizer_config import named_config

    from preframr_experiments.audit import unified_pitch as U
    from preframr_experiments.audit.melody_channel_render import IRQ, _frame, _set

    train, test = split(seqs, seed)
    vocab, resid, inv = build_desc_vocab(train)
    alpha = {}
    streams = [(s["dump"], stream(s["notes"], vocab, resid)) for s in train + test]
    for _, st in streams:
        for t in st:
            alpha.setdefault(t, len(alpha))
    inv_alpha = {i: t for t, i in alpha.items()}
    td = {s["dump"] for s in train}
    model = mc_train(
        [(d, st[:maxlen], None) for d, st in streams if d in td],
        epochs,
        dev,
        seed,
        len(alpha),
        maxlen,
        alpha,
    )
    tune = max(test, key=lambda s: len(s["notes"]))
    recs_gt = [{"skel": n["skel"], "desc": n["desc"]} for n in tune["notes"][:80]]
    ids = [alpha[t] for t in stream(tune["notes"], vocab, resid)[: 3 * 80]]
    g = mc_generate(
        model,
        ids[: max(2, len(ids) // 3)],
        len(ids),
        len(alpha),
        dev,
        temp=1.0,
        seed=seed,
    )
    recs_pred, cur = [], None
    for tid in (inv_alpha[i] for i in g):
        if SKEL_BASE <= tid < DESC_BASE:
            if cur is not None:
                recs_pred.append(cur)
            cur = {"skel": tid - SKEL_BASE, "desc": "PLAIN"}
        elif DESC_BASE <= tid < VIB_BASE and cur is not None:
            cur["desc"] = inv.get(tid - DESC_BASE, "RESID")
        # VIB tokens (>= VIB_BASE) carry sub-semitone depth; not used in this LUT render
    if cur:
        recs_pred.append(cur)

    def render(recs, path, nf=8):
        rows = []
        base = 60
        for r in recs:
            base = max(U.MIDI_LO, min(U.MIDI_HI, base + r["skel"]))
            for fi, m in enumerate(U._desc_frames(r["desc"], base, nf)):
                mm = max(U.MIDI_LO, min(U.MIDI_HI, m))
                fn = U.LUT[mm]
                rows.append(_frame())
                if fi == 0:  # note onset: gate-on triangle
                    rows += [
                        _set(0, fn & 0xFF),
                        _set(1, (fn >> 8) & 0xFF),
                        _set(5, 0x00),
                        _set(6, 0xFA),
                        _set(4, 0x11),
                    ]
                else:  # ornament freq write within the held note
                    rows += [_set(0, fn & 0xFF), _set(1, (fn >> 8) & 0xFF)]
        render_df_to_wav(pd.DataFrame(rows), IRQ, named_config("baseline"), Path(path))

    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    render(recs_gt, od / "unified_gt.wav")
    render(recs_pred, od / "unified_pred.wav")
    print(
        f"  render: gt={len(recs_gt)} notes, pred={len(recs_pred)} notes -> {od}/unified_{{gt,pred}}.wav"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--render-dir", default=None)
    cli = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = json.load(open(cli.data))["seqs"]
    print(f"{cli.data.name}: {len(seqs)} seqs")
    accs = []
    for seed in range(cli.seeds):
        base, skel, emit, cemit, js, vemit, vcorpus, vjs = run(
            seqs, cli.epochs, dev, seed
        )
        accs.append(skel)
        print(
            f"  [seed {seed}] SKELETON held-out next-interval acc={skel:.3f} (2-gram ceiling={base:.3f}) "
            f"| ORNAMENT emission={emit:.3f} (corpus {cemit:.3f}) JS(type)={js:.3f} "
            f"| VIBRATO emission={vemit:.3f} (corpus {vcorpus:.3f}) JS(depth)={vjs:.3f}"
        )
    if len(accs) > 1:
        a = np.array(accs)
        print(
            f"  MEAN skeleton acc={a.mean():.3f} (range {a.min():.3f}..{a.max():.3f})"
        )
    if cli.render_dir:
        render_demo(seqs, cli.render_dir, cli.epochs, dev, 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
