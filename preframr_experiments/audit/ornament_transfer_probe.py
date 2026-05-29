"""Ornament-transfer A/B probe (design/ornament_transfer.md): does a PARAMETRIC per-note ornament
descriptor get GENERATED and TRANSFER to held-out tunes, where the RAW per-frame encoding collapses?

One extraction (`extract_sid_melody --ornament`: per-note {skel, offs, desc}). Two encodings of the
SAME tunes, each a per-note stream aligned to the skeleton:
  RAW   : [skel] + raw note-relative offset tokens (variable per note)  -- the current encoding
  PARAM : [skel, descriptor]                                            -- one parametric token/note

Train each (same llama3_2 mini), free-run from a 1/3-note prompt over held-out tunes, then classify
every generated note's ornament into a descriptor TYPE (PLAIN/ARP/SLIDE/VIB/RESID) — for RAW by
re-fitting the generated offsets, for PARAM directly — and compare to the held-out actual type
distribution. Two metrics (NOT exact-token, P6):
  emission  = fraction of generated notes that carry ornament (vs corpus ~0.23)
  JS(type)  = Jensen-Shannon divergence (bits) between generated and held-out type histograms

Decision: PARAM emits at ~corpus rate with low JS while RAW under-emits / high JS -> parametric
ornament transfers; build it. Both collapse -> ornament is driver/tune-specific; score it as texture.

Usage (xpt image, GPU):
  python3 -m preframr_experiments.audit.ornament_transfer_probe --data /data/mini_ornament.json \
      --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from preframr_experiments.audit.extract_sid_melody import _fit_ornament
from preframr_experiments.audit.melody_channel_render import generate as mc_generate
from preframr_experiments.audit.melody_channel_render import train as mc_train

SKEL_BASE, RAW_BASE, DESC_BASE = 1000, 3000, 5000
TYPES = ["PLAIN", "ARP", "SLIDE", "VIB", "RESID"]


def split(seqs, seed):
    dumps = sorted(s["dump"] for s in seqs)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(dumps, max(1, len(dumps) // 5), replace=False))
    return (
        [s for s in seqs if s["dump"] not in held],
        [s for s in seqs if s["dump"] in held],
    )


def build_desc_vocab(train, topn=48):
    c = Counter(n["desc"] for s in train for n in s["notes"])
    keep = {d: i for i, (d, _) in enumerate(c.most_common(topn))}
    resid = len(keep)  # catch-all id for out-of-vocab descriptors
    inv = {i: d for d, i in keep.items()}
    inv[resid] = "RESID"
    return keep, resid, inv


def stream_raw(notes, cap):
    out = []
    for n in notes[:cap]:
        out.append(SKEL_BASE + n["skel"])
        out += [RAW_BASE + o for o in n["offs"]]
    return out


def stream_param(notes, cap, vocab, resid):
    out = []
    for n in notes[:cap]:
        out.append(SKEL_BASE + n["skel"])
        out.append(DESC_BASE + vocab.get(n["desc"], resid))
    return out


def js_bits(p, q):
    p = np.asarray(p, float) / max(np.sum(p), 1)
    q = np.asarray(q, float) / max(np.sum(q), 1)
    m = 0.5 * (p + q)

    def kl(a, b):
        ok = a > 0
        return float(np.sum(a[ok] * np.log2(a[ok] / b[ok])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def desc_type(s):
    t = s.split("|")[0]
    return t if t in TYPES else "RESID"


def parse_param(ids, inv_desc):
    """Generated PARAM ids (raw ints) -> per-note descriptor types."""
    types = []
    i = 0
    while i < len(ids):
        if RAW_BASE > ids[i] >= SKEL_BASE:  # skeleton -> note start
            if i + 1 < len(ids) and ids[i + 1] >= DESC_BASE:
                types.append(desc_type(inv_desc.get(ids[i + 1] - DESC_BASE, "RESID")))
                i += 2
                continue
        i += 1
    return types


def parse_raw(ids):
    """Generated RAW ids (raw ints) -> per-note descriptor types (re-fit the offsets)."""
    types = []
    cur = None
    for tid in ids:
        if RAW_BASE > tid >= SKEL_BASE:
            if cur is not None:
                types.append(desc_type(_fit_ornament(cur, max(2, len(cur) * 2))))
            cur = []
        elif DESC_BASE > tid >= RAW_BASE and cur is not None:
            cur.append(tid - RAW_BASE)
    if cur is not None:
        types.append(desc_type(_fit_ornament(cur, max(2, len(cur) * 2))))
    return types


def type_hist(types):
    c = Counter(types)
    return [c.get(t, 0) for t in TYPES]


def run_arm(train, test, build, parse, epochs, dev, seed, cap, maxlen):
    streams = [(s["dump"], build(s["notes"])) for s in train + test]
    alpha = {}
    for _, st in streams:
        for t in st:
            alpha.setdefault(t, len(alpha))
    inv = {i: t for t, i in alpha.items()}
    vocab = len(alpha)
    train_dumps = {x["dump"] for x in train}
    arm = [(d, st[:maxlen], None) for d, st in streams if d in train_dumps]
    model = mc_train(arm, epochs, dev, seed, vocab, maxlen, alpha)
    gen_types = []
    for s in test:
        st = build(s["notes"])[:maxlen]
        if len(st) < 6:
            continue
        prompt = [alpha[t] for t in st[: max(2, len(st) // 3)]]
        gen = mc_generate(model, prompt, len(st), vocab, dev, temp=1.0, seed=seed)
        gen_types += parse([inv[i] for i in gen])
    actual_types = [desc_type(n["desc"]) for s in test for n in s["notes"][:cap]]
    emit = 1 - (Counter(gen_types).get("PLAIN", 0) / max(len(gen_types), 1))
    js = js_bits(type_hist(gen_types), type_hist(actual_types))
    return emit, js, Counter(gen_types)


def _desc_to_offs(desc):
    """Synthesize a note-relative offset series from a parametric descriptor (for audition)."""
    p = desc.split("|")
    if p[0] == "ARP":
        oset = [int(x) for x in p[1].split(",")]
        return (oset * 6)[:8]
    if p[0] == "SLIDE":
        mag = 7 if p[2] == "big" else 3
        sign = 1 if p[1] == "+" else -1
        return list(range(sign, sign * (mag + 1), sign))
    if p[0] == "VIB":
        d = int(p[1])
        return [0, d, 0, -d, 0, d, 0, -d]
    return []


def _render_notes(notes_po, path, note_nf=12):
    """Render [(abs_pitch, [note-relative offsets])] as triangle notes with mid-note ornament."""
    import pandas as pd
    from preframr_audio.fidelity import render_df_to_wav
    from preframr_tokens.tokenizer_config import named_config

    from preframr_experiments.audit.melody_channel_render import (
        IRQ,
        PITCH_HI,
        PITCH_LO,
        _frame,
        _set,
        midi_fn,
    )

    rows = []
    for pitch, offs in notes_po:
        p = max(PITCH_LO, min(PITCH_HI, pitch))
        fn = midi_fn(p)
        for f in range(note_nf):
            rows.append(_frame())
            if f == 0:
                rows += [
                    _set(0, fn & 0xFF),
                    _set(1, (fn >> 8) & 0xFF),
                    _set(5, 0x00),
                    _set(6, 0xFA),
                    _set(4, 0x11),
                ]
            elif offs and 0 < f <= len(offs):
                fo = midi_fn(max(PITCH_LO, min(PITCH_HI, p + offs[f - 1])))
                rows += [_set(0, fo & 0xFF), _set(1, (fo >> 8) & 0xFF)]
    return render_df_to_wav(
        pd.DataFrame(rows), IRQ, named_config("baseline"), Path(path)
    )[0]


def render_demo(seqs, out_dir, epochs, dev, cap, maxlen, seed=0):
    """Audition: a held-out tune's ornament as GROUND TRUTH vs RAW-per-note vs PARAM-per-note
    model continuations (1/3 prompt). Hear that per-note ornament now generates."""
    train, test = split(seqs, seed)
    vocab_d, resid, inv_desc = build_desc_vocab(train)
    tune = max(test, key=lambda s: len(s["notes"]))
    notes = tune["notes"][:cap]
    base = 60

    def to_po(skel_offs):
        out = []
        p = base
        for sk, offs in skel_offs:
            p = max(40, min(76, p + sk))
            out.append((p, offs))
        return out

    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    _render_notes(
        to_po([(n["skel"], n["offs"]) for n in notes]), od / "ornament_gt.wav"
    )

    for name, build, parse_to_skeloffs in [
        ("raw", lambda n: stream_raw(n, cap), None),
        ("param", lambda n: stream_param(n, cap, vocab_d, resid), None),
    ]:
        streams = [(s["dump"], build(s["notes"])) for s in train + test]
        alpha = {}
        for _, st in streams:
            for t in st:
                alpha.setdefault(t, len(alpha))
        inv = {i: t for t, i in alpha.items()}
        td = {x["dump"] for x in train}
        model = mc_train(
            [(d, st[:maxlen], None) for d, st in streams if d in td],
            epochs,
            dev,
            seed,
            len(alpha),
            maxlen,
            alpha,
        )
        st = build(notes)[:maxlen]
        prompt = [alpha[t] for t in st[: max(2, len(st) // 3)]]
        gen = [
            inv[i]
            for i in mc_generate(
                model, prompt, len(st), len(alpha), dev, temp=1.0, seed=seed
            )
        ]
        # decode generated stream -> [(skel, offs)] per note
        skel_offs = []
        cur_skel = None
        cur_offs = []
        for tid in gen:
            if RAW_BASE > tid >= SKEL_BASE:
                if cur_skel is not None:
                    skel_offs.append((cur_skel, cur_offs))
                cur_skel, cur_offs = tid - SKEL_BASE, []
            elif DESC_BASE > tid >= RAW_BASE and cur_skel is not None:
                cur_offs.append(tid - RAW_BASE)
            elif tid >= DESC_BASE and cur_skel is not None:
                cur_offs = _desc_to_offs(inv_desc.get(tid - DESC_BASE, "RESID"))
        if cur_skel is not None:
            skel_offs.append((cur_skel, cur_offs))
        _render_notes(to_po(skel_offs), od / f"ornament_{name}_pred.wav")
    print(f"  ornament audition WAVs in {od} (gt / raw_pred / param_pred)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=80, help="max notes/tune")
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--render-dir", default=None)
    cli = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = json.load(open(cli.data))["seqs"]
    corpus_emit = 1 - sum(
        1 for s in seqs for n in s["notes"] if n["desc"] == "PLAIN"
    ) / sum(len(s["notes"]) for s in seqs)
    print(f"{cli.data.name}: {len(seqs)} seqs, corpus ornament rate={corpus_emit:.3f}")
    for seed in range(cli.seeds):
        train, test = split(seqs, seed)
        vocab, resid, inv_desc = build_desc_vocab(train)
        ra_emit, ra_js, _ = run_arm(
            train,
            test,
            lambda n: stream_raw(n, cli.cap),
            parse_raw,
            cli.epochs,
            dev,
            seed,
            cli.cap,
            cli.maxlen,
        )
        pa_emit, pa_js, pc = run_arm(
            train,
            test,
            lambda n: stream_param(n, cli.cap, vocab, resid),
            lambda ids: parse_param(ids, inv_desc),
            cli.epochs,
            dev,
            seed,
            cli.cap,
            cli.maxlen,
        )
        print(f"  [seed {seed}]")
        print(f"    RAW   emission={ra_emit:.3f}  JS(type)={ra_js:.3f} bits")
        print(
            f"    PARAM emission={pa_emit:.3f}  JS(type)={pa_js:.3f} bits  gen={dict(pc)}"
        )
    if cli.render_dir:
        render_demo(seqs, cli.render_dir, cli.epochs, dev, cli.cap, cli.maxlen, 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
