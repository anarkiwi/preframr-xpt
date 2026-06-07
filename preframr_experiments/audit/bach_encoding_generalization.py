"""Can THIS encoding generalize real music (Bach)? — the strong test.

The synthetic tests proved the encoding carries a *deterministic* rule. This asks the
harder question with *real music*: transcode public-domain Bach chorales (music21, BSD;
SATB reduced to SID's 3 voices = Soprano/Alto/Bass) into the CURRENT encoding
(clean voice + op45 FREQ_TRAJ onsets; MIDI pitch -> freq -> the freq-mapper cent bin
that the model actually trains on; pulse/triangle waveforms), train on a chorale split,
and measure next-onset pitch prediction on HELD-OUT chorales.

Bach chorales have genuine, learnable musical structure (unlike the multi-modal real-SID
cent magnitude). So:
- model held-out onset acc >> chance and approaching the n-gram structure baseline,
  with a small train/val gap  => the encoding carries real musical structure
  => ENCODING SUFFICIENT for music (the real-SID melody failure is that corpus's data,
  not the encoding).
- model fails to generalize Bach => the encoding's pitch representation is deficient.

A held-out chorale is then greedily continued and rendered to WAV (Bach-on-SID).

Input: /scratch/tmp/bach_chorales.json (from extract_bach.py via music21).
Run in the xpt/preframr image:
  python3 -m preframr_experiments.audit.bach_encoding_generalization --out /scratch/tmp/enc_audition
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from preframr.train.model.bodies import get_llama3_2

IRQ = 19656
PAL_CLOCK = 985248
VOICES = [
    (0x41, 0),
    (0x11, 7),
    (0x11, 14),
]  # (waveform|gate, base reg) S=pulse, A/B=triangle
AD, SR = 0x09, 0xF0


def _freqmap():
    from preframr_tokens.reg_mappers import FreqMapper

    return FreqMapper(cents=50, clock=PAL_CLOCK)


def _midi_to_bin(fm):
    """MIDI pitch -> SID Fn -> freq-mapper cent bin (the model-facing pitch token)."""
    out = {}
    for m in range(24, 108):
        hz = 440.0 * 2 ** ((m - 69) / 12.0)
        fn = int(round(hz * 16777216 / PAL_CLOCK))
        out[m] = int(fm.fi_map.get(min(fn, 65535), 0))
    return out


def transcode(chorales, m2bin):
    """Each chorale -> (atoms, onset_mask, boundary_mask). atoms=(op,reg,subreg,val)."""
    MAX_FRAMES = 160  # bound seq length (O(n^2) attn); ~10 bars of musical context
    tunes = []
    for ch in chorales:
        pitch = np.array(ch["pitch"])[:MAX_FRAMES]  # [nf,3] midi, 0=rest
        onset = np.array(ch["onset"])[:MAX_FRAMES]  # [nf,3] bool
        nf = len(pitch)
        atoms, om, bm = [], [], []
        seen = [False, False, False]

        def push(a, o=False, b=False):
            atoms.append(a)
            om.append(o)
            bm.append(b)

        for t in range(nf):
            push((0, -128, -1, 0))  # FRAME tick
            for vi, (wf, base) in enumerate(VOICES):
                push((0, -126, -1, vi))  # VOICE marker
                p = int(pitch[t, vi])
                if p > 0 and onset[t, vi]:
                    b = m2bin.get(p, 0)
                    is_b = not seen[
                        vi
                    ]  # first onset of this voice = unpredictable start
                    seen[vi] = True
                    push((45, base, 0, 2))  # FLAGS
                    push((45, base, 1, (b >> 8) & 0xFF))  # V0_HI (0 for this range)
                    push((45, base, 2, b & 0xFF), o=True, b=is_b)  # V0_LO = pitch bin
                    push((45, base, 3, 0))
                    push((45, base, 4, 1))
                    push((0, base + 5, -1, AD))
                    push((0, base + 6, -1, SR))
                    push((0, base + 4, -1, wf))  # gate on + waveform
                elif p == 0:
                    push((0, base + 4, -1, wf & ~1))  # rest: gate off
        tunes.append((atoms, np.array(om), np.array(bm)))
    return tunes


def build(tunes, n_val):
    alpha = {("PAD",): 0}
    for atoms, _, _ in tunes:
        for a in atoms:
            alpha.setdefault(a, len(alpha))
    L = max(len(a) for a, _, _ in tunes)

    def pack(split):
        x = np.zeros((len(split), L), np.int64)
        om = np.zeros((len(split), L), bool)
        bm = np.zeros((len(split), L), bool)
        for i, (atoms, o, b) in enumerate(split):
            ids = [alpha[a] for a in atoms]
            x[i, : len(ids)] = ids
            om[i, : len(o)] = o
            bm[i, : len(b)] = b
        return x, om, bm

    inv = {v: k for k, v in alpha.items()}
    tr, va = tunes[:-n_val], tunes[-n_val:]
    return (*pack(tr), *pack(va), len(alpha), inv, L)


def ngram_ceiling(tunes_pitch, k=2):
    """Cross-chorale n-gram ceiling on the per-voice pitch-bin onset sequence."""
    fit = tunes_pitch[: int(len(tunes_pitch) * 0.8)]
    test = tunes_pitch[int(len(tunes_pitch) * 0.8) :]
    ctx = defaultdict(Counter)
    for seqs in fit:
        for s in seqs:
            for j in range(k, len(s)):
                ctx[tuple(s[j - k : j])][s[j]] += 1
    model = {c: cnt.most_common(1)[0][0] for c, cnt in ctx.items()}
    h = t = 0
    for seqs in test:
        for s in seqs:
            for j in range(k, len(s)):
                t += 1
                h += model.get(tuple(s[j - k : j]), -1) == s[j]
    return h / max(t, 1)


def build_model(vocab, seq_len, c, device):
    a = argparse.Namespace(
        layers=c["layers"],
        heads=c["heads"],
        kv_heads=c["kv_heads"],
        embed=c["embed"],
        max_seq_len=seq_len,
        attn_dropout=0.1,
        norm_eps=1e-5,
        rope_base=500000,
        rope_scale=1.0,
        tie_word_embeddings=False,
    )
    return get_llama3_2(vocab, a).to(device)


def _fwd(m, x):
    o = m(x)
    return torch.cat(o, dim=1) if isinstance(o, list) else o


def onset_acc(model, x, om, bm, device, bs=8):
    model.eval()
    h = t = 0
    with torch.inference_mode():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i : i + bs]).to(device)
            pred = _fwd(model, xb).argmax(-1)[:, :-1]
            tgt = xb[:, 1:]
            mask = torch.from_numpy(om[i : i + bs, 1:] & ~bm[i : i + bs, 1:]).to(device)
            h += int(((pred == tgt) & mask).sum())
            t += int(mask.sum())
    return h / max(t, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--data", type=Path, default=Path("/scratch/tmp/bach_chorales.json")
    )
    ap.add_argument("--out", type=Path, default=Path("/scratch/tmp/enc_audition"))
    ap.add_argument("--quick", action="store_true")
    cli = ap.parse_args()
    cli.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chorales = json.load(open(cli.data))
    if cli.quick:
        chorales = chorales[:24]
    fm = _freqmap()
    m2bin = _midi_to_bin(fm)
    tunes = transcode(chorales, m2bin)
    n_val = max(2, len(tunes) // 5)
    x, om, bm, vx, vom, vbm, vocab, inv, L = build(tunes, n_val)
    # per-voice pitch-bin onset sequences for the n-gram baseline
    pv = []
    for atoms, o, _ in tunes:
        seqs = [[], [], []]
        cur = 0
        for a, onf in zip(atoms, o):
            if a[:3] == (0, -126, -1):
                cur = a[3]
            if onf:
                seqs[cur].append(a[3])
        pv.append(seqs)
    base2 = ngram_ceiling(pv, 2)
    c = dict(layers=6, heads=8, kv_heads=4, embed=288, epochs=40, batch_size=4)
    if cli.quick:
        c.update(layers=2, heads=4, kv_heads=2, embed=64, epochs=2)
    model = build_model(vocab, L, c, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = np.random.default_rng(0)
    for _ in range(c["epochs"]):
        model.train()
        for i in range(0, len(x), c["batch_size"]):
            b = rng.permutation(len(x))[i : i + c["batch_size"]]
            xb = torch.from_numpy(x[b]).to(device)
            lg = _fwd(model, xb)
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, vocab), xb[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    tr = onset_acc(model, x, om, bm, device)
    va = onset_acc(model, vx, vom, vbm, device)
    distinct_bins = len({m2bin[m] for m in range(36, 89)})
    print(
        f"Bach chorales: {len(tunes)} ({len(tunes)-n_val} train / {n_val} val), "
        f"vocab={vocab} seq_len={L} distinct_pitch_bins~{distinct_bins}"
    )
    print(
        f"  cross-chorale 2-gram pitch ceiling = {base2:.3f}  (chance ~ {1/max(distinct_bins,1):.3f})"
    )
    print(f"  MODEL train_onset_acc={tr:.3f}  HELDOUT_onset_acc={va:.3f}")
    verdict = (
        "ENCODING SUFFICIENT FOR MUSIC (generalizes Bach)"
        if va > 2.0 / max(distinct_bins, 1) and va > 0.5 * base2
        else "weak — see numbers"
    )
    print(f"VERDICT: {verdict}")

    # audition: continue a held-out chorale from its first third, render bins->freq
    try:
        from preframr_audio.fidelity import render_df_to_wav
        from preframr_tokens.tokenizer_config import named_config

        bin2fn = {b: fn for fn, b in fm.fi_map.items()}  # representative Fn per bin
        seq = vx[0]
        nz = int((seq != 0).sum())
        ids = seq[: nz // 3].tolist()
        with torch.inference_mode():
            while len(ids) < nz:
                ids.append(
                    int(_fwd(model, torch.tensor([ids], device=device))[0, -1].argmax())
                )

        def render(idseq, path):
            rows = []
            cur = 0
            for tid in idseq:
                a = inv[int(tid)]
                if a == ("PAD",):
                    continue
                op, reg, sr, val = a
                if (op, reg, sr) == (0, -126, -1):
                    cur = val
                if reg == -128:
                    rows.append(
                        dict(
                            op=0,
                            reg=-128,
                            subreg=-1,
                            val=0,
                            diff=IRQ,
                            irq=IRQ,
                            description=0,
                        )
                    )
                elif op == 45 and sr == 2:  # V0_LO bin -> Fn -> freq writes
                    base = VOICES[cur][1]
                    fn = int(bin2fn.get(val, 2000))
                    rows.append(
                        dict(
                            op=0,
                            reg=base + 0,
                            subreg=-1,
                            val=fn & 0xFF,
                            diff=0,
                            irq=IRQ,
                            description=0,
                        )
                    )
                    rows.append(
                        dict(
                            op=0,
                            reg=base + 1,
                            subreg=-1,
                            val=(fn >> 8) & 0xFF,
                            diff=0,
                            irq=IRQ,
                            description=0,
                        )
                    )
                elif op == 0 and reg >= 0 and sr == -1 and reg not in (-128, -126):
                    rows.append(
                        dict(
                            op=0,
                            reg=reg,
                            subreg=-1,
                            val=val,
                            diff=0,
                            irq=IRQ,
                            description=0,
                        )
                    )
            return render_df_to_wav(
                pd.DataFrame(rows), IRQ, named_config("baseline"), Path(path)
            )[0]

        ng = render(ids, cli.out / "bach_prediction.wav")
        ngt = render(seq[:nz].tolist(), cli.out / "bach_ground_truth.wav")
        print(
            f"audition: {cli.out}/bach_prediction.wav ({ng} samp), "
            f"{cli.out}/bach_ground_truth.wav ({ngt} samp)"
        )
    except Exception as e:
        print("audition skipped:", repr(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
