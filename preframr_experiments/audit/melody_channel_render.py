"""Audition render for the channel-factorization probe (design/melody_channel_factorization.md).
Trains the interleaved (skeleton+ornament) and skeleton-only models, free-runs a held-out tune's
continuation from a 1/3 prompt, and renders triangle-voice WAVs so the melody can be HEARD:

Single-voice (the probe's scope — within-voice note/ornament), interval-reconstructed from a
fixed base (contour, not key, is audible):
  channel_ground_truth.wav       held-out lead voice's true skeleton melody (note line)
  channel_interleaved_pred.wav   skeleton melody the interleaved model continues to (greedy)
  channel_skeleton_pred.wav      skeleton melody the skeleton-only model continues to
  channel_ground_truth_orn.wav   the SAME notes WITH ornament (vibrato/slide/arp) — the encoding
                                 PRESERVES ornamentation while the skeleton channel is the melody.
  channel_interleaved_pred_orn.wav  the interleaved model's SAMPLED continuation rendered
                                 ornamented (greedy emits ~no ornament; the model under-generates
                                 ornament even sampled — an honest finding, not a render choice).

3-voice GROUND-TRUTH polyphony reconstructed from the skeleton+ornament decomposition on the real
frame timeline with absolute pitches (needs --dumps); demonstrates the representation carries
polyphony + ornament — these are NOT model predictions:
  channel_polyphony_skeleton.wav all 3 voices, note-ons only (the melodic/harmonic skeleton)
  channel_polyphony_orn.wav      all 3 voices, notes + ornament (the full pitched signal)

Runs in the preframr image (torch + preframr_audio).

Usage (in the xpt/preframr image, GPU; mount the HVSC tree so staged-dump symlinks resolve):
  python3 -m preframr_experiments.audit.melody_channel_render \
      --data /data/mini_channels.json --dumps '/data/mini_dumps/*.dump.parquet' \
      --seed 1 --out-dir /data/audition
"""

from __future__ import annotations

import argparse
import glob
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from preframr.train.model.bodies import get_llama3_2
from preframr_audio.fidelity import render_df_to_wav
from preframr_tokens.tokenizer_config import named_config

from preframr_experiments.audit.extract_sid_melody import (
    MUX_ORN_BASE,
    MUX_SKEL_BASE,
    MUX_VOICE_STRIDE,
    ORN_OFFSET,
    _voice_l1_l2,
)
from preframr_experiments.audit.melody_channel_probe import load, split

IRQ = 19656
CLOCK = 985248
BASE = 60  # arbitrary tonic; intervals are key-invariant
PITCH_LO, PITCH_HI = 40, 76


def midi_fn(m):
    return int(round(440.0 * 2 ** ((m - 69) / 12.0) * 16777216 / CLOCK))


def _is_skel_val(v):
    return (
        v < ORN_OFFSET // 2
    )  # skeleton intervals are small signed; ornament is +ORN_OFFSET


def skel_pitches(values):
    """Skeleton interval token-values -> absolute pitch line from BASE (ornament ignored)."""
    p = [BASE]
    for v in values:
        if _is_skel_val(v):
            p.append(max(PITCH_LO, min(PITCH_HI, p[-1] + v)))
    return p


def _frame():
    return dict(op=0, reg=-128, subreg=-1, val=0, diff=IRQ, irq=IRQ, description=0)


def _freq(fn):
    return [
        dict(op=0, reg=0, subreg=-1, val=fn & 0xFF, diff=0, irq=IRQ, description=0),
        dict(
            op=0, reg=1, subreg=-1, val=(fn >> 8) & 0xFF, diff=0, irq=IRQ, description=0
        ),
    ]


def render_melody(pitches, path, nf=10):
    """One discrete triangle note per pitch (gate retrigger each note)."""
    rows = []
    for m in pitches:
        fn = midi_fn(m)
        for f in range(nf):
            rows.append(_frame())
            if f == 0:
                rows += _freq(fn) + [
                    dict(
                        op=0, reg=5, subreg=-1, val=0x00, diff=0, irq=IRQ, description=0
                    ),
                    dict(
                        op=0, reg=6, subreg=-1, val=0xFA, diff=0, irq=IRQ, description=0
                    ),
                    dict(
                        op=0, reg=4, subreg=-1, val=0x11, diff=0, irq=IRQ, description=0
                    ),
                ]
            elif f == nf - 2:
                rows.append(
                    dict(
                        op=0, reg=4, subreg=-1, val=0x10, diff=0, irq=IRQ, description=0
                    )
                )
    return render_df_to_wav(
        pd.DataFrame(rows), IRQ, named_config("baseline"), Path(path)
    )[0]


def render_ornamented(values, path, note_nf=10, orn_nf=2):
    """Replay the interleaved stream: a gated triangle note at each skeleton token, held while
    ornament tokens write mid-note freq (vibrato/slide/arp) — the note keeps ringing."""
    rows = []
    cur = BASE
    started = False
    for v in values:
        if _is_skel_val(v):
            cur = max(PITCH_LO, min(PITCH_HI, cur + v))
            fn = midi_fn(cur)
            for f in range(note_nf):
                rows.append(_frame())
                if f == 0:
                    rows += _freq(fn) + [
                        dict(
                            op=0,
                            reg=5,
                            subreg=-1,
                            val=0x00,
                            diff=0,
                            irq=IRQ,
                            description=0,
                        ),
                        dict(
                            op=0,
                            reg=6,
                            subreg=-1,
                            val=0xFA,
                            diff=0,
                            irq=IRQ,
                            description=0,
                        ),
                        dict(
                            op=0,
                            reg=4,
                            subreg=-1,
                            val=0x11,
                            diff=0,
                            irq=IRQ,
                            description=0,
                        ),
                    ]
            started = True
        elif started:
            fn = midi_fn(max(PITCH_LO, min(PITCH_HI, cur + (v - ORN_OFFSET))))
            for f in range(orn_nf):
                rows.append(_frame())
                if f == 0:
                    rows += _freq(fn)  # freq write only, no gate retrigger
    return render_df_to_wav(
        pd.DataFrame(rows), IRQ, named_config("baseline"), Path(path)
    )[0]


def _set(reg, val):
    return dict(op=0, reg=reg, subreg=-1, val=val, diff=0, irq=IRQ, description=0)


def render_polyphony(dump_path, path, skeleton_only=False, max_frames=1400):
    """Reconstruct ALL 3 SID voices of a real dump from the skeleton+ornament decomposition on
    the real frame timeline — true polyphony, absolute pitches (inter-voice harmony preserved).
    Skeleton = gate-on note-ons (retriggered triangle notes); ornament = the freq writes between
    note-ons (vibrato/slide/arp), rendered as mid-note freq writes unless skeleton_only.
    """
    d = pd.read_parquet(dump_path).sort_values("clock")
    per_frame: dict[int, list] = {}

    def at(fr, rows):
        per_frame.setdefault(fr, []).extend(rows)

    for v, (l1, l2) in enumerate(_voice_l1_l2(d)):
        lo, hi, ctrl, ad, sr = v * 7, v * 7 + 1, v * 7 + 4, v * 7 + 5, v * 7 + 6
        skel_frames = {fr for fr, _ in l2}
        note_frames = sorted(fr for fr, _ in l2 if fr < max_frames)
        for fr, m in l2:
            if fr >= max_frames:
                continue
            fn = midi_fn(m)
            at(
                fr,
                [
                    _set(lo, fn & 0xFF),
                    _set(hi, (fn >> 8) & 0xFF),
                    _set(ad, 0x00),
                    _set(sr, 0xFA),
                    _set(ctrl, 0x11),
                ],
            )
        for fr in note_frames[1:]:  # gate-off the frame before each retrigger
            if fr - 1 not in skel_frames:
                at(fr - 1, [_set(ctrl, 0x10)])
        if not skeleton_only:
            for fr, m in l1:
                if fr >= max_frames or fr in skel_frames:
                    continue
                fn = midi_fn(m)
                at(fr, [_set(lo, fn & 0xFF), _set(hi, (fn >> 8) & 0xFF)])

    last = max(per_frame) if per_frame else 0
    rows = []
    for fr in range(min(max_frames, last + 1)):
        rows.append(_frame())
        rows.extend(per_frame.get(fr, []))
    return render_df_to_wav(
        pd.DataFrame(rows), IRQ, named_config("baseline"), Path(path)
    )[0]


def render_multiplex_tokens(ids, path, voice_bases=(60, 48, 72), max_tokens=600):
    """Render a generated MULTIPLEX token stream (id encodes voice+channel+interval) as 3-voice
    polyphony: each token = one frame on a shared timeline applied to its voice (skeleton -> note
    gate-on; ornament -> mid-note freq write). Per-voice base pitches separate the lines audibly
    (absolute pitch is not in the interval stream). A model PREDICTION, not ground truth.
    """
    rows = []
    cur = list(voice_bases)
    for tid in ids[:max_tokens]:
        v, loc = tid // MUX_VOICE_STRIDE, tid % MUX_VOICE_STRIDE
        if v not in (0, 1, 2):
            continue
        lo, hi, ctrl, ad, sr = v * 7, v * 7 + 1, v * 7 + 4, v * 7 + 5, v * 7 + 6
        rows.append(_frame())
        if loc < (MUX_SKEL_BASE + MUX_ORN_BASE) // 2:  # skeleton
            cur[v] = max(PITCH_LO, min(PITCH_HI, cur[v] + (loc - MUX_SKEL_BASE)))
            fn = midi_fn(cur[v])
            rows += [
                _set(lo, fn & 0xFF),
                _set(hi, (fn >> 8) & 0xFF),
                _set(ad, 0x00),
                _set(sr, 0xFA),
                _set(ctrl, 0x11),
            ]
        else:  # ornament
            fn = midi_fn(max(PITCH_LO, min(PITCH_HI, cur[v] + (loc - MUX_ORN_BASE))))
            rows += [_set(lo, fn & 0xFF), _set(hi, (fn >> 8) & 0xFF)]
    return render_df_to_wav(
        pd.DataFrame(rows), IRQ, named_config("baseline"), Path(path)
    )[0]


def train(arm, epochs, dev, seed, vocab, maxlen, alpha):
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
    torch.manual_seed(seed)
    model = get_llama3_2(vocab, a).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    x = np.zeros((len(arm), maxlen), np.int64)
    for i, (_, t, _) in enumerate(arm):
        ids = [alpha[tok] for tok in t]
        x[i, : len(ids)] = ids
    rng = np.random.default_rng(1 + seed)
    for _ in range(epochs):
        model.train()
        for i in range(0, len(x), 16):
            b = rng.permutation(len(x))[i : i + 16]
            xb = torch.from_numpy(x[b]).to(dev)
            o = model(xb)
            lg = torch.cat(o, 1) if isinstance(o, list) else o
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, vocab), xb[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def generate(model, prompt_ids, target_len, vocab, dev, temp=0.0, seed=0):
    """Free-run continuation. temp=0 -> greedy (argmax, collapses away from the diffuse
    ornament distribution); temp>0 -> temperature sampling (lets ornament tokens surface).
    """
    ids = list(prompt_ids)
    gen = torch.Generator(device=dev).manual_seed(seed)
    model.eval()
    with torch.inference_mode():
        while len(ids) < target_len:
            o = model(torch.tensor([ids], device=dev))
            lg = torch.cat(o, 1) if isinstance(o, list) else o
            logits = lg[0, -1]
            if temp > 0:
                probs = F.softmax(logits / temp, dim=-1)
                ids.append(int(torch.multinomial(probs, 1, generator=gen)))
            else:
                ids.append(int(logits.argmax()))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument(
        "--temp", type=float, default=1.0, help="sampling temp for the ornamented pred"
    )
    ap.add_argument(
        "--dumps",
        default=None,
        help="glob matching the --channels extraction, for the "
        "3-voice polyphony render of the held-out tune (same sorted order as the json index)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    cli = ap.parse_args()
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    seqs = load(cli.data, cli.cap)
    alpha = {}
    for _, t, _ in seqs:
        for tok in t:
            alpha.setdefault(tok, len(alpha))
    inv = {i: v for v, i in alpha.items()}
    vocab = len(alpha)
    maxlen = max(len(t) for _, t, _ in seqs)
    skel_only = [
        (d, [t for t, s in zip(t_, sk) if s], [True] * sum(sk)) for d, t_, sk in seqs
    ]

    train_i, test_i = split(seqs, cli.seed)
    # Held-out tune for audition: take a 96-token window and prefer one that carries BOTH a
    # real melody (>=24 skeleton notes) and real ornament (so the interleaved-vs-skeleton and
    # the ornament-preserved demos are non-degenerate); score = min(skel, orn) in the window.
    win = 96
    cand = [(d, t[:win], sk[:win]) for d, t, sk in test_i if sum(sk[:win]) >= 24]
    tune = max(cand, key=lambda s: min(sum(s[2]), len(s[2]) - sum(s[2])))
    dump = tune[0]
    tune_s = (dump, [t for t, s in zip(tune[1], tune[2]) if s], None)
    print(
        f"held-out dump={dump} tokens={len(tune[1])} skel={sum(tune[2])} "
        f"orn={len(tune[2]) - sum(tune[2])}"
    )

    m_i = train(train_i, cli.epochs, dev, cli.seed, vocab, maxlen, alpha)
    m_s = train(
        [s for s in skel_only if s[0] != dump],
        cli.epochs,
        dev,
        cli.seed,
        vocab,
        maxlen,
        alpha,
    )

    # ground truth (full token values, and the skeleton line)
    gt_vals = tune[1]
    gt_pitches = skel_pitches(gt_vals)
    n_skel = len(gt_pitches) - 1

    # interleaved: prompt 1/3 of the FULL stream, free-run.
    p_i = [alpha[t] for t in tune[1][: max(2, len(tune[1]) // 3)]]
    # greedy run for the clean skeleton-line melody comparison (vs the skeleton model).
    gen_i = generate(m_i, p_i, len(tune[1]), vocab, dev)
    pred_i_pitches = skel_pitches([inv[i] for i in gen_i])
    # sampled run so the model's predicted ORNAMENT surfaces (greedy collapses to skeleton);
    # rendered ornamented so the model's notes + its vibrato/slide/arp are audible.
    gen_i_s = generate(m_i, p_i, len(tune[1]), vocab, dev, temp=cli.temp, seed=cli.seed)
    gen_i_s_vals = [inv[i] for i in gen_i_s]
    gi_orn = sum(1 for v in gen_i_s_vals if not _is_skel_val(v))

    # skeleton-only: prompt 1/3 of the skeleton line, free-run.
    p_s = [alpha[t] for t in tune_s[1][: max(2, n_skel // 3)]]
    gen_s = generate(m_s, p_s, n_skel, vocab, dev)
    pred_s_pitches = skel_pitches([inv[i] for i in gen_s])

    od = cli.out_dir
    render_melody(gt_pitches, od / "channel_ground_truth.wav")
    render_melody(pred_i_pitches, od / "channel_interleaved_pred.wav")
    render_melody(pred_s_pitches, od / "channel_skeleton_pred.wav")
    render_ornamented(gt_vals, od / "channel_ground_truth_orn.wav")
    render_ornamented(gen_i_s_vals, od / "channel_interleaved_pred_orn.wav")
    print(
        f"  interleaved sampled gen (temp={cli.temp}): {gi_orn} ornament writes emitted"
    )

    # 3-voice polyphony render of the held-out tune (real timeline, absolute pitches): hear
    # actual polyphony + ornament, both reconstructed from the skeleton+ornament decomposition.
    if cli.dumps:
        dump_path = sorted(glob.glob(cli.dumps))[dump]
        render_polyphony(
            dump_path, od / "channel_polyphony_skeleton.wav", skeleton_only=True
        )
        render_polyphony(
            dump_path, od / "channel_polyphony_orn.wav", skeleton_only=False
        )
        print(f"  polyphony (3 voices) from {Path(dump_path).name}")

    def dur(p):
        w = wave.open(str(p))
        return w.getnframes() / w.getframerate()

    print(
        f"  GT melody notes={len(gt_pitches)} dur={dur(od/'channel_ground_truth.wav'):.1f}s"
    )
    print(
        f"  interleaved pred notes={len(pred_i_pitches)} dur={dur(od/'channel_interleaved_pred.wav'):.1f}s"
    )
    print(
        f"  skeleton    pred notes={len(pred_s_pitches)} dur={dur(od/'channel_skeleton_pred.wav'):.1f}s"
    )
    print(
        f"  GT ornamented dur={dur(od/'channel_ground_truth_orn.wav'):.1f}s (notes+vibrato/slide/arp)"
    )
    print(
        f"  interleaved pred ORNAMENTED dur={dur(od/'channel_interleaved_pred_orn.wav'):.1f}s "
        f"({gi_orn} ornament writes)"
    )
    if cli.dumps:
        print(
            f"  POLYPHONY 3-voice: skeleton dur={dur(od/'channel_polyphony_skeleton.wav'):.1f}s "
            f"| ornamented dur={dur(od/'channel_polyphony_orn.wav'):.1f}s"
        )
    print(f"  WAVs in {od}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
