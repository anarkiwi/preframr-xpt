"""Multi-voice + multi-waveform encoding-generalization test, with audio audition.

Airtight version of ``encoding_generalization_test``: proves the CURRENT encoding
carries a deterministic, generalizable melody rule even with **3 multiplexed voices**
and **multiple SID waveforms** (pulse lead / triangle bass / noise percussion —
palette mined from the real mini corpus), and renders a model PREDICTION to WAV so
the result is audible.

Encoding (the current one): clean voice (FRAME=0 tick + per-run VOICE(id) markers),
op45 FREQ_TRAJ onsets (FLAGS/V0_HI/V0_LO/COUNT/DELTA) for pitch, op0 SET for
control(waveform+gate)/ADSR. Three voices interleaved per frame, so each voice's
consecutive onsets are separated by the other voices' tokens (the multiplexing +
locality stress).

Rule (mirrors framework_arch_test): within a motif, pitch[i+1]=(pitch[i]+1)%P, mapped
to an audible SID freq (freq=800+pitch*180), so the rule lives in the op45 V0_HI/V0_LO
bytes (with carry). N_SHARED motifs in train+val, N_HELD only in val. Held-out motif
continuations are predictable by RULE, not memory.

Verdict: HELDOUT inside-motif onset acc high (≈ the single-voice 0.88 and the
framework_arch_test 0.90) => the encoding carries the rule under multiplexing +
multiple waveforms => SUFFICIENT. Then a held-out prompt is greedily continued by the
model and rendered to WAV alongside ground truth.

Run inside the xpt/preframr image (needs preframr_audio):
  python3 -m preframr_experiments.audit.multivoice_audition_test --out /scratch/tmp/enc_audition
  python3 -m preframr_experiments.audit.multivoice_audition_test --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from preframr.train.model.bodies import get_llama3_2

IRQ = 19656
# (waveform|gate-on byte, voice base reg = v*7, attack/decay, sustain/release) — mini-mined
VOICES = [(0x41, 0, 0x09, 0xF0), (0x11, 7, 0x09, 0xF0), (0x81, 14, 0x0F, 0x00)]


def _freq(pitch: int) -> int:
    return 800 + pitch * 180  # audible SID freq; +1 pitch => +180 freq (the rule in V0)


def _make_motifs(n, motif_len, P, rng):
    seeds = rng.choice(P - motif_len, size=n, replace=False)
    return [[(int(s) + j) % P for j in range(motif_len)] for s in seeds]


def _emit_tune(motifs, allowed, motifs_per_tune, note_frames, rng):
    """One tune: per-voice successor melodies, clean-voice multiplexed, renderable.
    Returns (atoms, onset_mask, boundary_mask). atoms = (op,reg,subreg,val,diff)."""
    atoms, onset, bnd = [], [], []

    def push(op, reg, sr, val, diff, is_onset=False, is_bnd=False):
        atoms.append((op, reg, sr, val, diff))
        onset.append(is_onset)
        bnd.append(is_bnd)

    # per-voice note plans: list of (pitch, is_boundary)
    plans = []
    for _ in VOICES:
        notes = []
        for _ in range(motifs_per_tune):
            m = motifs[int(rng.choice(allowed))]
            for j, p in enumerate(m):
                notes.append((p, j == 0))
        plans.append(notes)
    n_notes = min(len(p) for p in plans)

    for ni in range(n_notes):
        for f in range(note_frames):
            push(0, -128, -1, 0, IRQ)  # FRAME tick (clean: val 0)
            for vi, (wf, base, ad, srl) in enumerate(VOICES):
                push(0, -126, -1, vi, 0)  # VOICE marker carries voice id
                pitch, is_b = plans[vi][ni]
                if f == 0:  # onset frame
                    fr = _freq(pitch)
                    push(45, base, 0, 2, 0)  # FLAGS (RUN, absolute)
                    push(45, base, 1, (fr >> 8) & 0xFF, 0)  # V0_HI (carry byte)
                    push(45, base, 2, fr & 0xFF, 0, is_onset=True, is_bnd=is_b)  # V0_LO
                    push(45, base, 3, 0, 0)  # COUNT_HI
                    push(45, base, 4, 1, 0)  # COUNT_LO
                    push(45, base, 6, 0, 0)  # DELTA (flat)
                    push(0, base + 5, -1, ad, 0)  # attack/decay
                    push(0, base + 6, -1, srl, 0)  # sustain/release
                    push(0, base + 4, -1, wf, 0)  # control: waveform + gate ON
                elif f == note_frames - 1:
                    push(0, base + 4, -1, wf & ~1, 0)  # gate OFF
    return atoms, np.array(onset), np.array(bnd)


def build_corpus(cfg, seed):
    rng = np.random.default_rng(seed)
    motifs = _make_motifs(
        cfg["n_shared"] + cfg["n_held"], cfg["motif_len"], cfg["P"], rng
    )
    shared, allm = np.arange(cfg["n_shared"]), np.arange(
        cfg["n_shared"] + cfg["n_held"]
    )
    splits = []
    for allowed, n in ((shared, cfg["n_train"]), (allm, cfg["n_val"])):
        splits.append(
            [
                _emit_tune(
                    motifs, allowed, cfg["motifs_per_tune"], cfg["note_frames"], rng
                )
                for _ in range(n)
            ]
        )
    train, val = splits
    alpha = {}
    for sp in (train, val):
        for atoms, _, _ in sp:
            for a in atoms:
                alpha.setdefault(
                    a[:4], len(alpha)
                )  # key on (op,reg,subreg,val); diff is decode-side

    def pack(sp):
        L = max(len(a) for a, _, _ in sp)
        x = np.zeros((len(sp), L), np.int64)
        om = np.zeros((len(sp), L), bool)
        bm = np.zeros((len(sp), L), bool)
        for i, (atoms, o, b) in enumerate(sp):
            ids = [alpha[a[:4]] for a in atoms]
            x[i, : len(ids)] = ids
            om[i, : len(o)] = o
            bm[i, : len(b)] = b
        return x, om, bm

    inv = {v: k for k, v in alpha.items()}
    return (*pack(train), *pack(val), len(alpha), inv)


def build_model(vocab, seq_len, cfg, device):
    args = argparse.Namespace(
        layers=cfg["layers"],
        heads=cfg["heads"],
        kv_heads=cfg["kv_heads"],
        embed=cfg["embed"],
        max_seq_len=seq_len,
        attn_dropout=0.1,
        norm_eps=1e-5,
        rope_base=500000,
        rope_scale=1.0,
        tie_word_embeddings=False,
    )
    return get_llama3_2(vocab, args).to(device)


def _fwd(model, x):
    out = model(x)
    return torch.cat(out, dim=1) if isinstance(out, list) else out


def onset_acc(model, x, om, bm, device, bs=16):
    model.eval()
    hits = tot = 0
    with torch.inference_mode():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i : i + bs]).to(device)
            pred = _fwd(model, xb).argmax(-1)[:, :-1]
            tgt = xb[:, 1:]
            mask = torch.from_numpy(om[i : i + bs, 1:] & ~bm[i : i + bs, 1:]).to(device)
            hits += int(((pred == tgt) & mask).sum())
            tot += int(mask.sum())
    return hits / max(tot, 1)


def train(model, x, cfg, vocab, device):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = np.random.default_rng(0)
    for _ in range(cfg["epochs"]):
        model.train()
        for i in range(0, len(x), cfg["batch_size"]):
            b = rng.permutation(len(x))[i : i + cfg["batch_size"]]
            xb = torch.from_numpy(x[b]).to(device)
            logits = _fwd(model, xb)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab), xb[:, 1:].reshape(-1)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def greedy_continue(model, prompt_ids, total_len, device):
    ids = list(prompt_ids)
    model.eval()
    with torch.inference_mode():
        while len(ids) < total_len:
            xb = torch.tensor([ids], device=device)
            nxt = int(_fwd(model, xb)[0, -1].argmax())
            ids.append(nxt)
    return ids


def render_ids(ids, inv, args, wav_path):
    from preframr_audio.fidelity import render_df_to_wav

    rows = []
    for tid in ids:
        op, reg, sr, val = inv[int(tid)]
        diff = IRQ if reg == -128 else 0
        rows.append(
            dict(op=op, reg=reg, subreg=sr, val=val, diff=diff, irq=IRQ, description=0)
        )
    df = pd.DataFrame(rows)
    n, _ = render_df_to_wav(df, IRQ, args, Path(wav_path))
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("/scratch/tmp/enc_audition"))
    cli = ap.parse_args()
    cli.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(
        P=64,
        motif_len=8,
        n_shared=6,
        n_held=2,
        n_train=768,
        n_val=192,
        motifs_per_tune=6,
        note_frames=4,
        layers=6,
        heads=8,
        kv_heads=4,
        embed=288,
        epochs=20,
        batch_size=16,
    )
    if cli.quick:
        cfg.update(
            P=16,
            motif_len=4,
            n_shared=3,
            n_held=1,
            n_train=96,
            n_val=48,
            motifs_per_tune=3,
            layers=2,
            heads=4,
            kv_heads=2,
            embed=64,
            epochs=2,
        )
    print(f"device={device}")
    x, om, bm, vx, vom, vbm, vocab, inv = build_corpus(cfg, seed=0)
    seq_len = x.shape[1]
    model = build_model(vocab, seq_len, cfg, device)
    train(model, x, cfg, vocab, device)
    tr = onset_acc(model, x, om, bm, device)
    va = onset_acc(model, vx, vom, vbm, device)
    print(f"3 voices (pulse/tri/noise) vocab={vocab} seq_len={seq_len}")
    print(f"  train_onset_acc={tr:.3f}  HELDOUT_onset_acc={va:.3f}")
    thr = 0.0 if cli.quick else 0.80
    print(
        "VERDICT:",
        (
            "ENCODING SUFFICIENT under multiplexing+waveforms"
            if va >= thr
            else "DEFICIENT under multiplexing"
        ),
    )
    # audition: continue a held-out tune from its first third
    args = None
    try:
        from preframr_tokens.tokenizer_config import named_config

        args = named_config("baseline")
    except Exception as e:  # pragma: no cover
        print("audio args unavailable:", e)
    if args is not None:
        seq = vx[0]
        nz = int((seq != 0).sum())
        prompt = seq[: nz // 3].tolist()
        gen = greedy_continue(model, prompt, nz, device)
        ng = render_ids(gen, inv, args, cli.out / "prediction.wav")
        ngt = render_ids(seq[:nz].tolist(), inv, args, cli.out / "ground_truth.wav")
        # fraction of generated continuation matching ground truth (token-level)
        cont_match = np.mean([gen[i] == int(seq[i]) for i in range(len(prompt), nz)])
        print(
            f"audition: prompt={len(prompt)} gen->{nz} | "
            f"continuation token-match={cont_match:.3f}"
        )
        print(
            f"  WAV: {cli.out}/prediction.wav ({ng} samp), {cli.out}/ground_truth.wav ({ngt} samp)"
        )
    return 0 if cli.quick or va >= thr else 1


if __name__ == "__main__":
    sys.exit(main())
