#!/usr/bin/env python3
"""Pre-launch GPU smoke: one Lightning fit step at prodlike shape."""

from __future__ import annotations

import argparse
import sys

import torch
from torch.utils.data import DataLoader, Dataset


def _prodlike_args():
    """Mirror ``prodlike_train_args()`` + the spec's seq_len + tkvocab."""
    return argparse.Namespace(
        model="llama3_2",
        layers=16,
        heads=12,
        kv_heads=4,
        embed=768,
        intermediate=2048,
        max_seq_len=8192,
        attn_dropout=0.1,
        norm_eps=1e-5,
        rope_base=1e4,
        rope_scale=1.0,
        tie_word_embeddings=True,
        precision="high",
        max_autotune=False,
        accumulate_grad_batches=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        label_smoothing=0.0,
        compile=True,
        token_weighting=False,
        structural_loss_lambda=0.0,
        num_output_chunks=0,
    )


class _SyntheticDataset(Dataset):
    """One-block synthetic dataset matching the (x, y) shape Lightning's
    training_step sees: x = token ids, y = next-token ids, both
    ``(seq_len,)`` long tensors.
    """

    def __init__(self, n_vocab, seq_len, n):
        if n < 1:
            raise ValueError(f"_SyntheticDataset: n must be >= 1, got {n}")
        self.n_vocab = n_vocab
        self.seq_len = seq_len
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        x = torch.randint(0, self.n_vocab, (self.seq_len,), dtype=torch.long)
        y = torch.randint(1, self.n_vocab, (self.seq_len,), dtype=torch.long)
        return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help=(
            "Per-device batch size to smoke. Default 2 matches the "
            "post-2026-05-12 prodlike spec; pre-fix prodlike used 4 "
            "(and OOM'd on the unembed at peak ~22.7 GiB). Pass 4 to "
            "verify the OOM cliff is still present below the fix."
        ),
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=16,
        help=(
            "Forwarded for parity with prodlike_train_args. The smoke "
            "runs one micro-batch so this only affects Lightning's "
            "trainer setup, not the peak per-step memory."
        ),
    )
    parser.add_argument(
        "--tkvocab",
        type=int,
        default=32768,
        help=(
            "Vocab size to smoke (default 32768 matches current prodlike "
            "specs incl. voice_traj_distributed_set_diff_freq_prodlike "
            "combined arm; pass 131072 to reproduce pre-2026-05-18 shape)."
        ),
    )
    cli = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA unavailable; smoke needs a GPU", file=sys.stderr)
        return 1

    import pytorch_lightning as pl
    from preframr.train.model import Model, cuda_compile

    args = _prodlike_args()
    args.accumulate_grad_batches = cli.accumulate_grad_batches
    n_vocab = cli.tkvocab
    batch_size = cli.batch_size
    seq_len = args.max_seq_len

    print(
        f"[smoke] prodlike Model: layers={args.layers} embed={args.embed} "
        f"n_vocab={n_vocab} seq_len={seq_len} batch={batch_size} "
        f"accumulate={cli.accumulate_grad_batches} num_output_chunks=auto",
        flush=True,
    )

    model = Model(args, n_vocab=n_vocab, tokens=None, tkmodel=None, metadata=None)
    print(
        f"[smoke] num_output_chunks set to {model.num_output_chunks}",
        flush=True,
    )
    model = cuda_compile(args, model)

    dataset = _SyntheticDataset(n_vocab, seq_len, n=batch_size)
    train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    torch.cuda.reset_peak_memory_stats()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=1,
        num_sanity_val_steps=2,
        limit_val_batches=1,
        accumulate_grad_batches=cli.accumulate_grad_batches,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    print(
        "[smoke] trainer.fit(max_steps=1, num_sanity_val_steps=2) ...",
        flush=True,
    )
    trainer.fit(model, train_dl, val_dl)
    torch.cuda.synchronize()

    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1 << 30)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1 << 30)
    print(
        f"[smoke] OK peak_alloc={peak_alloc_gb:.2f} GiB "
        f"peak_reserved={peak_reserved_gb:.2f} GiB",
        flush=True,
    )
    if peak_reserved_gb > 23.0:
        print(
            f"FAIL: peak reserved {peak_reserved_gb:.2f} GiB > 23 GiB headroom",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
