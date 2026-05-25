#!/usr/bin/env python3
"""Profile predict-time forward + per-token decode via torch.profiler."""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preframr.args import add_args  # noqa: E402
from preframr_tokens import (  # noqa: E402
    precompute_subtoken_arrays,
    precompute_vocab_arrays,
)
from preframr.inference.predict import Predictor, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument(
        "--profile-steps",
        type=int,
        default=8,
        help="Number of incremental decode steps to profile (after the prompt).",
    )
    ap.add_argument(
        "--profile-warmup",
        type=int,
        default=2,
        help="Number of warmup decode steps before the profiled pass.",
    )
    ap.add_argument(
        "--profile-trace",
        default=None,
        help="Optional Chrome trace output path (large; only set when needed).",
    )
    ap.add_argument(
        "--profile-rows",
        type=int,
        default=25,
        help="Top-N rows in each report table.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("profile_predict")

    dataset, model, device, _model_compiler = load_model(args, logger)
    model = model.to(device)

    vocab_arrays = None
    if getattr(args, "constrained_decode", False):
        if getattr(args, "tkvocab", 0) and dataset.tokenizer.tkmodel is not None:
            vocab_arrays = precompute_subtoken_arrays(
                dataset.tokenizer.tokens, dataset.tokenizer
            )
        else:
            vocab_arrays = precompute_vocab_arrays(dataset.tokenizer.tokens)
        logger.info("constrained-decode vocab=%u", vocab_arrays["n_vocab"])

    predictor = Predictor(
        args, dataset, model, device, vocab_arrays=vocab_arrays, logger=logger
    )

    seq, seq_meta = dataset.getseq(args.start_seq, block_j=args.start_block)
    prompt = seq[: args.prompt_seq_len].unsqueeze(0).long()
    irq = seq_meta.irq
    logger.info("prompt from %s irq=%u", seq_meta.df_file, irq)

    n_profile = args.profile_steps
    n_warmup = args.profile_warmup
    logger.info(
        "device=%s prompt_len=%d warmup_steps=%d profile_steps=%d",
        device,
        prompt.size(-1),
        n_warmup,
        n_profile,
    )

    if n_warmup > 0:
        t0 = time.perf_counter()
        _ = predictor.predict(prompt, n=n_warmup, temperature=0.1, top_k=1, irq=irq)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("warmup wallclock %.2fs", time.perf_counter() - t0)
        model.model.reset_caches()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    with profile(activities=activities, record_shapes=False) as prof:
        with record_function("predict_full"):
            t0 = time.perf_counter()
            _output = predictor.predict(
                prompt, n=n_profile, temperature=0.1, top_k=1, irq=irq
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0

    logger.info(
        "profile wallclock %.2fs -- %.1f ms/token over %d tokens "
        "(includes one prompt-step forward of %d tokens)",
        wall,
        1000.0 * wall / n_profile,
        n_profile,
        prompt.size(-1),
    )
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        logger.info(
            "cuda peak: allocated=%.1f MB, reserved=%.1f MB", peak_mb, reserved_mb
        )

    if torch.cuda.is_available():
        print("\n=== top ops by self CUDA time ===")
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=args.profile_rows
            )
        )
    print("\n=== top ops by self CPU time ===")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=args.profile_rows
        )
    )

    if args.profile_trace:
        prof.export_chrome_trace(args.profile_trace)
        logger.info("chrome trace saved to %s", args.profile_trace)


if __name__ == "__main__":
    main()
