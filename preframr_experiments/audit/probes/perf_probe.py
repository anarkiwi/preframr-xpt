#!/usr/bin/env python3
"""Time the real constrained-decode loop on whatever device load_model picks
(XPU when /dev/dri is present, else CPU). Bypasses the corpus dataset: the
constrained decoder generates structurally-valid tokens from scratch, so we
self-generate a prompt, then time steady-state per-token decode."""

import argparse
import logging
import time

import numpy as np
import torch

from preframr.args import add_args
from preframr_tokens import precompute_subtoken_arrays, precompute_vocab_arrays
from preframr.inference.predict import Predictor, load_model


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--probe-prompt", type=int, default=256)
    ap.add_argument("--probe-warmup", type=int, default=8)
    ap.add_argument("--probe-steps", type=int, default=128)
    ap.add_argument("--probe-irq", type=int, default=19656)
    ap.add_argument("--probe-profile", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("perf_probe")

    dataset, model, device, _ = load_model(args, logger)
    model = model.to(device)

    vocab_arrays = None
    if getattr(args, "constrained_decode", False):
        if getattr(args, "tkvocab", 0) and dataset.tokenizer.tkmodel is not None:
            vocab_arrays = precompute_subtoken_arrays(
                dataset.tokenizer.tokens, dataset.tokenizer
            )
        else:
            vocab_arrays = precompute_vocab_arrays(dataset.tokenizer.tokens)
        logger.info("constrained vocab=%u", vocab_arrays["n_vocab"])

    predictor = Predictor(
        args, dataset, model, device, vocab_arrays=vocab_arrays, logger=logger
    )
    irq = args.probe_irq

    start_tok = 0
    if vocab_arrays is not None:
        fm = np.nonzero(vocab_arrays["is_frame_marker"])[0]
        if len(fm):
            start_tok = int(fm[0])
    seed = torch.tensor([[start_tok]], device=device, dtype=torch.long)

    def sync():
        if device.type == "xpu":
            torch.xpu.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    logger.info(
        "generating %d-token prompt (also compile warmup)...", args.probe_prompt
    )
    t0 = time.perf_counter()
    pg = predictor.predict(seed, n=args.probe_prompt, temperature=0.1, top_k=1, irq=irq)
    sync()
    logger.info("prompt gen %.2fs", time.perf_counter() - t0)
    prompt = torch.cat([seed.squeeze(0), pg]).unsqueeze(0)
    model.model.reset_caches()

    _ = predictor.predict(
        prompt, n=args.probe_warmup, temperature=0.1, top_k=1, irq=irq
    )
    sync()
    model.model.reset_caches()

    if getattr(args, "probe_profile", False):
        from torch.profiler import profile, ProfilerActivity

        acts = [ProfilerActivity.CPU]
        if device.type == "xpu" and hasattr(ProfilerActivity, "XPU"):
            acts.append(ProfilerActivity.XPU)
        elif device.type == "cuda":
            acts.append(ProfilerActivity.CUDA)
        with profile(activities=acts) as prof:
            _ = predictor.predict(
                prompt, n=args.probe_steps, temperature=0.1, top_k=1, irq=irq
            )
            sync()
        key = "self_xpu_time_total" if device.type == "xpu" else "self_cpu_time_total"
        try:
            print(prof.key_averages().table(sort_by=key, row_limit=20))
        except Exception:
            print(
                prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
            )

    t0 = time.perf_counter()
    _ = predictor.predict(prompt, n=args.probe_steps, temperature=0.1, top_k=1, irq=irq)
    sync()
    wall = time.perf_counter() - t0
    logger.info(
        "RESULT device=%s constrained=%s compile=%s : %.3f ms/token "
        "(%d tokens, %.2fs)",
        device.type,
        getattr(args, "constrained_decode", False),
        getattr(args, "compile", True),
        1000.0 * wall / args.probe_steps,
        args.probe_steps,
        wall,
    )


if __name__ == "__main__":
    main()
