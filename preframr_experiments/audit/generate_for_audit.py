#!/usr/bin/env python3
"""Generate token streams (real-prompt + random-prompt) from a ckpt for audits."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from preframr.args import add_args, apply_pipeline_spec_to_args
from preframr.inference.predict import Predictor, load_model
from preframr.train.regdataset import get_prompt
from preframr.utils import get_logger


def _build_args(cli):
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args([])
    args.model_state = str(cli.ckpt)
    args.tb_logs = ""
    args.df_map_csv = str(cli.work_dir / "df-map.csv")
    args.dataset_csv = str(cli.work_dir / "dataset.csv.zst")
    args.token_csv = str(cli.work_dir / "tokens.csv")
    args.reglogs = str(cli.work_dir / "train" / "*" / "*.dump.parquet")
    args.eval_reglogs = ""
    args.require_pq = False
    args.max_seq_len = cli.max_seq_len
    args.prompt_seq_len = cli.prompt_seq_len
    args.seq_len = cli.max_seq_len
    args.tkvocab = cli.tkvocab
    args.batch_size = 1
    args.compile = False
    args.constrained_decode = False
    args.min_acc = 0.0
    args.temperature = cli.temperature
    args.top_k = cli.top_k if cli.top_k > 0 else None
    args.predictions = 1
    args.pipeline_spec = ""
    apply_pipeline_spec_to_args(args)
    return args


def _write_tokens(path: Path, tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for t in tokens:
            w.writerow([int(t)])


def _reset_predict_state(model, dataset):
    if hasattr(model.model, "reset_caches"):
        model.model.reset_caches()
    if hasattr(dataset, "block_mapper"):
        dataset.block_mapper.block_metas = []
        dataset.block_mapper.finalize()


def _generate_real(args, dataset, model, predictor, logger, start_seq):
    _reset_predict_state(model, dataset)
    args.start_seq = start_seq
    args.start_block = 0
    dataset.predict_load()
    args.start_seq = 0
    args.start_block = 0
    irq, n, prompt, _prompt_compare, _reg_start, _prompt_df = get_prompt(
        args, dataset, logger
    )
    out = predictor.predict(
        prompt, n, temperature=args.temperature, top_k=args.top_k, irq=irq
    )
    return out.tolist()


def _generate_random(args, dataset, model, predictor, _logger, rng_seed):
    _reset_predict_state(model, dataset)
    g = torch.Generator(device="cpu")
    g.manual_seed(rng_seed)
    n_vocab = dataset.n_vocab
    prompt = torch.randint(
        1, n_vocab, (1, args.prompt_seq_len), generator=g, dtype=torch.long
    )
    n = args.max_seq_len - args.prompt_seq_len
    out = predictor.predict(
        prompt, n, temperature=args.temperature, top_k=args.top_k, irq=None
    )
    return out.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=("real", "random"), required=True)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--prompt-seq-len", type=int, default=2048)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--tkvocab", type=int, default=32768)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", default="")
    cli = ap.parse_args()
    args = _build_args(cli)
    logger = get_logger("INFO")
    dataset, model, device, _compiler = load_model(args, logger)
    model = model.to(device)
    if hasattr(model.model, "set_num_output_chunks"):
        model.model.set_num_output_chunks(0)
    predictor = Predictor(
        args, dataset, model, device, vocab_arrays=None, logger=logger
    )
    tag = f"_{cli.tag}" if cli.tag else ""
    n_done = 0
    seq_attempts = 0
    seq_cap = cli.n_samples * 6
    while n_done < cli.n_samples and seq_attempts < seq_cap:
        try:
            if cli.mode == "real":
                tokens = _generate_real(
                    args, dataset, model, predictor, logger, start_seq=seq_attempts
                )
            else:
                tokens = _generate_random(
                    args, dataset, model, predictor, logger, rng_seed=seq_attempts
                )
        except (AssertionError, IndexError, ValueError) as e:
            logger.warning("skip attempt=%u: %s", seq_attempts, e)
            seq_attempts += 1
            continue
        out_path = (
            cli.out_dir / f"stream_{cli.mode}{tag}_T{cli.temperature}_{n_done:02d}.csv"
        )
        _write_tokens(out_path, tokens)
        logger.info("wrote %s (%u tokens)", out_path, len(tokens))
        n_done += 1
        seq_attempts += 1
    if n_done < cli.n_samples:
        logger.warning("generated %u / %u samples", n_done, cli.n_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
