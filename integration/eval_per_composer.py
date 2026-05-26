"""Per-composer val_acc + val_loss breakdown for a trained ckpt."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

_orig_torch_load = torch.load


def _trusted_torch_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_torch_load(*a, **kw)


torch.load = _trusted_torch_load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preframr.args import add_args  # noqa: E402
from preframr.train.regdataset import RegDataset, get_val_loader  # noqa: E402
from preframr.train.model import get_model  # noqa: E402
from preframr.utils import get_logger  # noqa: E402


def main():
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument("--ckpt", required=True, help="path to .ckpt to load")
    parser.add_argument(
        "--label", required=True, help="label for the report row (composer name)"
    )
    parser.add_argument(
        "--eval-basename-dir",
        required=True,
        help="directory of per-composer dump symlinks/files to keep in val mapper",
    )
    args = parser.parse_args()

    logger = get_logger("INFO")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    logger.info("ckpt: %s", args.ckpt)
    logger.info("eval-basename-dir: %s", args.eval_basename_dir)

    src_df_map = args.df_map_csv
    df_map = pd.read_csv(src_df_map)
    keep_basenames = {
        f for f in os.listdir(args.eval_basename_dir) if f.endswith(".dump.parquet")
    }
    logger.info("filtering val mapper to %u dumps", len(keep_basenames))
    is_val = df_map["kind"] == "val"
    val_basenames = df_map.loc[is_val, "dump_file"].apply(os.path.basename)
    val_keep = is_val & val_basenames.isin(keep_basenames)
    train_keep = ~is_val
    filtered = df_map[train_keep | val_keep].reset_index(drop=True)
    n_val_rows = int(val_keep.sum())
    logger.info("kept %u train rows + %u val rows", int(train_keep.sum()), n_val_rows)
    if n_val_rows == 0:
        raise RuntimeError("No val rows matched the per-composer eval-basename-dir")

    with tempfile.TemporaryDirectory() as tmpd:
        tmp_map = os.path.join(tmpd, "df-map.csv")
        filtered.to_csv(tmp_map, index=False)
        from preframr_tokens import reg_widths_path as _reg_widths_path

        os.symlink(_reg_widths_path(src_df_map), _reg_widths_path(tmp_map))
        args.df_map_csv = tmp_map
        args.eval_reglogs = os.path.join(args.eval_basename_dir, "*.dump.parquet")

        dataset = RegDataset(args, logger=logger)
        dataset.preload()
        assert dataset.tokenizer.token_metadata()
        dataset.load()
        val_dataloader = get_val_loader(args, dataset)
        if val_dataloader is None:
            raise RuntimeError("get_val_loader returned None -- val_block_mapper empty")
        logger.info("val_block_mapper has %u sequences", len(dataset.val_block_mapper))
        model = get_model(dataset, args, logger)

        trainer = pl.Trainer(
            precision=args.trainer_precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            max_epochs=1,
        )
        results = trainer.validate(
            model, dataloaders=val_dataloader, ckpt_path=args.ckpt
        )
        if not results:
            raise RuntimeError("Trainer.validate returned no results")
        metrics = results[0]
        val_acc = float(metrics.get("val_acc", float("nan")))
        val_loss = float(metrics.get("val_loss", float("nan")))
        print(f"{args.label}\t{val_loss:.4f}\t{val_acc:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
