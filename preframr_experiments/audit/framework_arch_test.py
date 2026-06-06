"""Framework integration test for the core model architecture.

Question: can the torchtune ``llama3_2`` body (the body the preframr trainer
wraps) learn and generalize from clean structured data in IDEAL conditions?

Task: deterministic-motif copy-then-continue.
- Vocabulary: 64 tokens (no SID semantics).
- Each "tune": 8 motifs of 8 tokens, randomly ordered. 64 tokens total.
- Within a motif: ``t[i+1] = (t[i] + 1) % MOTIF_LEN`` from a fixed seed.
- N_SHARED motifs appear in train + held-out. N_HELD motifs appear ONLY
  in held-out. Within-motif rule is identical for all motifs, so the
  model can predict the held-out continuations by *rule*, not memory.

Pass criteria (full mode):
- train inside-motif acc > 0.95 (model can fit)
- val inside-motif acc > 0.80 (model learned the rule, not just memorised)
- val_inside_acc within 0.10 of train_inside_acc by the last epoch (no
  catastrophic generalization gap). Total val_loss is NOT a sanity gate:
  motif boundaries are unpredictable by design (irreducible loss), so
  a confident model can keep raising boundary loss while inside-motif
  accuracy holds -- this is correct behaviour, not divergence.

Model dimensions = mini body=large (matches preframr_experiments.base.
mini_train_args body=large): llama3_2 layers=6 heads=8 kv_heads=4 embed=288.

``--quick`` mode reduces everything for a build-time smoke (tests live
generalization mechanics; doesn't verify the headline pass thresholds).

Run inside the xpt image:
  python3 -m preframr_experiments.audit.framework_arch_test
  python3 -m preframr_experiments.audit.framework_arch_test --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# Import via the framework's wrapper so the torchtune submodule load order
# matches production (a raw torchtune.models.llama3_2 import transitively
# pulls in torchtune.data, which fails on the host's pyarrow ABI).
from preframr.train.model.bodies import get_llama3_2


@dataclass
class Config:
    vocab: int = 64
    motif_len: int = 8
    motifs_per_tune: int = 8
    n_train: int = 1024
    n_val: int = 256
    n_shared_motifs: int = 6
    n_held_motifs: int = 2
    layers: int = 6
    heads: int = 8
    kv_heads: int = 4
    embed: int = 288
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    train_acc_threshold: float = 0.95
    val_acc_threshold: float = 0.80
    max_generalization_gap: float = 0.10  # train_acc - val_acc upper bound

    @property
    def tune_len(self) -> int:
        return self.motif_len * self.motifs_per_tune

    @classmethod
    def quick(cls) -> "Config":
        return cls(
            vocab=32,
            motif_len=4,
            motifs_per_tune=4,
            n_train=128,
            n_val=64,
            n_shared_motifs=3,
            n_held_motifs=1,
            layers=2,
            heads=4,
            kv_heads=2,
            embed=64,
            epochs=2,
            batch_size=16,
            train_acc_threshold=0.0,
            val_acc_threshold=0.0,
            max_generalization_gap=1.0,
        )


def _make_motifs(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    seeds = rng.choice(
        cfg.vocab - cfg.motif_len,
        size=cfg.n_shared_motifs + cfg.n_held_motifs,
        replace=False,
    )
    return np.stack(
        [np.array([s + j for j in range(cfg.motif_len)], dtype=np.int64) for s in seeds]
    )


def _synth_corpus(
    cfg: Config,
    motifs: np.ndarray,
    n_tunes: int,
    motif_idxs: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((n_tunes, cfg.tune_len), dtype=np.int64)
    boundary = np.zeros((n_tunes, cfg.tune_len), dtype=np.bool_)
    for i in range(n_tunes):
        choices = rng.choice(motif_idxs, size=cfg.motifs_per_tune)
        for j, m in enumerate(choices):
            s, e = j * cfg.motif_len, (j + 1) * cfg.motif_len
            out[i, s:e] = motifs[m]
            boundary[i, s] = True
    return out, boundary


def build_corpus(cfg: Config, seed: int):
    rng = np.random.default_rng(seed)
    motifs = _make_motifs(cfg, rng)
    shared = np.arange(cfg.n_shared_motifs)
    held_or_shared = np.arange(cfg.n_shared_motifs + cfg.n_held_motifs)
    train_x, train_b = _synth_corpus(cfg, motifs, cfg.n_train, shared, rng)
    val_x, val_b = _synth_corpus(cfg, motifs, cfg.n_val, held_or_shared, rng)
    return train_x, train_b, val_x, val_b


def _epoch_iter(
    x: np.ndarray,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
):
    idxs = np.arange(len(x))
    if shuffle:
        rng.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        yield x[idxs[i : i + batch_size]], idxs[i : i + batch_size]


def _inside_motif_acc(
    logits: torch.Tensor,
    x_batch: np.ndarray,
    b_batch: np.ndarray,
) -> tuple[int, int]:
    pred = logits.argmax(-1)
    gt = torch.from_numpy(x_batch).to(pred.device)
    target = gt[:, 1:]
    pred = pred[:, :-1]
    inside = ~torch.from_numpy(b_batch[:, 1:]).to(pred.device)
    inside[:, 0] = False  # exclude very first position bias
    n = int(inside.sum().item())
    if n == 0:
        return 0, 0
    hits = int(((pred == target) & inside).sum().item())
    return hits, n


def build_model(cfg: Config, device: str) -> torch.nn.Module:
    args = argparse.Namespace(
        layers=cfg.layers,
        heads=cfg.heads,
        kv_heads=cfg.kv_heads,
        embed=cfg.embed,
        max_seq_len=cfg.tune_len,
        attn_dropout=0.1,
        norm_eps=1e-5,
        rope_base=500000,
        rope_scale=1.0,
        tie_word_embeddings=False,
    )
    return get_llama3_2(cfg.vocab, args).to(device)


def _forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, list):
        out = torch.cat(out, dim=1)
    return out


def run_epoch_train(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    cfg: Config,
    train_x: np.ndarray,
    train_b: np.ndarray,
    rng: np.random.Generator,
    device: str,
) -> tuple[float, float]:
    model.train()
    loss_sum, n = 0.0, 0
    hits, denom = 0, 0
    for batch_x, idx in _epoch_iter(train_x, cfg.batch_size, True, rng):
        batch_b = train_b[idx]
        x = torch.from_numpy(batch_x).to(device)
        logits = _forward(model, x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab), x[:, 1:].reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)
        with torch.inference_mode():
            h, d = _inside_motif_acc(logits, batch_x, batch_b)
        hits += h
        denom += d
    return loss_sum / max(n, 1), hits / max(denom, 1)


def run_epoch_eval(
    model: torch.nn.Module,
    cfg: Config,
    val_x: np.ndarray,
    val_b: np.ndarray,
    rng: np.random.Generator,
    device: str,
) -> tuple[float, float]:
    model.eval()
    loss_sum, n = 0.0, 0
    hits, denom = 0, 0
    with torch.inference_mode():
        for batch_x, idx in _epoch_iter(val_x, cfg.batch_size, False, rng):
            batch_b = val_b[idx]
            x = torch.from_numpy(batch_x).to(device)
            logits = _forward(model, x)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, cfg.vocab), x[:, 1:].reshape(-1)
            )
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)
            h, d = _inside_motif_acc(logits, batch_x, batch_b)
            hits += h
            denom += d
    return loss_sum / max(n, 1), hits / max(denom, 1)


def run(
    cfg: Config, seed: int, device: str | None = None, verbose: bool = True
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    train_x, train_b, val_x, val_b = build_corpus(cfg, seed)
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(
            f"corpus train={train_x.shape}, val={val_x.shape}, "
            f"motifs={cfg.n_shared_motifs} shared + {cfg.n_held_motifs} held-out, "
            f"device={device}"
        )
        print(
            f"model: llama3_2 layers={cfg.layers} heads={cfg.heads} "
            f"kv_heads={cfg.kv_heads} embed={cfg.embed} "
            f"({n_params/1e6:.2f}M params)"
        )
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    rng = np.random.default_rng(seed)
    val_losses: list[float] = []
    train_acc = val_acc = 0.0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch_train(
            model, opt, cfg, train_x, train_b, rng, device
        )
        val_loss, val_acc = run_epoch_eval(model, cfg, val_x, val_b, rng, device)
        val_losses.append(val_loss)
        dt = time.time() - t0
        if verbose:
            print(
                f"  ep{epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_inside_acc={train_acc:.3f} val_inside_acc={val_acc:.3f} ({dt:.1f}s)"
            )
    result = {
        "final_train_inside_acc": float(train_acc),
        "final_val_inside_acc": float(val_acc),
        "final_val_loss": float(val_losses[-1]) if val_losses else float("nan"),
        "val_losses": [float(v) for v in val_losses],
        "generalization_gap": float(train_acc - val_acc),
        "n_params": int(n_params),
        "device": device,
    }
    return result


def verdict(cfg: Config, result: dict) -> tuple[bool, list[str]]:
    fails: list[str] = []
    if result["final_train_inside_acc"] < cfg.train_acc_threshold:
        fails.append(
            f"train_inside_acc {result['final_train_inside_acc']:.3f} "
            f"< {cfg.train_acc_threshold}"
        )
    if result["final_val_inside_acc"] < cfg.val_acc_threshold:
        fails.append(
            f"val_inside_acc {result['final_val_inside_acc']:.3f} "
            f"< {cfg.val_acc_threshold}"
        )
    if result["generalization_gap"] > cfg.max_generalization_gap:
        fails.append(
            f"generalization gap {result['generalization_gap']:.3f} "
            f"> {cfg.max_generalization_gap} "
            f"(train fit but doesn't generalize)"
        )
    return (not fails), fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--quick", action="store_true", help="Tiny smoke run (no headline gate)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    cli = ap.parse_args()
    cfg = Config.quick() if cli.quick else Config()
    result = run(cfg, cli.seed, device=cli.device)
    ok, fails = verdict(cfg, result)
    print("\n=== verdict ===")
    print(
        f"  train_inside_acc: {result['final_train_inside_acc']:.3f} "
        f"(want > {cfg.train_acc_threshold})"
    )
    print(
        f"  val_inside_acc:   {result['final_val_inside_acc']:.3f} "
        f"(want > {cfg.val_acc_threshold})"
    )
    print(
        f"  generalization gap:    {result['generalization_gap']:+.3f}  "
        f"(want < {cfg.max_generalization_gap})"
    )
    if ok:
        print("\n  CORE MODEL GENERALIZES: PASS.")
        return 0
    print("\n  CORE MODEL FAILED to generalize:")
    for f in fails:
        print(f"    - {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
