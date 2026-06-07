"""Build-time smoke for the framework-arch diagnostic. Runs the quick-mode
synthetic-corpus path through llama3_2 to make sure (a) the script imports,
(b) the corpus builder + forward + backward all work, (c) val_loss strictly
improves from epoch 0 to 1. The full diagnostic (which sets the >0.80
held-out generalization bar) is invoked manually --
``python3 -m preframr_experiments.audit.framework_arch_test``."""

from __future__ import annotations

import pytest

from preframr_experiments.audit import framework_arch_test as fat


def test_quick_config_runs_on_cpu_and_loss_drops():
    cfg = fat.Config.quick()
    result = fat.run(cfg, seed=0, device="cpu", verbose=False)
    assert result["n_params"] > 0
    assert result["val_losses"], "expected at least one val epoch"
    assert result["val_losses"][-1] < result["val_losses"][0], (
        f"val_loss did not improve in {cfg.epochs} epochs of quick mode: "
        f"{result['val_losses']}"
    )


def test_corpus_motif_partition():
    """Held-out motifs never leak into the training set, and the within-motif
    rule (token[i+1] = token[i] + 1) holds across every motif slot."""
    cfg = fat.Config.quick()
    train_x, _train_b, val_x, _val_b = fat.build_corpus(cfg, seed=0)
    train_motifs = train_x.reshape(-1, cfg.motif_len)
    val_motifs = val_x.reshape(-1, cfg.motif_len)
    train_seeds = {int(r[0]) for r in train_motifs}
    val_seeds = {int(r[0]) for r in val_motifs}
    assert len(train_seeds) == cfg.n_shared_motifs
    held_in_val_only = val_seeds - train_seeds
    assert len(held_in_val_only) <= cfg.n_held_motifs
    for motif in train_motifs:
        for i in range(1, cfg.motif_len):
            assert (
                int(motif[i]) == int(motif[0]) + i
            ), "within-motif rule broken in train"
    for motif in val_motifs:
        for i in range(1, cfg.motif_len):
            assert int(motif[i]) == int(motif[0]) + i, "within-motif rule broken in val"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_inside_motif_accuracy_helper_is_correct(seed):
    """Synthesize logits that always predict the correct next token; the
    accuracy helper should return 1.0 over inside-motif positions."""
    import numpy as np
    import torch

    cfg = fat.Config.quick()
    rng = np.random.default_rng(seed)
    motifs = fat._make_motifs(cfg, rng)
    x, b = fat._synth_corpus(cfg, motifs, 4, np.arange(cfg.n_shared_motifs), rng)
    one_hot = np.zeros((x.shape[0], x.shape[1], cfg.vocab), dtype=np.float32)
    # Shift right: predict x[:,1:] given x[:,:-1].
    for i in range(x.shape[1] - 1):
        for n in range(x.shape[0]):
            one_hot[n, i, x[n, i + 1]] = 10.0
    logits = torch.from_numpy(one_hot)
    hits, denom = fat._inside_motif_acc(logits, x, b)
    assert denom > 0
    assert (
        hits == denom
    ), f"perfect logits should hit all inside positions ({hits}/{denom})"
