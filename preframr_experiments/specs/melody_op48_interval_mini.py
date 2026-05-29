"""Mini 2-arm A/B: interval-code the op48 FREQ_ONSET channel.

Phase-0/voice work localised the remaining melodic-onset residual to pitch-VALUE
entropy: op48 FREQ_ONSET (isolated single-frame freq onsets) is stored ABSOLUTE
(204-way, ~6.8 bits) and reads 0.000 acc, even on the clean de-merged stream where
op45 V0_HI already hits 0.66. A per-voice entropy probe on the tokenized corpus
showed op48 absolute 6.5b (~100 eff) -> mod-256 interval 5.3b (~40 eff), INT better
on all three voices (correcting the old cross-voice-pooled "intervals are worse"
finding). `--freq-onset-interval` stores each FREQ-reg op48 onset as a signed
interval (mod 256) from the previous freq onset on that reg, marked subreg=1; the
first per reg stays absolute. PW/filter onsets stay absolute (timbre is absolute).
Round-trip validated byte-exact (incl. leaps + byte wraps).

Both arms also run --voice-order-on-marker (FRAME voice-order encoding ELIMINATED;
voice lives only on the VOICE reg, FRAME is a pure 0 tick). op48-interval is the
only A/B variable.

Arms (only the op48 flag differs), both on the clean bare-melody stream:
- op48_interval: clean substrate + freq stack + voice-order-marker + --tkvocab 0 + --freq-onset-interval
- baseline:      clean substrate + freq stack + voice-order-marker + --tkvocab 0

Decisive read: op x subreg split -- op48 subreg=1 (interval) acc vs the baseline's
op48 subreg=-1 (absolute, 0.000). Expect a partial lift (~0.2-0.4, V0_LO-like, since
40 eff values remain), not the V0_HI collapse. 3 seeds, mini body=large, 60 epochs.

Requires PREFRAMR_DATASET_CACHE_DISABLE=1 (arms tokenize differently) and
PREFRAMR_BIND_SRC=1 (--freq-onset-interval + the FreqOnsetPass/decoder/state changes
are working-tree, not yet baked).
"""

from pathlib import Path

from preframr_experiments.audit.ablate_pwfilter import ablate_staged_dumps
from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

_IMAGE = "anarkiwi/preframr:0.2.11"

_BASE_TRANSFORMS = [
    {"name": "freq_trajectory"},
    {"name": "preset"},
    {"name": "hard_restart"},
    {"name": "legato_per_cluster", "params": {"clusters": [2, 4]}},
    {"name": "voice_block_order"},
    {"name": "ctrl_bigram"},
    {"name": "loop"},
]

_STACK = (
    "--trajectory-anchor-pass --freq-v0-interval --freq-onset-pass "
    "--voice-order-on-marker --tkvocab 0"
)

_TRAIN_ARGS = mini_train_args(body="large").replace(
    "--max-epochs 160", "--max-epochs 60"
)


def _ablate_hook(arm: Arm, work_dir: Path) -> None:
    train_root = Path(work_dir) / "train"
    if not train_root.exists():
        raise FileNotFoundError(
            f"train dumps not staged at {train_root}; ablation must run after stage_dumps"
        )
    n = ablate_staged_dumps(Path(work_dir))
    print(f"{arm.label} ablation: rewrote {n} staged dumps (PW+filter dropped)")


spec = ExperimentSpec(
    name="melody_op48_interval_mini",
    doc=(
        "Mini 2-arm A/B: --freq-onset-interval (per-reg mod-256 interval on op48 "
        "FREQ_ONSET) vs absolute baseline, both clean substrate + freq stack + "
        "--tkvocab 0. Does interval-coding the absolute op48 onset channel move it "
        "off 0.000? Decisive: op48 subreg=1 acc via op x subreg split. 3 seeds, "
        "mini body=large, 60 epochs. Requires PREFRAMR_DATASET_CACHE_DISABLE=1 and "
        "PREFRAMR_BIND_SRC=1."
    ),
    tier="mini",
    image=_IMAGE,
    arms=[
        Arm(label="op48_interval", extra_cargs=f"{_STACK} --freq-onset-interval"),
        Arm(label="baseline", extra_cargs=_STACK, baseline=True),
    ],
    metrics=[
        "alphabet_size",
        "encoded_tokens_per_song",
        "val_loss_best",
        "val_acc_at_best_loss",
        "epochs_to_best_val_loss",
        "wallclock_train_min",
    ],
    seeds=3,
    seq_len=4096,
    tkvocab=32768,
    max_perm=1,
    train_args=_TRAIN_ARGS,
    pipeline_spec={"transforms": list(_BASE_TRANSFORMS)},
    pre_run_hook=_ablate_hook,
)
