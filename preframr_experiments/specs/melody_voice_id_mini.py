"""Mini 2-arm A/B: local voice id on the VOICE reg vs the current FRAME-packed order.

Built on the validated bare-melody stream from `melody_no_unigram_mini` (clean
PW+filter-ablated substrate + freq stack + `--tkvocab 0`, where op45 V0_HI pitch
onset already reads 0.658). This isolates the voice-encoding lever: does surfacing
voice identity LOCALLY on the VOICE (-126) marker help, now that the merge weld is
gone?

Today voice is implicit: a write's voice = (FRAME.val >> 2*marker_count) & 3 — a
stateful "read header + count markers" decode. `--voice-id-on-marker` instead writes
the voice id onto each surviving VOICE marker's val (currently zeroed), so the model
reads voice from the adjacent delimiter. Round-trip-safe: decode (`remove_voice_reg`)
still derives voice from FRAME.val and drops VOICE rows, so the val change is discarded
on decode (verified: byte-identical reconstruction; VOICE.val 0 -> {1,2}).

LIMITATION (documented, this is the minimal version): only the non-lead voices
(frame slots 1+) get an explicit local id; the slot-0 lead voice's marker is dropped
by the encoder, so its voice stays in the adjacent FRAME.val low bits. FRAME also still
redundantly carries the full order (decode reads it). The fully-clean version
(FRAME -> pure tick, explicit per-voice markers incl. slot 0, decode-from-VOICE.val)
needs a decode rewrite and is the follow-up if this shows signal. See
design/voice_encoding_reference.md.

Arms (only the voice flag differs):
- voice_id: clean substrate + freq stack + --tkvocab 0 + --voice-id-on-marker
- baseline: clean substrate + freq stack + --tkvocab 0  (reproduces no_unigram)

Decisive read: op x subreg split (pool per_class by vocab_atom) — does op45 V0_HI rise
above the 0.658 baseline, and do the residual voiced atoms (V0_LO / DELTA / op48) move?
3 seeds, mini body=large, 60 epochs, image 0.2.11.

Requires PREFRAMR_DATASET_CACHE_DISABLE=1 (arms tokenize differently) and
PREFRAMR_BIND_SRC=1 (the --voice-id-on-marker flag + encode change are working-tree,
not yet baked: preframr args.py + preframr-tokens reglogparser.py).
"""

from pathlib import Path

from preframr_experiments.audit.ablate_pwfilter import ablate_staged_dumps
from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

_IMAGE = "anarkiwi/preframr:0.2.11"

_BASE_MACROS = (
    "freq_trajectory_pass",
    "preset_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c4",
    "voice_canonical_block_order",
    "ctrl_bigram_pass",
    "loop_pass",
    "loop_transposed",
)

_STACK_MACROS = _BASE_MACROS + (
    "trajectory_anchor_pass",
    "freq_v0_interval",
    "freq_onset_pass",
)

_STACK = "--tkvocab 0"

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
    name="melody_voice_id_mini",
    doc=(
        "Mini 2-arm A/B: --voice-id-on-marker (local voice id on the VOICE reg) vs "
        "the FRAME-packed-order baseline, both on the clean substrate + freq stack + "
        "--tkvocab 0 (the validated bare-melody stream). Does local voice identity "
        "help once the Unigram weld is gone? Decisive: op x subreg split on op45 "
        "V0_HI + residual voiced atoms. 3 seeds, mini body=large, 60 epochs. "
        "Requires PREFRAMR_DATASET_CACHE_DISABLE=1 and PREFRAMR_BIND_SRC=1."
    ),
    tier="mini",
    image=_IMAGE,
    arms=[
        Arm(
            label="voice_id",
            extra_cargs=f"{_STACK} --voice-id-on-marker",
            macro_flags=_STACK_MACROS,
        ),
        Arm(
            label="baseline",
            extra_cargs=_STACK,
            macro_flags=_STACK_MACROS,
            baseline=True,
        ),
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
    pre_run_hook=_ablate_hook,
)
