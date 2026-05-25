#!/bin/bash
# Overnight calibration + sweep batch.
#
# Current batch: re-arc STAGE 1 (mini triage) on the post-FREQ_TRAJ tokenizer
# at the vocab-trimmed config (--tkvocab 8192, UNK=0). 7 model-side specs (see
# SPECS below); baseline + refuted re-run after full_macros_prodlike PASSED.
# Prodlike stage (B=4/accum=8, effective batch 32) follows once mini is clean.
#
# Outputs land under /scratch/tmp/preframr_experiments/.
# Status / progress: tail -f /scratch/tmp/preframr_experiments/overnight_batch.log
# Completion marker: /scratch/tmp/preframr_experiments/overnight_batch.done

set -u

REPO_ROOT="/scratch/anarkiwi/preframr"
WORK_ROOT="/scratch/tmp/preframr_experiments"
LOG="${WORK_ROOT}/overnight_batch.log"
DONE_MARKER="${WORK_ROOT}/overnight_batch.done"

mkdir -p "${WORK_ROOT}"
rm -f "${DONE_MARKER}"
cd "${REPO_ROOT}"

# NOTE: the encoder-branch validate_branches.sh gate is skipped here. This
# re-arc batch is MODEL-SIDE specs whose baseline/variant arms share an
# identical parse (they differ only in model flags), so the encoder-branch
# firing check does not apply. Per-spec runner preflight_check (HVSC + OOM
# smoke) is the safety gate. Restore validate_branches for encoder-A/B batches.

# Re-arc STAGE 1 (mini triage) on the post-FREQ_TRAJ tokenizer at the
# vocab-trimmed config (--tkvocab 8192, UNK=0). Prodlike stage follows once
# mini triage is clean. full_macros_prodlike PASSED content-confirmed.
SPECS=(
    "content_floor_check"
    "per_tier_heads_mini_body_large"
    "content_diffusion_mini_body_large"
    "contrastive_mini_body_large"
    "mask_structural_loss_mini"
    "voice_permutation_mini_body_large"
    "cluster_content_mini_body_large"
)

{
echo "==== batch started $(date -Iseconds) host=$(hostname) ===="
for spec in "${SPECS[@]}"; do
    echo "==== ${spec} starting $(date -Iseconds) ===="
    # cluster spec's pre_run_hook must run each seed; disable the dataset cache for it
    if [[ "${spec}" == cluster_content* ]]; then
        export PREFRAMR_DATASET_CACHE_DISABLE=1
    else
        unset PREFRAMR_DATASET_CACHE_DISABLE
    fi
    if python3 -m preframr_experiments.run "${spec}" --root "${WORK_ROOT}" --tkvocab 8192; then
        echo "==== ${spec} done $(date -Iseconds) ===="
    else
        echo "==== ${spec} FAILED rc=$? $(date -Iseconds) ===="
    fi
done
echo "==== batch finished $(date -Iseconds) ===="
} > "${LOG}" 2>&1

touch "${DONE_MARKER}"
