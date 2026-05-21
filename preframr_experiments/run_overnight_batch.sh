#!/bin/bash
# Overnight calibration + sweep batch.
#
# Current batch: voice_traj_distributed_set_diff_freq_prodlike. Prodlike-tier
# confirmation of the mini PASS (val_acc +0.0417 on combined arm). 2 arms ×
# 1 seed; ~12-22 hr total wallclock.
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

# Pre-flight: prove every encoder A/B branch actually fires on a
# representative SID before spending hours on training. We learned the
# hard way (fuzzy_loop_ab 2026-05-10) that a silently-disabled branch
# produces a uninformative A/B that looks like a real result.
if ! preframr_experiments/validate_branches.sh > "${WORK_ROOT}/validate_branches.log" 2>&1; then
    echo "==== batch ABORTED: validate_branches failed ===="
    cat "${WORK_ROOT}/validate_branches.log"
    exit 1
fi

SPECS=(
    "voice_traj_distributed_set_diff_freq_prodlike"
)

{
echo "==== batch started $(date -Iseconds) host=$(hostname) ===="
for spec in "${SPECS[@]}"; do
    echo "==== ${spec} starting $(date -Iseconds) ===="
    if python3 -m preframr_experiments.run "${spec}" --root "${WORK_ROOT}"; then
        echo "==== ${spec} done $(date -Iseconds) ===="
    else
        echo "==== ${spec} FAILED rc=$? $(date -Iseconds) ===="
    fi
done
echo "==== batch finished $(date -Iseconds) ===="
} > "${LOG}" 2>&1

touch "${DONE_MARKER}"
