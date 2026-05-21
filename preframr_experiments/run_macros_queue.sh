#!/bin/bash
# 2026-05-16 macros-queue batch.
#
# Runs the three sequential mini-tier macro A/Bs that landed
# fuzzy_loop_ab refuted at training: legato_per_cluster (5 arms ×
# 1 seed) -> hard_restart_ab (2 arms × 2 seeds) -> global_instr_ids
# Phase A rerun (2 arms × 2 seeds, with the canonical-palette
# bind-mount fix in flight). Estimated wallclock ~10-11 hours on a
# single RTX 4090; legato_per_cluster fastest (~75 min), Phase A
# longest (~4-5 h).
#
# Pre-flight gates:
#   * validate_branches.sh must pass (legato_per_cluster + hard_restart_ab
#     entries; fuzzy_fingerprint failure is pre-existing and tolerated
#     because that spec isn't in this queue).
#   * preflight_check inside each spec asserts HVSC version match.
#
# Status: preframr_experiments/check_macros_queue.sh
# Per-spec reports land at ${WORK_ROOT}/results/<spec>/report.md
# Completion marker: ${WORK_ROOT}/macros_queue.done
# Wrapper log: ${WORK_ROOT}/macros_queue.log

set -u

REPO_ROOT="/scratch/anarkiwi/preframr"
WORK_ROOT="/scratch/tmp/preframr_experiments"
LOG="${WORK_ROOT}/macros_queue.log"
DONE_MARKER="${WORK_ROOT}/macros_queue.done"
PID_FILE="${WORK_ROOT}/macros_queue.pid"

mkdir -p "${WORK_ROOT}"
rm -f "${DONE_MARKER}"
echo "$$" > "${PID_FILE}"
cd "${REPO_ROOT}"

# Pre-flight: each spec's validate_branches entries must report
# byte-distinct base/var parquets before any GPU is spent. The script
# returns non-zero on ANY failure; the 2026-05-11 fuzzy_fingerprint
# entry is a pre-existing fail we tolerate (the spec isn't queued).
# Filter for the queue's specs only.
{
    preframr_experiments/validate_branches.sh 2>&1 \
        | tee "${WORK_ROOT}/validate_branches.log"
    rc=${PIPESTATUS[0]}
} > /dev/null 2>&1

queue_specs_failed=$(grep -cE "^\[(legato_per_cluster_c|hard_restart_ab).*\] FAIL" "${WORK_ROOT}/validate_branches.log" || true)
if [ "${queue_specs_failed}" -gt 0 ]; then
    echo "==== batch ABORTED: validate_branches failure in queue spec ===="
    cat "${WORK_ROOT}/validate_branches.log"
    exit 1
fi

SPECS=(
    "legato_per_cluster"
    "hard_restart_ab"
    "global_instr_ids_phase_a"
)

{
echo "==== macros-queue batch started $(date -Iseconds) host=$(hostname) ===="
for spec in "${SPECS[@]}"; do
    echo "==== ${spec} starting $(date -Iseconds) ===="
    spec_start=$(date +%s)
    if python3 -m preframr_experiments.run "${spec}" \
            --root "${WORK_ROOT}"; then
        spec_elapsed=$(($(date +%s) - spec_start))
        echo "==== ${spec} done $(date -Iseconds) (${spec_elapsed}s) ===="
        # Stamp per-spec done marker so check_macros_queue.sh can show
        # incremental progress without parsing the wrapper log.
        touch "${WORK_ROOT}/results/${spec}/done.marker" 2>/dev/null || true
    else
        rc=$?
        echo "==== ${spec} FAILED rc=${rc} $(date -Iseconds) ===="
        # Don't abort on per-spec failure -- subsequent specs are
        # independent and should still run.
    fi
done
echo "==== macros-queue batch finished $(date -Iseconds) ===="
} > "${LOG}" 2>&1

touch "${DONE_MARKER}"
