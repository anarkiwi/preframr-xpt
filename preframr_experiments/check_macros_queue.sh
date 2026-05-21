#!/bin/bash
# One-shot status read for the macros-queue batch.
# Usage: preframr_experiments/check_macros_queue.sh

WORK_ROOT="/scratch/tmp/preframr_experiments"
LOG="${WORK_ROOT}/macros_queue.log"
DONE_MARKER="${WORK_ROOT}/macros_queue.done"
PID_FILE="${WORK_ROOT}/macros_queue.pid"

SPECS=(legato_per_cluster hard_restart_ab global_instr_ids_phase_a)

echo "==== macros-queue status $(date -Iseconds) ===="

if [ -f "${DONE_MARKER}" ]; then
    echo "STATE: finished ($(stat -c %y "${DONE_MARKER}"))"
elif [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "STATE: running (wrapper pid $(cat "${PID_FILE}"), elapsed $(ps -o etime= -p "$(cat "${PID_FILE}")" | tr -d ' '))"
else
    live="$(pgrep -af 'bash[^ ]* /scratch/anarkiwi/preframr/preframr_experiments/run_macros_queue.sh' | awk '{print $1}' | head -1)"
    if [ -n "${live}" ]; then
        echo "STATE: running (rediscovered pid ${live}; pid file stale)"
    else
        echo "STATE: not running, no done marker. Wrapper crashed -- inspect log."
    fi
fi

echo ""
echo "---- per-spec progress ----"
for spec in "${SPECS[@]}"; do
    d="${WORK_ROOT}/results/${spec}"
    if [ -f "${d}/done.marker" ]; then
        echo "  ${spec}: DONE ($(stat -c %y "${d}/done.marker"))"
    elif [ -d "${d}" ]; then
        seeds_done="$(find "${d}" -name metrics.json 2>/dev/null | wc -l)"
        running_dir="$(find "${d}" -maxdepth 2 -name "seed*" -type d 2>/dev/null | tail -1)"
        echo "  ${spec}: in progress (${seeds_done} arm-seed metrics.json landed)"
        if [ -n "${running_dir}" ]; then
            train_log="${running_dir}/logs/train.log"
            if [ -f "${train_log}" ]; then
                cur="$(tr '\r' '\n' < "${train_log}" | grep -oE 'Epoch [0-9]+:[[:space:]]+[0-9]+%' | tail -1)"
                if [ -n "${cur}" ]; then
                    echo "    latest arm: $(basename "$(dirname "${running_dir}")")/$(basename "${running_dir}") -- ${cur}"
                fi
            fi
        fi
    else
        echo "  ${spec}: queued (no results dir yet)"
    fi
done

echo ""
echo "---- wrapper log tail ----"
tail -20 "${LOG}" 2>/dev/null || echo "(log not found at ${LOG})"

echo ""
echo "---- next actions ----"
if [ -f "${DONE_MARKER}" ]; then
    echo "  - Read each report:  cat ${WORK_ROOT}/results/<spec>/report.md"
    echo "  - Fold verdicts into AGENTS.md Forward-looking work."
else
    echo "  - Re-run this helper for live progress."
    echo "  - Full log: less ${LOG}"
fi
