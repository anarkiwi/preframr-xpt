#!/bin/bash
# One-shot status read for the overnight batch.
# Usage: preframr_experiments/check_overnight_batch.sh

WORK_ROOT="/scratch/tmp/preframr_experiments"
LOG="${WORK_ROOT}/overnight_batch.log"
DONE_MARKER="${WORK_ROOT}/overnight_batch.done"
PID_FILE="${WORK_ROOT}/overnight_batch.pid"

echo "==== overnight batch status $(date -Iseconds) ===="

if [ -f "${DONE_MARKER}" ]; then
    echo "STATE: finished ($(stat -c %y "${DONE_MARKER}"))"
elif [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "STATE: running (wrapper pid $(cat "${PID_FILE}"), elapsed $(ps -o etime= -p "$(cat "${PID_FILE}")" | tr -d ' '))"
else
    # Wrapper exited without writing the done marker => crashed or was killed.
    if pgrep -af "run_overnight_batch.sh" >/dev/null; then
        live="$(pgrep -af "bash /scratch/anarkiwi/preframr/preframr_experiments/run_overnight_batch.sh" | awk '{print $1}')"
        echo "STATE: running (rediscovered pid ${live}; pid file stale)"
    else
        echo "STATE: not running, no done marker. Wrapper crashed -- inspect log."
    fi
fi

echo ""
echo "---- per-spec reports landed (results/<name>/report.md) ----"
for d in "${WORK_ROOT}"/results/*/; do
    name="$(basename "${d}")"
    if [ -f "${d}/report.md" ]; then
        echo "  ${name}: report.md present"
    elif [ -d "${d}" ]; then
        seeds_done="$(find "${d}" -name metrics.json 2>/dev/null | wc -l)"
        echo "  ${name}: in progress (${seeds_done} arm-seeds completed)"
    fi
done

echo ""
echo "---- wrapper log tail ----"
tail -30 "${LOG}" 2>/dev/null || echo "(log not found)"

echo ""
echo "---- next actions ----"
if [ -f "${DONE_MARKER}" ]; then
    echo "  - Read each report:  cat ${WORK_ROOT}/results/<spec>/report.md"
    echo "  - Compare baseline noise to inter-arm Delta in each spec."
    echo "  - Log decisions in AGENTS.md Forward-looking work."
else
    echo "  - Re-run this helper for live progress."
    echo "  - Full log: less ${LOG}"
fi
