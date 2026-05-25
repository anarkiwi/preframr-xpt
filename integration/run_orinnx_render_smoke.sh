#!/bin/bash
# Orin NX render smoke (build-tier integration test).
#
# Verifies the predict -> WAV chain end-to-end on the Jetson by
# rendering one audition WAV from a finished experiment arm. Pairs
# with the x86 path (audition_arm_wavs.sh) -- same code, smaller
# context window so the unified-memory + compile budget on Orin NX
# fits.
#
# Usage:
#   integration_tests/run_orinnx_render_smoke.sh [SPEC_ROOT]
#
# SPEC_ROOT defaults to the most recently completed arm directory
# under /scratch/tmp/preframr_experiments/results/<spec>/.
#
# Exit 0 = a non-empty WAV landed.
# Exit 1 = orinnx unreachable, no finished arm available, or
#          predict.py / docker failed before any wav write.
#
# Wallclock target: ~5-10 min (one render at PROMPT_SEQ_LEN=512,
# MAX_SEQ_LEN=1024; cached inductor / FX graph assumed warm).

set -u

REPO=${REPO:-/scratch/anarkiwi/preframr}
RESULTS_ROOT=${RESULTS_ROOT:-/scratch/tmp/preframr_experiments/results}
OUT_DIR=${OUT_DIR:-/scratch/tmp/preframr_render_smoke}
ORINNX_HOST=${ORINNX_HOST:-orinnx-wifi}

SPEC_ROOT=${1:-}
if [ -z "${SPEC_ROOT}" ]; then
    # Pick the most recent spec dir that contains a finished arm
    # (i.e. at least one seed* with a best-*-val_loss=*.ckpt).
    SPEC_ROOT=$(find "${RESULTS_ROOT}" -mindepth 4 -maxdepth 4 \
        -path '*/tb_logs/preframr' -type d 2>/dev/null | \
        head -1 | xargs -r -I{} dirname {} | xargs -r -I{} dirname {} | \
        xargs -r -I{} dirname {} | xargs -r -I{} dirname {})
    if [ -z "${SPEC_ROOT}" ]; then
        echo "[orinnx-render-smoke] FAIL: no finished arm under ${RESULTS_ROOT}"
        exit 1
    fi
fi
if [ ! -d "${SPEC_ROOT}" ]; then
    echo "[orinnx-render-smoke] FAIL: SPEC_ROOT not a directory: ${SPEC_ROOT}"
    exit 1
fi
echo "[orinnx-render-smoke] spec_root=${SPEC_ROOT}"

# Probe reachability before the slow path -- fail fast if the Jetson
# is down or the NFS mount isn't symmetric.
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${ORINNX_HOST}" \
        "test -d '${SPEC_ROOT}'" 2>/dev/null; then
    echo "[orinnx-render-smoke] FAIL: ${ORINNX_HOST} unreachable or " \
         "${SPEC_ROOT} not visible via NFS"
    exit 1
fi

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

t0=$(date +%s)
"${REPO}/integration_tests/profile/audition_arm_wavs_orinnx.sh" \
    "${SPEC_ROOT}" "${OUT_DIR}" 1
rc=$?
elapsed=$(( $(date +%s) - t0 ))

# Pass criterion: at least one non-empty WAV exists. The bisect-on-
# invalid patch in predict.py means a small WAV is success (the model
# emitted an invalid macro and we recovered a partial prefix); both
# are valid for "predict -> wav chain works on orinnx" smoke.
wav_count=$(find "${OUT_DIR}" -name '*.wav' -size +0 | wc -l)
if [ "${wav_count}" -ge 1 ]; then
    echo "[orinnx-render-smoke] PASS ${wav_count} wav(s), ${elapsed}s"
    find "${OUT_DIR}" -name '*.wav' -size +0 -printf '  %s bytes %p\n'
    exit 0
fi
echo "[orinnx-render-smoke] FAIL no wav produced (rc=${rc}, ${elapsed}s)"
tail -30 "${OUT_DIR}"/*.log 2>/dev/null | head -60
exit 1
