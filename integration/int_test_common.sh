#!/bin/bash
# Shared helpers for run_*_int_test.sh.
#
# Source this file from a sibling test script:
#     source "$(dirname "$0")/int_test_common.sh"
#
# All tunables are exposed as env-overridable variables so individual
# tests can override before sourcing (memorize uses TRAIN_MAX_MEM=12g
# while generalize wants 16g, etc.). The functions below assume the
# caller has set ROOT and LOG_DIR before invoking them.

# ----- Static defaults -----
LOCAL_HVSC=${LOCAL_HVSC:-/scratch/hvsc}
IMG=${IMG:-anarkiwi/preframr}
LIMITCYCLES=${LIMITCYCLES:-600000000}      # ~10 min @ ~1MHz fallback
PAL_HZ=${PAL_HZ:-985248}
LIMITCYCLES_MARGIN_PCT=${LIMITCYCLES_MARGIN_PCT:-10}
SONGLENGTHS_DB=${SONGLENGTHS_DB:-${LOCAL_HVSC}/DOCUMENTS/Songlengths.md5}

# Container resource limits. Capping host RAM + disabling container
# swap forces the OOM-killer to fire inside the container instead of
# taking the whole machine down (an earlier integration run hung the
# host this way).
DUMP_MAX_MEM=${DUMP_MAX_MEM:-4g}
# Parse runs ``cpu_count`` worker processes, each materialising one
# song's parsed df (largest are tens of MB live, plus pandas /
# pyarrow overhead and palette state). A 4g cap was enough for
# memorize's 4-song corpus but OOM-killed the whole container at
# all-Goto80's 157-song scale. 32g leaves headroom even with 32+
# concurrent workers on rich-palette songs.
PARSE_MAX_MEM=${PARSE_MAX_MEM:-32g}
TRAIN_MAX_MEM=${TRAIN_MAX_MEM:-12g}
# Predict needs more RAM than train: Lightning streams train batches and
# frees them, but predict materialises the block mapper + checkpoint +
# KV cache + torch.compile artifacts simultaneously. Generalize tests
# override this default upward (predict_load is lean but the larger
# alphabet still grows the embedding table). 32g is the memorize-test
# default and stays unchanged.
PREDICT_MAX_MEM=${PREDICT_MAX_MEM:-32g}
SHM_SIZE=${SHM_SIZE:-2g}                   # PyTorch DataLoader workers
LIMITS_DUMP="--memory=${DUMP_MAX_MEM} --memory-swap=${DUMP_MAX_MEM}"
LIMITS_PARSE="--memory=${PARSE_MAX_MEM} --memory-swap=${PARSE_MAX_MEM}"
LIMITS_TRAIN="--memory=${TRAIN_MAX_MEM} --memory-swap=${TRAIN_MAX_MEM} --shm-size=${SHM_SIZE} --oom-kill-disable=false"
LIMITS_PREDICT="--memory=${PREDICT_MAX_MEM} --memory-swap=${PREDICT_MAX_MEM} --shm-size=${SHM_SIZE} --oom-kill-disable=false"

# ----- Helpers -----

# Look up a SID's listed runtime in HVSC's Songlengths.md5 and emit
# the cycle count to feed vsiddump's -limitcycles. The DB format:
#     ; /PATH/TO/SONG.sid
#     md5=M:SS[.mmm] [M:SS[.mmm] ...]
# The first duration is for tune 1 (which vsiddump always uses here).
# Falls back to ${LIMITCYCLES} if the song isn't listed.
song_length_cycles() {
    local hvsc_path="$1"
    if [ ! -f "${SONGLENGTHS_DB}" ]; then
        echo "${LIMITCYCLES}"
        return
    fi
    local target="; /${hvsc_path}"
    local dur
    # Strip CRLF (Songlengths.md5 ships with \r\n) before comparing.
    dur=$(awk -v t="${target}" '
        { sub(/\r$/, "") }
        $0 == t { getline; sub(/\r$/, ""); sub(/^[a-f0-9]+=/, ""); print $1; exit }
    ' "${SONGLENGTHS_DB}")
    if [ -z "${dur}" ]; then
        echo "${LIMITCYCLES}"
        return
    fi
    local minutes seconds
    minutes=${dur%%:*}
    local rest=${dur#*:}
    seconds=${rest%%.*}
    # ``10#`` prefix forces base-10 interpretation; without it bash
    # treats leading-zero strings (e.g. "08" from "1:08") as octal
    # and errors on digits >= 8.
    if [ "${rest}" != "${seconds}" ]; then
        seconds=$((10#${seconds} + 1))
    fi
    local total_sec=$((10#${minutes} * 60 + 10#${seconds}))
    local with_margin=$((total_sec * (100 + LIMITCYCLES_MARGIN_PCT) / 100 + 2))
    echo $((with_margin * PAL_HZ))
}

# Reset and create ROOT + LOG_DIR. Caller sets ROOT and LOG_DIR.
prepare_root() {
    if [[ -d "${ROOT}" ]]; then
        sudo chown -R "$(id -u)" "${ROOT}"
        rm -rf "${ROOT}"
    fi
    mkdir -p "${ROOT}" "${LOG_DIR}"
    echo "logs in ${LOG_DIR}"
}

# Dump one SID via vsiddump. Args:
#   $1 = HVSC-relative path (e.g. MUSICIANS/G/Goto80/Truth.sid)
#   $2 = subdir under ROOT to dump to (empty = ROOT itself)
# The dump container streams into ${ROOT}/${subdir}; the vsiddump
# container always sees /scratch/preframr as that subdir, so the
# generated *.dump.parquet lands next to the source SID.
#
# Looks for a pre-staged dump under ``${TRAINING_DUMPS}/<hvsc_path>``
# first; if present, copies the cached ``*.dump.parquet`` instead of
# re-running vsiddump. Pre-parsed ``.{i}.parquet`` siblings in the
# cache are *not* copied -- the parse stage re-derives those each
# run so changes to the macro pipeline take effect without a
# stale-cache footgun.
dump_one() {
    local sid="$1" subdir="${2:-}"
    local bsid
    bsid=$(basename "${sid}")
    local stem="${bsid%.sid}"
    local sid_dir
    sid_dir=$(dirname "${sid}")
    local dest_dir="${ROOT}"
    local log_label="${bsid}"
    if [ -n "${subdir}" ]; then
        dest_dir="${ROOT}/${subdir}"
        log_label="${subdir}.${bsid}"
        mkdir -p "${dest_dir}"
    fi
    # Cache hit? The pre-staged corpus uses a vsiddump output naming
    # scheme of ``<stem>.<tune>.dump.parquet`` (e.g. Skybox.1.dump.parquet)
    # plus parsed sidecars. ``ls`` glob is enough -- there's at most
    # one dump per (sid, tune).
    local training_dumps="${TRAINING_DUMPS:-/scratch/preframr/training-dumps}"
    local cache_dir="${training_dumps}/${sid_dir}"
    local cached_dump
    cached_dump=$(ls "${cache_dir}/${stem}".*.dump.parquet 2>/dev/null | head -1)
    if [ -n "${cached_dump}" ]; then
        echo "  ${bsid}: cache hit (${cached_dump##*/})"
        cp "${cached_dump}" "${dest_dir}/"
        return 0
    fi
    local localsid="${LOCAL_HVSC}/${sid}"
    local outsid="${dest_dir}/${bsid}"
    if [[ -f "${localsid}" ]]; then
        cp "${localsid}" "${outsid}"
    else
        wget -O"${outsid}" "http://www.hvsc.c64.org/download/C64Music/${sid}"
    fi
    local cycles
    cycles=$(song_length_cycles "${sid}")
    local cycles_sec=$((cycles / PAL_HZ))
    echo "  ${bsid}: -limitcycles ${cycles} (~${cycles_sec}s)"
    docker run --rm ${LIMITS_DUMP} -v "${dest_dir}":/scratch/preframr -t \
        anarkiwi/headlessvice /usr/local/bin/vsiddump.py \
        --dumpdir=/scratch/preframr --sid /scratch/preframr/"${bsid}" \
        -tune 1 -limitcycles "${cycles}" \
        > "${LOG_DIR}/dump.${log_label}.log" 2>&1
}

# Bounded-concurrency dump driver. Replaces the unbounded
# ``for sid in $SIDS; do dump_one & done; wait`` pattern that
# OOM-killed the host once corpus size exceeded ~100 SIDs (each
# dump_one spawns a vsiddump docker container worth several GB
# of address space). Caps in-flight dumps at ``DUMP_WORKERS``
# (default $(nproc)); needs bash 4.3+ for ``wait -n``.
#
# Usage: ``dump_pool train SID1 SID2 ...``
dump_pool() {
    local subdir="$1"
    shift
    local n_workers="${DUMP_WORKERS:-$(nproc)}"
    local active=0
    local sid
    for sid in "$@"; do
        dump_one "${sid}" "${subdir}" &
        active=$((active + 1))
        if [ "${active}" -ge "${n_workers}" ]; then
            # ``wait -n`` returns once any one backgrounded job
            # finishes; bounds the in-flight set without blocking
            # on the slowest dump.
            wait -n
            active=$((active - 1))
        fi
    done
    wait
}

# Run preframr/parse.py over a glob of dump.parquet files. Produces
# the cached ``.{i}.parquet`` per-rotation parsed dfs that downstream
# stages (tokenize / train / predict) consume without re-deriving the
# macro pipeline. Caller supplies the in-container glob (e.g.
# ``/scratch/preframr/train/*.dump.parquet``); ``--reglogs`` accepts
# comma-separated globs too.
do_parse() {
    local reglogs="$1"
    local label="${2:-parse}"
    local extra="${3:-}"
    docker run ${LIMITS_PARSE} --rm --name "preframr-${label}" \
        -v "${ROOT}":/scratch/preframr "${IMG}" \
        /preframr/core/parse.py \
        --no-require-pq --max-files 999999 ${extra} \
        --reglogs "${reglogs}" \
        > "${LOG_DIR}/${label}.log" 2>&1
}

# Run preframr/core/stftokenize.py to build the alphabet, encode blocks to
# ``.{i}.blocks.npy``, write tokens.csv + df-map.csv (with irq /
# n_rotations columns) + reg_widths.json sidecar. Train + predict
# consume those artifacts directly; no re-parse needed.
# Caller supplies CARGS-style flags (seq-len, tkvocab, df-map-csv,
# token-csv, dataset-csv, max-perm, ...) via $1.
do_tokenize() {
    local cargs="$1"
    local reglogs="$2"
    local eval_reglogs="${3:-}"
    local label="${4:-tokenize}"
    local extra=""
    if [ -n "${eval_reglogs}" ]; then
        extra="--eval-reglogs ${eval_reglogs}"
    fi
    docker run ${LIMITS_TRAIN} --rm --name "preframr-${label}" \
        -v "${ROOT}":/scratch/preframr "${IMG}" \
        /preframr/core/stftokenize.py ${cargs} \
        --reglogs "${reglogs}" \
        ${extra} \
        > "${LOG_DIR}/${label}.log" 2>&1
}

# Start the tensorboard sidecar container with mounted tb_logs.
start_tensorboard() {
    docker rm -f tensorboard-test || true
    docker run -v "${ROOT}/tb_logs":/tb_logs --rm --name tensorboard-test \
        -d -p 6006:6006 -ti anarkiwi/tensorboard
}

# Set ``FLAGS`` to "--gpus=all" if any nvidia GPU is visible, else "".
detect_gpu() {
    FLAGS=""
    local nvgpus
    nvgpus=$(nvidia-smi -L 2>/dev/null || true)
    if [[ -n "${nvgpus}" ]]; then
        FLAGS="--gpus=all"
    fi
}
