#!/usr/bin/env bash
# Pre-encode the entire HVSC corpus into in-place .atoms.zst sidecars (the
# tkvocab-independent event atom-id streams), so experiment runs reuse the encode
# instead of re-running stream.encode + its self-verify on every tkvocab change.
# Runs on fogbank by default -- the corpus physically lives there, so it is local
# disk I/O. Re-run after an HVSC upgrade (or bump ATOM_CACHE_VERSION in
# preframr_tokens to invalidate every sidecar).
#
#   preencode_corpus.sh                 # full corpus, anarkiwi/preframr:latest, on fogbank
#   ONLY_MISSING=1 preencode_corpus.sh  # incremental: skip dumps whose sidecar is current
#   HOST= preencode_corpus.sh           # run on the local host (no ssh)
#
# Env: HVSC (/scratch/preframr/hvsc), IMAGE (anarkiwi/preframr:latest),
#      CPUS (48; cap so nfsd keeps serving defroster), WORKERS (=CPUS),
#      HOST (fogbank; empty = local), XPT_SRC (/scratch/anarkiwi/preframr-xpt),
#      ONLY_MISSING (0).
set -euo pipefail

HVSC="${HVSC:-/scratch/preframr/hvsc}"
IMAGE="${IMAGE:-anarkiwi/preframr:latest}"
CPUS="${CPUS:-48}"
WORKERS="${WORKERS:-${CPUS}}"
HOST="${HOST-fogbank}"
XPT_SRC="${XPT_SRC:-/scratch/anarkiwi/preframr-xpt}"
ONLY_MISSING="${ONLY_MISSING:-0}"

extra=""
[ "${ONLY_MISSING}" = "1" ] && extra="--only-missing"

run="docker run --rm --network host --cpus=${CPUS} --memory=64g \
  -v ${HVSC}:/dumps \
  -v ${XPT_SRC}/preframr_experiments:/xpt/preframr_experiments:ro \
  -e PYTHONPATH=/xpt \
  ${IMAGE} python3 -m preframr_experiments.preencode_corpus \
  --reglogs '/dumps/**/*.dump.parquet' --workers ${WORKERS} \
  --failures /dumps/.preencode_failures.txt ${extra}"

echo "pre-encode: HVSC=${HVSC} IMAGE=${IMAGE} CPUS=${CPUS} WORKERS=${WORKERS} HOST=${HOST:-local} only_missing=${ONLY_MISSING}"
if [ -n "${HOST}" ]; then
  ssh "${HOST}" "${run}"
else
  eval "${run}"
fi
