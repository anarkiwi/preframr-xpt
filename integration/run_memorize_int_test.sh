#!/bin/bash
#
# Memorize-back smoke test (build gate). Thin wrapper around
# ``preframr-experiments-run memorize`` that keeps the
# legacy invocation working:
#
#     bash integration_tests/run_memorize_int_test.sh > .../log 2>&1
#
# The spec lives at ``preframr_experiments/memorize.py``;
# data tier (smoke.list) and metrics are pinned there. Adjusting the
# memorize config means editing the spec, not this script.

set -e

ROOT=${ROOT:-/scratch/tmp/preframr_experiments}
SRC_ROOT=${SRC_ROOT:-/scratch/preframr/training-dumps}
PREFRAMR_XPT=${PREFRAMR_XPT:-/scratch/anarkiwi/preframr-xpt}

# build the docker image so the spec runner has an image to run.
./build.sh

PYTHONPATH="${PREFRAMR_XPT}" \
    python3 -m preframr_experiments.run memorize \
        --root "${ROOT}" \
        --src-root "${SRC_ROOT}"
