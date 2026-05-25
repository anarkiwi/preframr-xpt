#!/bin/bash
#
# Generalize harness (build gate). Thin wrapper around
# ``preframr-experiments-run generalize`` that preserves
# the legacy invocation:
#
#     bash integration_tests/run_generalize_int_test.sh > .../log 2>&1
#
# Calibration mode (MIN_VAL_ACC=0) by default; set
# ``GENERALIZE_MIN_VAL_ACC`` to enforce a floor once the canonical
# baseline lands per ``untracked/TODO.md`` (Pipeline test coverage
# holes -> generalize harness §1).
#
# The spec lives at ``preframr_experiments/generalize.py``;
# data tier (canonical/) and metrics are pinned there.

set -e

ROOT=${ROOT:-/scratch/tmp/preframr_experiments}
SRC_ROOT=${SRC_ROOT:-/scratch/preframr/training-dumps}
PREFRAMR_XPT=${PREFRAMR_XPT:-/scratch/anarkiwi/preframr-xpt}

./build.sh

PYTHONPATH="${PREFRAMR_XPT}" \
    python3 -m preframr_experiments.run generalize \
        --root "${ROOT}" \
        --src-root "${SRC_ROOT}"
