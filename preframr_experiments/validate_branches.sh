#!/bin/bash
# Pre-overnight branch-firing gate.
#
# For each (spec, validation-SID) pair, parse the SID with the spec's
# baseline arm and its variant arm, then diff the produced parsed
# parquet sizes / op counts. If they're byte-identical, the variant's
# encoder branch did not fire and the A/B would be uninformative --
# we found this the hard way with fuzzy_loop_ab (2026-05-10), where
# the set_fastpath bug silently disabled fuzzy across the entire mini
# corpus.
#
# Exit code 0 = all branches fire; non-zero = abort the overnight.
#
# Usage:
#   preframr_experiments/validate_branches.sh
#
# Each spec validated here MUST be in run_overnight_batch.sh.
# Adding a new encoder-A/B spec means adding a SPEC + arm + flag
# triplet here too.

set -u

REPO=${REPO:-/scratch/anarkiwi/preframr}
DUMPS=${DUMPS:-/scratch/preframr/training-dumps}
WORK=${WORK:-/tmp/preframr_branch_validate}

# (spec, baseline-cargs, variant-cargs, validation-SID-relpath)
SPECS=(
    "fuzzy_loop_ab|--no-fuzzy-loop-pass|--fuzzy-loop-pass|MUSICIANS/G/Goto80/Acid_10000.1.dump.parquet"
    "fuzzy_fingerprint|--fuzzy-loop-pass|--fuzzy-loop-pass --fuzzy-fp-adsr|MUSICIANS/H/Hubbard_Rob/Commando.1.dump.parquet"
    "loop_lookahead|--loop-lookahead 1|--loop-lookahead 3|MUSICIANS/G/Goto80/Truth.1.dump.parquet"
)

# hard_restart_ab entries. Validation SIDs picked by the
# b2_unblock_prototype scan (2026-05-16): Whittaker 180.1 and
# Soccer_Skills.1 both produce hundreds of HARD_RESTART_OP emissions
# on the hr_on arm and zero on hr_off, giving byte-distinct parses.
# The earlier Commando.1 / Arkanoid.1 entries were broken: Commando.1
# has zero HR pairs at parser input (Layer-0 audit hr_total=0), and
# Arkanoid.1 trips the parser's digi-like vol-density skip. Both
# SIDs produced byte-identical or no-rotation outputs that were
# previously mis-attributed to a GateMacroPass swallow. See
# integration_tests/design/b2_unblock_prototype_verdict.md (main repo).
SPECS+=(
    "hard_restart_ab|--no-hard-restart-pass|--hard-restart-pass|MUSICIANS/W/Whittaker_David/180.1.dump.parquet"
    "hard_restart_ab_soccer|--no-hard-restart-pass|--hard-restart-pass|MUSICIANS/W/Whittaker_David/4_Soccer_Sims_Soccer_Skills.1.dump.parquet"
)

# legato_per_cluster: one validation SID per PASS cluster (2/3/4/7).
# Each variant arm enables only its cluster's predicate, so the parse
# should produce byte-distinct output from the legato-off baseline.
SPECS+=(
    "legato_per_cluster_c2||--legato-pass-c2|MUSICIANS/M/Mibri/64_Counts_of_Battery.1.dump.parquet"
    "legato_per_cluster_c4||--legato-pass-c4|MUSICIANS/J/Jammer/80squares.1.dump.parquet"
    "legato_per_cluster_c7||--legato-pass-c7|MUSICIANS/H/Hubbard_Rob/Commando.1.dump.parquet"
)

# preset_pass: --preset-pass snaps wide-val plain SETs on regs 2/9/16
# (PW_LO, grid=128) and reg 21 (FC packed, grid=256) to per-reg preset
# tables. Hubbard_Rob/Commando.1 exercises both branches per the commit
# spot-check (PWM_PRESET=4766 emissions with reg=2 SET tail draining
# from 2279 -> 0 under --preset-pass).
SPECS+=(
    "preset_pass|--no-preset-pass|--preset-pass|MUSICIANS/H/Hubbard_Rob/Commando.1.dump.parquet"
)

# voice_canonical_block_order: per-frame voice-block reorder + PERM_REG
# atom. Variant arm enables the pass; baseline disables it. The pass
# fires on any frame whose voices have non-equal sort keys (gate /
# waveform / pitch). Commando.1 has multi-voice frames throughout so
# the parsed parquet differs from the baseline.
SPECS+=(
    "voice_canonical_block_order|--no-voice-canonical-block-order|--voice-canonical-block-order|MUSICIANS/H/Hubbard_Rob/Commando.1.dump.parquet"
)

SPECS+=(
    "ctrl_bigram_pass|--no-ctrl-bigram-pass|--ctrl-bigram-pass|MUSICIANS/H/Hubbard_Rob/Commando.1.dump.parquet"
)

# voice_trajectory_pass: inserts VOICE_TRAJ_REG annotations after each
# VOICE_REG marker (see design/voice_trajectory_design.md). Baseline
# disables; variant enables with default K=8 window. Commando.1 is a
# multi-voice tune; the trajectory annotations differ row-by-row from
# the baseline.
SPECS+=(
    "voice_trajectory_pass|--no-voice-trajectory-pass|--voice-trajectory-pass|MUSICIANS/H/Hubbard_Rob/Commando.1.dump.parquet"
)

# set_to_diff_pass: converts bare SETs into signed-delta DIFFs after the
# first per-(voice, reg) anchor. Variant enables; baseline disables.
# Commando.1 has thousands of bare SETs throughout that get rewritten
# to DIFFs, so the parsed parquet bytes diverge from baseline.
SPECS+=(
    "set_to_diff_pass|--no-set-to-diff-pass|--set-to-diff-pass|MUSICIANS/H/Hubbard_Rob/Commando.1.dump.parquet"
)

rm -rf "${WORK}"
mkdir -p "${WORK}"

failed=0
for entry in "${SPECS[@]}"; do
    IFS='|' read -r spec base_cargs var_cargs sid <<< "${entry}"
    src="${DUMPS}/${sid}"
    if [ ! -f "${src}" ]; then
        echo "[${spec}] SKIP (validation SID missing: ${sid})"
        continue
    fi
    sid_base=$(basename "${sid}")
    # Preserve the composer dir so engine_fp_cluster (composer ->
    # cluster via composer_from_dump_path -> parent dir name)
    # resolves correctly inside the container. A flat staging dir
    # collapses the composer to ``preframr`` and routes every SID to
    # UNKNOWN_CLUSTER, which silently breaks any cluster-gated
    # encoder pass (e.g. LegatoPerClusterPass).
    composer=$(basename "$(dirname "${sid}")")
    base_dir="${WORK}/${spec}/base"
    var_dir="${WORK}/${spec}/var"
    mkdir -p "${base_dir}/${composer}" "${var_dir}/${composer}"
    cp "${src}" "${base_dir}/${composer}/"
    cp "${src}" "${var_dir}/${composer}/"

    docker run --rm \
        -v "${base_dir}":/scratch/preframr \
        anarkiwi/preframr \
        /preframr/parse.py --no-require-pq --max-files 1 \
            ${base_cargs} \
            --reglogs '/scratch/preframr/*/*.dump.parquet' > "${base_dir}/parse.log" 2>&1 &
    base_pid=$!
    docker run --rm \
        -v "${var_dir}":/scratch/preframr \
        anarkiwi/preframr \
        /preframr/parse.py --no-require-pq --max-files 1 \
            ${var_cargs} \
            --reglogs '/scratch/preframr/*/*.dump.parquet' > "${var_dir}/parse.log" 2>&1 &
    var_pid=$!
    wait "${base_pid}" "${var_pid}"

    # Containers wrote artefacts as root; chown back to the host
    # user so the post-run ``rm -rf "${WORK}"`` on the next gate
    # invocation doesn't trip EPERM.
    docker run --rm \
        -v "${base_dir}":/work \
        anarkiwi/preframr chown -R "$(id -u):$(id -g)" /work \
        > /dev/null 2>&1 || true
    docker run --rm \
        -v "${var_dir}":/work \
        anarkiwi/preframr chown -R "$(id -u):$(id -g)" /work \
        > /dev/null 2>&1 || true

    # Compare rotation 0 only (sufficient signal; rotations 1/2 follow
    # the same shape). Parquet now lands under the composer subdir.
    base_pq="${base_dir}/${composer}/${sid_base%.dump.parquet}.0.parquet"
    var_pq="${var_dir}/${composer}/${sid_base%.dump.parquet}.0.parquet"
    if [ ! -f "${base_pq}" ] || [ ! -f "${var_pq}" ]; then
        echo "[${spec}] FAIL (parse produced no rotation 0 parquet)"
        failed=1
        continue
    fi
    if cmp -s "${base_pq}" "${var_pq}"; then
        echo "[${spec}] FAIL: byte-identical parsed parquets -- variant branch did not fire on ${sid}"
        failed=1
    else
        base_sz=$(stat -c %s "${base_pq}")
        var_sz=$(stat -c %s "${var_pq}")
        echo "[${spec}] OK (base=${base_sz} bytes, var=${var_sz} bytes)"
    fi
done

if [ "${failed}" -ne 0 ]; then
    echo "==== validate_branches: FAIL (one or more branches dead; aborting overnight)"
    exit 1
fi
echo "==== validate_branches: all branches fire"
exit 0
