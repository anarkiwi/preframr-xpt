# orinnx_audition — scoping note

## Status (2026-05-11)

Implemented and verified.

- `integration_tests/profile/audition_arm_wavs.sh` parametrised with
  `IMAGE`, `GPU_FLAGS`, `COMPILE_FLAGS`, `INDUCTOR_CACHE_DIR`,
  `PROMPT_SEQ_LEN`, `MAX_SEQ_LEN`.
- `integration_tests/profile/audition_arm_wavs_orinnx.sh` ssh-wraps
  the above with Jetson-shaped flags (`anarkiwi/preframr-jetson`,
  `--runtime=nvidia`, `--no-max-autotune` retained so compile +
  Triton kernels run with FX-graph cache; PROMPT_SEQ_LEN=512,
  MAX_SEQ_LEN=1024).
- `integration_tests/run_orinnx_render_smoke.sh` is the build-tier
  test entry: one render per arm, PASS = any non-empty WAV. Listed
  under AGENTS.md wallclock anchors at ~10 min.
- `preframr/predict.py` patches that landed alongside:
  - n_vocab > 16384 skips the multiclass_accuracy compute (which
    needs num_classes^2 ints; 137 GB @ tkvocab=131072 instantly
    OOM-kills 16 GB unified-memory hosts).
  - prepare_df_for_audio bisects on invalid model macro emissions
    to recover the largest renderable prefix.

Verified 2026-05-11: 2 wavs from fuzzy_loop_ab (best fz_on seed +
best fz_off seed) in 570 s total (~285 s / arm) with the FX-graph
cache warm.

## Goal

Make `integration_tests/profile/audition_arm_wavs.sh` runnable on the
Orin NX so audition WAVs can be rendered from a trained arm's
checkpoint without occupying the RTX 4090 (which is the overnight
batch's primary tenant). Same code path as the x86 render; only the
docker invocation needs Jetson-shaped flags.

## What we have

- `orinnx-wifi` reachable via SSH from defroster; user is the same
  unprivileged account; `docker` runnable without sudo.
- `/scratch` is NFS-mounted from defroster (192.168.5.2) onto
  `/scratch` on orinnx. Work dirs (`/scratch/tmp/preframr_experiments/results/<spec>/<arm>/seed*/`)
  and the audition output dir (`/scratch/tmp/preframr_wavs/`) are the
  same paths on both hosts — no path translation needed.
- `anarkiwi/preframr-jetson:latest` already built (14.5 GB,
  jetson-triton v2.11.0 base).
- Orin GPU visible to docker via `--runtime=nvidia` (verified —
  `nvidia-smi -L` inside the container reports `Orin (nvgpu)`).
- 15.6 GB unified memory (RAM + VRAM shared); ~14 GB free at idle.
- `predict-nv.sh` already auto-detects `Orin` from `nvidia-smi` and
  switches image to `anarkiwi/preframr-jetson` and flags to
  `--runtime=nvidia`. Pattern to copy.
- `predict.py` writes WAV via `scipy.io.wavfile.write`; no audio
  device needed for offline render (`/dev/snd` only matters for live
  playback through `render-jetson.sh`).

## What needs changing

1. **Parametrise `audition_arm_wavs.sh`** for image + GPU flags. Today
   it hardcodes:
   ```
   docker run --rm \
       -v "${REPO_ROOT}/preframr":/preframr \
       -v "${seed_work}":/scratch/preframr \
       -v "${OUT_DIR}":/wavs \
       --gpus=all \
       "${IMAGE}" \
       /preframr/predict.py ...
   ```
   Replace `--gpus=all` with `${GPU_FLAGS:-"--gpus=all"}` and keep
   `IMAGE` as the existing env-overridable variable. Add `--no-compile`
   to the predict invocation (already present; Inductor OOMs the
   Orin Nano per AGENTS.md, and is unnecessary overhead for an
   audition on Orin NX).

2. **Add a thin orinnx wrapper** at
   `integration_tests/profile/audition_arm_wavs_orinnx.sh` that
   SSHs into the Jetson and runs the generic script with the right
   env:
   ```bash
   ssh orinnx-wifi \
       IMAGE=anarkiwi/preframr-jetson \
       GPU_FLAGS=--runtime=nvidia \
       REPO_ROOT=/scratch/anarkiwi/preframr \
       bash /scratch/anarkiwi/preframr/integration_tests/profile/audition_arm_wavs.sh \
       <spec_root> <out_dir> [N]
   ```
   The repo dir is on the NFS share, so the script body and the
   repo's `preframr/` source (mounted into the container) are the
   same content the x86 path uses.

3. **No predict.py changes** beyond what landed for the x86 audition
   (multiclass_accuracy try/except for the 137 GB bincount; bisect
   on invalid macro emission to recover a partial WAV). Both fixes
   are in `preframr/predict.py` and live via the `-v
   ${REPO_ROOT}/preframr:/preframr` mount, which works identically
   on orinnx.

## Risks / open questions

- **KV cache fit at full context.** `--seq-len 8192 --tkvocab 131072`
  was the training shape. AGENTS.md flags this as the standing
  full-context blocker: *"Smaller tkvocab is the cleanest unblocker
  for full-context Jetson predict."* For the audition path
  specifically, we can dodge this two ways:
  - Hold `seq-len 8192` but reduce `--max-seq-len` (the generation
    cap) — predict.py's `setup_caches(decoder_max_seq_len=args.max_seq_len)`
    sizes the KV cache from that. `--max-seq-len 4096` halves the
    cache and is plenty for a 2048-prompt + 2048-generated audition.
  - Accept that full-context render may OOM; surface this as a
    smoke-test result, not a design blocker.
- **Wallclock.** Orin NX FP16 ≈ 30 TFLOPS vs RTX 4090 ≈ 330 TFLOPS
  (~10×). Plus the audition's `--no-compile` overhead. Expect per-
  render wallclock of ~30–60 min on Orin NX at the current config
  (vs ~5 min on the 4090 we just measured). With `--max-seq-len 4096`
  that drops to ~10–20 min. Reasonable for a one-off audition; do not
  expect to render six in a session.
- **Same-NFS write contention.** Both hosts may write to
  `/scratch/tmp/preframr_wavs/` simultaneously if a future workflow
  runs them in parallel. Out-of-scope for this scoping note; the
  current shape is sequential.
- **Quantization.** Not in scope here; predict.py's
  `--model-precision bfloat16` should fit. A future follow-up may
  add `int8` for tighter VRAM budget at full tkvocab.

## Smoke-test recipe (manual, pre-script)

Before plumbing the wrapper, validate end-to-end with one render:

```bash
ssh orinnx-wifi 'docker run --rm \
    --runtime=nvidia \
    -v /scratch/anarkiwi/preframr/preframr:/preframr \
    -v /scratch/tmp/preframr_experiments/results/fuzzy_loop_ab/fz_on/seed0:/scratch/preframr \
    -v /scratch/tmp/preframr_wavs:/wavs \
    anarkiwi/preframr-jetson \
    /preframr/predict.py \
        --no-require-pq --no-max-autotune --no-compile \
        --seq-len 8192 --max-seq-len 4096 --tkvocab 131072 --max-perm 2 \
        --df-map-csv /scratch/preframr/df-map.csv \
        --reglogs "/scratch/preframr/train/*.dump.parquet" \
        --eval-reglogs "/scratch/preframr/eval/*.dump.parquet" \
        --dataset-csv /scratch/preframr/dataset.csv.zst \
        --token-csv /scratch/preframr/tokens.csv \
        --tb-logs /scratch/preframr/tb_logs \
        --predict-set val --start-seq 0 --start-block 0 \
        --constrained-decode --temperature 0.8 --top-k 50 \
        --wav /wavs/orinnx_smoke.wav'
```

Records: load wallclock, generation wallclock, peak unified memory
(`free -h` snapshot mid-run), final WAV size, whether the bisect
patch fired. If the smoke test passes, plumb the parametrised
script + wrapper; if it OOMs, drop `--max-seq-len` to 3072 or 2048
and re-try before changing model precision.

## Effort

- Parametrise audition_arm_wavs.sh: ~5 min.
- Write orinnx wrapper: ~5 min.
- Smoke test + tune `--max-seq-len`: 20–60 min wallclock (waiting on
  one render).
- AGENTS.md note: ~5 min.

Total scoping → working state: ~1.5 hr including the smoke render.
No model retraining or tkvocab reduction needed for the audition
use case (those remain open for the full-context-prod-predict path
AGENTS.md tracks separately).
