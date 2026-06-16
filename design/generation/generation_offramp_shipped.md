# Shipped generation path — the free-running off-ramp (constrained decode + Tier-1 caps)

**Status: SHIPPED (2026-06-16).** This is the usable-but-imperfect generation path the free-running
arc landed on after the [remediation ladder](free_running_pathology_remediation_design.md) and
[DAgger triage](dagger_recanonicalization_design.md) concluded that open-ended free-running is **not
fixed** by data (Tier-3, M1 not M4) or by re-canonicalisation (the triage: recanon does not restore
recoverability; the model emits ~99% non-canonical surface). It is an honest deliverable — you can
generate audible, structured-ish SID audio from a trained checkpoint today — not a "fix" for the
free-running pathology.

## What it is

`preframr/inference/event_render.py` — load a checkpoint, continue event-token prompts from
`.blocks.npy`, **grammar-constrained decode** (`EventConstraint`/`EventStreamState`, so every token
keeps the stream renderable) with **Tier-1 anti-collapse caps** (repetition penalty + no-repeat-ngram,
preframr #167), then decode → reSID → WAV. Prompts are snapped to whole-frame `unit_starts` boundaries
(preframr #168/#169) so they decode (the silent-WAV bug is fixed). The audition CLI now **defaults to
the off-ramp settings** (`repetition_penalty 1.3`, `no_repeat_ngram_size 4`, `top_k 8`); override as
needed.

## Run it

```
docker run --rm --gpus=all -v <ckpt-workdir>:/scratch/preframr \
  -v /scratch/anarkiwi/preframr/preframr:/preframr \
  -v /scratch/anarkiwi/preframr-tokens/preframr_tokens:/root/.local/lib/python3.12/site-packages/preframr_tokens \
  anarkiwi/preframr:latest python3 /preframr/inference/event_render.py \
    --tb-logs /scratch/preframr/tb_logs --token-csv /scratch/preframr/tokens.csv \
    --df-map-csv /scratch/preframr/df-map.csv --reglogs '/scratch/preframr/train/**/*dump.parquet' \
    --blocks-glob '/scratch/preframr/eval_b_*/**/*.blocks.npy' \
    --predict-set train --no-compile --seq-len 8192 --max-seq-len 8192 \
    --prompt-seq-len 128 --gen-tokens 512 --n-prompts 3 --wav-dir <out>
```

(The local-source mounts are needed until a preframr image is rebuilt past #167/#168/#169; the tokens
mount supplies `events.constrained` absent from the released 0.51.0.)

## What you get (honest)

Demonstrated on `instrument_full` (2026-06-16): with caps, the best auditions are **sustained and
audible** (e.g. 9.6 s at 0% silent, 36.4 s at 4% silent); the caps suppress the empty-frame drone that
raw free-running falls into (same model, no caps: 32.8 s at 78% silent). **It is inconsistent** — some
prompts still drift to a near-silent drone (one cap'd audition was 85% silent). So: usable for offline
auditions and demos, not for unattended high-quality generation. Quality is bounded by the free-running
pathology (M1 exposure bias), which remains open.

## Knobs

- `--repetition-penalty` (1.3) / `--no-repeat-ngram-size` (4) — the anti-collapse caps; raise the
  penalty / n for more aggressive de-droning, at the cost of more forced novelty.
- `--top-k` (8) / `--temperature` (1.0) — sampling; greedy (`top_k 1`) collapses, so keep some sampling.
- `--gen-tokens`, `--prompt-seq-len`, `--n-prompts`, `--frame-cycles` (PAL 19656).

## Remaining levers (if quality must improve — see the DAgger design)

The off-ramp is the floor, not the ceiling. The one untried model-side bet is **objective 2
(consistency / fixed-point):** train the model so its *own* output is canonical (minimise the
recanon-delta, currently ≈1.0) — motivated by the triage finding, needs no off-distribution re-prompt.
A grammar-aware hard frame/DELAY budget cap in `EventStreamState` (forbid runaway empty frames at the
source rather than via the loop penalty) is the cheapest decode-side improvement still on the table.
