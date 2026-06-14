# WORK ORDER: audition-from-generation (render an event model's GENERATION to WAV), then delete this file

**Mission:** make `ckpt → generate → WAV` work end-to-end for the event models, so a model's
*generation* can be heard. The render half is DONE and verified; the open piece is that a generalize
model's generation does not currently decode to a renderable stream. Self-contained — assumes zero
prior context. Work on `/scratch` (defroster = GPU; fogbank = CPU/builds). Stack is **v2**:
preframr-tokens 0.51.0 / preframr 0.2.30 / xpt `BASE=0.2.30` / `anarkiwi/preframr:latest`
(`EVENT_FORMAT_VERSION=2`).

## Background (what works, what doesn't)

The event model encodes a SID dump as a frame-resolution atom stream; a checkpoint generates atom
ids; those decode to `(frame, reg, val)` writes; writes render to WAV via reSID.

- **Render half — DONE + verified** (preframr `preframr/inference/event_render.py`, PR #163, merge
  it first or bind-mount). `render_tokens_to_wav(tokenizer, ids, wav)` turns ANY VALID event ids
  into a faithful WAV (verified: a tune's canonical writes → an identical-duration 134.1s WAV;
  `tests/predict/test_event_render.py` green). The one non-obvious piece it solved: absolute cycle
  timing (`writes_to_timed_dump_df`: `clock = frame * frame_cycles`, PAL `frame_cycles=19656`), so
  the standard `RegLogParser` reparse → `prepare_df_for_audio` → `render_to_wav` chain accepts it.
  **Do NOT re-litigate the render path or re-hand-roll timing — it is correct.**
- **predict.py is old-substrate** for this — its `_state_df` keys on `FRAME_REG` and token→(reg,val),
  neither of which exists in the event alphabet. Use the event-native path only.

## NO GPU NEEDED (captured generations provided)

Real generations are saved at **`/scratch/tmp/audition_sample_gen.npz`** (4 prompts from the v2
baseline, constrained decode). Per `i` in 0..3: `prompt_i` (128 atoms), `truth_i` (512 grammatical
ground-truth atoms), `gen_i` (512 model-generated atoms — the ones that fail to decode), `path_i`.
Debug decode/render against these **entirely on CPU**: `truth_i` is a known-good stream (verify the
decode/render fix works on a valid window), `gen_i` is the failing case to fix. The only step that
needs a (brief, ~3 GB) generation is the final end-to-end re-confirm on a fresh ckpt — optional until
the very end, and shareable with the GPU batch or run after it.

## The bug (reproduce this first)

The audition CLI `run_render`/`main` in `event_render.py` (load ckpt → generate → constrained
decode → render) is scaffolded but its generation does not decode. Repro on the v2 baseline (its run
dir has the ckpt + `tokens.csv`/`df-map.csv`/`*.blocks.npy`):

```
V2=/scratch/tmp/preframr_experiments/unigram_atoms_v2/results/generalize/default/seed0
docker run --rm --gpus=all -v /scratch:/scratch -v "$V2":/scratch/preframr \
  -v /scratch/anarkiwi/preframr/preframr/inference/event_render.py:/preframr/inference/event_render.py \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e RUST_MIN_STACK=2000000000 \
  anarkiwi/preframr:latest python3 /preframr/inference/event_render.py \
    --tb-logs /scratch/preframr/tb_logs --token-csv /scratch/preframr/tokens.csv \
    --df-map-csv /scratch/preframr/df-map.csv --reglogs '/scratch/preframr/train/**/*dump.parquet' \
    --predict-set train --no-compile --seq-len 1200 --max-seq-len 1200 \
    --prompt-seq-len 128 --gen-tokens 256 --n-prompts 1 --wav-dir /scratch/tmp/auditions
```

Observed: model loads + generates, then **`no decodable writes (ValueError: expected varint digit at
128)`** — the failure is at index 128, i.e. the prompt/generation boundary (`--prompt-seq-len 128`).
No WAV is written. Constrained decode (the grammar mask) is already enabled in `run_render` via
`precompute_vocab_arrays` and `predict(..., irq=frame_cycles)`, and it did NOT fix this.

## Two hypotheses (the actual work is to confirm which, then fix)

1. **Complete-block decoding.** Decoding must operate on a COMPLETE self-contained block, not a
   truncated window. A prompt = the first 128 non-zero atoms of a `.blocks.npy` row, cut mid-frame.
   `event_gate.decode_tolerant` trims from the END with `min_keep = len(prompt_ids)` (128), so it can
   never trim back into the prompt to reach a complete-frame boundary, and the blocks may carry
   KEYFRAME conditioning prefixes that `ids_to_writes` (strict canonical-stream parser) does not
   expect without `stream.strip_keyframes` first. (See the resolved-log note: "the gate must decode
   COMPLETE self-contained blocks, not truncated windows".)
2. **Constrained-decode prompt priming.** `preframr_tokens/events/constrained.py` `EventStreamState`
   starts FRESH (expects the frame-count header, then per-voice preambles, then frame groups). If the
   constrained decoder does not REPLAY the prompt through the state before sampling the first
   generated token, the mask at the boundary is wrong → the model samples a token the *decoder*
   (`ids_to_writes`) then rejects. `EventStreamState`'s docstring claims it "mirrors the stream
   decoder's field reads exactly" — verify that, and verify priming.

**Decisive first probe (CPU, ~1 min, use the saved `prompt_0`/`truth_0`/`gen_0`):** does the PROMPT
ALONE decode? does `prompt_0 + truth_0` (a real grammatical window) decode? does `prompt_0 + gen_0`?
`from preframr_tokens.events.generate import tokens_to_writes; tokens_to_writes(tokenizer, list(ids))`
(build the events tokenizer with `tkvocab=0` via `events.dataset.make_tokenizer`; no model/GPU).
- Fails at 128 too → it's hypothesis 1 (truncated-window/keyframe decode), independent of generation.
- Decodes fine → it's the generation/priming (hypothesis 2): the model's first generated token is
  invalid because the constrained mask isn't primed from the prompt's grammar state.

## Code map

- Render (done): `preframr/inference/event_render.py` — `render_tokens_to_wav`, `render_writes_to_wav`,
  `writes_to_timed_dump_df`, and the `run_render`/`main` CLI scaffold.
- Generate: `preframr/inference/predict.py` — `Predictor.predict` (unconstrained if `vocab_arrays is
  None`, else `_predict_constrained` which calls `state.mask_logits(logits)`); `run_predict` builds
  `vocab_arrays` via `precompute_vocab_arrays(tokens)` (atoms-only, tkvocab=0) /
  `precompute_subtoken_arrays(tokens, tokenizer)` (BPE), gated on `args.constrained_decode`.
- Grammar mask: `preframr_tokens/events/constrained.py` — `EventStreamState` (`valid_mask`, `push`).
- Decode: `preframr_tokens/events/generate.py` — `tokens_to_writes` → `dataset.ids_to_writes` →
  `stream.decode` (strict parser; `stream.strip_keyframes`, `stream.unit_starts` are siblings).
- Tolerant decode: `preframr/inference/event_gate.py` — `decode_tolerant(tokenizer, ids, min_keep)`,
  `load_prompts`. **`event_gate` works on MEMORIZE models** (gate green, decoded-gen-frac 0.991) —
  the difference there is the model reproduces the grammatical truth; on a GENERALIZE model free
  generation diverges. Study why the gate is happy and the audition is not.

## The fix (likely shape — adapt to what the probe shows)

- If hyp 1: decode the largest COMPLETE-frame prefix of the full stream (strip keyframes if present;
  allow trimming below `prompt_len`; trim back to a frame/unit boundary via `stream.unit_starts`).
  Rendering the prompt's complete prefix + the grammatical generated extension is a valid audition.
- If hyp 2: prime `EventStreamState` by `push`-ing the prompt atoms before the first generated step
  so the mask is correct at the boundary; ensure `_predict_constrained` does this (it may already for
  the gate's prompts but not for these). Confirm the mask and `ids_to_writes` agree at every step
  (add a debug assert that every sampled token is in `valid_mask()`).
- Keep `render_tokens_to_wav` as the render sink — it is correct.

## Done = these gates

1. The repro command writes **≥1 non-empty WAV** to `--wav-dir`, decoding a non-trivial fraction of
   the generation (log `decoded_gen_frac`; aim ≥ the gate's 0.5).
2. Audition the WAV (by ear / spectrally) — it is coherent SID audio, not silence/noise.
3. `tests/predict/test_event_render.py` still green; add a test for the priming/complete-block fix
   (CPU; a synthetic prompt that decodes through a primed state).
4. Re-run on a context-arc ckpt once those land
   (`/scratch/tmp/preframr_experiments/ctx_atoms_v2_sl*/.../checkpoints/best-*.ckpt`).

## Gotchas

- **GPU: NOT needed for the core work** — the saved `audition_sample_gen.npz` lets you do the probe,
  diagnosis, fix, and CPU verification (against `truth_i`/`gen_i`) with no model. Only an optional
  final end-to-end re-confirm needs a brief (~3 GB) generation; defroster runs the context-arc
  overnight batch (~12–16 h from 2026-06-13 05:31 UTC, marker
  `/scratch/tmp/preframr_experiments/ctx_overnight.done`), so share briefly or wait.
- v2 baseline ckpt: `/scratch/tmp/v2_atoms_baseline.ckpt`; run dir (tokens.csv/df-map/blocks):
  `/scratch/tmp/preframr_experiments/unigram_atoms_v2/results/generalize/default/seed0`.
- `event_render.py` lives in preframr PR #163 (merge first, or bind-mount as in the repro).
- `load_model` with `--token-csv` short-circuits the dump glob (no raw dumps needed for generation).
- NFS hygiene per AGENTS.md (fogbank IS the /scratch server; cap pools, no lingering `tail -f`).
- When done, `git rm WORK_ORDER_audition_from_generation.md`, commit + PR per repo conventions.
