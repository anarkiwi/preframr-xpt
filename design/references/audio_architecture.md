# preframr-audio architecture

**Status:** Pointer (2026-06-12). The full API reference — lazy-import contract, module
table, render pipeline, fidelity gates (strict RMS + perceptual), features/fingerprint/
batch, ASID/live playback, and the SID chip-behavior facts with their pinning tests —
now lives in the **[preframr-audio README](https://github.com/anarkiwi/preframr-audio)**.

xpt-relevant seams (not in the README):

- **Cross-repo seam:** `prepare_df_for_audio` lives in preframr-tokens
  (`reglogparser.py`), not in preframr-audio — audio only ever sees literal writes +
  FRAME/DELAY markers after tokens-side decode. A new tokenizer op needs a tokens-side
  decoder; audio needs no change as long as decode yields literal SETs.
- The model's prediction path (`preframr/inference/predict.py`) builds the audio df via
  `prepare_df_for_audio` and renders with `render_to_wav`; `--predict-dump` saves the
  prediction-window `df_audio` parquet for re-rendering/scoring (see
  [`framework_architecture.md`](framework_architecture.md)).
- Release: PyPI `v*`-tag model, same as preframr-tokens (see
  [`release_build_cache.md`](release_build_cache.md)).
