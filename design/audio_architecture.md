# preframr-audio architecture

SID **audio rendering**, **fidelity comparison**, and **fingerprinting**. Turns a
parsed register-write DataFrame into PCM/WAV via a real SID emulator (resid-fp),
and provides the audio-equivalence oracle the tokenizer's byte-exact tests rely
on. Depends on `preframr-tokens` (for `prepare_df_for_audio`); the framework
(`preframr`) and audits call it to render predictions.

## Soft-dependency design (important)

The public API in `preframr_audio/__init__.py` is **lazy** (PEP 562
`__getattr__`): `import preframr_audio` is cheap and pulls in **neither
`pyresidfp` nor `scipy`**. Those load only when a render-backed symbol is first
*accessed*. So the render-free path — `compare_renders`, `mel_features`,
fingerprinting — works without a SID emulator installed, and `render_*` raises a
clear "pyresidfp not installed" only when actually invoked. Don't break this:
import submodules lazily, keep `_EXPORTS` the single registry of public names.

## Modules

| module | role |
|---|---|
| **`audio_driver.py`** | the render core: `AudioRenderBuffer`, `ResidWorker` (clocks resid per frame → samples), output sinks (`WavSampleSink`, `SampleRing`), `df_to_packets` (df → per-frame register-write packets), and the ASID live drivers (`ResidAlsaDriver`, `AsidMidiDriver`, `AsidServer`) for real-hardware/MIDI playback. `render_to_samples` / `render_to_wav` / `render_per_voice` are the entry points. |
| **`sidwav.py`** | low-level SID helpers: `default_sid()`, `sidq()` (the resid sample quantum / clock), `write_reg(sid, reg, val, reg_widths)`. |
| **`fidelity.py`** | the audio oracle (below): `compare_renders`, `render_df_to_wav`, `dfs_render_equivalent`, `assert_dfs_render_equivalent`, `per_frame_rel_rms`. |
| **`features.py`** | `mel_features`, `spectral_features`, `raw_pcm_features` (render-free spectral reads). |
| **`fingerprint.py`** | `fingerprint_writes` / `fingerprint_batch` / `canonical_scaffold` — engine/driver fingerprints from the write stream. |
| **`batch.py`** | `render_batch`, `verify_equivalent_batch` (corpus-scale render/verify). |
| **`_reg_mappers.py`, `_sid_constants.py`, `live_animator.py`** | internal reg mapping, SID constants, live visualisation. |

## Render pipeline

```
parsed df (literal SETs + FRAME/DELAY markers)
   │  prepare_df_for_audio(df, reg_widths, irq, sidq())   ← lives in preframr-tokens
   ▼  (decodes any remaining macro ops to literal writes; fills reg_widths; validates)
df_audio, reg_widths
   │  df_to_packets(df_audio, reg_widths)                 ← group writes into per-frame packets
   ▼
FramePacket stream → ResidWorker
   │  per frame: apply register writes to a pyresidfp SoundInterfaceDevice,
   ▼  clock it for the frame's cycles, collect samples
PCM samples → WavSampleSink / SampleRing → .wav
```

- **`render_df_to_wav(df, irq, args, wav_path) -> (n_samples, reg_widths)`** is the
  one-call render used everywhere (audits, predict). It calls
  `prepare_df_for_audio` then the sample render. `args.cents` sets the pitch
  tolerance / quantum.
- The model's prediction path (`preframr/inference/predict.py`) builds the audio
  df via `prepare_df_for_audio` and renders with `render_to_wav`; `--predict-dump`
  saves the prediction-window `df_audio` as parquet for re-rendering/scoring.
- **Cross-repo seam:** `prepare_df_for_audio` is in **preframr-tokens**
  (`reglogparser.py`), not here — audio imports it. The decode of macro ops
  (op45/48/…) to literal register SETs happens via the tokens `expand_ops`; audio
  only sees literal writes + markers. So a new tokenizer op needs its tokens-side
  decoder (see tokens `ARCHITECTURE.md`); audio needs no change as long as decode
  yields literal SETs.

## Fidelity oracle

`compare_renders(a, b)` lag-aligns two sample arrays (`_estimate_lag`, ±2048) and
returns the **worst per-frame relative RMS** as `AudioFidelityResult` (tolerance
`FRAME_RMS_TOLERANCE`); `per_frame_rel_rms` exposes the full per-frame curve.
`dfs_render_equivalent(df_a, df_b, args)` / `assert_dfs_render_equivalent` render
two dfs and compare — this is the **audio-level** equivalence check the
tokenizer's byte-exact tests use (alongside the tokens-side per-frame
register-state oracle in `preframr-tokens` `test_full_pipeline_fidelity`). Long
songs drift on absolute RMS; the per-frame/register-state checks are the reliable
signal (audits prefer the register-state read).

## ASID / live playback

`AsidMidiDriver` / `AsidServer` stream register writes to a local resid SID or out
over MIDI sysex (`encode_asid_update` / `decode_asid_update`) for real-time
playback on hardware (e.g. via an ASID-capable device). These are
`# pragma: no cover` (hardware/threaded) and gated on `pyresidfp`.

## Release

PyPI lib (same `release.yml` / `v*` tag model as preframr-tokens; `fallback_version`
tracks the tag). Current floor consumed by preframr: `preframr-audio>=0.5.1`.
