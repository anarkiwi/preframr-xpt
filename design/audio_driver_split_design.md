# preframr-audio: split `audio_driver.py` (render core vs live playback)

**Status:** Drafted, pending review. Refactor #2 from the preframr-audio survey
(companion to the landed #1 façade + #3 in-memory equivalence + #4 batch API on
`feat/audio-augmentation-scaling`). Behaviour-preserving module split; no API
change for consumers (the `__init__` façade + a back-compat shim keep every
existing import working).

## Problem

`audio_driver.py` is 1499 LoC mixing two unrelated concerns:

- **Offline render core (~600 LoC, tested):** `AudioRenderBuffer`, `WavSampleSink`,
  `SampleRing`, `ResidWorker`, `FrameOp`/`FramePacket`, `_OpCollector`,
  `df_to_packets`, `_drive`, `render_to_samples`/`render_to_wav`/
  `render_per_voice`. This is what `fidelity`, `fingerprint`, `batch`, and the
  augmentation prebake import.
- **Live playback + ASID/MIDI + CLI (~900 LoC, all `# pragma: no cover`):**
  `ResidAlsaDriver`, `AsidWorker`, `AsidMidiDriver`, `AsidServer`,
  `play_samples`/`play_via_wav`/`play_via_aplay*`, `demo_frame_producer`,
  `encode_asid_update`/`decode_asid_update`, `main`. Pulls soft `alsaaudio` /
  `mido` / socket deps (the three top-level `try:` blocks).

The untested live half drags its optional deps and surface into the import path
of the augmentation/fidelity hot path, and makes the render core harder to read
and maintain. The render core is the only part the corpus-scale augmentation
work touches.

## Approach

- **`render.py`** ← the render core (list above). Import-safe without
  `pyresidfp` (keeps the `_HAVE_RESID` soft guard + raise-at-call); no
  `alsaaudio`/`mido`/socket imports.
- **`live.py`** ← the playback/ASID/MIDI/CLI half, with the `alsaaudio`/`mido`
  soft `try:` blocks. Keeps `main()` (the `python -m preframr_audio.live` CLI).
- **`audio_driver.py`** → thin back-compat re-export shim:
  `from preframr_audio.render import *` (+ the live names), so existing
  `from preframr_audio.audio_driver import render_to_samples` keeps working. The
  `__init__` façade (#1) re-points its `_EXPORTS` render entries at `render`.
- Shared `_sid_constants` / `_reg_mappers` stay where they are (both halves
  import them).

## Why now

The augmentation prebake (`render_batch`/`verify_equivalent_batch`, this branch)
and the fingerprint path import only the render core; isolating it means the
fogbank prebake workers don't import the alsa/midi/socket machinery, and the
~600-LoC core is independently reviewable. It also mirrors the preframr-tokens
0.15.0 public-API tidy-up (one stable surface, internals split by concern).

## Risk + validation

- **Back-compat:** consumers (`main-repo preframr/inference`, `fidelity`,
  `fingerprint`, tests) import `from preframr_audio.audio_driver import …`. The
  shim preserves every name; add a test asserting the legacy import path still
  resolves. The `__init__` façade is the preferred new path.
- **Soft-import behaviour:** `render.py` must stay import-safe without resid
  (pinned by the existing `test_init.py::test_import_is_safe_without_resid` +
  render tests' `importorskip`); `live.py` keeps the `alsaaudio`/`mido` guards.
- **Tests:** `test_audio_driver{,_unit}.py` reference `audio_driver` symbols —
  they pass unchanged through the shim, or re-point to `render`. The full
  201-test suite + ruff/black/pylint gate the refactor; it adds no behaviour.
- Mechanical but large diff; single behaviour-preserving commit. Low risk given
  the test coverage, but touches the most-imported module so worth its own PR.

## References

- preframr-audio survey (this session): #1 façade, #3 `dfs_render_equivalent`,
  #4 `render_batch`/`verify_equivalent_batch` — landed on
  `feat/audio-augmentation-scaling`.
- `melody_transfer_augmentation_design.md` (the prebake consumer of the render
  core).
