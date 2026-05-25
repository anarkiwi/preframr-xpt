# audio_fidelity_helper — design note

## What

Lift the audio-render-and-compare machinery currently buried in
``integration_tests/test_render_parsed_vs_dump.py`` into a shared
helper, ``integration_tests/audio_fidelity.py``, with its own unit
tests. Used by:

- The existing ``test_render_parsed_vs_dump`` regression suite
  (refactored to call into the helper).
- The "audio fidelity smoke" layer of every new macro validation
  (vol_flip_ab / transpose_xframe / palette_pwm / global_instr_ids,
  plus future macros).

## Why

Today, the cadence-aware comparison logic (about 200 lines:
``_estimate_lag``, ``_per_frame_rms``, ``_compare_with_cadence_diagnostic``,
``FRAME_RMS_TOLERANCE``, etc.) is ``_private`` to one test file. Every
macro's audio-fidelity step currently says "render variant, spot-check
vs baseline" -- which is either copy-pasted helpers (drift) or
unimplemented (skipped under time pressure). The validation framework
proposal called for layered checks; this is the missing layer 3 / 4
common substrate.

Concretely: if vol_flip_ab regresses MODE_VOL_REG decoding in a
musically-subtle way (wrong nibble pick on burst replay) and we have
no automated audio comparison, that regression slips through every
existing test -- byte-level encoded form is correct, decoder
round-trip on register stream is correct, only the **rendered audio**
diverges. The current ``_compare_with_cadence_diagnostic`` catches
exactly that class of bug; making it a shared, tested helper is
table stakes for the Phase-3 macros to be safely landed.

## Public API

Two layers. The low-level layer is the unit-testable comparison
primitive; the high-level layer is the macro-validation convenience.

### Low level

```python
from integration_tests.audio_fidelity import (
    compare_renders,
    AudioFidelityResult,
)

result: AudioFidelityResult = compare_renders(
    samples_a, samples_b, sample_rate,
    tolerance=0.05,    # default; per-frame RMS as fraction of peak
)
# result.passed: bool
# result.shape: one of "PASS", "DURATION_MISMATCH", "FRAME_CADENCE_BREAK",
#               "INITIAL_STATE_DIVERGENCE", "DRIFTING_DIVERGENCE",
#               "CONSTANT_DIVERGENCE"
# result.diagnostic: human-readable string (or empty on pass)
# result.worst_frame_rel_rms: float
# result.lag: int
```

``AudioFidelityResult`` is a frozen dataclass. ``compare_renders``
takes two int16 sample arrays + sample rate + tolerance; returns
the result. No I/O, no rendering, no asserts -- this is the
testable core.

### High level

```python
from integration_tests.audio_fidelity import (
    render_df_to_wav,
    assert_dfs_render_equivalent,
)

n_samples, df_audio = render_df_to_wav(df, irq, args, wav_path)

assert_dfs_render_equivalent(
    df_a, df_b, args,        # args carries cents
    tmp_path,                # for the intermediate WAV files
    label_a="baseline", label_b="variant",   # for error messages
    tolerance=0.05,
)
```

``assert_dfs_render_equivalent`` is what macro-fidelity smokes call:
it renders both dfs to disk via the shared render path, reads the
WAVs, computes ``compare_renders``, and raises ``AssertionError`` with
the diagnostic string on fail. One-line call from each macro test.

The convenience layer hides the ``prepare_df_for_audio`` / ``sidq()``
/ ``reg_widths`` plumbing that's currently boilerplate in the existing
test.

### Macro-level convenience (optional, evaluate after layer 1+2)

If the same "parse with arm_a flags, parse with arm_b flags, render,
compare" pattern appears in 3+ macro tests, lift it into a one-liner:

```python
assert_dump_renders_equivalent(
    dump_path,
    args_baseline, args_variant,
    tmp_path,
)
```

This is opt-in based on how much the macro tests actually repeat.
Don't ship up-front; let the second macro test that wants it
extract.

## Where it lives

```
integration_tests/
    audio_fidelity.py            # NEW
    test_audio_fidelity.py       # NEW (unit tests for the helper)
    test_render_parsed_vs_dump.py    # refactored to call into helper
```

``audio_fidelity.py`` imports from ``preframr_audio.audio_driver``,
``preframr.reglogparser``, and stdlib only (numpy, scipy.io.wavfile).
No cyclic-import concerns: this module is leaf-of-leaves.

## Test surface

``test_audio_fidelity.py`` covers each diagnostic shape:

1. **Pass path**. Two identical sample arrays → ``shape=PASS``,
   ``diagnostic=""``. Repeat with sample arrays that differ within
   tolerance (tiny gaussian noise scaled to 1% of peak) → still
   PASS.
2. **DURATION_MISMATCH**. Two arrays of different length →
   ``shape=DURATION_MISMATCH``, diagnostic mentions both lengths.
3. **FRAME_CADENCE_BREAK**. Take an array, shift it by 2 frames
   (insert zeros at the head, truncate the tail to keep equal
   length) → ``shape=FRAME_CADENCE_BREAK``, lag close to +2 frames.
4. **INITIAL_STATE_DIVERGENCE**. Inject a noise burst into the
   first 200 samples of one array, leave the rest matching →
   ``shape=INITIAL_STATE_DIVERGENCE``, head-RMS >> tail-RMS.
5. **DRIFTING_DIVERGENCE**. Inject growing noise (envelope
   linear in stream position) → ``shape=DRIFTING_DIVERGENCE``.
6. **CONSTANT_DIVERGENCE**. Inject uniform noise across the whole
   stream above tolerance → ``shape=CONSTANT_DIVERGENCE``.
7. **Tolerance boundary**. RMS exactly at tolerance → PASS;
   slightly above → fail. Catches off-by-one tolerance bugs.
8. **Empty / degenerate inputs**. Two zero-length arrays;
   single-sample arrays; arrays shorter than one frame window.
   Helper must not crash, must give a sensible PASS or NO-INFO.

These are all synthetic-array tests -- no SID rendering required.
They run in milliseconds and don't need the preframr docker image.
Move them to ``tests/`` (unit tier) since they're pure-numpy.

Plus 2 integration-tier tests that DO render:

9. **Same-df-twice round-trip**. Render the swm_authored fixture
   twice; ``assert_dfs_render_equivalent`` passes. Existing test;
   refactored to use the new helper. Lives under
   ``integration_tests/``.
10. **Parsed-vs-raw-dump round-trip**. The existing
    ``test_render_matches_raw_dump_path`` recast to use
    ``assert_dfs_render_equivalent``. Lives under
    ``integration_tests/``.

## Effort

- Extract helpers + frozen dataclass + docstrings: ~2 hr.
- Unit tests (8 synthetic + 2 integration): ~2 hr.
- Refactor existing test to call helper, verify still passes: ~1 hr.
- Update each macro design note's "Layer 3/4 audio fidelity" to
  point at the helper: ~30 min (mechanical).
- Total: ~half day.

## Open questions

1. **Where do unit tests live**, ``tests/`` or ``integration_tests/``?
   The pure-numpy unit tests are unit-tier; the rendering tests are
   integration-tier. Split as above.
2. **Should the helper own a fixture cache** for "expected good WAV"
   per macro? Caching a known-good baseline WAV per macro means the
   variant rendering can be compared against a stable target, not
   against a fresh baseline render every time. Adds reproducibility
   but bloats the repo with WAV files. Defer to a follow-up; the
   current "render both, compare" pattern is good enough.
3. **Tolerance per failure mode**. The current ``FRAME_RMS_TOLERANCE
   = 0.05`` is global. Should INITIAL_STATE_DIVERGENCE have a
   different threshold than CONSTANT? Probably yes long-term;
   ship with global tolerance, add per-mode tuning if a
   real-world test trips. (KISS.)
4. **Two-render perceptual hash** as a faster pre-check before
   per-frame RMS. Speculative; only worth it if total comparison
   wallclock becomes a problem in CI. Defer.

## Open question: should I implement now?

This scope is small enough (~half day) that the principled answer is
"yes, before the next overnight queues another batch with macros
that need it". But it doesn't block the *current* overnight
(``mini_baseline_seeds`` + ``fuzzy_loop_ab`` + ``fuzzy_fingerprint``
+ ``loop_lookahead``) -- those four don't introduce new ops and
their fidelity smoke is covered by existing tests via the
fastpath-fix regen.

Recommend: queue the current overnight first; build the helper
while it runs; have it ready before any Phase-3 macro with new ops
(vol_flip_ab onwards) starts implementation.
