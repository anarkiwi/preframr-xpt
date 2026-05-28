# Freq passes re-fire inside iter_self_contained_row_blocks

**Status:** LANDED (preframr-tokens 0.29.0 / preframr 0.2.11).
**Discovered:** 2026-05-28 while building `audit/melody_gap_distribution.py`.

## The bug

`iter_self_contained_row_blocks` (preframr-tokens/macros/blocks.py:92) called
`expand_to_literal_form(df)` on every block. That helper does what it says — fully
decompiles every macro (op45 FREQ_TRAJ, op47 FREQ_NUDGE, op48 FREQ_ONSET, op49
RELEASE_UPDATE) back to literal SET rows. Then `run_passes(slice_df)` re-applied only
the macros listed in `PASSES`, which was:

```
PASSES = [PresetPass, GateSlopeShiftPass, Flip2Pass, TransposePass,
          DedupSetPass, DedupSetPass, HardRestartPass, LegatoPerClusterPass,
          CtrlTriplePass, CtrlBigramPass, SubregPass]
```

Notably missing: every freq pass (FreqTrajectoryPass, FreqOnsetPass, FreqNudgePass,
TrajectoryAnchorPass, ReleaseUpdatePass). PW + filter + control re-encoded; **freq
pitch entered the tokenizer as raw op=0 reg=0 SETs**.

The on-disk `.parquet` (output of `RegLogParser.parse`) had freq atoms; the
`.blocks.npy` (encoded for training) had none. Decoded train block op-distribution
on `freq_onset_channel_mini`: `{0: 65.8%, 35: 21.7%, 36: 6.0%, 50: 4.7%, 42: 1.2%}`.
Zero op=45/47/48 across 872k tokens.

Consequence: the entire melody encoding stack (`--trajectory-anchor-pass`,
`--freq-v0-interval`, `--freq-onset-pass`, `--melody-merge-split`) operated at a
layer the encoder strips before tokenization. Every mini A/B was flat on "V0-onset
acc = 0.000" because the levers weren't reaching the layer the model trains on.
The real melody signal lives in op=0 reg=0 SETs (~12% of stream).

## The fix

Introduce `FREQ_BLOCK_PASSES` + `run_freq_block_passes(df, args)` in
preframr-tokens/macros/__init__.py:

```python
FREQ_BLOCK_PASSES = [
    TrajectoryAnchorPass(),
    FreqTrajectoryPass(),
    FreqOnsetPass(),
    PerRegBurstPass(),
    ReleaseUpdatePass(),
]

def run_freq_block_passes(df, args=None):
    for macro_pass in FREQ_BLOCK_PASSES:
        df = macro_pass.apply(df, args=args)
    return df
```

Wire it into `iter_self_contained_row_blocks` **before** `run_passes`:

```python
slice_df = run_freq_block_passes(slice_df, args=args)
block    = run_passes(slice_df, args=args)
```

`parse()` is left untouched — it already runs the freq passes once at lines 894-902
on the whole-song df, and they're kept out of `PASSES` so they don't double-fire
inside the rotation loop (double-firing broke the existing
`test_freq_trajectory_lossless` fidelity test).

## Also: leaked predicate move

`_is_freq_onset_atom` in `preframr/train/model/tier_map.py` (used by
`_build_vocab_onset_weight` for the onset-loss-weight buffer) duplicated logic that
should live in preframr-tokens next to `is_melody_pitch_atom`. Moved to
`preframr_tokens.regtokenizer` as `is_freq_onset_atom`; exported from
`preframr_tokens.__init__`. Preframr now imports it.

## Regression guards

Three new tokens-side tests:

- `tests/test_is_freq_onset_atom.py` — predicate semantics, strict-subset-of-melody invariant.
- `tests/macros/test_freq_passes_in_blocks.py::TestFreqBlockPassesContract`:
  - `test_required_freq_passes_present_in_freq_block_passes` — guards against
    silently dropping a pass from the list.
  - `test_freq_block_passes_not_in_passes_list` — guards against re-adding them to
    `PASSES` and silently breaking `test_freq_trajectory_lossless`.
- `tests/macros/test_freq_passes_in_blocks.py::TestIterSelfContainedRowBlocksPreservesFreqOps`:
  - `test_literal_freq_ramp_yields_freq_traj_atoms` — synthetic 12-frame SET
    ramp on reg=0 → expects op=45 atoms in the output blocks. This is the
    end-to-end invariant: if `run_freq_block_passes` ever stops getting called
    inside `iter_self_contained_row_blocks`, this test fails immediately with
    a clear message naming the bug it guards.

## Versions

- preframr-tokens: 0.28.0 → **0.29.0** (PASSES contract change + new `is_freq_onset_atom` symbol).
- preframr: 0.2.10 → **0.2.11** (floor `preframr-tokens>=0.29.0`; remove leaked predicate).

## Implications for the melody arc

Re-baselining is mandatory: every prior experiment trained on a stream with zero
op=45/47/48 atoms. The encoded stream now contains them, so per-class accuracy at
op=45 / op=48 finally measures what the design assumed. Open question: is the
melody learnability story still scale-bound, or did the structural levers we built
(anchor / interval V0 / onset pass / merge split) actually work — we just couldn't
see them because they had no atoms to operate on?
