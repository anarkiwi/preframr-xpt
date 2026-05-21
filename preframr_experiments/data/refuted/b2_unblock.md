# `hard_restart_ab` B2 unblock — REFUTED (2026-05-16)

**Status:** refuted at prototype stage; design closed. The B2
unblock implementation is NOT scheduled.

## Hypothesis

`GateMacroPass` swallows the universal SID hard-restart 2-CTRL-
write pair into a single `GATE_REPLAY_OP` token before
`HardRestartPass` can collapse it into `HARD_RESTART_OP`. The
proposed fix (`design/hard_restart_ab_unblock_design.md`, Path B2)
adds a new `GATE_REPLAY_NOCTRL_OP`: when `GateMacroPass` detects
an HR-pair frame, it drops only the AD/SR portion of the bundle
and leaves the CTRL pair literal for `HardRestartPass` to
collapse downstream.

## Refutation

Standalone prototype at
`integration_tests/profile/b2_unblock_prototype.py` (commit
`95df58e`) subclasses `GateMacroPass` with the proposed B2
behaviour and runs both pipelines in lockstep over real SIDs.

Sample: 300+ Hubbard SIDs (subset of mini train), plus per-SID
probes on Whittaker `180.1`, `4_Soccer_Sims_Soccer_Skills.1`,
Hubbard `Commando.1`, `ACE_II.1`, `5_Title_Tunes.5`.

**Result: 0 SIDs where B2 lifts the `HARD_RESTART_OP` emission
count.** On every Hubbard SID tested, `GateMacroPass` fires
hundreds-to-thousands of times but the encoder's HR-pair count is
zero — because the post-`_read_df` data has **0 same-frame HR
pairs to collapse**. On every Whittaker SID tested,
`GateMacroPass` doesn't fire at all but `HardRestartPass` already
emits 300+ `HARD_RESTART_OP` tokens without any unblock.

The "GateMacroPass swallows the pair" hypothesis required both
passes to be in conflict on the same frames. That conflict appears
zero times in the sampled corpus.

## Root cause of the original mis-diagnosis

The 2026-05-11 diagnostic ("0 HARD_RESTART_OPs / 1386 GATE_REPLAY_
OPs on both arms" on Commando.1) was interpreted as
"GateMacroPass swallowed the pair". The actual story is
**absent-from-data**: Commando.1 has zero same-frame HR pairs at
parser input. `hard_restart_layer0_audit._scan_dump` re-confirms
`hr_total=0` for Commando.1.

A second contributing error: the Layer-0 audit reported 118,981
HR sequences on mini train (the "GO" headline) but conflated
same-frame and next-frame pairs. `HardRestartPass` only fires on
same-frame (`frames[i] == frames[i+1]`). Same-frame breakout
landed `e22cc97` 2026-05-16: 63,975 same-frame + 55,006 cross-
frame; recalibrated projection 4.27% (SOFT, expect Hold-not-Flip).

## Evidence

- `integration_tests/design/b2_unblock_prototype_verdict.md` — full
  prototype verdict.
- `integration_tests/profile/b2_unblock_prototype.py` — re-runnable
  prototype (kept around as a diagnostic).
- `integration_tests/profile/hard_restart_layer0_audit.py` — now
  reports same-frame / cross-frame breakout in `totals.hr_same_frame`
  / `hr_next_frame`.

## Do not revisit without

- A SID where the prototype reports `b2_hr > live_hr` (the
  scenario the design assumed exists).
- OR a counter-argument that the prototype is incorrectly
  mirroring `GateMacroPass`'s decision logic (review the walker's
  `on_frame_end` HR-pair detection against `GateMacroPass`'s
  palette-slot lookup).

`hard_restart_ab` Phase 1 A/B is **schedulable today** against
Whittaker `180.1` / `4_Soccer_Sims_Soccer_Skills.1` —
`HardRestartPass` already fires there without any core change.
The B2 design solved a problem that doesn't exist in the sampled
corpus.
