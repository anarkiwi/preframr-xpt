# Lane-demux learnability triage — voice-form (v2 corpus, 2026-06-14)

`learnability_triage` proxies on the canonical train sample, frame-major (current) vs voice-major
event ordering. Song-mode (40 tunes) and window-mode (8192-window, 60 tunes / 326 windows) agree.

## Window-mode (the doc-specified mode; 326 windows, 60 tunes)

| ordering | h_k/frame k=0..4 | induction-copy |
|---|---|---|
| frame-major | 87.8 / 52.0 / 38.9 / 28.8 / 22.3 | 0.9457 |
| voice-major | 83.4 / 53.9 / 39.0 / 27.1 / 19.2 | **0.9469** |

## Song-mode (40 tunes, for reference — over-credits whole-tune copy)
| ordering | h_k/frame k=0..4 | induction-copy |
|---|---|---|
| frame-major | 56.9 / 33.4 / 24.5 / 17.7 / 13.3 | 0.9846 |
| voice-major | 54.2 / 34.6 / 24.5 / 16.5 / 11.4 | 0.9850 |

## Verdict: does NOT clear the gate
The gate (`design/encoding/lane_demux_hypothesis.md`): voice-form wins only if per-frame h_k drops
AND induction-copy rises. **Induction-copy is flat** (0.946->0.947), so the gate fails. h_k drops
only at high memory-depth (k=4 -14%, k=3 -6%) and RISES at k=1 -- a weak, mixed signal. Consistent
with the refuted `sequence_order_normalization` (~5% recovery). The frame-major interleave (M2) is
NOT the binding learnability constraint; induction-copy is already ~0.946 regardless of ordering
(the copy-dominance M4 is corpus-inherent, not interleave-caused). Role-form (the truer target,
+0.294 bits in prior measurements) remains untested but needs the role segmenter, and voice-form's
flat copy tempers expectations. Tool: `/scratch/tmp/lane_demux_triage_win.py` (song-mode `/scratch/tmp/lane_demux_triage.py`).
