# Probe archive (reference only)

Committed verbatim snapshots of one-shot diagnostics that back a documented finding,
kept so the claim stays reproducible after `/scratch/tmp` is cleaned. **Not**
parameterised, **not** tested, **not** imported by the package — many hardcode a path or
import torch. If one becomes a recurring read, promote it to a tested module in the parent
dir (see `../README.md` for the triage table and the promoted tools).

- Anchor validation (back `design/trajectory_anchoring.md`): `gate_anchor_probe.py`,
  `raw_gate_anchor_confirm.py`, `intrinsic_anchor_probe.py`, `freqtraj_interval_probe.py`,
  `interval_from_dataset.py`, `raw_atom_diag.py`, `inspect_frames.py`, `adf_probe.py`.
- Predict-host (backs `design/orin_inference_optimization_design.md`): `perf_probe.py`.
- Refuted-motif regression repro (backs `data/refuted/motif_pass.md`): `motif_v1_hang_repro.py`.
