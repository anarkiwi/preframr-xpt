# `legato_ab` — REFUTED (2026-05-10) / PARTIALLY FLIPPED (2026-05-16)

**Status (2026-05-10):** refuted at Layer-0 entropy probe. Pass
retained at default-OFF for reproducibility; spec +
validate_branches entry removed.

**Status (2026-05-16):** original single-rule refutation FLIPPED
on 4 of 6 mini clusters by the per-cluster re-probe
(`08597a1`, `legato_layer0_per_cluster_verdict.md` (removed)).
Clusters 2 (Mibri), 3 (Whittaker), 4 (Jammer), 7 (Hubbard) PASS at
fire-rate 16-29% + entropy Δ +0.6 to +1.1 bits. Cluster 1
(DRAX/Crisps) PARTIAL (high fire-rate, +0.020 bits). Cluster 6
(Galway) confirmed REFUTE — Galway's sub-100-cycle CTRL bursts are
hard-restart pairs, not legato (`8090987`,
`galway_sustain_hold_probe_verdict.md` (removed)).

The single-rule refutation below was aggregated over a smoke
corpus dominated by cluster 7 (Hubbard, where it WOULD have
fired under the per-cluster rule but the original probe used the
wrong predicate). Treat the 2026-05-10 finding as "single-rule
refutation stands"; advance to `LEGATO_OP_CLUSTER_<n>` per the
`legato_per_cluster_design.md` ladder for the 4 PASS clusters.

## Hypothesis

Articulated-onset legato transitions could compress into a single
`LEGATO_OP` token whose embedding captures both the destination CTRL
byte and the timing context, lifting `val_acc` by reducing the CTRL
SET byte's entropy distribution.

## Refutation

Layer-0 entropy probe (`profile/legato_entropy.py`) on the smoke
tier:

- 61% CTRL byte mismatches between LEGATO_OP val and baseline CTRL
  SET byte — the proposed encoding doesn't preserve CTRL semantics.
- Mean entropy Δ -0.666 bits: LEGATO_OP val is BROADER than the
  baseline CTRL SET byte (the opposite of what the hypothesis
  required).
- Hubbard ``Commando.1`` fires **zero** LEGATO_OP — the engine
  doesn't write CTRL during legato passages on the canonical
  legato-cohort track. The pass is a no-op on the strongest
  candidate corpus subset.

Net: the LEGATO_OP encoding does not exist in a form that satisfies
the val_acc-primary hypothesis. Re-encoding would require defining
"legato" without a CTRL-write footprint, which the current macro
framework can't observe without source-level engine reverse
engineering.

## Evidence

- `legato_ab_design.md` (removed) §Layer-0 — full
  probe writeup with histograms.
- `integration_tests/profile/legato_entropy.py` — reproducible
  probe script.

## Do not revisit without

- New evidence that the LEGATO_OP boundary is identifiable in
  Hubbard-class engines (e.g. a per-frame fingerprint that
  distinguishes legato transitions from articulated onsets without
  reading CTRL bytes).
- OR a redefinition of `LEGATO_OP` that fires on non-CTRL signals
  (e.g. ADSR replay cadence + frequency continuation).

Both are open research questions; neither has prior art in the
HVSC reverse-engineering literature surveyed for the original
design.
