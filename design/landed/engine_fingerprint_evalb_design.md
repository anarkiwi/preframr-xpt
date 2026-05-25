# Engine-fingerprint Eval-B expansion

## Why

`prodlike` Eval-B is currently Daglish_Ben (16) + Follin_Tim (16),
both C64-engine-family adjacent (per `prodlike_tier_design.md`
§Risks). The prodlike spec's "cross-composer transfer" decision
rule treats per-Eval-B-* breakouts as evidence of off-distribution
generalisation, but two same-family composers underpower that
claim -- a candidate that lifts Eval-B-Daglish + Eval-B-Follin may
still fail on a structurally-different engine (e.g. Galway's
chord-arpeggio, JCH's tracker-style sequence engine, Hubbard's
gate-based legato).

Without an objective measure of engine-family separation, picking
"more composers for Eval-B" is style-guessing. This note specifies
the fingerprint comparison, the candidate pool, and the decision
rule for the eventual Eval-B re-pin.

## Method

### Engine fingerprint

Reuse and extend the picker's existing prefix fingerprint (first
2000 raw register writes, SHA-256) into a *family* fingerprint:

1. For each candidate SID, extract a feature vector from the first
   K register writes (K target: 2000 -- 8000; tune to whichever
   length captures engine-startup behaviour without overfitting
   to subtune transitions).
2. Features capture *patterns*, not values:
   - Per-register-address write density (24-element histogram).
   - Inter-write delta histogram (log-bucketed; captures
     player-tick cadence).
   - CTRL-byte transition n-grams (n=2, n=3) over voice-0..2 CTRL.
   - Filter-register touch ratio (engines with no filter routine
     have a near-zero column).
3. Normalise + concatenate into a fixed-dim engine vector.

### Family clustering

1. Compute the engine vector across every composer's top-N SIDs
   (post leak-audit filter).
2. Aggregate per-composer (mean engine vector).
3. Pairwise cosine / L2 distance matrix over composers.
4. Hierarchical cluster (Ward linkage) into ~6-8 engine families.
5. Cross-check clusters against known SID-engine documentation
   (existing player attribution lists; HVSC `Documents/Players/`).

### Eval-B candidate selection

Pick one composer per cluster, preferring composers that:

1. Have >= 16 SIDs surviving the 30s duration + leak-audit filters.
2. Are NOT in the current prodlike train set (this is hard:
   25 composers in train; picking representatives outside the
   top-25 may mean small post-filter pools).
3. Have a clean fingerprint distance > median to ALL train
   composers (i.e. the holdout is genuinely off-distribution).

Target: 6 Eval-B subsets x 8-16 SIDs each ~ 48-96 SIDs total
(vs current 32). Smaller per-subset N is acceptable since the
breakout is for direction-of-effect, not sigma-tight estimates.

## Artefacts to produce

1. `integration_tests/profile/engine_fingerprint.py` --
   computes the engine vector for a single SID, dumps to JSON.
2. `integration_tests/profile/audit_engine_families.py` --
   batch-runs `engine_fingerprint.py` over a composer list,
   clusters, dumps distance matrix + cluster assignment + per-
   cluster representative pick.
3. `integration_tests/data/prodlike/engine_families.json` --
   distance matrix + cluster labels committed alongside the
   eval-B lists.
4. Re-pin: `eval-B-<family>.list` x 6 (or whatever count survives
   filtering) replacing the current Daglish + Follin pair.
5. `base.py.resolve_data_layout` extension to fan the new
   `eval-B-<family>` subsets through into per-subset metrics.

## Decision rules

The Eval-B re-pin lands once:

1. The fingerprint clustering produces stable 6-8 cluster
   assignments under bootstrap resampling (cluster membership
   agreement >= 80% over 100 bootstrap draws).
2. At least 5 of the 6-8 clusters have a viable composer
   candidate (>= 16 SIDs surviving filters, > median fingerprint
   distance to train).
3. The leak-audit re-run on the expanded eval-B set produces no
   aborts and < 5 warnings at the 5% n-gram threshold.

Until those land, `loop_lookahead_prodlike` results should be
read with the explicit caveat that "Eval-B" means "two same-family
composers" rather than "cross-engine holdout."

## Scope

In scope:

- Eval-B re-pin only. Train composer list is left untouched
  (re-pinning train would invalidate the leak-audit baseline
  for the loop_lookahead_prodlike result that motivated this
  expansion).
- Single one-shot fingerprint script + cluster audit. Not a
  permanent runtime feature.

Out of scope:

- Per-engine encoder branching (engine-specific macro passes).
  Engine-aware encoding is a separate question that should be
  driven by Phase A of `global_instr_ids_design.md`
  (engine-fingerprint palette), not by Eval-B re-pinning.
- Frontier-tier corpus expansion. The Eval-B re-pin is sized
  for prodlike-scale generalisation testing.

## Wallclock estimate

- Fingerprint script + audit: ~1 day (mostly feature design +
  empirical cluster validation).
- Composer candidate vetting + re-pin: ~0.5 day.
- `base.py.resolve_data_layout` extension + smoke verification:
  ~0.5 day.

Total: ~2 days, fits comfortably under the `loop_lookahead_prodlike`
~36-66 hr run.

## Open questions

- Does the fingerprint need to be subtune-aware (per-subtune
  feature vector aggregated per-SID), or is per-SID enough? The
  picker's prefix fingerprint is per-subtune; this design
  defaults to per-SID-mean for clustering.
- Is the engine vector dimensionality fixed (e.g. 64) or
  variable-length? Fixed is easier to cluster; variable risks
  overfitting to particular feature axes.
- Should the train composer list be re-fingerprinted to check
  that the existing 25 cover at most ~4 of the 6-8 clusters
  (i.e. train is genuinely missing engine families)? If the
  train set already spans 6+ clusters, the Eval-B expansion is
  weaker evidence than expected.
