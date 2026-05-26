# HVSC version pinning enforcement — design note

Framework follow-up from AGENTS.md §Framework follow-ups: startup
check on HVSC corpus version mismatch. Today each tier's pinned
data list is committed alongside a `HVSC_VERSION` file
(`integration_tests/data/{mini,canonical,prodlike}/HVSC_VERSION`)
but nothing enforces that the dump cache the runner reads from
actually corresponds to that version. Operator drift (upgrading
HVSC without re-pinning) is silent.

Impact bound: the `engine_fingerprint_evalb` audit's leak-audit
baseline (`leak_audit.json`) was computed against HVSC v84; a v85
upgrade can perturb cross-set fingerprint overlap without
re-running the audit, invalidating downstream decisions about
eval-B membership.

## Status today

```
integration_tests/data/mini/HVSC_VERSION       -> "84"
integration_tests/data/canonical/HVSC_VERSION  -> "84"
integration_tests/data/prodlike/HVSC_VERSION   -> "84"
```

Dump cache at `/scratch/preframr/training-dumps/` carries
`MUSICIANS/<L>/<Composer>/<SID>.dump.parquet` files. The cache
doesn't currently record which HVSC version produced it.

HVSC tree at `/scratch/preframr/hvsc/` carries
`DOCUMENTS/HVSC.txt` whose header contains the canonical version
declaration:

```
                                  Release 84

                               December 25, 2025
```

Other version signals:
- Highest-numbered `DOCUMENTS/Update*.hvs` file (`Update84.hvs`
  for v84).
- `readme.1st` (the file `audit_engine_families.py` currently
  reads via a generic `[0-9]+` regex; that regex returned `#2`
  on v84 — wrong, picked the bullet number "1. ... (2) ...").
  The audit JSON's `hvsc_version: "#2"` is the symptom.

## Required behaviour

At runner startup (before `preflight_check`):

1. Read `data/<tier>/HVSC_VERSION` for the spec's tier.
2. Read the actual HVSC tree's version (parser below).
3. If they disagree, abort with a clear error.

Optional: stamp the dump cache itself with the version at
build-time so cache-version mismatch is independently checkable
(adds the cache as a fifth signal source).

## Version parser

Single source of truth: `DOCUMENTS/HVSC.txt` header line matching
`^\s*Release\s+(\d+)\s*$`.

Fallbacks if `HVSC.txt` is missing or malformed (informational
only — never the primary signal):
- Highest-numbered `DOCUMENTS/Update*.hvs` filename. Regex
  `Update(\d+)\.hvs$`.
- README scrape (current `audit_engine_families` approach,
  brittle, deprecated).

Both fallbacks should re-derive the same integer; mismatch
between them is itself worth flagging.

## Implementation

### `integration_tests/profile/hvsc_version_check.py`

Standalone probe + library. Used at runner startup AND can be
invoked manually for diagnosis.

```python
#!/usr/bin/env python3
"""Read the HVSC release version from a checked-out HVSC tree.

Primary: DOCUMENTS/HVSC.txt header "Release NN".
Fallback (informational): max(Update<NN>.hvs) under DOCUMENTS/.

CLI:
  python -m integration_tests.profile.hvsc_version_check \\
      --hvsc-root /scratch/preframr/hvsc
  -> exits 0 with version on stdout; non-zero if parse fails.

  python -m integration_tests.profile.hvsc_version_check \\
      --hvsc-root /scratch/preframr/hvsc \\
      --expected 84
  -> exits 0 if match, 1 if mismatch (with diagnostic).

Library:
  from integration_tests.profile.hvsc_version_check import (
      read_hvsc_version, HvscVersionMismatch,
  )
  version = read_hvsc_version(Path("/scratch/preframr/hvsc"))
  # raises HvscVersionMismatch if HVSC.txt is unreadable.
"""
```

Two functions:

```python
def read_hvsc_version(hvsc_root: Path) -> int:
    """Parse DOCUMENTS/HVSC.txt header. Raises HvscVersionMismatch
    if file is missing/unparseable. Returns int (e.g. 84)."""

def assert_hvsc_version(
    hvsc_root: Path,
    expected: int,
    logger: logging.Logger | None = None,
) -> None:
    """Raises HvscVersionMismatch if the tree's version differs
    from ``expected``. Logs a confirmation line on match."""
```

### Runner wiring

`integration_tests/experiments/base.py` adds an `hvsc_root` field
to `ExperimentSpec` (default `/scratch/preframr/hvsc`) and a
check at `preflight_check`:

```python
def preflight_check(spec, work_root, logger):
    # NEW: HVSC version gate
    hvsc_root = Path(spec.hvsc_root)
    expected = int(read_tier_hvsc_version_pin(spec.tier))
    try:
        assert_hvsc_version(hvsc_root, expected, logger)
    except HvscVersionMismatch as e:
        raise RuntimeError(
            f"hvsc version mismatch: {e}. Re-pin the tier or update "
            f"the HVSC checkout to match."
        )

    # existing: train_preflight_smoke ...
```

`read_tier_hvsc_version_pin(tier)` reads the appropriate
`data/<tier>/HVSC_VERSION`.

### `audit_engine_families.py` correction

The existing `_hvsc_version` function in
`profile/audit_engine_families.py:276-285` parses `readme.1st`
with a generic regex (returns `#2` on v84). Replace with
`read_hvsc_version(hvsc_root)` from the new module. Re-run the
audit; commit updated `engine_families.json` with correct
`hvsc_version: 84`. This is a no-op for the decision
(version is metadata in the JSON, not used in clustering) but
removes a misleading data point from the audit record.

## Failure modes caught

1. **Operator upgrades HVSC without re-pinning.** v84 → v85;
   dump cache regenerated from v85; pin file still says 84.
   Runner aborts at preflight.
2. **Operator downgrades HVSC (rollback).** Same mechanism.
3. **Dump cache stale vs HVSC tree.** If the cache was built
   from v83 but HVSC tree is v84, the runner reads the v84 pin
   ("84") and compares to the HVSC tree's "84"; passes. The
   stale-cache failure is NOT caught by this check.
   Mitigation: stamp the dump cache too (Phase 2).

## Phase 2 (optional): dump cache stamp

Stamp `/scratch/preframr/training-dumps/HVSC_VERSION` at cache
build time. Runner reads both the pin AND the cache stamp; both
must agree with the live HVSC tree.

Requires updating the cache-build tooling (out-of-tree script,
not in this repo). Defer Phase 2 until the in-tree pin check is
landed and proven.

## Validation strategy

**L0 — unit (`tests/test_hvsc_version_check.py`):**
- Fixture: synthesise a `DOCUMENTS/HVSC.txt` with "Release 84"
  header in a tmpdir. Assert `read_hvsc_version` returns 84.
- Negative: same fixture with "Release ??" or missing file.
  Assert `HvscVersionMismatch` raised with a clear message.

**L1 — integration (`tests/test_runner_hvsc_check.py`):**
- Synthetic spec with `tier="mini"`; tmpdir HVSC tree at v84,
  pin file at "84". Assert preflight passes.
- Same setup but pin file at "85". Assert preflight raises.

**L2 — live tree probe:** run
`python -m integration_tests.profile.hvsc_version_check \
--hvsc-root /scratch/preframr/hvsc --expected 84` against the
production tree. Assert exit 0.

**L3 — audit JSON correction:** re-run `audit_engine_families.py`
post-fix; assert the regenerated `engine_families.json` has
`"hvsc_version": 84` (int) instead of `"#2"`.

## Effort

- `profile/hvsc_version_check.py` (probe + library): **~0.2 day**.
- `read_tier_hvsc_version_pin` + `preflight_check` wiring in
  `base.py`: **~0.1 day**.
- `_hvsc_version` correction in `audit_engine_families.py`:
  **~0.1 day**.
- L0-L3 tests + audit re-run: **~0.3 day**.

Total: **~0.7 day**. Lands after `loop_lookahead_prodlike`
completes (touches `experiments/base.py`).

The standalone probe (`profile/hvsc_version_check.py`) is
runner-independent and could land mid-run without violating the
mid-run-edit rule. Useful as immediate operator tooling.

## Connection to other infra

- **`--resume` design.** Per the sibling resume design, stage
  resume keys include the data-tier list. Adding HVSC version to
  the cache key catches the cross-version-resume failure mode
  (operator upgrades HVSC, runs `--resume`; stage caches keyed on
  the old HVSC artifacts would silently mix old + new data).
- **`engine_fingerprint_evalb` audit.** The audit JSON's
  `hvsc_version` field becomes load-bearing once Phase 2 (cache
  stamp) lands. Today it's informational.
- **`_stage_dumps` composer-subdir layout** (sibling design G):
  orthogonal — HVSC version covers WHAT (corpus version),
  composer-subdir covers HOW (staging mechanism).

## Out of scope

- **Auto-upgrading HVSC.** Operator does the checkout; the runner
  only checks.
- **Multi-HVSC-version coexistence.** A spec pinned to v82 cannot
  run on a v84 tree under this scheme. Acceptable: re-pin the
  spec to v84 if it's worth re-running on the newer corpus.
- **Dump cache rebuild detection.** If the operator re-runs the
  cache build with the same HVSC version but different filter
  parameters, the cache content differs but the version stamp
  matches. Out of scope; cache stamps could carry a build-args
  hash in a future iteration.

## Order of operations

1. Land this design (reviewer pass).
2. Land `profile/hvsc_version_check.py` standalone (no runner
   change). Can land mid-run if desired.
3. Verify against the live tree (L2).
4. Run audit re-run; commit corrected `engine_families.json`.
5. **Wait for `loop_lookahead_prodlike` to complete.**
6. Land `preflight_check` wiring in `base.py` (Phase 1).
7. AGENTS.md update: move §Framework follow-up entry to Resolved.
8. Future: Phase 2 (dump cache stamp) if version-drift incidents
   recur.
