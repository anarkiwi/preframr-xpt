# preframr-tokens test hardening — real-pipeline tests + fixture policy

**Status:** Pending impl (the real-pipeline harness, retargeted 2026-06-12 to the event codec). The
original 2026-05-29 backlog is OBE (items landed or dissolved with the macro zoo / generator
pipeline; history in git). What survives is the **testing discipline**, which applies to any
pipeline. All paths under `/scratch/anarkiwi/preframr-tokens/`.

## The false-green trap (why real-pipeline tests)

Synthetic-df unit tests that feed a hand-built frame to one component bypass the real input path
(register settling, lo/hi combining, frame derivation) and **ship false greens** — the canonical
case: a pass reading a derived column instead of the raw 16-bit freq was green in unit tests while
doing nothing on real tunes. The fix: assert through the **full real pipeline** — for v3 that is
dump df → `oracle.ordered_writes` → `stream.encode(verify=True)` → `decode` — never a hand-built
intermediate.

## The one live item: real-pipeline structural + balance tests

A `tests/test_event_pipeline_smoke.py` driving the real codec end-to-end:

1. **`_synthetic_dump()` helper** — a raw dump df (`clock, irq, chipno, reg, val`) with separate
   lo+hi freq byte writes, per-frame writes, and on voice 0: held notes, an octave arp, a vibrato,
   a slide, gate on/offs, plus PW and a global write. Deterministic, a few hundred rows.
2. **Atom-mix assertions** — encode and assert the expected event kinds actually appear
   (`NI_STEP`/`NI_RAMP` for the melody, `FD_*` for the vibrato, `FLD_NOTE_ON` per gate-on, `PW_*`,
   `G_*`) — the event-model analogue of the op-mix check that would have caught the no-op pass.
3. **Round-trip** — `decode(encode(ow)) == canonical_writes(ow)` on the synthetic dump (the encode
   self-verify makes this structural; assert it fires).
4. **Balance assertion** — guard channel-drowning: `count_kind_a / max(count_kind_b,1) <=
   BALANCE_MAX` at CI.
5. **Real-tune cross-check** — the fidelity round-trip on a cached, untracked driver dump from
   `sid_fixtures.ensure_dumps`. **Regenerate-or-fail, never skip.**

Prove the tests bite: they must FAIL when run against a known-broken codec revision.

## Fixture policy (HARD CONSTRAINTS)

- **No tunes in the repo** — never commit `.sid` or real `.dump.parquet` (copyrighted HVSC). Real
  fixtures are cached locally, regenerated on demand (`tests/sid_fixtures.py` → headlessvice render
  → `$PREFRAMR_SID_FIXTURE_CACHE`).
- **No skipping on missing fixtures** — a silent skip is the same false-green; regenerate, and FAIL
  loudly if regeneration is impossible.
- **Split the suite:** synthetic core (copyright-free, deterministic, always runs in the docker
  gate) carries the structural/balance assertions; real-tune fidelity is a second layer that runs
  where the fixture cache is mounted.

Legacy note: the parse-domain pass framework (Pass + Decoder + Transform triple, `DECODERS` decode
tripwire, flag registry) still exists for audits/constrained-decode; its discipline notes live in
git history of this file. Tokens lint: ≤5-line docstrings, no non-directive `#` comments.
