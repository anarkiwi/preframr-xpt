# preframr-tokens test hardening — real-pipeline tests + fixture policy

**Status:** Partially landed (2026-06-12, with the v3-hardening PR). The real-pipeline round-trip
harness shipped in preframr-tokens as `tests/test_events_corpus.py` — raw dump df → `Corpus.preload`
→ per-dump `.blocks.npy` → `ids_to_writes` == `canonical_writes`, on both the raw-block and
BPE-trained paths — and the 5-driver-fixture + corpus-sample fidelity cross-checks live in
`tests/test_events_roundtrip.py` and `tests/test_events_stream.py`. **Only the dedicated synthetic
atom-mix + balance assertions remain open** (items 2 and 4 below). The original 2026-05-29 backlog is
OBE; the surviving **testing discipline** applies to any pipeline. All paths under
`/scratch/anarkiwi/preframr-tokens/`.

## The false-green trap (why real-pipeline tests)

Synthetic-df unit tests that feed a hand-built frame to one component bypass the real input path
(register settling, lo/hi combining, frame derivation) and **ship false greens** — the canonical
case: a pass reading a derived column instead of the raw 16-bit freq was green in unit tests while
doing nothing on real tunes. The fix: assert through the **full real pipeline** — for v3 that is
dump df → `oracle.ordered_writes` → `stream.encode(verify=True)` → `decode` — never a hand-built
intermediate.

## Remaining live item: synthetic atom-mix + balance assertions

The real-pipeline round-trip (item 3) and the real-tune fidelity cross-check (item 5) **landed** (see
Status). What is still missing is the synthetic-core structural test — a
`tests/test_event_pipeline_smoke.py` (or a fold into `test_events_corpus.py`) carrying the atom-mix
and balance assertions on a rich hand-built dump, the event-model analogue of the op-mix check that
would have caught the no-op pass:

1. **`_synthetic_dump()` helper** — a raw dump df (`clock, irq, chipno, reg, val`) with separate
   lo+hi freq byte writes, per-frame writes, and on voice 0: held notes, an octave arp, a vibrato,
   a slide, gate on/offs, plus PW and a global write. Deterministic, a few hundred rows.
2. **Atom-mix assertions** — encode and assert the expected event kinds actually appear
   (`NI_STEP`/`NI_RAMP` for the melody, `FD_*` for the vibrato, `FLD_NOTE_ON` per gate-on, `PW_*`,
   `G_*`) — the event-model analogue of the op-mix check that would have caught the no-op pass.
3. **Round-trip** — `decode(encode(ow)) == canonical_writes(ow)` on the synthetic dump (the encode
   self-verify makes this structural; assert it fires). **Landed:** `test_events_corpus.py` asserts
   it through the full `Corpus.preload` → `.blocks.npy` → `ids_to_writes` pipeline (raw-block + BPE).
4. **Balance assertion** — guard channel-drowning: `count_kind_a / max(count_kind_b,1) <=
   BALANCE_MAX` at CI.
5. **Real-tune cross-check** — the fidelity round-trip on a cached, untracked driver dump from
   `sid_fixtures.ensure_dumps`. **Regenerate-or-fail, never skip.** **Landed:** the 5-driver-fixture
   + corpus-sample round-trips in `test_events_stream.py` / `test_events_roundtrip.py` — though they
   currently `pytest.skip` when the fixture cache is absent rather than regenerate-or-fail, so
   enforcing that policy is the residual on this item.

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
