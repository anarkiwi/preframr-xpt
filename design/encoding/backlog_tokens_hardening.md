# preframr-tokens test hardening — real-pipeline tests + fixture policy

**Status:** Pending impl (the real-pipeline test harness below). The rest of the original
2026-05-29 backlog is **OBE**: the RESID-completeness items (driver-truth fixtures, the
fast-melodic-run gap, the Antony-Crowther/Trap driver, the per-driver→common-ornament
collapse) and the dead-wood removal were either landed or **subsumed by the generator
pipeline** — the per-pass zoo (incl. `SkeletonPass`) is deleted and the encoding is
residual-zero *by construction*, gated by `test_whole_chip_no_singleton_set` (see
[`../references/verification_and_audits.md`](../references/verification_and_audits.md)
property B). What survives is the **testing discipline** below, which applies to any
pipeline. (Item history is in git; the driver mechanism×primitive matrix lives in
[`../references/sid_driver_ornament_reference.md`](../references/sid_driver_ornament_reference.md).)
All paths below are under `/scratch/anarkiwi/preframr-tokens/`.

## The false-green trap (why real-pipeline tests)

Synthetic-df unit tests (`Pass.apply(hand_built_df)`) bypass the real parser's
`_combine_regs` + `_quantize_freq_to_cents` and **ship false greens**: a pass can be a
complete no-op on real data yet pass its hand-built-df test (the canonical case: a pass
that read the cent-indexed `val` instead of the 16-bit `freq_unq` was green in unit tests
while doing nothing on real tunes). Memory `test-through-real-parse`. The fix is to assert
through the **full `RegLogParser.parse()`**, not a hand-built frame.

## The one live item: real-pipeline structural + balance tests

A `tests/test_parse_pipeline_smoke.py` that drives the **real parser** end-to-end:

1. **`_synthetic_dump()` helper** — build a raw dump DataFrame (`clock, irq, chipno, reg,
   val`) that goes through the FULL parser: emit **separate lo+hi byte writes** so
   `_combine_regs` runs, and **per-frame** freq writes. On voice 0, include held notes (a
   melody), an octave arp, a vibrato (±few-cent wobble), a slide (monotone ramp), and a
   couple of raw PW writes. Deterministic, a few hundred rows.
2. **Per-config op-mix assertions** — for each gating flag, parse the synthetic dump with
   that flag on vs off and assert the **op-mix actually changes** in the direction the flag
   claims (this is what would have caught the cent-index no-op). Use the torch-free
   `pipeline_trace` instrumentation (per-stage which-flag-gated-it / did-it-fire / op-mix
   delta) rather than trusting docs. Update the asserted ops to the **current** op set
   (`GEN_*` / `SWEEP_OP` etc.) — the old `op54/op55/op45/op48` (`SKEL`/`ORN`/`FREQ_TRAJ`/
   `FREQ_ONSET`) belonged to the deleted zoo.
3. **Round-trip** — assert register-log equivalence on the synthetic dump via the fidelity
   oracle (`PREFRAMR_PARSE_AUDIT=raise`; same regs/order/delay within `freq_tol` — no
   render; see verification_and_audits.md).
4. **Balance assertion** — guard against channel-drowning (one op-class swamping another)
   with a `count_a / max(count_b,1) <= BALANCE_MAX` check at CI.
5. **Real-tune cross-check (cached, not committed, no-skip)** — run the fidelity oracle on a
   real driver dump from `sid_fixtures.ensure_dumps` (locally cached, untracked). The
   synthetic round-trip (item 3) is the copyright-free always-runs core; this is the
   cross-check against actual driver output. **Regenerate-or-fail, never skip.**

**Gate:** the shared docker gate (below). Prove the tests bite — they must FAIL if reverted
onto a known-broken parser version.

## Fixture policy (HARD CONSTRAINTS)

- **No tunes in the repo.** Never `git add` a `.sid` or a real SID `.dump.parquet` —
  copyrighted HVSC data must NOT be tracked. Real-tune fixtures are **cached locally,
  untracked**, regenerated on demand from HVSC via `tests/sid_fixtures.py` (`ensure_dumps`
  → downloads the `.sid`, renders a dump in the `anarkiwi/headlessvice` image, caches under
  `$PREFRAMR_SID_FIXTURE_CACHE`; slice it small).
- **No skipping on missing fixtures.** A test must NEVER `self.skipTest(...)` because a
  fixture is absent — a silent skip is the same false-green this whole effort fights. The
  test **regenerates** the fixture, and **FAILS loudly** if regeneration is impossible (no
  network / no docker).
- **Split the suite:** the always-runnable core is **synthetic** (generated register
  streams — no copyright, deterministic, never skip) carrying the structural/balance
  assertions in the plain docker gate; **real-tune** fidelity tests are a second layer that
  runs where the fixture cache is mounted (`-v $PREFRAMR_SID_FIXTURE_CACHE:...`), regenerate
  -or-fail, never skip.

## The pass-framework 3-layer model (reference)

An op exists only when **Pass + Decoder + Transform** line up (see
[`../references/tokens_architecture.md`](../references/tokens_architecture.md)):

1. **Pass** — `MacroPass` subclass in `preframr_tokens/macros/<name>_pass.py` (or
   `passes.py`), `GATE_FLAGS={"<flag>"}`; listed in a run list in `macros/__init__.py`
   (`FREQ_BLOCK_PASSES` / `PASSES` / `POST_NORM_PRE_VOICE_PASSES`) and/or called inline in
   `reglogparser.py:RegLogParser.parse()`.
2. **Decoder** — `MacroDecoder` `op_code=<OP>` in `macros/decoders.py`, registered in the
   `DECODERS` tuple. `macros/decode.py:expand_ops` **asserts `DECODERS.get(op) is not
   None`** — a dangling op hard-crashes decode (the tripwire that catches a botched removal).
3. **Transform** — `@register("<name>")` `PassBackedTransform` in `macros/transforms_*.py`,
   tying `OP_CODES`/`LOSS_TIER`/`REQUIRES_ARGS`/`PASS_CLASS`/`DECODER_CLASS`. Flag names
   auto-derive from `GATE_FLAGS`/`REQUIRES_ARGS` via `macros/flag_registry.py`.

Ops + subregs are in `stfconstants.py`; the default pipeline is
`macros/default_pipeline.py:DEFAULT_PIPELINE_SPEC`.

## Shared docker gate

Use the **baked cache image** `anarkiwi/preframr-tokens-test` (deps pre-installed; editable
`--no-deps` is instant and picks up working-tree edits → ~6s/run vs ~90s reinstalling):

```
docker run --rm -v "$PWD":/src -v /scratch:/scratch -w /src anarkiwi/preframr-tokens-test bash -c "
  git config --global --add safe.directory /src
  pip install -e . --no-deps -q
  black --check preframr_tokens tests && pylint preframr_tokens tests && pyright preframr_tokens \
    && pytest -q --cov=preframr_tokens --cov-report=term-missing --cov-fail-under=85
"
```

Rebuild the cache image only when **deps** change, **through the proxpi mirror** (memory
`docker-build-cache`): on host `defroster` docker can't reach the mirror on the bridge —
use `--network host` + `PIP_OPTS` from `preframr/.env`:
`docker build --network host --build-arg PIP_OPTS="--index-url http://192.168.5.1:5001/index/ --trusted-host 192.168.5.1" -f Dockerfile.tokenstest -t anarkiwi/preframr-tokens-test .`
Tokens lint (`tests/test_lint.py`): ≤5-line one-paragraph docstrings; **no non-directive
`#` comments**. `tests/test_flag_registry.py` fails if a pass reads a boolean arg with no
declaration — relevant when removing passes.
