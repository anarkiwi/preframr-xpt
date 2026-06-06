# Staged tracker round-trip tests (destined for `preframr-tokens/tests/`)

These tests prove the deployed **generator-MDL** encoding losslessly round-trips the register stream
that real **SID-Wizard (SWM)** and **defMON** players emit. They are the §7B Tier-1 / §7.2 cross-driver
tests the generator work order specified but the landing PRs (#62–#68) shipped **without** (the agent
never installed `pysidwizard`/`pydefmon`).

They live here, **outside `preframr-tokens`**, on purpose — written and verified now, to be moved into
`preframr-tokens/tests/` later (see the AGENTS.md follow-up). Nothing here is committed into the tokens
repo yet.

## What it checks

`module → register log → tokenizer(generator_pass) → decode == player output`

- `tracker_render.py` renders a module through its **own verified player** (`pysidwizard.SWMPlayer` /
  `pydefmon.DefmonPlayer`) into a preframr dump (`clock,irq,chipno,reg,val` parquet), framing each tick
  the way a real `$D4xx` logger would (one `irq` per tick, constant PAL period, empty ticks → DELAY gaps;
  multispeed flattened to one-tick-per-frame so every `irq` stays inside the parser's admission window).
- `test_tracker_round_trip.py` parses that log under the deployed default (`full_macros` = `generator_pass`
  + kept passes) with **`parse_audit='raise'`**. The parser fires the fidelity oracle after every pass, so
  completing the parse **is** the byte-exact / same-output guarantee. We never hand-roll a `register_state`
  diff — see `design/verification_and_audits.md` "THE TRAP".

Equivalence = **same output** (the user's definition), enforced by the project's own byte-exact oracle.

## Status (verified 2026-06-06, tokens `main` @ 632f498)

`11 passed, 1 skipped, 1 xfailed` — 6 SWM + 5 defMON forward round-trips lossless. The skip is a
`$0801` BASIC-stub `.prg` the defMON player can't load (fixture provenance, not a round-trip failure).
The xfail is the **reverse** half: `log → SWM → log` (the recompiler in
`design/log_to_swm_recompiler_design.md`) is designed but **not built** — `strict=True` so it flips to a
hard failure the moment the recompiler exists.

## Run it (in place, from the xpt tree)

On fogbank, in the xpt image, with a tokens-main checkout on `PYTHONPATH`:

```
ssh fogbank 'docker run --rm --network host -v /scratch:/scratch \
  -e PYTHONPATH=/scratch/tmp/tokens-main-triage \
  -e TRACKER_RT_MAX_FIXTURES=6 -e TRACKER_RT_NFRAMES=1200 \
  -w /scratch/anarkiwi/preframr-xpt/staging/tokens_tests anarkiwi/preframr-xpt:latest \
  python3 -m pytest test_tracker_round_trip.py -v'
```

`conftest.py` adds the `pysidwizard`/`pydefmon`/`preframr_tokens` source paths only when they aren't
already importable (env-overridable: `PYSIDWIZARD_SRC`, `PYDEFMON_SRC`, `PREFRAMR_TOKENS_SRC`), so an
explicit `PYTHONPATH` always wins.

Env knobs: `TRACKER_RT_MAX_FIXTURES` (per tracker), `TRACKER_RT_NFRAMES` (ticks/render),
`PREFRAMR_SWM_DIR`, `PREFRAMR_DEFMON_DIR`.

## To land in `preframr-tokens/tests/`

1. Add `pysidwizard` + `pydefmon` as **test-only** deps (the tokens runtime stays torch-free /
   tracker-free).
2. Drop `conftest.py`'s source-path shim (deps are now installed) and move
   `tracker_render.py` + `test_tracker_round_trip.py` under `tests/`.
3. **Provision fixtures** that travel: reuse `pysidwizard.tests._swm_cache` (fetches the 4 verified
   SID-Wizard 1.94 example SWMs by SHA-256) + `pydefmon`'s bundled `build/fixtures/*.prg` (skip the
   non-`$1800` packed ones). This matches xpt's "fixtures provisioned locally, untracked" convention —
   no SID binaries in git.
4. The forward tests gate in CI; keep the reverse `xfail` as the recompiler's tracking test.
