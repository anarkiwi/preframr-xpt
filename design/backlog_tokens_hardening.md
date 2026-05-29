# Backlog: preframr-tokens hardening — precise implementation instructions

**Status:** QUEUED 2026-05-29, blocked on the ornament-codebook/parametric work landing
(op-code churn must settle first). Three independent items: (#9) dead-wood removal,
(#10) real-pipeline structural/balance tests, (#11) driver-truth RESID-completeness fixtures.
Reference: [`tokens_architecture.md`](tokens_architecture.md) (the pass framework + parse
pipeline) and [`sid_driver_ornament_reference.md`](sid_driver_ornament_reference.md) (driver
mechanics). All paths below are under `/scratch/anarkiwi/preframr-tokens/`.

Shared docker gate — use the **baked cache image** `anarkiwi/preframr-tokens-test` (deps
pre-installed; editable `--no-deps` is instant and picks up working-tree edits → ~6s/run vs
~90s reinstalling). ~the gate:
```
docker run --rm -v "$PWD":/src -v /scratch:/scratch -w /src anarkiwi/preframr-tokens-test bash -c "
  git config --global --add safe.directory /src
  pip install -e . --no-deps -q
  black --check preframr_tokens tests && pylint preframr_tokens tests && pyright preframr_tokens \
    && pytest -q --cov=preframr_tokens --cov-report=term-missing --cov-fail-under=85
"
```
Rebuild the cache image only when **deps** change, and do it **through the proxpi mirror** (see
memory `docker-build-cache`): on host `defroster` docker can't reach the mirror on the bridge —
use `--network host` + `PIP_OPTS` from `preframr/.env`:
`docker build --network host --build-arg PIP_OPTS="--index-url http://192.168.5.1:5001/index/ --trusted-host 192.168.5.1" -f Dockerfile.tokenstest -t anarkiwi/preframr-tokens-test .`
(Same `--network host` + `PIP_OPTS` applies to the `anarkiwi/preframr` image builds — I bypassed
the cache with empty `PIP_OPTS` this session; don't.)
Tokens lint (`tests/test_lint.py`): ≤5-line one-paragraph docstrings; **no non-directive `#`
comments** (only `pylint:`/`noqa`/`type: ignore`/`fmt:`/shebang). `tests/test_flag_registry.py`
fails if a pass reads a boolean arg with no declaration — relevant when removing passes.

---

## ## Fixture policy (HARD CONSTRAINTS — read before #10/#11)
- **No tunes in the repo.** Never `git add` a `.sid` or a real SID `.dump.parquet` — copyrighted
  HVSC data must NOT be tracked (AGENTS: "SID songs must NOT be tracked here"). Real-tune fixtures
  are **cached locally, untracked**, regenerated on demand from HVSC via the
  `tests/sid_fixtures.py` helper (`ensure_dumps` → downloads the `.sid`, renders a dump in the
  `anarkiwi/headlessvice` image, caches under `$PREFRAMR_SID_FIXTURE_CACHE`; reduce/slice it small).
- **No skipping on missing fixtures.** A test must NEVER `self.skipTest(...)` because a fixture is
  absent — a silent skip is the same false-green this whole effort is fighting. Instead: the test
  **regenerates** the fixture via the helper, and **FAILS loudly** if regeneration is impossible
  (no network / no docker). The current `FixtureUnavailable → skip` path in `sid_fixtures.py` must
  be changed so real-tune tests **fail rather than skip** (or the cache is pre-populated for CI).
- **Therefore split the suite:** the **always-runnable core is SYNTHETIC** — generated register
  streams (no copyright, deterministic, never skip) carry the structural/balance/driver-mechanism
  assertions and run in the plain `python:3.12` docker gate. **Real-tune** fidelity/RESID tests are
  a second layer that runs where the fixture cache is available (host, or the gate with the cache
  mounted `-v $PREFRAMR_SID_FIXTURE_CACHE:...`); they regenerate-or-fail, never skip. Run the gate
  with the cache mounted so the real-tune layer executes there too.

## The pass-framework 3-layer model (you touch all three per op)
An op exists only when **Pass + Decoder + Transform** line up (see `tokens_architecture.md`):
1. **Pass** — `MacroPass` subclass in `preframr_tokens/macros/<name>_pass.py` (or `passes.py`),
   `GATE_FLAGS={"<flag>"}`. Listed in one of the run lists in `macros/__init__.py`
   (`FREQ_BLOCK_PASSES` / `PASSES` / `POST_NORM_PRE_VOICE_PASSES`) and/or called inline in
   `reglogparser.py:RegLogParser.parse()`.
2. **Decoder** — `MacroDecoder` `op_code=<OP>` in `macros/decoders.py`, registered in the
   `DECODERS = {d.op_code: d for d in (...)}` tuple. `macros/decode.py:expand_ops` **asserts
   `DECODERS.get(op) is not None`** — a dangling op hard-crashes decode (this is the tripwire).
3. **Transform** — `@register("<name>")` `PassBackedTransform` in
   `macros/transforms_*.py`, ties `OP_CODES`/`LOSS_TIER`/`REQUIRES_ARGS`/`PASS_CLASS`/
   `DECODER_CLASS`. Flag names auto-derive from `GATE_FLAGS`/`REQUIRES_ARGS` via
   `macros/flag_registry.py`.
Ops + subregs are in `stfconstants.py`. The default-on pipeline is
`macros/default_pipeline.py:DEFAULT_PIPELINE_SPEC`.

---

## #9 — Remove dead-wood transforms/macros

**Targets** (refuted/unused; confirm zero refs per the procedure before each removal):

| transform / pass | files | op(s) freed |
|---|---|---|
| `set_to_diff` | `macros/transforms_set_to_diff.py` (+ any `set_to_diff` pass) | (verify: may reuse `DIFF_OP=1` — frees nothing) |
| `voice_trajectory` | `macros/transforms_voice_trajectory.py` | `TRACK_REF_OP=46`, `VOICE_TRAJ_REG=-123` (verify decoder `TrackRefDecoder`) |
| `voice_trajectory_distributed` | `macros/transforms_voice_trajectory_distributed.py` | — |
| `super_frame` | `macros/transforms_superframe.py` | `SUPER_FRAME_REG=-124` |
| `ctrl_update` | `macros/ctrl_update_pass.py` | `CTRL_UPDATE_OP=51` |
| `flip2` | `macros/passes.py` (`Flip2Pass`) + its transform | `FLIP2_OP=7` (verify `FlipDecoder`/`Flip2Decoder`) |
| `motif` | `macros/motif_pass.py` + `motif_mine.py` | `MOTIF_OP=52`, `MOTIF_ARG=53` |

**Per-target procedure:**
1. **Verify dead** (do NOT skip):
   - `grep -rn "<name>" /scratch/anarkiwi/preframr-xpt/preframr_experiments/specs/` → must be 0
     (motif: only its own refuted-experiment specs — delete those specs too, or leave them and
     skip motif).
   - Not in `DEFAULT_PIPELINE_SPEC` (`macros/default_pipeline.py`).
   - Not in `FREQ_BLOCK_PASSES`/`PASSES`/`POST_NORM_PRE_VOICE_PASSES` (`macros/__init__.py`) nor
     the inline `parse()` calls in `reglogparser.py`.
2. **Remove**, in order: the `@register` transform class (`transforms_*.py`); the pass class +
   file; the decoder class + its entry in the `DECODERS` tuple (`decoders.py`); the op/reg/subreg
   constants in `stfconstants.py`; exports in `preframr_tokens/__init__.py` (`import` + `__all__`)
   and `macros/__init__.py`; the tests (`tests/test_<name>*.py`).
3. **Op-code policy:** prefer to **leave a one-line reserved comment** at the freed op number
   (`# 46 reserved (was TRACK_REF)`) rather than renumbering survivors — renumbering churns every
   vocab. Only the removed op's atoms disappear.
4. **Framework + xpt fallout** (motif only): remove `preframr/preframr/mine_motifs.py` entrypoint,
   the `--motif-dict` arg in `preframr/preframr/args.py`, and any motif `pre_run_hook` in xpt specs.
5. **Gate** (shared gate above) — `expand_ops` assert + `test_flag_registry` + `test_full_pipeline_fidelity`
   are the ones that catch a botched removal. Coverage may rise (less code) — fine.
6. **Release:** bump `pyproject.toml fallback_version`; CHANGELOG `### Removed` — **BREAKING:
   op-code/vocab change, re-cut corpora/checkpoints, no metric transfer** (all removed work is
   refuted, so nothing to lose). One PR, ~1.6k LOC net.

**Do NOT remove** `ctrl_triple` or `freq_nudge` (still spec-referenced; borderline, separate call).

---

## #10 — Real-pipeline structural + balance tests

**Why:** synthetic-df unit tests (`Pass.apply(hand_built_df)`) bypass `_combine_regs` +
`_quantize_freq_to_cents` and shipped a **false green** while SkeletonPass was a no-op on real
data (it read the cent-indexed `val`, not 16-bit `freq_unq`). See memory `test-through-real-parse`.

**New file `tests/test_parse_pipeline_smoke.py`:**
1. **`_synthetic_dump()` helper** — build a raw dump DataFrame with columns
   `clock, irq, chipno, reg, val` that goes through the FULL parser (emit **separate lo+hi byte
   writes** so `_combine_regs` runs; emit **per-frame** freq writes). Include, on voice 0 (reg
   0/1 freq, reg 4 ctrl, reg 5/6 ADSR): several held notes (a melody), one **octave arp**
   (alternate note / note+12 each frame), one **vibrato** (±a few-cent wobble around a semitone),
   one **slide** (monotone freq ramp), plus a couple raw PW writes. ~a few hundred rows, a few
   seconds of `clock`. Keep it deterministic.
2. **Per-config parse assertions** — for each, build `args` (a `SimpleNamespace` with
   `cents=50, exclude_list=None` + the flags), `RegLogParser(args).parse(path, reparse=True)`:
   - `skeleton_pass=True, freq_trajectory_pass=False, freq_onset_pass=False, trajectory_anchor_pass=True`
     → assert `op54 (SKEL) > 0` **and** `op55 (ORN) > 0` and `op45==0 and op48==0`.
   - defaults (`freq_trajectory_pass=True`) → assert `op45 > 0`.
   - `freq_onset_pass=True` (no skeleton) → assert `op48 > 0`.
   (This config matrix is what would have caught the cent-index no-op.)
3. **Round-trip** — assert `audio_bit_exact`: `register_state(parsed)` (or
   `preframr_audio.assert_dfs_render_equivalent`) matches the parsed input's settled per-frame
   state within tolerance, for the skeleton config.
4. **Encoding-balance assertion** — in skeleton mode assert `op55_count / max(op54_count,1) <=
   BALANCE_MAX` (start `BALANCE_MAX=6`). This flags channel-drowning (the op55:op54 13:1 bug) at
   CI. Port the op-count logic from the xpt probes `audit/probes/op48_probe.py` /
   `op48_context.py` (torch-free) — move them into `tests/` helpers.
5. **Real-tune round-trip (cached, not committed, no-skip)** — run the fidelity oracle on a real
   driver dump obtained via `sid_fixtures.ensure_dumps` (locally cached, untracked — see the
   Fixture policy above). Do NOT commit the dump. The test must **regenerate-or-fail**, never
   skip; mount `$PREFRAMR_SID_FIXTURE_CACHE` into the gate so it runs there. The **synthetic**
   round-trip in item 3 is the copyright-free, always-runs core; this real-tune layer is the
   cross-check against actual driver output.

**Gate:** shared gate. These tests must FAIL if reverted onto tokens 0.31.0 (the cent-index bug)
— verify that to prove they bite.

---

## #11 — Driver-truth fixtures: RESID≈0 as the completeness metric

**Principle:** we know what each driver does; the transforms claim to model it. A known-driver
fixture that **leaks to RESID** means the encoding is **missing a driver mechanism** — a coverage
gap to close, not a tune to tolerate. RESID share per driver is the completeness metric.

**New `tests/test_driver_coverage.py` + fixtures:**
1. **Synthetic driver-output streams** (most controlled — known expected primitive). A generator
   per mechanism emits the exact register stream the driver produces, then asserts the parse
   classifies it correctly AND RESID==0:
   - **octave arp** (Hubbard fx bit2: note / note+12 @50Hz) → must classify `ORN_TYPE_OCTAVE`.
   - **table arp** (note-relative offset cycle, e.g. `[0,+4,+7]` major) → `ORN_TYPE_ARP` with the
     correct period.
   - **vibrato** (sub-semitone depth/rate wobble) → `ORN_TYPE_VIB` with the right depth bucket.
   - **slide/portamento** (freq ramp toward target) → `ORN_TYPE_SLIDE` with target≈next note.
   - **plain held note** → `ORN_TYPE_PLAIN`.
   Each: assert `0` RESID notes. (Build the streams from `sid_driver_ornament_reference.md`.)
2. **Curated real per-driver fixtures (cached, not committed, no-skip)** — one known tune per
   driver (`{hubbard,galway,sidwizard,defmon}`), obtained via `sid_fixtures.ensure_dumps`
   (locally cached, untracked — never `git add`), identified via `engine_fingerprint` / composer
   dir. Assert the **dominant ornament type matches the driver's known mechanism** and
   `RESID_share <= RESID_MAX` (start `0.10`). Regenerate-or-fail, never skip.
3. **RESID-as-signal:** a fixture exceeding `RESID_MAX` is a **failing completeness test** — the
   fix is to model the missing mechanism (extend `fit_descriptor`), NOT to raise the threshold.
   Document each known-acceptable RESID source (genuinely-aperiodic noise sweeps) inline so the
   threshold is principled.

**Gate:** shared gate, with `$PREFRAMR_SID_FIXTURE_CACHE` mounted so the real-tune fixtures
regenerate/run (never skip). Keep fixtures tiny **in the local cache, never committed** — slice to
the needed rows (see `tests/sid_fixtures.py` `_REDUCE_MASKS` for the canonical reduction). The
synthetic generators (#11.1) are the copyright-free, always-runs core.

---

## Sequencing
Do **#9 first** (op-code space clean) → then **#10** (the structural harness) → then **#11**
(driver fixtures build on the #10 parse-assertion helpers). All three are tokens-only, torch-free,
and CI-gateable.
