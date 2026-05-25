# Permanent tokenizer profiling tooling (efficiency + correctness)

**Status (2026-05-25):** Drafted, **not yet implemented** (checked
`feat/freq-traj-and-profiling`: `audit_primitives` still has only its original 3
fns; no `tokenizer_config` / `tokenizer_profile` / `trajectory_coverage`). The
other agent is finishing it; the 0.16.0 CHANGELOG pre-stage *claims* profiling
tools that don't exist yet. **Blocks FREQ_TRAJ validation:** the
`unified_oscillation_primitive_design.md` Phase-0 coverage (≥40% oscillatory
motion) + efficiency (FREQ atoms/song < current) gates need
`tokenizer_profile`/`trajectory_coverage` — **I run those gates once this lands.**
The op-tier/decode reductions can settle now that the API + FREQ_TRAJ op set are
stable.

## Problem

Assessing a tokenizer change (firing rate, atoms/song, lossless round-trip,
structural coverage) keeps spawning throwaway `/scratch/tmp/*_probe.py` scripts.
Three recurring costs:

1. **Boilerplate duplication.** Every probe re-implements the args namespace
   (a hand-rolled `SimpleNamespace` + `_BASE`/`_FLAGS` flag list) and the
   dump-glob → parse loop. The flag list goes stale on every arg rename — the
   `FREQ_TRAJ` rework alone retires `--slope-pass`/`--oscillate-env-pass`/
   `--vibrato-env-pass`/`--freq-run-pass`.
2. **Internals-poking.** The throwaway probes reached into `expand_ops`,
   `FREQ_REGS_BY_VOICE`, `remove_voice_reg`, raw op constants — exactly the
   surface the API rework is reshaping, so they break on every rework.
3. **Partial duplication of tools that already exist.**
   `integration_tests/profile/audit_macro_fidelity_probe.py` is already the
   register-state correctness tool, and `audit_strict_no_diff_coverage.py` the
   residual-coverage gate — both on the **public** `preframr_tokens` surface
   (`RegLogParser`, `prepare_df_for_audio`, `classify_carveout`, `reg_class`).
   The throwaway probes reinvented slices of these.

There is no fast, standalone **token-efficiency profiler**: the experiment runner
computes `encoded_tokens_per_song` + `alphabet_size`, but only via a full
parse+tokenize+train arm. A host-side op-histogram / atoms-per-song tool that
A/Bs two flag sets on a corpus sample in minutes is the genuine gap.

## What the throwaway probes map to

| throwaway probe | permanent home |
|---|---|
| `firing_probe.py` (op histogram, %atoms) | **new** `tokenizer_profile.py` |
| `freq_magnitude_probe.py` (value/delta byte-width) | `tokenizer_profile.py` (payload-width report) |
| `freq_traj_probe.py` (decoded per-frame trajectory, oscillation %) | **new** `trajectory_coverage.py` + shared register-state reduction with `audit_macro_fidelity_probe.py` |
| `freq_struct_probe.py` (run/gap/alternation/periodicity) | `trajectory_coverage.py` |
| `osc_sweep.py` (gate × cost model) | **stays throwaway** — a design-time what-if; once params lock, "before/after `tokenizer_profile --compare`" replaces it |

## Design

**Placement: in preframr-tokens** (torch-free; `RegLogParser`,
`blocks.glob_dumps`, pandas/numpy are all already there). Work order:
`preframr-tokens:PROFILING_TOOLS.md`.

### Dependency direction (the load-bearing constraint)

The tools cannot import `preframr.*` — that inverts the dependency (preframr-tokens
is the library the main repo consumes). So they cannot use
`preframr.args.add_args`. Instead **preframr-tokens owns the tokenizer-arg source
of truth**: a torch-free `tokenizer_config.default_tokenizer_args()` /
`named_config()` (promoted from the `_BASE`/`_ALL_FLAGS` hand-roll already in
`tests/test_full_pipeline_fidelity.py`). Main-repo `args.add_args` consuming that
builder is a follow-up, tracked main-side — not a blocker.

### Two reductions in `audit_primitives` (next to `tier_accuracy`/`distinct_n`)

```
op_atom_profile(xdf) -> dict
    # op-id histogram + %atoms, total atoms, atoms/frame, per-tier atom budget
    # (op->tier via the public tier map), payload byte-width.

register_state(xdf) -> np.ndarray   # (n_frames, 25)
    # per-frame decoded register state; the SAME reduction
    # test_full_pipeline_fidelity + audit_macro_fidelity_probe use — factor here
    # so all three share one impl.

trajectory_coverage(xdf, tier="freq") -> dict
    # from register_state: per-op capture vs mop-up fraction + run/gap/
    # alternation/periodicity distributions.
```

`register_state` + `op_atom_profile` are canonical; everything composes from them.
Corpus iteration reuses the existing `blocks.glob_dumps` + `RegLogParser.parse`.

### CLI 1 — `python -m preframr_tokens.tokenizer_profile` (efficiency)

`op_atom_profile` over a corpus sample for one flag set, or `--compare A B` to
diff two flag sets / pipeline specs on the **same** sample. Output: per-op
`%atoms`, atoms/song, atoms/frame, per-tier budget, alphabet size, payload
byte-width; the compare mode prints net atom delta per op. This is the
before/after that replaces `osc_sweep`'s cost model with a *measurement* of the
real implementation, and the Phase-0 gate in `unified_oscillation_primitive_design.md`
("FREQ atoms/song < current") points here instead of a `/scratch/tmp` script.

### CLI 2 — `python -m preframr_tokens.trajectory_coverage` (structural coverage)

From `register_state`, reconstruct per-voice per-tier motion and report, for a
chosen tier (FREQ first, extensible), what fraction of motion each structural op
captures vs falls to the mop-ups (the "is the primitive firing on its
phenomenon" question), plus run-length / gap / alternation / periodicity
distributions. Generalizes the freq probes to any future structural primitive.
Correctness is delegated: `register_state(macros) == register_state(raw)` is the
byte-exact oracle, with diff-by-register-class so divergence localises to the
offending op.

### Reuse, do not duplicate

- `audit_macro_fidelity_probe.py` (main repo) stays the canonical **correctness**
  tool (raw vs leave-one-out, render top offenders), but imports `register_state`
  from `preframr_tokens.audit_primitives` instead of re-implementing it. So does
  `test_full_pipeline_fidelity.py`.
- `audit_strict_no_diff_coverage.py` stays the **residual-coverage** gate.
- The experiment runner keeps owning `encoded_tokens_per_song`/`alphabet_size`
  as A/B metrics; `tokenizer_profile` is the fast pre-experiment estimate, not a
  replacement.

## Internal surface these build on

All public in preframr-tokens: `RegLogParser`, `blocks.glob_dumps`, the op→tier
map (`tier_classify`), and the atoms→per-frame register-state decode (today
`expand_ops` + `remove_voice_reg` — `register_state` wraps it so nothing else
imports those internals). The API rework owns keeping these stable; the
`FREQ_TRAJ` rework changes the op set the per-tier/coverage parts read.

## Phase / sequencing

This work lives in preframr-tokens, the same repo as the in-flight API + FREQ_TRAJ
reworks, so it **lands after they settle** (no parallel churn on the same
modules):

1. `tokenizer_config` builder + `op_atom_profile` + `tokenizer_profile` CLI
   (op-id histogram / atoms-per-song / `--compare`). Robust to op renames; repoint
   the FREQ_TRAJ Phase-0 gate here.
2. `register_state` factored into `audit_primitives`; refactor
   `test_full_pipeline_fidelity` + `audit_macro_fidelity_probe` onto it.
3. `trajectory_coverage` (assumes the final FREQ_TRAJ op taxonomy).

## Non-goals

- Replacing the experiment-runner metrics or `audit_macro_fidelity_probe` /
  `audit_strict_no_diff_coverage` (reuse + share the reduction, don't fork).
- A new audio-fidelity gate — the planned `compare_renders` corpus gate
  (AGENTS.md "Land any time") is orthogonal; this tooling is register-state +
  atom-level, which is the drift-free signal.
- Making the `osc_sweep` cost model permanent — it is a design-time what-if;
  the permanent equivalent is measuring the real impl with `tokenizer_profile`.

## References

- Existing: `integration_tests/profile/{audit_macro_fidelity_probe,
  audit_strict_no_diff_coverage,seq_budget_coverage,parse,macros}.py`;
  `preframr_tokens.audit_primitives`; `preframr.args.add_args`.
- Throwaway probes (now deleted from `/scratch/tmp`; their findings are captured
  in `unified_oscillation_primitive_design.md`, and this doc's table maps each to
  its permanent home): `firing_probe`, `freq_struct_probe`, `freq_traj_probe`,
  `freq_magnitude_probe`, `osc_sweep`.
- Consumers: `unified_oscillation_primitive_design.md` Phase 0 gate;
  `preframr-tokens:OSCILLATE_REWORK.md`.
