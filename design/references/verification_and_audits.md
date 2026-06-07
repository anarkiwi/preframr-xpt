# Verification & audits — how to check the tokenizer, and the ONE tool for each property

**Status:** Reference (authoritative, 2026-06-06). Written to end recurring confusion between several
overlapping "is it correct?" checks. There are **two distinct properties** people conflate, each with **one
canonical tool**. **Do not hand-roll register comparisons** — the trap is documented below (it produced four
false "divergence" results in one session before the control caught it). Complements
[`sid_render_fidelity_contract.md`](sid_render_fidelity_contract.md) (which *defines* what "same output"
means); this doc says *how to check it* and *which tool*.

## The two properties (do not conflate)

| property | question | canonical tool | what counts |
|---|---|---|---|
| **A. Byte-exact losslessness** | does `decode(parse(dump)) == dump` per-frame? | **`PREFRAMR_PARSE_AUDIT=raise`** (one tune / in tests) → **`cb_div_audit.py`** (corpus) | EXACT_REGS (CTRL/AD/SR/23/24) byte-exact in input order; FREQ within `freq_tol` cents on audible frames; frame-aligned (`best_offset`) + skip-init. |
| **B. Residual-zero** | does any raw `SET` survive (any unmodeled write)? | **`tests/test_whole_chip_no_singleton_set.py`** (gate) → **`/scratch/tmp/wholechip_census.py`** (corpus census) | count of `op==SET` rows on non-FREQ (and now all) regs; target 0. |

A pipeline can be byte-exact but not residual-zero (a raw `SET` that decodes fine); residual-zero implies the
modeling is byte-exact. **They are different metrics — name which one you mean.**

## A. Byte-exactness — the ONE path, and why not to hand-roll
The check is layered; **only the top two are user-facing — never call the bottom two directly for a
verdict:**
1. **`cb_div_audit.py`** (corpus runner, `/scratch/preframr/cb_div_audit.py`) — parses every Nth tune with
   `parse_audit='raise'` in parallel and groups DIRTY by `(diverging-pass, reg)`. **This is THE corpus
   byte-exact gate.** `python3 cb_div_audit.py <STEP> <WORKERS>` (run on fogbank, in the tokens-test image
   with `PYTHONPATH` to the source under test).
2. **`PREFRAMR_PARSE_AUDIT=raise`** (env) or `args.parse_audit='raise'` — the in-parse mechanism
   (`parse_audit.py`/`make_pass_audit`). The parser fires it **after every pass** (`audit.after(df, pass)` in
   `reglogparser.py`), so a raise names the *exact pass* that broke byte-exactness. **This is THE single-tune
   / unit-test check.** A tune that parses without raising is byte-exact.
3. *(primitive, do not use for a verdict)* `sid_frame_diff.diff_dump_vs_pipeline(path, xdf)` /
   `diff_states(ref, test)` — the oracle: aligns by `best_offset`, EXACT_REGS exact, FREQ cent-tol on audible
   frames. It is what `parse_audit` calls **at the right stage on the right df**.
4. *(primitive)* `audit_primitives.register_state(xdf)` → `(F,25)` decoded per-frame state (expands loops);
   `sid_frame_diff.dump_frame_state(path)` → raw-dump per-frame state.

**THE TRAP (do not repeat): never compute your own `register_state` diff for a verdict.** Calling
`diff_dump_vs_pipeline(path, next(parser.parse(path)))` from outside the parser gives **false divergences** —
`next()` returns only the first block, and the raw-dump-vs-decoded framing / skip-init / alignment is not what
the in-parse audit applies. **Proof (2026-06-06):** a `baseline` parse (plain `SET`, lossless *by
construction*) "diverged" on ~65% of tunes under that external usage, yet is **197/197 CLEAN** under
`PREFRAMR_PARSE_AUDIT=raise`. The control is the tell — if your method flags baseline as dirty, your method is
wrong. **Always use parse_audit / cb_div_audit; never a bespoke register_state comparison.**

**WAV is never the fidelity gate.** Fidelity is entirely a register-level property: the same registers in the
same input order with the same nominal `_MIN_DIFF` delay render *identically by construction*
([`sid_render_fidelity_contract.md`](sid_render_fidelity_contract.md)). The tools above ARE the gate — for
byte-exact changes AND for deliberately-lossy content-tier changes (the latter must still land within the
contract's FREQ/PW/filter `freq_tol` tolerance, which `parse_audit`/`cb_div_audit` already apply). A change that
diverges beyond that tolerance is invalid; there is no WAV render or listening step. (Rendering/listening only
makes sense for judging *generated* model output — a separate quality question, not a fidelity check.)

### Related but narrower (keep for debugging, not for the verdict)
- **`PREFRAMR_ARBITER_STRICT=1`** — raises if any *single pass's claim* changes `register_state` mid-arbitration.
  A per-pass dev guard; subsumed by `parse_audit` for the end-to-end check. Use it to localize which claim a
  specific pass mis-emits.

## B. Residual-zero — the ONE path
- **`tests/test_whole_chip_no_singleton_set.py`** (in `preframr-tokens`) — the gate: zero raw `SET` on the
  deployed default. Lives with the code, runs in CI.
- **`/scratch/tmp/wholechip_census.py`** — the corpus census (per register-class + per-mechanism). Always
  `reparse=True` (stale `.pq` caches lie). Retire the other census variants (`residual_set_census`,
  `residual_mechanism.py`) — they duplicate this.

## Current status (2026-06-06)
**Deployed default (`full_macros`, `generator_pass`) is byte-exact: 197/197 CLEAN, 0 DIRTY** under
`PREFRAMR_PARSE_AUDIT=raise` (baseline control also 197/197). Residual-zero is gated by
`test_whole_chip_no_singleton_set` (the 23/24 tail is drained by `generator_pass`; the gate was un-xfailed in
PART D).

## Convergence + cleanup actions
1. **`cb_div_audit.py` is THE corpus byte-exact tool** — updated to parse under the **deployed default**
   (`named_config("full_macros")`) instead of the stale hard-coded codebook flag set, so it audits what ships.
2. **`wholechip_census.py` is THE residual-zero census** — the others are retired.
3. **Deleted ad-hoc scratch verification scripts** (`/scratch/tmp/cmp_*.py` and the superseded one-offs) — they
   were the "confusing duplication." Do not re-create them; use the two tools above.
4. **Anyone adding a correctness check** uses `parse_audit`/`cb_div_audit` (byte-exact) or the whole-chip
   gate/census (residual-zero). New bespoke `register_state` diffs are a smell — see THE TRAP.

## Where these live (single-source)
- Primitives + oracle + in-parse: `preframr-tokens/preframr_tokens/{audit_primitives,sid_frame_diff,parse_audit}.py`.
- Corpus byte-exact runner: `/scratch/preframr/cb_div_audit.py` (loose script — candidate to move into
  `preframr-tokens` as a `python3 -m preframr_tokens.cb_div_audit` CLI so it versions with the code it audits).
- Residual-zero gate: `preframr-tokens/tests/test_whole_chip_no_singleton_set.py`; census `/scratch/tmp/wholechip_census.py`.
