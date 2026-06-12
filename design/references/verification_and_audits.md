# Verification & audits — how to check the tokenizer, and the ONE tool for each property

**Status:** Reference (rewritten 2026-06-11 for the v3 canonical contract; reduced to
property pointers 2026-06-12 — the tool details live in the downstream repo READMEs).

## The two properties (do not conflate)

| property | question | where documented |
|---|---|---|
| **A. Canonical fidelity** | does `decode(encode(dump)) == canonical_writes(dump)` exactly? | **[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens)** ("Fidelity contract"): `stream.encode(ow, verify=True)` self-verifies on every encode; corpus scale = the 200-tune roundtrip + per-driver tests in preframr-tokens `tests/`. Scope guard: single-speed (`stream.single_speed`), non-digi (`dump_meta.is_digi`) — an out-of-scope failure is a scope bug, not a fidelity bug. Residual-zero is structural in v3: no literal/escape path exists in the grammar; the decoder's strict grammar is the residue gate. |
| **B. Canonicalization soundness** | is `canonical_writes(dump)` audibly identical to the dump? | **[preframr-audio README](https://github.com/anarkiwi/preframr-audio)** ("SID programming facts"): every canonicalization rule cites the pyresidfp test that measured it. The A/B methodology (write-count-matched variants, per-write clocking, ENV3 reads, ±8 nondeterminism floor) is in the test files themselves. |

Property A is mechanical and runs on every encode. Property B is where the science
lives: each rule of the canonical form exists because a preframr-audio test measured it
faithful — and the rules changed on 2026-06-11 when measurement falsified one (the
fixed AD,SR-before-gate onset order).

## Operating rules (xpt-internal)

- **A new canonicalization needs a new measurement** — never extend the canonical form
  by convention. Collapsed-timing A/Bs mask placement effects entirely; per-write
  clocking always.
- **The end-to-end check is the perceptual raw-vs-canonical A/B**
  (`fidelity.perceptual_distance` + sample-level stats — never the distance alone;
  it destabilizes on near-silent windows). Productizing it as a corpus-wide audit is
  the open follow-up; it is the v3 successor to the old corpus byte-exact runner.
- **WAV/listening is never the gate for encoding changes**; rendering judges
  *generated* model output only (a quality question).
- Never hand-compare raw vs canonical write lists — use the encode self-verify
  (hand-rolled comparisons reintroduce the framing/alignment trap).

## Historical note (the old substrate's tools)

`parse_audit` / `cb_div_audit.py` / `PREFRAMR_ARBITER_STRICT` / the residual-zero
census (`test_whole_chip_no_singleton_set`, `wholechip_census.py`) verified the retired
(op,reg,subreg,val) + macro-pass pipeline and its byte-order oracle. They are not the
gates for the event model and should not be revived for it.
