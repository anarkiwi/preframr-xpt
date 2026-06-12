# Verification & audits — how to check the tokenizer, and the ONE tool for each property

**Status:** Reference (authoritative, rewritten 2026-06-11 for the v3 event-model canonical
contract; the previous revision documented the retired (op,reg,subreg,val) substrate's tools —
`parse_audit` / `cb_div_audit` / residual-zero census — which no longer exist as gates).
Complements [`sid_render_fidelity_contract.md`](sid_render_fidelity_contract.md) (which *defines*
what "same output" means); this doc says *how to check it* and *which tool*.

## The two properties (do not conflate)

| property | question | canonical tool | what counts |
|---|---|---|---|
| **A. Canonical fidelity** | does `decode(encode(dump)) == canonical_writes(dump)` exactly? | **`stream.encode(ow, verify=True)`** (the default — every encode self-verifies, fail-loudly) → events test suites (5 drivers + 200-tune corpus roundtrip in gen2 `tests/test_events_stream.py` / `test_events_roundtrip.py`) | Exact list equality of ordered `(frame, reg, val)` triples vs the canonical form. Zero drops: canonical is an intra-frame permutation + derivation of the dump's writes. |
| **B. Canonicalization soundness** | is `canonical_writes(dump)` audibly identical to the dump? | **the chip-semantics reference suites** (preframr-audio: `test_gate_adsr_reference`, `test_adsr_write_liveness_matrix`, `test_release_write_position`, `test_register_canonicalization`, `test_freq_write_audibility`, `test_sid_same_value_writes`) + the **perceptual raw-vs-canonical A/B render** (reSID, per-write clocking, `fidelity.perceptual_distance` + sample stats) | Every canonicalization rule cites a measured chip fact; the A/B renders at the reSID noise floor (max&nbsp;Δ ≤ the ±8 render-nondeterminism floor) on the driver fixtures. |

Property A is mechanical and runs on every encode. Property B is where the science lives: each
rule of the canonical form (settled freq/PW first, globals last, same-value drops, derived
gate-offs, NOTE_ON envelope folds, recorded gate-edge sides) exists because a preframr-audio test
measured it faithful — and the rules changed on 2026-06-11 when measurement falsified one
(the fixed AD,SR-before-gate onset order; see the contract doc).

## A. Canonical fidelity — the ONE path

1. **`stream.encode(ow)`** self-verifies by default (`verify=True`): it decodes its own token
   stream and asserts exact equality with `stream.canonical_writes(ow)`. A tune that encodes
   without raising IS canonically faithful. This replaces the old `PREFRAMR_PARSE_AUDIT=raise`.
2. **Corpus scale:** gen2 `tests/test_events_stream.py::test_corpus_sample_canonical_roundtrip`
   (200-tune in-scope sample) + the per-driver roundtrip tests. This replaces `cb_div_audit.py`.
3. *(primitive, not a verdict)* `stream.canonical_writes(ow)` and `events/oracle.ordered_writes(df)`
   are the building blocks; comparing them by hand reintroduces the old framing/alignment trap —
   use the self-verify.

**Scope guard:** the contract covers single-speed non-digi tunes (`stream.single_speed`,
`dump_meta.is_digi`). Corpus globs MUST filter; an out-of-scope tune failing the roundtrip is a
scope bug, not a fidelity bug.

**The residual-zero property is structural now.** The old "does any raw `SET` survive?" census is
meaningless under v3: there is no literal/escape path in the grammar at all — every write value
derives from modeled state by construction. Nothing to audit; the decoder's strict grammar
(malformed streams raise) is the residue gate.

## B. Canonicalization soundness — measurement, not convention

The canonical form re-orders and derives writes. Each liberty is licensed by a pinned
measurement; the authoritative set lives in **preframr-audio's test suites**, indexed in
[`sid_render_fidelity_contract.md`](sid_render_fidelity_contract.md). The operating rules:

- **A new canonicalization needs a new measurement.** The method that works (2026-06-11):
  write-count-matched A/B variants (same-value pads so chunk boundaries and write offsets are
  identical), real per-write clocking (~32 cycles between writes — collapsed-timing A/Bs MASK
  placement effects entirely), ENV3 reads for envelope-state verdicts, and the equivalence floor
  calibrated to the measured ±8 resampler nondeterminism. See the methodology notes in
  `test_adsr_write_liveness_matrix.py`.
- **The end-to-end check is the perceptual A/B**: render the dump's raw writes and
  `canonical_writes` under identical clocking and compare (`fidelity.perceptual_distance`
  + max-|Δ| / %-samples-over-500). Currently a gen2 tmp probe (`tmp/order_fix_audibility.py`)
  run on the 5 drivers; **productizing it as a corpus-wide audit is the open follow-up** —
  it is the v3 successor to the old corpus byte-exact runner.
  Caveat: `perceptual_distance` destabilizes on near-silent windows (dither dominates the
  log-band features) — read it jointly with the sample-level stats, never alone.
- **WAV/listening is still never the gate** for *encoding* changes — but the gate moved one level
  up: from "same registers in input order ⟹ same render by construction" to "canonical form
  measured render-equivalent, then exact equality to the canonical form." Rendering only judges
  *generated* model output (a quality question).

## Historical note (the old substrate's tools)

`parse_audit` / `cb_div_audit.py` / `PREFRAMR_ARBITER_STRICT` / the residual-zero census
(`test_whole_chip_no_singleton_set`, `wholechip_census.py`) verified the retired
(op,reg,subreg,val) + macro-pass pipeline and its byte-order oracle. They are not the gates for
the event model and should not be revived for it; their documented trap (never hand-roll a
`register_state` diff for a verdict) survives in spirit — under v3 the equivalent trap is
hand-comparing raw vs canonical write lists without the self-verify's exact framing.
