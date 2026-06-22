# SID render fidelity contract

**Status:** Pointer (chip-facts index; the fidelity gate is now the BACC residual-zero contract —
`recover_from_sid`'s render == ground-truth dump byte-exact over all 25 registers, by construction).

The two halves of the contract now live with the code that owns them:

- **Chip-behavior facts and their pinning tests** — the render timing model
  (per-write clocking is mandatory), the complete envelope/ADSR-bug mechanism, the
  write-liveness matrix, release-write placement, write-order audibility, the two
  distinct hard-restart mechanisms, sexy-start, and the full proven-facts → tests
  table — are in the **[preframr-audio README](https://github.com/anarkiwi/preframr-audio)**
  ("SID programming facts" section). The tests are the single source of truth; cite
  them, don't paraphrase.
- **The BACC fidelity gate** — `residual = 0` by construction: the program recovered by
  `recover_from_sid` renders byte-exact to the ground-truth dump over all 25 registers, so any SID
  write not explained by score+generator is a non-zero residual to trace, never an escape lane. What
  the recovered program must reproduce (ordered CTRL/AD/SR change activity, per-nibble gate-edge sides,
  HR prep side; settled freq/PW first, globals last) is in the
  **[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens)** ("Fidelity contract" section).

Scope: non-digi tunes; multiple AD/SR/CTRL writes per voice per frame are in scope and order-preserved.
Multispeed-aware framing has LANDED (sub-frame play-period framing, lossless per play-call).

xpt-internal context: the measurement methodology that produced these rules
(write-count-matched A/B variants, ENV3 reads, ±8 nondeterminism floor) and the
decisive 2026-06-11 falsification of the fixed AD,SR-before-gate onset order are
discussed in [`verification_and_audits.md`](verification_and_audits.md) §B. Settled
`register_state` is order- and timing-blind — necessary, never sufficient; the old
per-pass `parse_audit`/`cb_div_audit` gates belonged to the retired substrate.
