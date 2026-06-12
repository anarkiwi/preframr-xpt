# SID render fidelity contract

**Status:** Pointer (2026-06-12; previously the chip-facts index, updated 2026-06-11
when the contract moved from "preserve byte order" to the v3 canonical form).

The two halves of the contract now live with the code that owns them:

- **Chip-behavior facts and their pinning tests** — the render timing model
  (per-write clocking is mandatory), the complete envelope/ADSR-bug mechanism, the
  write-liveness matrix, release-write placement, write-order audibility, the two
  distinct hard-restart mechanisms, sexy-start, and the full proven-facts → tests
  table — are in the **[preframr-audio README](https://github.com/anarkiwi/preframr-audio)**
  ("SID programming facts" section). The tests are the single source of truth; cite
  them, don't paraphrase.
- **The v3 canonical form** — what `canonical_writes` preserves (ordered CTRL/AD/SR
  change activity, per-nibble gate-edge sides, HR prep side) vs canonicalizes
  (settled freq/PW first, globals last, same-value drops, derived gate-offs), and the
  encode-time self-verification `decode(encode(ow)) == canonical_writes(ow)` — is in
  the **[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens)**
  ("Fidelity contract" section).

Scope for both: single-speed, non-digi tunes; multiple AD/SR/CTRL writes per voice per
frame (~17% of single-speed tunes) are in scope and order-preserved.

xpt-internal context: the measurement methodology that produced these rules
(write-count-matched A/B variants, ENV3 reads, ±8 nondeterminism floor) and the
decisive 2026-06-11 falsification of the fixed AD,SR-before-gate onset order are
discussed in [`verification_and_audits.md`](verification_and_audits.md) §B. Settled
`register_state` is order- and timing-blind — necessary, never sufficient; the old
per-pass `parse_audit`/`cb_div_audit` gates belonged to the retired substrate.
