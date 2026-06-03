# Work order: capture the note-trigger envelope (hard-restart + first-occurrence AD/SR)

**Status:** Work order, scoped 2026-06-03. Ready to schedule; not started. Source: forensic on the
codebook-arm raw-SET residual (Camerock, Baggis), tracing every residual AD/SR write to its cause.

**Learnability framing.** Bundling the envelope trigger fragments into one event removes hidden trigger-timing state ([`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md) Principles 1/3) — a learnability win, not only fidelity.

## Problem
In the skeleton+codebook arm, note-trigger **envelope** writes (AD reg+5 / SR reg+6) leak to raw SET.
Small by volume (Camerock 4 AD + 28 SR; Baggis 7 AD + 8 SR) but structurally significant: the
note-trigger event is **fragmented** — `HardRestartPass` captures the ctrl/gate half, patch/stamp
capture the freq, but the **envelope half is left raw** on a large, consistent class of notes.

## Root cause (evidence-grounded)
Traced each residual AD/SR write in the encoded stream. Two distinct causes, both in the
envelope-bundling logic:

1. **Hard-restart double-load (dominant).** A hard restart writes the envelope twice in one frame
   (kill-envelope then reload) to retrigger ADSR. Example (Camerock f975, voice 0):
   `AD=7,SR=132,CTRL` then `AD=3,SR=136,CTRL` in the same frame. `PatchPass._events`
   (`patch_pass.py:92`) bails on any frame with `ad_count != 1 or sr_count != 1` — it refuses
   multi-load frames. `HardRestartPass` only collapses the **CTRL** pair (`target_regs =
   CTRL_REGS_BY_VOICE`), never the envelope. So the hard-restart envelope reload falls in the **seam
   between the two passes** and leaks. Hard restart is on ~every retriggered/percussive note.
2. **First-occurrence / non-recurring envelope.** `PatchPass` emits PATCH_DEF/SET only for (ad,sr)
   pairs recurring `>= PATCH_MINREP` (=2) times (`patch_pass.py:113`). A one-off timbre, or the first
   occurrence before a second exists, is below threshold → declined → leaks. Examples: Camerock f794
   (AD=0,SR=203), f1047 (AD=15,SR=0).

Leaked AD/SR are then nibble-split by `SubregPass` (→ `op=SET subreg=0/1`), which is the form seen in
the stream.

## Why the current fallback is wrong
`release_update_pass` (off in this arm) would mop these up — but it relabels by **isolation**, so it
tags a note-**onset attack** envelope as `RELEASE_UPDATE`. Semantically wrong (it's attack, not
release) and still un-bundled from the trigger. It's a validator band-aid, not a fix.

## Fix options
- **(A) Teach `PatchPass` the hard restart.** Handle multi-load-per-frame: capture the
  `(kill, reload)` envelope pair (or the settled reload) as a unit; consider a hard-restart-aware
  PATCH variant co-located with `HardRestartPass`. Also revisit `PATCH_MINREP` for trigger envelopes
  (first occurrence should still be capturable, e.g. DEF-on-first even if reused 0 times, or a
  lower threshold for envelope-at-gate).
- **(B) Unified note-trigger event.** Bundle ctrl(hard-restart) + envelope(AD/SR) + freq into one
  trigger descriptor, so the envelope is part of the note rather than three fragments. Larger; the
  "note-bundle completion" the residual analysis keeps pointing at.

Recommend (A) first (localized, attacks the dominant hard-restart cause); (B) if the fragmentation
shows up as a learnability cost.

## Acceptance criteria
- Hard-restart and first-occurrence AD/SR no longer leak as raw SET in the codebook arm (residual
  AD/SR → ~0) WITHOUT enabling `release_update_pass`.
- Byte-exact: register-exact via `arbitrate(validate=True)`; gate on `cb_div_audit`.
- A real-corpus check first: confirm the hard-restart double-load is the dominant AD-leak cause
  (vs first-occurrence) to size (A) vs (B).

## Related / also surfaced
- `release_update_pass` + `lonely_catch_all` are **omitted** from the `codebook_distribution_mini`
  arm flag list (present in `full_macros`). This inflated that arm's SR/AD residual vs full_macros —
  the cross-arm distribution read was apples-to-oranges on envelope handling. Decide if intentional.
- Digi detection gap (PWM) is tracked separately in `digi_detection_reference.md`.
