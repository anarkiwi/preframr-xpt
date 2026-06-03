# Oscillating SWEEP — capture mechanism-B vibrato / LFO

**Status:** Proposed (2026-06-03). Extends the existing `SweepPass`/`SWEEP` macro to absorb the
residual raw-freq tail measured in `codebook_distribution_mini` (codebook arm: ~0.9% of atoms are
raw `FREQ_lo` SET). Grounded in [`sid_driver_ornament_reference.md`](sid_driver_ornament_reference.md)
(mechanism B) and a forensic on DRAX `Billig_Oel`.

**Learnability framing.** Collapsing a per-frame value-domain oscillation into one parametric atom removes an implicit counter ([`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md) Principles 1/3).

## The gap (forensic, not a hypothesis)
Decoding `Billig_Oel` voice 0, frames 1403–1407, gate held (`ctrl=0x41`), triangle wave:

```
freq:  1387  1403  1419  1435  1419   (Δ +16 +16 +16 +16 −16)
```

A held note with a **±16 freq-word oscillation** — textbook **mechanism-B value-domain vibrato**
(driver doc: *"a value swept per frame … added to the freq word … not note-relative and not a small
codebook"*; Hubbard depth byte / Galway `FMG/FMD` gradient / defMON auto-reversing `ps_depth`). The
skeleton captures the note **onset**; the per-frame modulation has no descriptor and falls to raw SET.

This is a real, bounded, representable driver feature that the stack **does not model** — an unmodelled
feature, not noise. ORN covers note-relative *semitone* arps (mechanism A); SWEEP covers *monotonic*
value-domain ramps; **neither covers value-domain oscillation.**

## Why SWEEP is the right macro to extend (not ORN)
The modulation is **value-domain** (a ±N freq-*word* wobble, sub-semitone — ±16 ≈ a few cents), the
driver doc's mechanism B, which it groups with PW and filter sweeps — exactly `SweepPass`'s domain
(freq + `pw_sweep` + `filter_sweep`). ORN is note-relative semitone space and cannot express a
fractional value-domain wobble. SWEEP already carries the right fields — `START`, signed `DELTA`,
`LEN`, **`PERIOD`** — and already runs the per-frame `pending_set_writes` drain. The only thing missing
is the **waveform shape**: today `SweepDecoder` does `step = k % period` (a **sawtooth** — ramps then
snaps to start); vibrato/PW-LFO is a **triangle** (ramps then **reverses** at the bound).

## The extension: an auto-reversing (triangle) SWEEP mode
1. **Encoding** — add one discriminator atom `SWEEP_SUBREG_SHAPE` (e.g. subreg 6), emitted before the
   `LEN` trigger: `0 = ramp` (current behaviour, default), `1 = triangle` (auto-reverse). Reuse all
   existing fields; `PERIOD` becomes the half-cycle length (frames from one extreme to the other).
   No new op, no change to the codebook-id story (SWEEP is not a define→ref op).
2. **Decoder** (`SweepDecoder.expand`) — replace the single `step` rule with a shape switch:
   ```
   ramp     (shape 0):  step = k % period   if period else k        # unchanged
   triangle (shape 1):  c = k % (2*period); step = c if c <= period else 2*period - c
   ```
   `value = (start + step*delta) & 0xFFFF`, drained one per frame as today. Byte-exact: it reproduces
   the exact `1387,1403,1419,1435,1419,…` sequence.
3. **Miner** (`SweepPass`) — today it greedily extends a **constant-delta** run. Add: when a run hits a
   sign flip whose magnitude matches the run's `|delta|` and the reversal recurs on a fixed `period`
   (a bounded triangle), emit ONE triangle SWEEP `(start, delta, period, len)` instead of breaking the
   run. Detection is the same per-frame-delta scan plus a reversal/period check; bounds come from the
   observed extremes. Persists across gate-on (the doc: PW/filter/vibrato sweeps are **not**
   note-aligned), so it skips the skeleton RESID gate like the other sweep sub-flags.
4. **Validation / byte-exactness** — register-exact via the existing `arbitrate(..., validate=True)`
   path (one `Claim`/run); a triangle that isn't byte-exact stays literal. Gate on the corpus
   `cb_div_audit` (currently dirty≈0).
5. **Flag** — a default-OFF sub-flag `sweep_osc` (or `vib`) under `sweep_pass`, mirroring
   `pw_sweep`/`filter_sweep`. Applies to **freq (vibrato), PW (PW-LFO), and filter (cutoff wobble)** —
   all the same mechanism-B auto-reversing sweep.

## Expected payoff
- Absorbs the ~0.9% raw `FREQ_lo` residual (and the analogous PW/filter oscillation tail) into one
  low-cardinality parametric atom per run — same fidelity×budget×learnability profile as the monotonic
  SWEEP that already landed, and the same parametric form the driver renders from (`center/depth/rate`).
- Learnability: a triangle SWEEP is a handful of fixed-meaning fields (mechanism B's depth/rate/period),
  not a per-tune codebook — transferable by construction.
- **Non-claims:** does not touch the percussion/drum residual (that's a separate `stamp_pass`
  MINREP/variance miss); not a generalization bet on its own — a coverage/fidelity completion of the
  mechanism-B story the SWEEP line already started.

## Reconciliation: why the three existing oscillation paths don't already cover it (2026-06-03)
Pinned the root cause precisely — three mechanisms touch oscillation, none fits the codebook-arm gap:
- **`FreqTrajectoryPass` FT_SUBTYPE_OSCILLATE** *is* a lossless value-domain oscillator (v0 + delta
  steps, with a periodic mode) — but `freq_trajectory_pass` and `skeleton_pass` are **mutually-exclusive
  freq substrates** (`codebook_distribution_mini`: the codebook arm "drops the FREQ_TRAJ op for
  SKEL+ORN+codebooks"). So in the arm where the residual appears, OSCILLATE is **off by design.**
- **Skeleton `ORN_TYPE_VIB`** is **lossy by design**: `vib_frame_offsets` replays offset **0** — "the
  content-tier floor drops the sub-semitone wobble; depth/rate carry the learnable signal, not bytes."
  ORN is semitone-offset space and cannot express a fractional value-domain wobble.
- **Monotonic `SweepPass` freq path** is sawtooth-only AND gated to "skydive pitch sweeps the skeleton's
  SLIDE can't fit" — it fires only on skeleton-**unfittable** runs. A vibrato'd held note is
  skeleton-**fittable**, so the gate **suppresses** SWEEP there → wobble falls to raw FREQ_lo SET.

## Integration refinement (the part to get right)
The triangle decoder is the easy half. The hard half is **where freq-vibrato is mined**, because a
vibrato'd held note is simultaneously a NOTE (skeleton's learnable identity) and a WOBBLE
(value-domain). Split by domain:
- **PW / filter LFO** — not note-attached; the value-domain triangle SWEEP is clean and correct as
  written (persists across gate, no skeleton coupling). Do this first; lowest risk.
- **Freq vibrato on a held note** — the existing freq-SWEEP gate is the WRONG gate (it excludes
  skeleton-fittable notes). Two options:
  - **(A) value-domain triangle SWEEP with a NEW gate** firing on the *sub-semitone residual* of a
    skeleton-fitted note (opposite of the skydive gate), partitioning sustain freq writes from
    skeleton's onset. Risk: double-representation / losing the SKEL note identity if writes aren't
    cleanly partitioned.
  - **(B) make the note's vibrato descriptor byte-exact** — keep SKEL for identity, reconstruct the
    wobble in the value domain *relative to the note base* (center = note freq word; depth/period =
    triangle). Preserves note identity + learnability, no write-partition conflict; freq-only, couples
    to skeleton/ORN.
  Recommendation: ship PW/filter (value-domain SWEEP) first; for freq, prefer (B) over pre-empting the
  SKEL note. Decide via a forensic on whether the residual sustain writes partition cleanly from
  skeleton's onset write.

## References
`sid_driver_ornament_reference.md` (mechanism B; defMON `ps_depth` auto-reverse, Galway gradient,
Hubbard depth/pulsework), `landed/` SWEEP (PW/filter sweep mining, tokens 0.42.0).
