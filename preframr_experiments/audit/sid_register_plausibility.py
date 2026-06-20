"""SID register-plausibility judge: a hardware-grounded, automated signal for
recognizing poor generalization in generated output.

Real SID music only drives the chip in ways that actually produce the intended
sound. A model that generalizes poorly emits register sequences that *cannot*
make sound, or that contradict how the SID is really used:

- **gate-on with no waveform** -- the envelope opens but no oscillator waveform is
  selected, so the "note" is silent (the flagship nonsense case);
- **oscillator held in test for many frames** -- the test bit resets/holds the
  oscillator; real tunes pulse it for ~1-3 frames (hard restart), not sustained;
- **ring-mod off a dead source** -- ring modulation with a non-oscillating source
  voice silences the carrier (carrier x 0), proven on the emulator;
- **noise combined with another waveform** -- feeds 0s into the LFSR so the voice
  decays despite a held gate (the noise-lock quirk).

Each check is calibrated to fire ~never on the real HVSC tunes (they *define*
plausible -- all four are 0 across the calibration corpus) and to catch the
nonsensical sequences a degenerate decode produces. The per-frame ``(n_frames,
25)`` register state is the same one the audio render sees, so "plausible" means
"would actually sound the way the tokens claim".

Note: master volume (reg 24) is NOT carried in this render-prep state (it is
applied globally at render time), so a "volume 0 while playing" check is not
assessable here and is deliberately omitted -- a lesson from calibration, where a
naive volume check fired on 77%+ of real tunes purely from the missing register.

Grounded in the emulator-proven behaviours pinned in preframr-audio
(test_freq_write_audibility / test_sid_feature_behaviors /
test_register_canonicalization).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GATE = 0x01
SYNC = 0x02
RING = 0x04
TEST = 0x08
TRI = 0x10
SAW = 0x20
PULSE = 0x40
NOISE = 0x80
WAVE = 0xF0
TONE_WAVES = TRI | SAW | PULSE

CTRL_REGS = (4, 11, 18)
FREQ_LO_REGS = (0, 7, 14)
# the SID wires each voice's sync/ring SOURCE to the previous voice (mod 3):
# voice 0 <- voice 2, voice 1 <- voice 0, voice 2 <- voice 1.
RING_SOURCE = {0: 2, 1: 0, 2: 1}

# verdict thresholds on the per-frame violation rate (calibrated so the real-tune
# corpus -- which sits at 0 on every check -- is comfortably PASS while a degenerate
# decode trips WARN/IMPLAUSIBLE; see calibrate_corpus / the unit test).
WARN_RATE = 0.02
IMPLAUSIBLE_RATE = 0.10
STUCK_TEST_FRAMES = 8

HARD_CHECKS = (
    "gate_no_waveform",
    "ring_idle_source",
    "noise_plus_wave",
    "stuck_test",
)


@dataclass
class PlausibilityReport:
    """Per-check violated-frame counts + rates and an overall verdict. ``rate`` is
    the fraction of frames with ANY hard violation -- the headline poor-generalization
    signal (lower is better; real tunes are ~0)."""

    n_frames: int
    counts: dict[str, int] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)
    rate: float = 0.0
    verdict: str = "PASS"

    def as_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "rate": round(self.rate, 5),
            "verdict": self.verdict,
            "counts": self.counts,
            "rates": {k: round(v, 5) for k, v in self.rates.items()},
        }


def _freq_word(state: np.ndarray, v: int) -> np.ndarray:
    lo = FREQ_LO_REGS[v]
    return state[:, lo].astype(np.int64) | (state[:, lo + 1].astype(np.int64) << 8)


def _run_lengths_ge(mask: np.ndarray, k: int) -> np.ndarray:
    """Boolean per-frame: True where ``mask`` is in a True-run of length >= k."""
    out = np.zeros_like(mask, dtype=bool)
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= k:
            out[i - k + 1 : i + 1] = True
    return out


def check_masks(state: np.ndarray) -> dict[str, np.ndarray]:
    """Per-check boolean per-frame violation masks for a ``(n_frames, 25)`` state."""
    n = len(state)
    if n == 0:
        return {c: np.zeros(0, dtype=bool) for c in HARD_CHECKS}
    ctrl = {v: state[:, CTRL_REGS[v]].astype(np.int64) for v in range(3)}
    gate = {v: (ctrl[v] & GATE) != 0 for v in range(3)}
    test = {v: (ctrl[v] & TEST) != 0 for v in range(3)}
    wave = {v: ctrl[v] & WAVE for v in range(3)}

    # a voice that is meant to be sounding: gate on, not held in test, with a waveform
    audible = {v: gate[v] & ~test[v] & (wave[v] != 0) for v in range(3)}

    gate_no_waveform = np.zeros(n, dtype=bool)
    ring_idle_source = np.zeros(n, dtype=bool)
    noise_plus_wave = np.zeros(n, dtype=bool)
    stuck_test = np.zeros(n, dtype=bool)
    for v in range(3):
        gate_no_waveform |= gate[v] & ~test[v] & (wave[v] == 0)
        src_idle = _freq_word(state, RING_SOURCE[v]) == 0
        ring_idle_source |= ((ctrl[v] & RING) != 0) & audible[v] & src_idle
        noise_plus_wave |= ((ctrl[v] & NOISE) != 0) & ((ctrl[v] & TONE_WAVES) != 0)
        stuck_test |= _run_lengths_ge(test[v], STUCK_TEST_FRAMES)

    return {
        "gate_no_waveform": gate_no_waveform,
        "ring_idle_source": ring_idle_source,
        "noise_plus_wave": noise_plus_wave,
        "stuck_test": stuck_test,
    }


def plausibility_report(state: np.ndarray) -> PlausibilityReport:
    """Score a ``(n_frames, 25)`` per-frame SID register state for hardware
    plausibility. Use on a decoded GENERATION to judge model quality, or on a real
    dump to calibrate (real tunes verdict PASS)."""
    n = len(state)
    rep = PlausibilityReport(n_frames=n)
    if n == 0:
        rep.verdict = "EMPTY"
        return rep
    masks = check_masks(state)
    any_hard = np.zeros(n, dtype=bool)
    for name, m in masks.items():
        c = int(m.sum())
        rep.counts[name] = c
        rep.rates[name] = c / n
        if name in HARD_CHECKS:
            any_hard |= m
    rep.rate = float(any_hard.sum()) / n
    if rep.rate >= IMPLAUSIBLE_RATE:
        rep.verdict = "IMPLAUSIBLE"
    elif rep.rate >= WARN_RATE:
        rep.verdict = "WARN"
    return rep


def reg_state_from_dump(dump, args=None, irq=None) -> np.ndarray:
    """Read a raw dump and return its ``(n_frames, 25)`` per-frame register state via the BACC codec's ``per_frame_state`` (the same state the renderer + verifier see). ``args`` is accepted but unused (legacy parser-config slot); ``irq`` overrides the frame clock, else it is taken from the ``.meta.txt`` sidecar."""
    # pylint: disable=import-outside-toplevel
    from preframr_tokens import cpf_from_meta, per_frame_state

    base = dump[: -len(".dump.parquet")] if dump.endswith(".dump.parquet") else dump
    cpf = irq if irq is not None else cpf_from_meta(base)
    state = per_frame_state(dump, cpf, 10**9)
    if state is None or len(state) == 0:
        return np.zeros((0, 25), dtype=np.int64)
    return np.asarray(state, dtype=np.int64)


def calibrate_corpus(dumps, args=None) -> dict:
    """Run the judge over real dumps; return per-dump reports + the corpus max rate. Real tunes define plausible, so the corpus max bounds where thresholds must sit."""
    reports = {}
    worst = 0.0
    for dump in dumps:
        state = reg_state_from_dump(dump, args)
        rep = plausibility_report(state)
        reports[str(dump)] = rep.as_dict()
        worst = max(worst, rep.rate)
    return {"corpus_max_rate": worst, "reports": reports}


def _main(argv=None):
    """CLI: score raw dumps (or a generated dump) for SID-register plausibility."""
    import argparse  # pylint: disable=import-outside-toplevel
    import json  # pylint: disable=import-outside-toplevel

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dumps", nargs="+", help="raw dump parquet path(s)")
    args = ap.parse_args(argv)
    out = calibrate_corpus(args.dumps)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    _main()
