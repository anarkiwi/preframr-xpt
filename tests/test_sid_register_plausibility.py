"""Tests for the SID register-plausibility judge -- the hardware-grounded signal
for catching poor generalization (register sequences that cannot make the sound
the tokens claim). Each detector must fire on a constructed implausible state and
stay silent on a plausible one; and where the real-tune corpus is available, real
music must verdict PASS (it defines plausible)."""

from __future__ import annotations

import glob
import os
import unittest

import numpy as np

from preframr_experiments.audit.sid_register_plausibility import (
    GATE,
    IMPLAUSIBLE_RATE,
    NOISE,
    PULSE,
    RING,
    STUCK_TEST_FRAMES,
    TEST,
    TRI,
    WARN_RATE,
    check_masks,
    plausibility_report,
    reg_state_from_dump,
)

CORPUS_DIR = "/scratch/preframr/sid_fixture_cache"


def _blank(n):
    return np.zeros((n, 25), dtype=np.int64)


def _plausible(n=40):
    """A voice gated on a pulse waveform with a real freq -- the canonical sounding
    note; trips no check."""
    st = _blank(n)
    st[:, 4] = GATE | PULSE  # voice 0 ctrl: gate + pulse
    st[:, 0] = 0x00
    st[:, 1] = 0x20  # freq word 0x2000
    st[:, 2] = 0x00
    st[:, 3] = 0x08  # pulse width
    return st


class TestPlausibilityDetectors(unittest.TestCase):
    def test_plausible_state_passes(self):
        rep = plausibility_report(_plausible())
        self.assertEqual(rep.verdict, "PASS")
        self.assertEqual(rep.rate, 0.0)
        self.assertTrue(all(c == 0 for c in rep.counts.values()), rep.counts)

    def test_gate_on_no_waveform_fires(self):
        """The flagship nonsense case: gate on, no waveform selected -> silent note."""
        st = _blank(40)
        st[:, 4] = GATE  # gate, no waveform bits, no test
        masks = check_masks(st)
        self.assertTrue(masks["gate_no_waveform"].all())
        # and a test frame (gate+test, no waveform) must NOT count -- that is hard
        # restart, not a silent note
        st2 = _blank(40)
        st2[:, 4] = GATE | TEST
        self.assertFalse(check_masks(st2)["gate_no_waveform"].any())

    def test_ring_with_idle_source_fires(self):
        """Ring-mod on an audible voice whose source oscillator is idle (freq 0)
        silences it -- carrier x 0."""
        st = _blank(40)
        st[:, 4] = GATE | TRI | RING  # voice 0 audible + ring
        # source for voice 0 is voice 2 (regs 14/15); leave its freq at 0 (idle)
        masks = check_masks(st)
        self.assertTrue(masks["ring_idle_source"].all())
        # giving the source a frequency clears the violation
        st[:, 14] = 0x10
        self.assertFalse(check_masks(st)["ring_idle_source"].any())

    def test_noise_plus_waveform_fires(self):
        """Noise combined with a tone waveform feeds 0s into the LFSR (noise-lock)."""
        st = _blank(40)
        st[:, 4] = GATE | NOISE | PULSE
        self.assertTrue(check_masks(st)["noise_plus_wave"].all())

    def test_stuck_test_fires_only_when_sustained(self):
        """Test held for >= STUCK_TEST_FRAMES frames is a stuck oscillator; a short
        hard-restart pulse is fine."""
        st = _blank(40)
        st[:, 4] = TEST  # held the whole time
        self.assertTrue(check_masks(st)["stuck_test"].any())
        short = _blank(40)
        short[: STUCK_TEST_FRAMES - 1, 4] = TEST  # brief pulse
        self.assertFalse(check_masks(short)["stuck_test"].any())

    def test_verdict_thresholds(self):
        """Rate between WARN and IMPLAUSIBLE -> WARN; above IMPLAUSIBLE -> IMPLAUSIBLE."""
        n = 100
        warn = _plausible(n)
        k = int((WARN_RATE + IMPLAUSIBLE_RATE) / 2 * n)
        warn[:k, 4] = GATE  # k silent-gated frames
        self.assertEqual(plausibility_report(warn).verdict, "WARN")

        bad = _plausible(n)
        bad[: int(IMPLAUSIBLE_RATE * n) + 5, 4] = GATE
        self.assertEqual(plausibility_report(bad).verdict, "IMPLAUSIBLE")

    def test_empty_state(self):
        self.assertEqual(plausibility_report(_blank(0)).verdict, "EMPTY")


@unittest.skipUnless(
    glob.glob(os.path.join(CORPUS_DIR, "*.dump.parquet")),
    f"real-tune corpus not present at {CORPUS_DIR}",
)
class TestPlausibilityCorpusCalibration(unittest.TestCase):
    def test_real_tunes_are_plausible(self):
        """Calibration anchor: real HVSC tunes define plausible, so every one must
        verdict PASS. If this fails, a detector has drifted into flagging real usage
        (the failure mode that the dropped volume check exhibited)."""
        from preframr_tokens.tokenizer_config import named_config

        args = named_config("baseline")
        dumps = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.dump.parquet")))
        for dump in dumps:
            state = reg_state_from_dump(dump, args)
            rep = plausibility_report(state)
            self.assertEqual(
                rep.verdict, "PASS", f"{os.path.basename(dump)}: {rep.as_dict()}"
            )


if __name__ == "__main__":
    unittest.main()
