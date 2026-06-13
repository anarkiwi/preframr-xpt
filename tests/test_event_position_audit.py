"""Tests for the position-within-event probe (pure role/offset/bucketing)."""

from __future__ import annotations

import unittest

from preframr_experiments.audit.event_position_audit import (
    atom_role,
    event_offsets,
    finalize,
    merge_hits,
    position_hits,
)

# Synthetic compact alphabet (same shape as the real one, smaller ranges).
R = {
    "var_base": 0,
    "var_end": 4,  # digits 0-3
    "reg_base": 4,
    "reg_end": 8,  # regs 4-7
    "voice_base": 8,
    "voice_end": 12,  # voices 8-11
    "headers": frozenset((12, 13, 14)),
    "kinds_lo": 15,
    "kinds_hi": 18,
    "shape_lo": 19,
    "shape_hi": 20,
    "nib_wave": 21,
    "nib_art": 37,
    "nib_env": 53,
    "keyframe": 69,
}


class TestAtomRole(unittest.TestCase):
    def test_each_range(self):
        cases = {
            0: "digit",
            4: "reg",
            8: "voice_tag",
            12: "header",
            15: "kind",
            19: "shape",
            21: "nib_wave",
            37: "nib_art",
            53: "nib_env",
            69: "keyframe",
            100: "other",
        }
        for tok, role in cases.items():
            self.assertEqual(atom_role(tok, R), role, tok)


class TestOffsets(unittest.TestCase):
    def test_offsets_from_markers(self):
        roles = ["header", "digit", "kind", "digit", "digit", "voice_tag"]
        self.assertEqual(event_offsets(roles), [0, 1, 0, 1, 2, 0])

    def test_pre_marker_is_negative(self):
        self.assertEqual(
            event_offsets(["digit", "digit", "kind", "digit"]), [-1, -1, 0, 1]
        )


class TestHits(unittest.TestCase):
    def test_role_and_offset_split(self):
        block = [8, 15, 1, 2]  # voice_tag, kind, digit, digit
        tf_pred = [15, 1, 9]  # predicts block[1..3]; block[3] missed
        out = finalize(position_hits(block, tf_pred, R))
        self.assertEqual(out["role"]["kind"]["acc"], 1.0)
        self.assertEqual(out["role"]["digit"]["acc"], 0.5)  # one hit, one miss
        self.assertEqual(out["offset"]["0"]["acc"], 1.0)  # the kind marker (offset 0)
        self.assertEqual(out["offset"]["2"]["acc"], 0.0)  # second field missed

    def test_merge(self):
        acc = {}
        merge_hits(acc, position_hits([8, 15, 1], [15, 1], R))
        merge_hits(acc, position_hits([8, 15, 1], [15, 9], R))
        out = finalize(acc)
        self.assertEqual(out["role"]["digit"]["n"], 2)
        self.assertEqual(out["role"]["digit"]["acc"], 0.5)


if __name__ == "__main__":
    unittest.main()
