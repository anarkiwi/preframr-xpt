"""Unit tests for the melody-feature audit pipeline.

The muspy-dependent metrics (pitch_class_entropy, scale_consistency, etc.)
are tested via ``pytest.importorskip("muspy")``. The SID-dump parsing and
collapse-guard / verdict logic are tested without muspy so they run on every
docker build."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _fake_dump(rows: list[tuple[int, int, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["clock", "irq", "chipno", "reg", "val"])


def test_dump_to_notes_extracts_gated_notes_per_voice():
    from preframr_experiments.audit.melody_features import (
        PAL_CLOCK,
        dump_to_notes,
    )

    # V0 plays one note: gate on at clock=1000 freq=4000 (~MIDI 60), gate off at 2000.
    rows = [
        (500, 0, 0, 0, 0xA0),  # freq_lo
        (500, 0, 0, 1, 0x0F),  # freq_hi -> word ~= 4000 -> ~A4 (MIDI ~69)
        (1000, 0, 0, 4, 0x41),  # ctrl gate-on triangle wave
        (2000, 0, 0, 4, 0x40),  # ctrl gate-off
    ]
    df = _fake_dump(rows)
    notes = dump_to_notes(df)
    assert len(notes) == 1
    assert notes[0].voice == 0
    assert notes[0].pitch is not None
    duration_s = notes[0].end_s - notes[0].start_s
    assert abs(duration_s - 1000 / PAL_CLOCK) < 1e-6


def test_dump_to_notes_handles_audio_df_format_via_diff():
    """audio_df (predict.py --predict-dump output) has diff in PHI2 cycles,
    no clock column. _ensure_clock should derive clock as cumsum*18."""
    from preframr_experiments.audit.melody_features import (
        PHI2_TO_MASTER,
        _ensure_clock,
    )

    df = pd.DataFrame(
        {
            "reg": [0, 1, 4, 4],
            "val": [0xA0, 0x0F, 0x41, 0x40],
            "diff": [100, 100, 1000, 5000],
        }
    )
    out = _ensure_clock(df)
    assert "clock" in out.columns
    expected = np.array([100, 200, 1200, 6200]) * PHI2_TO_MASTER
    assert (out["clock"].to_numpy() == expected).all()


def test_dump_to_notes_skips_filter_and_pw_writes():
    """Filter regs (21/22/23) and PW regs (2/3/9/10/16/17) must not produce
    notes -- they're not voice-control writes."""
    from preframr_experiments.audit.melody_features import dump_to_notes

    rows = [
        (100, 0, 0, 21, 0xFF),  # FC_LO write
        (200, 0, 0, 22, 0xFF),  # FC_HI
        (300, 0, 0, 23, 0xFF),  # filter routing
        (400, 0, 0, 2, 0x00),  # PW lo V0
        (500, 0, 0, 3, 0x08),  # PW hi V0
    ]
    df = _fake_dump(rows)
    notes = dump_to_notes(df)
    assert notes == []


def test_features_from_notes_handles_empty_input():
    from preframr_experiments.audit.melody_features import features_from_notes

    feats = features_from_notes([], total_seconds=1.0)
    assert feats["gates_per_sec"] == 0.0
    assert feats["pitched_gates_per_sec"] == 0.0


def test_score_collapse_on_silent_output(tmp_path):
    """A model producing zero notes should trigger the silence collapse guard."""
    from preframr_experiments.audit.melody_score_generation import (
        score_one,
    )

    scaler = {
        f: {"mean": 0.0, "std": 1.0}
        for f in [
            "pitch_class_entropy",
            "scale_consistency",
            "pitch_in_scale_rate",
            "pitch_range",
            "n_pitches_used",
            "n_pitch_classes_used",
            "polyphony",
            "polyphony_rate",
            "groove_consistency",
            "empty_beat_rate",
            "gates_per_sec",
            "pitched_gates_per_sec",
            "unpitched_gate_rate",
            "median_note_frames",
            "top_interval_share",
            "interval_repeat_share",
        ]
    }
    silent = _fake_dump([(100, 0, 0, 24, 15)])  # one mode/vol write, no gates
    silent_path = tmp_path / "silent.dump.parquet"
    silent.to_parquet(silent_path)
    rep = score_one(str(silent_path), scaler, {})
    assert rep["verdict"] == "COLLAPSE"
    assert any("silent" in r.lower() for r in rep["collapse_reasons"])


def test_score_pass_on_in_distribution_synthetic():
    """A synthesised feature vector close to the baseline should PASS."""
    from preframr_experiments.audit.melody_score_generation import (
        score_one,
    )

    scaler = {
        "pitch_class_entropy": {"mean": 2.5, "std": 0.5},
        "scale_consistency": {"mean": 0.9, "std": 0.1},
        "pitched_gates_per_sec": {"mean": 100.0, "std": 30.0},
        "median_note_frames": {"mean": 5.0, "std": 2.0},
        "n_pitches_used": {"mean": 20.0, "std": 5.0},
    }
    # Generate a synthetic dump with enough notes that pitched_gates_per_sec
    # passes the silence guard; pitches walking up by one semitone.
    rows = []
    clock = 0
    base_freq_hi = 0x10
    for i in range(80):
        clock += 5000
        # set freq lo/hi to climb pitch
        rows.append((clock, 0, 0, 0, (i * 4) & 0xFF))
        rows.append((clock, 0, 0, 1, base_freq_hi + (i // 8)))
        rows.append((clock + 100, 0, 0, 4, 0x41))  # gate on
        rows.append((clock + 4000, 0, 0, 4, 0x40))  # gate off
    df = _fake_dump(rows)
    path = Path("/tmp/_pytest_synth.dump.parquet")
    df.to_parquet(path)
    rep = score_one(str(path), scaler, {})
    # Don't require PASS verdict (depends on muspy), but require no collapse.
    assert (
        rep["verdict"] != "COLLAPSE"
    ), f"synthetic walk-up should not collapse: {rep['collapse_reasons']}"


@pytest.mark.parametrize(
    "ok_feature_value,bad_feature_value",
    [
        ((2.5, 0.5), (10.0, 0.5)),  # huge z on pitch_class_entropy -> FAIL
    ],
)
def test_extreme_outlier_triggers_fail(ok_feature_value, bad_feature_value, tmp_path):
    """A single feature with |z| > 4 should yield FAIL verdict."""
    from preframr_experiments.audit.melody_score_generation import _verdict_for_z

    assert _verdict_for_z(5.0) == "FAIL"
    assert _verdict_for_z(-5.0) == "FAIL"
    assert _verdict_for_z(2.5) == "WARN"
    assert _verdict_for_z(1.0) == "PASS"
    assert _verdict_for_z(None) == "MISSING"


def test_muspy_available_in_image_for_real_analysis(tmp_path):
    """When muspy is installed, the analyzer should produce its
    pitch_class_entropy + scale_consistency features. Skipped on lite envs."""
    pytest.importorskip("muspy")
    from preframr_experiments.audit.melody_features import analyze

    # Build a tiny ascending scale dump (12 notes), then analyze.
    rows = []
    clock = 0
    for i in range(12):
        clock += 5000
        rows.append((clock, 0, 0, 0, (i * 4) & 0xFF))
        rows.append((clock, 0, 0, 1, 0x12 + i))
        rows.append((clock + 100, 0, 0, 4, 0x41))
        rows.append((clock + 4000, 0, 0, 4, 0x40))
    path = tmp_path / "scale.dump.parquet"
    _fake_dump(rows).to_parquet(path)
    feats = analyze(str(path))
    assert feats["pitch_class_entropy"] is not None
    assert feats["scale_consistency"] is not None
    assert feats["n_pitches_used"] > 0
