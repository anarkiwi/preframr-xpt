"""Unit tests for the unified pitch encoder/decoder (design/unified_pitch_encoding.md): the semitone
LUT + residual factoring, level-change∪gate note segmentation, the unified ornament descriptor fitter,
and decode round-trip (PLAIN notes are LUT-exact)."""

from preframr_experiments.audit import unified_pitch as U


def test_lut_round_trips_through_note_resid():
    # LUT[m] decoded back to the nearest semitone is m, with ~0 residual.
    for m in (40, 52, 69, 88):
        note, resid = U.fn_to_note_resid(U.LUT[m])
        assert note == m
        assert abs(resid) < 1.0


def test_segment_notes_level_change_union_gate():
    # gate-on at frame 0 (note A); then a sustained new level (note B) held >= MIN_HOLD with no gate.
    a, b, c = U.LUT[60], U.LUT[64], U.LUT[60]
    # l1 holds only freq CHANGES: B is written at frame 10 and held until the next change at 25.
    l1 = [(0, a), (10, b), (25, c)]
    l2 = [(0, a)]  # only one gate-on
    onsets = U.segment_notes(l1, l2)
    frames = [fr for fr, _ in onsets]
    assert (
        0 in frames and 10 in frames
    )  # gate-on AND the held level-change both become notes
    assert dict(onsets)[10] == 64


def test_segment_notes_fast_arp_not_segmented():
    # a fast alternation that never holds MIN_HOLD frames is not split into a note per arp step
    # (only the gate-on, plus at most the trailing sustained value, become notes).
    l1 = [(i, U.LUT[60 if i % 2 == 0 else 72]) for i in range(12)]
    l2 = [(0, U.LUT[60])]
    onsets = U.segment_notes(l1, l2)
    assert len(onsets) <= 2  # not 12; the arp is ornament of the held note


def test_fit_descriptor_primitives():
    base = 60
    assert U._fit_descriptor(base, [])[0] == "PLAIN"
    assert U._fit_descriptor(base, [(1, U.LUT[72]), (2, U.LUT[60])])[0] == "OCTAVE|+"
    # non-monotonic small offset set -> ARP (monotonic would be SLIDE).
    arp = [(1, U.LUT[64]), (2, U.LUT[60]), (3, U.LUT[67]), (4, U.LUT[60])]
    assert U._fit_descriptor(base, arp)[0].startswith("ARP|")
    assert U._fit_descriptor(base, [(1, U.LUT[61]), (2, U.LUT[62]), (3, U.LUT[63])])[
        0
    ].startswith("SLIDE|+")


def test_vib_bucket_detects_sub_semitone_oscillation():
    # >=3 writes wobbling sub-semitone around one semitone -> non-zero vibrato bucket.
    seg = [
        (f, U.midi_to_fn_f(60 + c / 100.0))
        for f, c in [(1, -20), (2, 0), (3, 20), (4, 0)]
    ]
    assert U._vib_bucket(seg) >= 1
    # all on the exact semitone -> no vibrato.
    assert U._vib_bucket([(f, U.LUT[60]) for f in range(4)]) == 0
    # too few writes -> 0.
    assert U._vib_bucket([(1, U.LUT[60])]) == 0


def test_fn_from_note_cents_round_trips_within_quantum():
    # reconstructing note+cents stays within the cents quantum of the source freq.
    src = U.midi_to_fn_f(60 + 0.23)  # 23 cents sharp
    note, cents = U.fn_to_note_resid(src)
    recon = U.fn_from_note_cents(note, cents)
    rc = U.fn_to_note_resid(recon)
    assert (
        note == 60
        and abs((rc[0] + rc[1] / 100.0) - (note + cents / 100.0))
        <= U.CENTS_RES / 100.0 + 1e-6
    )


def test_encode_decode_plain_is_lut_exact():
    # two clean held notes (gate-on, no intra-note writes) -> PLAIN; decode returns LUT freqs.
    l1 = [(0, U.LUT[60]), (20, U.LUT[67])]
    l2 = [(0, U.LUT[60]), (20, U.LUT[67])]
    recs = U.encode_voice(l1, l2)
    assert [r["desc"] for r in recs] == ["PLAIN", "PLAIN"]
    assert recs[1]["skel"] == 7  # 67 - 60
    frames = U.decode_notes(recs, frames_per_note=4)
    assert all(fn == U.LUT[note] for note, fn in frames)  # LUT-exact
