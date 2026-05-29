"""Unit tests for the melody/ornament channel-factorization probe
(design/melody_channel_factorization.md): the extractor's interleaved skeleton+ornament
builder (interval coding, ornament offset, gate-on de-dup, ornament-before-note drop) and the
probe's torch-free scoring helpers (held-out-by-dump split, skeleton-only n-gram ceiling).
"""

import pandas as pd

from preframr_experiments.audit import extract_sid_melody as E
from preframr_experiments.audit import melody_channel_probe as P


def _df(rows):
    return pd.DataFrame(
        [{"clock": c, "reg": r, "val": v, "irq": 1} for c, r, v in rows]
    )


def test_voice_channels_labels_and_intervals():
    # voice 0: note A (gate on), two ornament freq writes, then note B (+intervals).
    df = _df(
        [
            (0, 1, 0x10),
            (0, 0, 0x00),
            (0, 4, 0x01),  # note A: freq + gate-on (skeleton)
            (2, 0, 0x40),
            (4, 0, 0x00),  # ornament freq wiggles (gate still on)
            (10, 4, 0x00),  # gate off
            (12, 1, 0x12),
            (12, 4, 0x01),  # note B: gate-on (skeleton)
        ]
    )
    toks, mask = E._voice_channels(df)[0]
    # first skeleton interval is 0; ornament tokens carry the ORN_OFFSET; B is a +interval.
    assert mask[0] is True and toks[0] == 0
    assert mask.count(True) == 2  # two gate-on notes
    assert all(t >= E.ORN_OFFSET for t, s in zip(toks, mask) if not s)
    assert all(abs(t) <= E.MAX_INTERVAL for t, s in zip(toks, mask) if s)


def test_voice_channels_gate_on_freq_write_not_double_counted():
    # the freq write coinciding with the gate-on IS the note; it must not also be an ornament.
    df = _df([(0, 1, 0x10), (0, 0, 0x00), (0, 4, 0x01)])
    toks, mask = E._voice_channels(df)[0]
    assert toks == [0] and mask == [True]


def test_voice_channels_ornament_before_first_note_dropped():
    # freq writes with no active note (gate never on) produce nothing.
    df = _df([(0, 0, 0x10), (2, 0, 0x20)])
    toks, mask = E._voice_channels(df)[0]
    assert toks == [] and mask == []


def test_clamp_interval_bounds():
    assert E._clamp_interval(100) == E.MAX_INTERVAL
    assert E._clamp_interval(-100) == -E.MAX_INTERVAL
    assert E._clamp_interval(3) == 3


def test_dump_multiplex_interleaves_voices_by_frame_and_ids_decode():
    # voice 0: note A (f0) + ornament (f2) + gate-off (f10) + note B (f12, real retrigger);
    # voice 1: one note (f0, regs 7/8/11).
    df = _df(
        [
            (0, 1, 0x10),
            (0, 0, 0x00),
            (0, 4, 0x01),
            (2, 0, 0x40),
            (10, 4, 0x00),
            (12, 1, 0x12),
            (12, 4, 0x01),
            (0, 8, 0x08),
            (0, 7, 0x00),
            (0, 11, 0x01),
        ]
    )
    tokens, is_skel, voice = E._dump_multiplex(df)
    # frame 0 carries both voices' note-ons (interleaved); ornament + voice-0 note B follow.
    assert voice[:2] == [0, 1] and is_skel[:2] == [True, True]
    assert is_skel == [True, True, False, True] and voice == [0, 1, 0, 0]
    # every id decodes back to (voice, channel, interval).
    for tid, sk, v in zip(tokens, is_skel, voice):
        assert tid // E.MUX_VOICE_STRIDE == v
        loc = tid % E.MUX_VOICE_STRIDE
        assert (loc < (E.MUX_SKEL_BASE + E.MUX_ORN_BASE) // 2) == sk


def test_voice_channels_unchanged_by_refactor():
    # the single-voice channel scheme (skeleton raw interval, ornament +ORN_OFFSET) is preserved.
    df = _df(
        [
            (0, 1, 0x10),
            (0, 0, 0x00),
            (0, 4, 0x01),
            (2, 0, 0x40),
            (12, 1, 0x12),
            (12, 4, 0x01),
        ]
    )
    toks, mask = E._voice_channels(df)[0]
    assert toks[0] == 0 and mask[0] is True
    assert all(t >= E.ORN_OFFSET for t, s in zip(toks, mask) if not s)


def test_split_is_held_out_by_dump():
    seqs = [(d, [0, 1, 2], [True, False, True]) for d in range(10)]
    train, test = P.split(seqs)
    train_dumps = {d for d, _, _ in train}
    test_dumps = {d for d, _, _ in test}
    assert train_dumps.isdisjoint(test_dumps)
    assert test_dumps and len(test_dumps) == 2  # 10 // 5


def test_skel_ngram_ceiling_scores_skeleton_positions_only():
    # context (1,2)->3 only ever precedes a skeleton 3; ornament target 9 is never scored.
    train = [(0, [1, 2, 3, 9, 1, 2, 3, 9], [False, False, True, False] * 2)]
    test = [(1, [1, 2, 3, 9], [False, False, True, False])]
    assert P.skel_ngram_ceiling(train, test, k=2) == 1.0
