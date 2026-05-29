"""Extract per-(dump,voice) melodic onset-pitch sequences from raw SID dumps for the
melody data-gap ladder (design/melody_data_gap_ladder.md). Host-side (pandas; no torch).

Ladder levels (onset definition):
  L1 all-freq   : every freq-register change is an onset (ornament-laden; the high
                  predictability here is a vibrato/sustain-repeat ARTIFACT, not melody).
  L2 gateanchor : onset only at control-reg gate-on transitions (the musical note-ons).
  L3 dearp      : L2 + minimum note duration (>=4 frames) — collapse fast arps/retriggers.
With --lead, keep only the most-melodic voice per dump (max distinct gate-on pitches).
With --composer NAME, restrict to that composer's subdir (homogeneity test, e.g. DRAX).

With --channels, emit instead the channel-factorization probe stream
(design/melody_channel_factorization.md): one temporally-ordered token sequence per
(dump, lead voice) interleaving SKELETON note-ons (interval from the previous skeleton note)
and ORNAMENT freq writes (interval from the active skeleton note), in disjoint id ranges,
with a per-token is_skel mask. Feeds `audit.melody_channel_probe`, which scores held-out
accuracy on skeleton positions only — to test whether multiplexing ornament into the melody
prediction position steals skeleton predictability. Intervals (both channels) are
key-invariant, matching the deployed interval-V0 encoding.

Pitch is MIDI (semitone) from SID Fn (equal temperament). For L1/L2/L3, feed the output json
to `melody_ladder` (measure on intervals — key-invariant — via to-intervals, see the doc).

Usage:
  python3 -m preframr_experiments.audit.extract_sid_melody \
      --dumps '<root>/*/*.dump.parquet' --level L2 --lead --out mini_L2lead.json
  python3 -m preframr_experiments.audit.extract_sid_melody \
      --dumps '<root>/*/*.dump.parquet' --channels --out mini_channels.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math

import pandas as pd

CLOCK_RATE = 985248


def fn_to_midi(fn: int) -> int:
    if fn < 8:
        return 0
    hz = fn * CLOCK_RATE / 16777216.0
    if hz < 16:
        return 0
    m = round(69 + 12 * math.log2(hz / 440.0))
    return int(m) if 24 <= m <= 108 else 0


# channel-factorization id ranges: skeleton tokens are the raw signed interval;
# ornament tokens are offset so the two channels never share an id (is_skel is the real
# channel signal, the offset just prevents accidental vocab collision in the shared alphabet).
ORN_OFFSET = 1000
MAX_INTERVAL = 24


def _voice_l1_l2(d):
    """Per-voice ``(l1, l2)`` raw onset lists ``[(frame, midi), ...]``: l1 = every freq
    change (ornament-laden), l2 = control-reg gate-on note-ons. Single-sourced by the level
    selector and the channel builder."""
    fc = int(d["irq"][d["irq"] > 0].mode().iloc[0]) if (d["irq"] > 0).any() else 19592
    reg = d["reg"].to_numpy()
    val = d["val"].to_numpy()
    clk = d["clock"].to_numpy()
    out = []
    for v in range(3):
        lo, hi, ctrl = v * 7, v * 7 + 1, v * 7 + 4
        cl = ch = gate = 0
        l1, l2 = [], []
        for i in range(len(reg)):
            r, x, fr = reg[i], int(val[i]), clk[i] // fc
            if r == lo and x != cl:
                cl = x
                m = fn_to_midi((ch << 8) | cl)
                if m:
                    l1.append((fr, m))
            elif r == hi and x != ch:
                ch = x
                m = fn_to_midi((ch << 8) | cl)
                if m:
                    l1.append((fr, m))
            elif r == ctrl:
                ng = x & 1
                if ng and not gate:
                    m = fn_to_midi((ch << 8) | cl)
                    if m:
                        l2.append((fr, m))
                gate = ng
        out.append((l1, l2))
    return out


def _voice_onsets(d, level):
    """Return list of 3 per-voice onset sequences [(frame, midi), ...] at the level."""
    out = []
    for l1, l2 in _voice_l1_l2(d):
        if level == "L1":
            seq = l1
        elif level == "L2":
            seq = l2
        else:  # L3 de-arp
            seq, last = [], -999
            for fr, m in l2:
                if fr - last >= 4:
                    seq.append((fr, m))
                    last = fr
        out.append(seq)
    return out


def _clamp_interval(iv):
    return max(-MAX_INTERVAL, min(MAX_INTERVAL, int(iv)))


def _voice_channel_events(l1, l2):
    """One voice's (frame, is_skel, interval) events in frame order. Skeleton = gate-on note-on,
    interval from the previous skeleton note (first = 0). Ornament = freq change between gate-ons,
    interval from the active skeleton note (dropped before the first note). A freq change
    coinciding with a gate-on IS the note write, so it is counted skeleton-only. Raw interval
    (no ORN_OFFSET) — callers tag the channel."""
    skel_frames = {fr for fr, _ in l2}
    events = [(fr, True, m) for fr, m in l2]
    events += [(fr, False, m) for fr, m in l1 if fr not in skel_frames]
    events.sort(key=lambda e: (e[0], not e[1]))  # skeleton first within a frame
    out = []
    prev_skel = cur_skel = None
    for fr, is_s, m in events:
        if is_s:
            iv = 0 if prev_skel is None else _clamp_interval(m - prev_skel)
            out.append((fr, True, iv))
            prev_skel = cur_skel = m
        elif cur_skel is not None:
            out.append((fr, False, _clamp_interval(m - cur_skel)))
    return out


def _voice_channels(d):
    """Per-voice interleaved (skeleton, ornament) interval-token streams + is_skel mask;
    ornament offset by ORN_OFFSET so the channels never share an id."""
    out = []
    for l1, l2 in _voice_l1_l2(d):
        toks, mask = [], []
        for _fr, is_s, iv in _voice_channel_events(l1, l2):
            toks.append(iv if is_s else ORN_OFFSET + iv)
            mask.append(is_s)
        out.append((toks, mask))
    return out


# multiplex id scheme: per-voice local code in [76, 324] (skeleton 100+iv, ornament 300+iv),
# global id = voice*1000 + local. voice/channel/interval all recoverable: voice=id//1000,
# local=id%1000, is_skel = local < 200, interval = local-100 (skel) | local-300 (orn).
MUX_VOICE_STRIDE = 1000
MUX_SKEL_BASE = 100
MUX_ORN_BASE = 300


def _dump_multiplex(d):
    """All 3 voices' channel events merged in frame order into ONE token stream (the deployment
    multiplexing condition) + per-token is_skel / voice arrays. Token id encodes voice+channel+
    interval (see MUX_* scheme), so the model can read voice from the token; the cost under test
    is interleaving 3 independent voice streams into one next-token position."""
    allev = []
    for v, (l1, l2) in enumerate(_voice_l1_l2(d)):
        for fr, is_s, iv in _voice_channel_events(l1, l2):
            allev.append((fr, v, is_s, iv))
    allev.sort(key=lambda e: (e[0], e[1], not e[2]))
    tokens, is_skel, voice = [], [], []
    for _fr, v, is_s, iv in allev:
        local = (MUX_SKEL_BASE + iv) if is_s else (MUX_ORN_BASE + iv)
        tokens.append(v * MUX_VOICE_STRIDE + local)
        is_skel.append(is_s)
        voice.append(v)
    return tokens, is_skel, voice


ORN_CLAMP = (
    48  # note-relative ornament offsets span multiple octaves (arps); clamp wider
)


def _fit_ornament(offs, dur):
    """Fit one held note's intra-note offset series (note-relative semitones) to a parametric
    descriptor STRING (design/ornament_transfer.md). plain / slide / arp(offset-set) / vibrato /
    residual, with a coarse fast|slow rate bucket. Note-relative => transposition-invariant.
    """
    if len(offs) < 2:
        return "PLAIN"
    distinct = sorted(set(offs))
    rng = max(offs) - min(offs)
    diffs = [b - a for a, b in zip(offs, offs[1:])]
    mono = all(x >= 0 for x in diffs) or all(x <= 0 for x in diffs)
    rate = "f" if len(offs) / max(1, dur) >= 0.4 else "s"
    if mono and rng >= 2:
        return f"SLIDE|{'+' if offs[-1] >= offs[0] else '-'}|{'big' if rng >= 7 else 'sml'}|{rate}"
    if len(distinct) <= 4:
        return "ARP|" + ",".join(str(x) for x in distinct) + f"|{rate}"
    if rng <= 3:
        return f"VIB|{min(rng, 3)}|{rate}"
    return "RESID"


def _dump_ornament_records(d):
    """Lead voice (most note-ons): per-note records {skel interval, raw note-relative offset
    series, parametric descriptor} for the ornament-transfer A/B probe."""
    vl = _voice_l1_l2(d)
    best = max(range(3), key=lambda v: len(vl[v][1]))
    l1, l2 = vl[best]
    if len(l2) < 2:
        return []
    notes = []
    prev_base = None
    for i, (a, base) in enumerate(l2):
        b = l2[i + 1][0] if i + 1 < len(l2) else a + 8
        skel = 0 if prev_base is None else _clamp_interval(base - prev_base)
        prev_base = base
        offs = [max(-ORN_CLAMP, min(ORN_CLAMP, m - base)) for fr, m in l1 if a < fr < b]
        notes.append({"skel": skel, "offs": offs, "desc": _fit_ornament(offs, b - a)})
    return notes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dumps", required=True, help="glob for *.dump.parquet")
    ap.add_argument("--level", choices=("L1", "L2", "L3"), default="L2")
    ap.add_argument(
        "--lead", action="store_true", help="keep only most-melodic voice/dump"
    )
    ap.add_argument("--composer", default=None, help="restrict to this composer subdir")
    ap.add_argument("--min-notes", type=int, default=12)
    ap.add_argument(
        "--channels",
        action="store_true",
        help="emit lead-voice interleaved skeleton+ornament probe stream (is_skel mask)",
    )
    ap.add_argument(
        "--multiplex",
        action="store_true",
        help="emit ALL-3-voice frame-multiplexed stream (tokens/is_skel/voice) for the "
        "cross-voice multiplexing probe",
    )
    ap.add_argument(
        "--ornament",
        action="store_true",
        help="emit per-note {skel, offs, desc} records (lead voice) for the ornament-transfer "
        "A/B probe (raw per-frame vs parametric descriptor)",
    )
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    files = sorted(glob.glob(cli.dumps))
    if cli.composer:
        files = [f for f in files if f"/{cli.composer}/" in f]
    seqs = []
    for di, f in enumerate(files):
        d = pd.read_parquet(f).sort_values("clock")
        if cli.ornament:
            notes = _dump_ornament_records(d)
            if len(notes) >= cli.min_notes:
                seqs.append({"dump": di, "notes": notes})
            continue
        if cli.multiplex:
            tokens, is_skel, voice = _dump_multiplex(d)
            if sum(is_skel) >= cli.min_notes:
                seqs.append(
                    {"dump": di, "tokens": tokens, "is_skel": is_skel, "voice": voice}
                )
            continue
        if cli.channels:
            cand = [
                (sum(mask), toks, mask)
                for toks, mask in _voice_channels(d)
                if sum(mask) >= cli.min_notes
            ]
            if not cand:
                continue
            _, toks, mask = max(cand, key=lambda z: z[0])
            seqs.append({"dump": di, "tokens": toks, "is_skel": mask})
            continue
        vseqs = _voice_onsets(d, cli.level)
        if cli.lead:
            cand = [
                (len({m for _, m in s}), s) for s in vseqs if len(s) >= cli.min_notes
            ]
            if not cand:
                continue
            _, s = max(cand, key=lambda z: z[0])
            seqs.append({"dump": di, "pitch": [m for _, m in s]})
        else:
            for s in vseqs:
                if len(s) >= cli.min_notes:
                    seqs.append({"dump": di, "pitch": [m for _, m in s]})
    json.dump({"seqs": seqs}, open(cli.out, "w"))
    if cli.ornament:
        notes = sum(len(s["notes"]) for s in seqs)
        orn = sum(1 for s in seqs for n in s["notes"] if n["desc"] != "PLAIN")
        from collections import Counter

        tc = Counter(n["desc"].split("|")[0] for s in seqs for n in s["notes"])
        print(
            f"{cli.out}: {len(seqs)} seqs, {notes} notes, {orn} ornamented "
            f"({100 * orn / max(notes, 1):.0f}%), types {dict(tc)} ({len(files)} dumps, ornament)"
        )
    elif cli.multiplex:
        ns = sum(sum(s["is_skel"]) for s in seqs)
        no = sum(len(s["is_skel"]) - sum(s["is_skel"]) for s in seqs)
        nv = sum(len(set(s["voice"])) for s in seqs)
        print(
            f"{cli.out}: {len(seqs)} seqs, {ns} skeleton + {no} ornament tokens, "
            f"{nv / max(len(seqs), 1):.1f} voices/seq avg ({len(files)} dumps, multiplex)"
        )
    elif cli.channels:
        ns = sum(sum(s["is_skel"]) for s in seqs)
        no = sum(len(s["is_skel"]) - sum(s["is_skel"]) for s in seqs)
        print(
            f"{cli.out}: {len(seqs)} seqs, {ns} skeleton + {no} ornament tokens "
            f"({len(files)} dumps, channels)"
        )
    else:
        n = sum(len(s["pitch"]) for s in seqs)
        print(
            f"{cli.out}: {len(seqs)} seqs, {n} onsets ({len(files)} dumps, level {cli.level})"
        )


if __name__ == "__main__":
    main()
