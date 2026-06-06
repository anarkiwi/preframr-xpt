"""Extract Bach chorales -> 3-voice (S,A,B) frame-quantized piano-roll. Public-domain
Bach via music21 (BSD). Saves pitch + onset arrays per chorale."""

import numpy as np, json
from music21 import corpus

FPQ = 4  # frames per quarter note (16th-note grid)
paths = corpus.getComposer("bach")
chorales = []
used = 0
for p in paths:
    try:
        s = corpus.parse(p)
    except Exception:
        continue
    parts = list(s.parts)
    if len(parts) < 3:
        continue
    sel = [parts[0], parts[1], parts[-1]]  # Soprano, Alto, Bass
    # total length in frames
    qlen = float(s.highestTime)
    nf = int(round(qlen * FPQ))
    if nf < 16 or nf > 800:
        continue
    pitch = np.zeros((nf, 3), np.int16)  # 0 = rest, else MIDI
    onset = np.zeros((nf, 3), bool)
    ok = True
    for vi, part in enumerate(sel):
        for n in part.flatten().notes:
            m = max(n.pitches).midi if n.isChord else n.pitch.midi
            t0 = int(round(float(n.offset) * FPQ))
            t1 = int(round((float(n.offset) + float(n.duration.quarterLength)) * FPQ))
            if t0 >= nf:
                continue
            t1 = min(t1, nf)
            pitch[t0:t1, vi] = m
            if t0 < nf:
                onset[t0, vi] = True
    chorales.append({"pitch": pitch.tolist(), "onset": onset.tolist()})
    used += 1
    if used >= 200:
        break
print("chorales extracted:", len(chorales))
allp = np.concatenate([np.array(c["pitch"]).reshape(-1) for c in chorales])
nz = allp[allp > 0]
print(
    "midi range:",
    int(nz.min()),
    "-",
    int(nz.max()),
    "| distinct pitches:",
    len(np.unique(nz)),
)
print("avg frames/chorale:", np.mean([len(c["pitch"]) for c in chorales]))
json.dump(chorales, open("bach_chorales.json", "w"))
print("saved bach_chorales.json")
