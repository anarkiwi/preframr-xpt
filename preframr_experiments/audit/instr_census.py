import glob, pandas as pd, statistics as st
from preframr_tokens.events.oracle import ordered_writes
from preframr_tokens.events import stream as S

TIMBRE = {S.FLD_CTRL, S.FLD_AD, S.FLD_SR, S.PW_STEP, S.PW_RAMP}
MELODY = {S.NI_STEP, S.NI_RAMP, S.FD_STEP, S.FD_RAMP}
STRUCT = {S.VOICE_BASE + i for i in range(5)} | {
    S.TUNING,
    S.NOTE_TABLE,
    S.TICK,
    S.G_STEP,
    S.G_RAMP,
    S.KEYFRAME,
}
ONSET = {S.FLD_NOTE_ON}


def cat(atom):
    if atom in TIMBRE:
        return "timbre"
    if atom in MELODY:
        return "melody"
    if atom in ONSET:
        return "onset"
    if atom in STRUCT:
        return "struct"
    return None  # value digit/nibble -> attributed to current field


rows = []
for dp in sorted(
    glob.glob("/scratch/preframr/hvsc/MUSICIANS/D/Daglish_Ben/*.dump.parquet")
)[:12]:
    try:
        df = pd.read_parquet(dp, columns=["clock", "irq", "chipno", "reg", "val"])
        ow = ordered_writes(df)
        if not S.single_speed(ow):
            continue
        a = S.encode(ow, verify=False)
    except Exception:
        continue
    counts = {"timbre": 0, "melody": 0, "onset": 0, "struct": 0}
    cur = "struct"
    for atom in a:
        c = cat(atom)
        if c is not None:
            cur = c
        counts[cur] += 1
    tot = len(a)
    onsets = sum(1 for x in a if x in ONSET)
    rows.append(
        (
            tot,
            onsets,
            counts["timbre"] / tot,
            counts["melody"] / tot,
            counts["onset"] / tot,
            counts["struct"] / tot,
        )
    )
m = lambda i: st.mean([r[i] for r in rows])
print(f"n={len(rows)} mean_atoms={m(0):.0f} mean_onsets={m(1):.0f}")
print(
    f"  atom fraction: TIMBRE={m(2):.1%}  MELODY={m(3):.1%}  ONSET={m(4):.1%}  STRUCT={m(5):.1%}"
)
print(
    f"  => timbre (recurring instrument program) is ~{m(2):.0%} of the stream; front-load+reference could reclaim most of it for context"
)
