"""Interval-reducibility on the post-transform atom sequence (dataset.csv.zst):
per (song, voice) extract reg-0 FREQ_TRAJ anchor pitches (V0/TERMINAL = HI<<8|LO),
convert to semitones, and compare ABSOLUTE-pitch vs successive-INTERVAL entropy.
Voice tracked via VOICE_REG(-126); intervals never cross voice/song boundaries."""
import sys, csv, math
from collections import Counter, defaultdict

MAX_SONGS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
FREQ_TRAJ_OP, FREQ_REG, VOICE_REG = 45, 0, -126

seqs = defaultdict(list)          # (song, voice) -> [anchor freq, ...]
cur_voice, cur_song = 0, None
pend = {}                          # hi/lo being assembled for current (song,voice)
songs_seen = set()

r = csv.DictReader(sys.stdin)
for row in r:
    i = row["i"]
    if i != cur_song:
        cur_song = i
        songs_seen.add(i)
        if len(songs_seen) > MAX_SONGS:
            break
        cur_voice, pend = 0, {}
    op, reg = int(row["op"]), int(row["reg"])
    if op == 0 and reg == VOICE_REG:
        cur_voice = int(row["val"])
        continue
    if op != FREQ_TRAJ_OP or reg != FREQ_REG:
        continue
    subreg, val = int(row["subreg"]), int(row["val"])
    if subreg == 0:          # FLAGS -> new event
        pend = {}
    elif subreg == 1:        # HI
        pend["hi"] = val
    elif subreg == 2:        # LO
        pend["lo"] = val
        if "hi" in pend:
            seqs[(i, cur_voice)].append((pend["hi"] << 8) | pend["lo"])
            pend = {}

abs_c, int_c = Counter(), Counter()
per_voice = defaultdict(lambda: [Counter(), Counter()])
for (song, voice), freqs in seqs.items():
    sem = [round(12 * math.log2(f)) for f in freqs if f > 0]
    coll = [sem[0]] if sem else []
    for s in sem[1:]:
        if s != coll[-1]:
            coll.append(s)
    ivals = [b - a for a, b in zip(coll, coll[1:])]
    abs_c.update(coll); int_c.update(ivals)
    per_voice[voice][0].update(coll); per_voice[voice][1].update(ivals)


def stat(c):
    N = sum(c.values())
    H = -sum((v / N) * math.log2(v / N) for v in c.values())
    return N, len(c), H, 2 ** H, max(c.values()) / N


print(f"songs={len(songs_seen)-1}  (song,voice) lines={len(seqs)}  "
      f"anchor events={sum(len(v) for v in seqs.values())}")
for name, c in (("ABSOLUTE semitone", abs_c), ("INTERVAL (relative)", int_c)):
    if not c:
        print(f"  {name}: empty"); continue
    N, k, H, eff, maj = stat(c)
    print(f"  {name:<22} N={N:>7} distinct={k:>4} entropy={H:>5.2f}b  "
          f"eff_vocab={eff:>5.0f}  majority_floor={maj:.3f}")
if abs_c and int_c:
    Ha, Hi = stat(abs_c)[2], stat(int_c)[2]
    ea, ei = stat(abs_c)[3], stat(int_c)[3]
    print(f"\n  relative encoding: entropy {Ha:.2f}->{Hi:.2f}b ({Ha-Hi:+.2f}); "
          f"eff_vocab {ea:.0f}->{ei:.0f}; majority floor "
          f"{stat(abs_c)[4]:.3f}->{stat(int_c)[4]:.3f}")
    tot = sum(int_c.values())
    print("  top intervals (semitone): " +
          ", ".join(f"{kk:+d}:{100*v/tot:.0f}%" for kk, v in int_c.most_common(9)))
print("\n  per-voice abs-entropy -> interval-entropy (bits):")
for v in sorted(per_voice):
    ac, ic = per_voice[v]
    if sum(ac.values()) < 50:
        continue
    print(f"    voice {v}: abs {stat(ac)[2]:.2f} (eff {stat(ac)[3]:.0f}) "
          f"-> int {stat(ic)[2]:.2f} (eff {stat(ic)[3]:.0f})  n={sum(ac.values())}")
