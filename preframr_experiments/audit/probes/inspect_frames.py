"""Inspect real post-full_macros atom streams to design the intra-frame reorder:
the (op,reg,subreg) vocabulary with tiers, sample inter-frame blocks verbatim,
and the per-atom diff (timing) distribution within blocks -- the diff semantics
decide whether reordering is well-defined / how to recompute timing."""
import argparse
from collections import Counter
from types import SimpleNamespace

from preframr.args import add_args, apply_macro_flags_to_args
from preframr.utils import get_logger
import preframr_tokens.motif_mine as MM
import preframr_tokens.vocab_signature as VS
from preframr_tokens.stfconstants import (
    FRAME_REG, DELAY_REG, VOICE_REG, SUPER_FRAME_REG, VOICE_TRAJ_REG,
    LOOP_OP_REG,
)

NAMED = {FRAME_REG: "FRAME", DELAY_REG: "DELAY", VOICE_REG: "VOICE",
         SUPER_FRAME_REG: "SUPERFRAME", VOICE_TRAJ_REG: "VOICETRAJ",
         LOOP_OP_REG: "LOOP", -1: "PAD"}
CTRL_REGS = {4, 11, 18}

CAP = {}
def _capture(streams, composers, **kw):
    CAP.setdefault("s", []).extend(streams)
    class _D:
        def __len__(self): return 0
    return _D()
MM.mine_motifs = _capture

base = add_args(argparse.ArgumentParser()).parse_args(
    ["--no-require-pq", "--macro-config", "full_macros", "--max-files", "999999"]
)
apply_macro_flags_to_args(base)
OT = VS._op_tier_map()
CAP["s"] = []
MM.mine_dict_from_dumps(base, "/work/eval_b_marquis/*/*.dump.parquet",
                        max_files=8, k=1, min_count=1, min_composers=1,
                        logger=get_logger("ERROR"))
streams = [list(s) for s in CAP["s"]]
print(f"streams={len(streams)} atoms={sum(len(s) for s in streams)}")

def lab(reg):
    return NAMED.get(reg, ("CTRL%d" % reg) if reg in CTRL_REGS else str(reg))

vocab = Counter()
for s in streams:
    for a in s:
        vocab[(a[0], a[1], a[2])] += 1
print("\n== (op,reg,subreg) vocab [tier] ==")
for (op, reg, sub), c in vocab.most_common():
    t = VS._row_tier(SimpleNamespace(op=op, reg=reg), OT)
    print(f"  op={op:>3} reg={lab(reg):>6} sub={sub:>3}  n={c:>6}  {t}")

def blocks(s):
    cur = []
    for a in s:
        if a[1] == FRAME_REG:
            if cur:
                yield cur
            cur = [a]
        else:
            cur.append(a)
    if cur:
        yield cur

print("\n== sample inter-frame blocks (op,reg,subreg,val,diff) ==")
shown = 0
for blk in blocks(streams[0]):
    if not (2 <= len(blk) <= 9):
        continue
    print("  [")
    for a in blk:
        print(f"    op={a[0]:>3} reg={lab(a[1]):>6} sub={a[2]:>3} "
              f"val={a[3]:>6} diff={a[4]:>6}")
    print("  ]")
    shown += 1
    if shown >= 6:
        break

print("\n== diff (timing) distribution ==")
fr = Counter(); nonfr = Counter()
for s in streams:
    for a in s:
        (fr if a[1] == FRAME_REG else nonfr)[a[4]] += 1
def summ(c, name):
    tot = sum(c.values())
    z = c.get(0, 0)
    top = c.most_common(6)
    print(f"  {name}: n={tot} zero-diff={z} ({100*z/tot:.0f}%) top={top}")
summ(fr, "FRAME atoms")
summ(nonfr, "non-FRAME atoms")
