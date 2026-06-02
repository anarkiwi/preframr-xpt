"""Diagnostic: what do raw parsed atoms look like? op histogram + per-op reg/subreg
samples, so we can locate frequency-register writes for the interval probe."""
import argparse, sys
from collections import Counter
from preframr_tokens.blocks import glob_dumps, iter_voiced_blocks
from preframr_tokens.macros.motif_pass import _atoms_of
from preframr_tokens.reglogparser import RegLogParser
from preframr.args import add_args, apply_macro_flags_to_args

ap = add_args(argparse.ArgumentParser())
ap.add_argument("--motif-out"); ap.add_argument("--motif-k", type=int, default=0)
ap.add_argument("--motif-min-count", type=int, default=0)
ap.add_argument("--motif-min-composers", type=int, default=0)
ap.add_argument("--motif-mine-version", type=int, default=1)
args = ap.parse_args(sys.argv[1:])
apply_macro_flags_to_args(args)
args.motif_pass = False
parser = RegLogParser(args, __import__("logging"))
bp = RegLogParser(args, __import__("logging"))
streams = []
for name in glob_dumps(args.reglogs, args.max_files, require_pq=False):
    try:
        for df in parser.parse(name, max_perm=1, require_pq=False, reparse=True):
            for v in iter_voiced_blocks(df, getattr(args, "seq_len", 4096), bp, {}, stride=None):
                if not v.empty:
                    streams.append(_atoms_of(v))
    except (AssertionError, ValueError, KeyError):
        pass

ops = Counter(); reg_by_op = {}
for atoms in streams:
    for op, reg, subreg, val, diff in atoms:
        ops[op] += 1
        reg_by_op.setdefault(op, Counter())[reg] += 1
print(f"blocks={len(streams)} total_atoms={sum(ops.values())}")
print("op histogram:", ops.most_common())
for op, _ in ops.most_common(4):
    print(f"\nop {op}: top regs {reg_by_op[op].most_common(12)}")
# sample SET (op 0) atoms across a few regs
print("\nsample atoms (op,reg,subreg,val) first block:")
for a in streams[0][:25]:
    print("  ", a)
