"""Inspect the audio df schema + whether render is sensitive to intra-frame
write order. If render(adf) == render(stable-sort-by-reg adf), intra-frame
reorder is inaudible by construction (adf is frame-quantized)."""

import sys
import numpy as np
from types import SimpleNamespace
from preframr_audio.audio_driver import render_to_wav
from preframr_audio.sidwav import sidq
from preframr_tokens import RegLogParser, prepare_df_for_audio, read_initial_irq

DUMP = sys.argv[1]
BASE = dict(
    cents=50,
    exclude_list=None,
    min_irq=int(1.5e4),
    max_irq=int(2.5e4),
    min_song_tokens=256,
    diffq=4,
    loop_lookahead=3,
    coarsen_min_len=16,
    voice_trajectory_window=8,
    macro_flags="",
    meta_exclude_digi=False,
    meta_irq_lo=0,
    meta_irq_hi=0,
    meta_require=False,
)
MACROS = (
    "freq_trajectory_pass",
    "preset_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c3",
    "legato_pass_c4",
    "legato_pass_c7",
    "voice_canonical_block_order",
    "ctrl_bigram_pass",
    "loop_pass",
    "loop_transposed",
    "fuzzy_loop_pass",
    "fuzzy_fp_adsr",
    "coarsen_pass",
    "mode_vol_flip_pass",
    "voice_trajectory_pass",
    "voice_trajectory_distributed_pass",
    "set_to_diff_pass",
    "freq_nudge_pass",
    "release_update_pass",
    "ctrl_triple_pass",
    "lonely_catch_all",
)
cfg = dict(BASE)
for m in MACROS:
    cfg[m] = False
args = SimpleNamespace(**cfg)

parser = RegLogParser(args=args)
df = next(parser.parse(DUMP, max_perm=1, require_pq=False, reparse=True), None)
irq = read_initial_irq(df)
adf, rw = prepare_df_for_audio(df, {}, irq, sidq(), strict=False)
print("adf columns:", list(adf.columns))
print("adf shape:", adf.shape, " reg_widths keys:", list(rw)[:8])
print(adf.head(14).to_string())

# does adf carry an intra-frame cycle/clock column? show per-frame row counts
import collections

fc = collections.Counter(adf["f"].to_numpy().tolist())
multi = [f for f, c in fc.items() if c > 2]
print(f"\nframes with >2 writes: {len(multi)} / {len(fc)}")
if multi:
    f0 = multi[0]
    print(f"frame {f0} rows:")
    print(adf[adf["f"] == f0].to_string())
