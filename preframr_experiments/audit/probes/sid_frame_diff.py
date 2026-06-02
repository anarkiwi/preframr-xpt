"""Frame-diff audit: compare the SID register bytes a raw dump presents to the chip against the bytes a
parse->decode->render pipeline would present, per frame, per register. Both sides are reduced to a settled
per-frame state (last write per byte-reg, forward-filled = a "minimal squeeze" of redundant writes), aligned
by frame ordinal, then diffed. Discrete registers (CTRL gate/waveform, ADSR) MUST match exactly and in the
same frame; FREQ/PW differ only by the cent-quantisation tolerance. Anything else is a pipeline bug.

Usage: sid_frame_diff.py <dump.parquet> [config]
  config: 'prod' (default_tokenizer_args), 'skel', 'base', 'stack'  (default 'stack')
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, "/tok")
sys.path.insert(0, "/tok/tests")
sys.path.insert(0, "/aud")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from preframr_tokens.reglogparser import (  # noqa: E402
    RegLogParser,
    prepare_df_for_audio,
    read_initial_irq,
)
from preframr_tokens.tokenizer_config import default_tokenizer_args  # noqa: E402
from preframr_audio.sidwav import sidq as sidq_fn  # noqa: E402
from preframr_audio.audio_driver import df_to_packets  # noqa: E402
from preframr_audio._reg_mappers import FreqMapper  # noqa: E402
from parse_probes import parse_args  # noqa: E402

PAL = 985248
# SID register byte -> human label / class
VOICE = {0: (0, 1, 2, 3, 4, 5, 6), 1: (7, 8, 9, 10, 11, 12, 13), 2: (14, 15, 16, 17, 18, 19, 20)}
CLASS = {}
for v, (flo, fhi, plo, phi, ctrl, ad, sr) in VOICE.items():
    CLASS[flo] = f"v{v}.FREQ_LO"; CLASS[fhi] = f"v{v}.FREQ_HI"
    CLASS[plo] = f"v{v}.PW_LO"; CLASS[phi] = f"v{v}.PW_HI"
    CLASS[ctrl] = f"v{v}.CTRL"; CLASS[ad] = f"v{v}.AD"; CLASS[sr] = f"v{v}.SR"
for r, n in {21: "FC_LO", 22: "FC_HI", 23: "RES_FILT", 24: "MODE_VOL"}.items():
    CLASS[r] = n
EXACT_REGS = [r for r, c in CLASS.items() if (".CTRL" in c or ".AD" in c or ".SR" in c
                                              or c in ("RES_FILT", "MODE_VOL"))]
FREQ_REGS = [r for r, c in CLASS.items() if "FREQ" in c]


def dump_frame_state(dump_path):
    """Raw dump -> (n_frames, 25) settled byte state. Rows sharing an ``irq`` value are one player call;
    settle the last write per byte-reg in each frame, forward-fill across frames (the chip holds last value)."""
    df = pd.read_parquet(dump_path)
    df = df[(df["chipno"] == 0)].sort_values(["irq", "clock"]).reset_index(drop=True)
    cur = np.zeros(25, dtype=np.int64)
    frames = []
    for _irq, grp in df.groupby("irq", sort=True):
        for reg, val in zip(grp["reg"].to_numpy(), grp["val"].to_numpy()):
            r = int(reg)
            if 0 <= r <= 24:
                cur[r] = int(val) & 0xFF
        frames.append(cur.copy())
    return np.asarray(frames)


def pipeline_frame_state(dump_path, cfg):
    """Parse + decode + the exact writes ``df_to_packets`` presents to the SID (freq cent-index mapped back
    via FreqMapper, lo/hi split), settled per frame -> (n_frames, 25)."""
    args = build_args(cfg)
    xdf = next(RegLogParser(args=args).parse(dump_path, max_perm=1, require_pq=False, reparse=True), None)
    assert xdf is not None, "pipeline produced no rows"
    irq = read_initial_irq(xdf)
    adf, rw = prepare_df_for_audio(xdf, {}, irq, sidq_fn(), strict=False)
    fm = FreqMapper(cents=50)
    cur = np.zeros(25, dtype=np.int64)
    frames = []
    for pkt in df_to_packets(adf, rw, fm, irq_cycles=int(irq), clock_frequency=PAL):
        for op in pkt.ops:
            r = int(op.reg)
            if 0 <= r <= 24:
                cur[r] = int(op.val) & 0xFF
        frames.append(cur.copy())
    return np.asarray(frames)


_BASE = dict(skeleton_pass=True, trajectory_anchor_pass=True, stamp_pass=True,
             sweep_pass=True, patch_pass=True, held_arp=True)
_STACK = dict(wavetable_pass=True, zero_plain=True, wt_short=True, wt_oneshot=True,
              slide_wide=True, slide_landing=True, sweep_loop=True)


def build_args(cfg):
    if cfg == "prod":
        return default_tokenizer_args(cents=50)
    if cfg == "skel":
        return parse_args(skeleton_pass=True, trajectory_anchor_pass=True, held_arp=True)
    if cfg == "base":
        return parse_args(**_BASE)
    if cfg == "stack":
        return parse_args(**{**_BASE, **_STACK})
    raise SystemExit(f"unknown config {cfg}")


def best_offset(ref, test, regs, span=24):
    """Find the frame shift (test delayed by k) minimising discrete-register mismatch — the dump's irq grid
    and the parser's FRAME_REG grid differ by a few leading init frames, so align before diffing."""
    best_k, best_bad = 0, None
    rr = ref[:, regs].astype(np.int64)
    for k in range(-span, span + 1):
        if k >= 0:
            a, b = rr[: len(rr) - k], test[k:, regs].astype(np.int64)
        else:
            a, b = rr[-k:], test[: len(test) + k, regs].astype(np.int64)
        m = min(len(a), len(b))
        if m < 100:
            continue
        bad = int((a[:m] != b[:m]).any(axis=1).sum())
        if best_bad is None or bad < best_bad:
            best_bad, best_k = bad, k
    return best_k


def report(ref, test, cfg):
    k = best_offset(ref, test, EXACT_REGS)
    if k > 0:
        test = test[k:]
    elif k < 0:
        ref = ref[-k:]
    n = min(len(ref), len(test))
    print(f"=== SID FRAME DIFF: raw dump vs '{cfg}' pipeline ===")
    print(f"frames: raw vs pipeline aligned at offset k={k}; compared={n}")
    ref, test = ref[:n], test[:n]
    diff = ref != test
    # per-register summary
    print(f"\n{'reg':>3} {'class':12} {'frames_diff':>11} {'first':>7} {'maxΔ':>6}  sample (frame: raw->pipe)")
    bad_exact = []
    skip = 4  # leading init frames where the irq grid and FRAME_REG grid settle
    for r in range(25):
        d = diff[:, r].copy()
        d[:skip] = False
        c = int(d.sum())
        if not c:
            continue
        fr = int(np.argmax(d))
        maxd = int(np.abs(ref[:, r].astype(int) - test[:, r].astype(int)).max())
        ex = [(int(f), int(ref[f, r]), int(test[f, r])) for f in np.where(d)[0][:3]]
        exs = " ".join(f"{f}:{a}->{b}" for f, a, b in ex)
        cls = CLASS.get(r, f"reg{r}")
        flag = ""
        if r in EXACT_REGS:
            flag = "  <<< MUST MATCH"
            bad_exact.append((r, cls, c))
        # distinguish cent-quant (small) from garbage (large) for freq
        if r in FREQ_REGS:
            big = int((np.abs(ref[skip:, r].astype(int) - test[skip:, r].astype(int)) > 4).sum())
            flag = f"  freq Δ>4 in {big} frames" + ("  <<< GARBAGE" if big > n // 10 else "")
        print(f"{r:>3} {cls:12} {c:>11} {fr:>7} {maxd:>6}  {exs}{flag}")
    # class rollup
    print("\n--- class rollup (frames with any diff in class) ---")
    for label, regs in (("CTRL (gate/wave)", [r for r in range(25) if CLASS.get(r, "").endswith("CTRL")]),
                        ("ADSR", [r for r in range(25) if CLASS.get(r, "")[-3:] in (").AD", "AD", "SR") and ("AD" in CLASS.get(r,"") or "SR" in CLASS.get(r,""))]),
                        ("FREQ", FREQ_REGS),
                        ("PW", [r for r in range(25) if "PW" in CLASS.get(r, "")])):
        if not regs:
            continue
        any_d = diff[:, regs].any(axis=1)
        print(f"  {label:18}: {int(any_d.sum())}/{n} frames differ")
    print("\nVERDICT:", "FAIL — discrete (gate/ADSR) registers diverge: " + ", ".join(
        f"{c}({n_})" for _, c, n_ in bad_exact) if bad_exact
        else "discrete regs (gate/ADSR) match exactly; only FREQ/PW differ (cent-quant expected)")


def main():
    dump = sys.argv[1] if len(sys.argv) > 1 else "/corpus/hvsc/MUSICIANS/J/Jammer/Grid_Runner.1.dump.parquet"
    cfg = sys.argv[2] if len(sys.argv) > 2 else "stack"
    ref = dump_frame_state(dump)
    test = pipeline_frame_state(dump, cfg)
    report(ref, test, cfg)


if __name__ == "__main__":
    main()
