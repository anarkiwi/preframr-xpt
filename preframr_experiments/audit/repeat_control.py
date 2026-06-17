import glob, numpy as np, torch, argparse, pandas as pd, statistics as st
from preframr.inference.predict import load_model
from preframr.args import add_args
from preframr.utils import get_logger
from preframr_tokens.events.oracle import ordered_writes
from preframr_tokens.events import stream as S

try:
    from torchtune.modules.common_utils import disable_kv_cache
except Exception:
    disable_kv_cache = None


def fin(rows):
    rows = sorted(rows, key=lambda r: (int(r["irq"]), int(r.get("clock", 0))))
    for i, r in enumerate(rows):
        r["clock"] = i
    return pd.DataFrame(rows)


def first_seg(df, H):
    u = sorted(set(int(x) for x in df["irq"]))[:H]
    rk = {ir: i for i, ir in enumerate(u)}
    return [
        {**r, "irq": rk[int(r["irq"])]}
        for r in df.to_dict("records")
        if int(r["irq"]) in rk
    ]


def atoms_of(df):
    return [x + 1 for x in S.encode(ordered_writes(df), verify=False)]


def tf_acc(model, device, ns, lo, hi):
    L = len(ns)
    x = torch.tensor(ns, dtype=torch.long, device=device).unsqueeze(0)
    pos = torch.arange(L, device=device).unsqueeze(0)
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device)).unsqueeze(0)
    ctx = disable_kv_cache(model.model) if disable_kv_cache else torch.no_grad()
    with torch.inference_mode(), ctx:
        try:
            out = model.model(x, input_pos=pos, mask=mask)
        except TypeError:
            out = model.model(x)
    pred = out[0, :-1].argmax(-1).cpu().numpy()
    tgt = np.array(ns[1:])
    corr = pred == tgt
    lo = max(0, lo)
    hi = min(len(corr), hi)
    return float(corr[lo:hi].mean()) if hi > lo else float("nan")


def main():
    p = add_args(argparse.ArgumentParser(conflict_handler="resolve"))
    p.add_argument("--dumps-glob")
    p.add_argument("--n-pairs", type=int, default=8)
    p.add_argument("--H-list", default="32,64,128,256")
    a = p.parse_args()
    log = get_logger("INFO")
    Hs = [int(h) for h in a.H_list.split(",")]
    _ds, model, device, _ = load_model(a, log)
    model = model.to(device)
    model.eval()
    dumps = []
    for dp in sorted(glob.glob(a.dumps_glob, recursive=True)):
        try:
            df = pd.read_parquet(dp, columns=["clock", "irq", "chipno", "reg", "val"])
            if S.single_speed(ordered_writes(df)) and len(set(df["irq"])) > 600:
                dumps.append(df)
        except Exception:  # pylint: disable=broad-except
            continue
        if len(dumps) >= a.n_pairs + 1:
            break
    log.info("loaded %d dumps; H %s", len(dumps), Hs)
    for H in Hs:
        aa = []
        ab = []
        dists = []
        for i in range(min(a.n_pairs, len(dumps) - 1)):
            A = first_seg(dumps[i], H)
            B = first_seg(dumps[i + 1], H)
            nsA1 = atoms_of(fin([dict(r) for r in A]))
            L1 = len(nsA1)
            AA = atoms_of(
                fin([dict(r) for r in A] + [{**r, "irq": int(r["irq"]) + H} for r in A])
            )
            AB = atoms_of(
                fin([dict(r) for r in A] + [{**r, "irq": int(r["irq"]) + H} for r in B])
            )
            if len(AA) < L1 + 50 or len(AB) < L1 + 50:
                continue
            aa.append(tf_acc(model, device, AA, L1, len(AA) - 1))
            ab.append(tf_acc(model, device, AB, L1, len(AB) - 1))
            dists.append(L1)
        m = lambda x: st.mean([v for v in x if v == v]) if x else float("nan")
        print(
            f"H={H:<4} repeat_dist~{m(dists):.0f}atoms  AA(repeat)_acc={m(aa):.3f}  AB(novel)_acc={m(ab):.3f}  lift={m(aa)-m(ab):+.3f}  n={len(aa)}"
        )


main()
