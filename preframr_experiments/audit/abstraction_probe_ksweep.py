import glob, numpy as np, torch, argparse, pandas as pd, random, statistics as st
from preframr.inference.predict import load_model
from preframr.args import add_args
from preframr.utils import get_logger
from preframr_tokens.events.oracle import ordered_writes
from preframr_tokens.events import stream as S

try:
    from torchtune.modules.common_utils import disable_kv_cache
except Exception:
    disable_kv_cache = None
FREQREG = {0: (0, 1), 1: (7, 8), 2: (14, 15)}
REG2V = {0: 0, 1: 0, 7: 1, 8: 1, 14: 2, 15: 2}


def _retime(rows):
    for i, r in enumerate(rows):
        r["clock"] = i
    return pd.DataFrame(rows)


def freq_shift(df, semis, rnd=False, seed=0):
    rng = random.Random(seed)
    rows = df.to_dict("records")
    out = []
    stt = {v: {"lo": 0, "hi": 0} for v in range(3)}
    for r in rows:
        reg = int(r["reg"])
        if reg in REG2V:
            v = REG2V[reg]
            lo, hi = FREQREG[v]
            if reg == lo:
                stt[v]["lo"] = int(r["val"])
            else:
                stt[v]["hi"] = int(r["val"])
            f = (stt[v]["hi"] << 8) | stt[v]["lo"]
            s = rng.uniform(-abs(semis), abs(semis)) if rnd else semis
            f2 = min(65535, max(0, round(f * (2.0 ** (s / 12.0)))))
            out.append({**r, "reg": lo, "val": f2 & 0xFF})
            out.append({**r, "reg": hi, "val": (f2 >> 8) & 0xFF})
        else:
            out.append(dict(r))
    return _retime(out)


def voiceperm(df):
    return _retime(
        [
            {
                **r,
                "reg": (
                    (int(r["reg"]) + 7) % 21 if int(r["reg"]) <= 20 else int(r["reg"])
                ),
            }
            for r in df.to_dict("records")
        ]
    )


def blockrev(df, K):
    uniq = sorted(set(int(x) for x in df["irq"]))
    if len(uniq) < K:
        return df.copy()
    segs = [uniq[i * len(uniq) // K : (i + 1) * len(uniq) // K] for i in range(K)]
    rank = {ir: i for i, ir in enumerate([x for seg in reversed(segs) for x in seg])}
    rows = df.to_dict("records")
    for r in rows:
        r["irq"] = rank.get(int(r["irq"]), int(r["irq"]))
    rows.sort(key=lambda r: (r["irq"], r["clock"]))
    return _retime(rows)


def atoms_of(df, W):
    return [x + 1 for x in S.encode(ordered_writes(df), verify=False)][:W]


def embed(model, device, ns):
    cap = {}
    h = model.model.norm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", o.detach())
    )
    try:
        L = len(ns)
        x = torch.tensor(ns, dtype=torch.long, device=device).unsqueeze(0)
        pos = torch.arange(L, device=device).unsqueeze(0)
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device)).unsqueeze(
            0
        )
        ctx = disable_kv_cache(model.model) if disable_kv_cache else torch.no_grad()
        with torch.inference_mode(), ctx:
            try:
                model.model(x, input_pos=pos, mask=mask)
            except TypeError:
                model.model(x)
    finally:
        h.remove()
    return cap["h"][0].float().mean(0).cpu().numpy()


def cos(a, b):
    return 1 - float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def surf(a, b):
    m = min(len(a), len(b))
    return 1 - sum(1 for x, y in zip(a, b) if x == y) / max(m, 1)


def main():
    p = add_args(argparse.ArgumentParser(conflict_handler="resolve"))
    p.add_argument("--dumps-glob")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--W", type=int, default=1536)
    p.add_argument("--k-list", default="2,4,8,16,32")
    a = p.parse_args()
    log = get_logger("INFO")
    Ks = [int(k) for k in a.k_list.split(",")]
    _ds, model, device, _ = load_model(a, log)
    model = model.to(device)
    model.eval()
    base = []
    for dp in sorted(glob.glob(a.dumps_glob, recursive=True)):
        try:
            df = pd.read_parquet(dp, columns=["clock", "irq", "chipno", "reg", "val"])
            ow = ordered_writes(df)
            if not S.single_speed(ow):
                continue
            ns = atoms_of(df, a.W)
            if len(ns) < 600:
                continue
            base.append((dp, df, ns))
        except Exception:  # pylint: disable=broad-except
            continue
        if len(base) >= a.n:
            break
    log.info("loaded %d passages, K-list %s", len(base), Ks)
    keys = ["id", "tr", "rnd", "vp"] + [f"sh{k}" for k in Ks]
    E = {k: [] for k in keys}
    surfs = {k: [] for k in ["tr", "rnd", "vp"] + [f"sh{k}" for k in Ks]}
    for dp, df, ns in base:
        E["id"].append(embed(model, device, ns))
        for nm, t in [
            ("tr", freq_shift(df, 5)),
            ("rnd", freq_shift(df, 5, True, 1)),
            ("vp", voiceperm(df)),
        ] + [(f"sh{k}", blockrev(df, k)) for k in Ks]:
            nt = atoms_of(t, a.W)
            E[nm].append(embed(model, device, nt))
            surfs[nm].append(surf(ns, nt))
    allE = np.stack([v for k in keys for v in E[k]])
    mu = allE.mean(0)
    C = {k: [v - mu for v in E[k]] for k in keys}
    m = lambda x: st.mean(x) if x else float("nan")
    dd = lambda k: [cos(C["id"][i], C[k][i]) for i in range(len(base))]
    diff = [
        cos(C["id"][i], C["id"][j])
        for i in range(len(base))
        for j in range(i + 1, len(base))
    ]
    print(f"RESULT n={len(base)} diff-passages={m(diff):.3f}")
    print(
        f"  LOCAL  transpose={m(dd('tr')):.3f}(surf{m(surfs['tr']):.2f}) random={m(dd('rnd')):.3f}(surf{m(surfs['rnd']):.2f}) voiceperm={m(dd('vp')):.3f}  | random/transpose={m(dd('rnd'))/max(m(dd('tr')),1e-6):.1f}"
    )
    for k in Ks:
        print(
            f"  STRUCT blockrev K={k:<2} rep={m(dd(f'sh{k}')):.3f} surf={m(surfs[f'sh{k}']):.2f}  vs-diff={m(dd(f'sh{k}'))/max(m(diff),1e-6):.2f}"
        )


main()
