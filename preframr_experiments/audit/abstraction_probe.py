import glob, numpy as np, torch, argparse, pandas as pd, random
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


def freq_shift_df(df, semis, randomize=False, seed=0):
    rng = random.Random(seed)
    rows = df.to_dict("records")
    out = []
    st = {v: {"lo": 0, "hi": 0} for v in range(3)}
    for r in rows:
        reg = int(r["reg"])
        if reg in REG2V:
            v = REG2V[reg]
            lo, hi = FREQREG[v]
            if reg == lo:
                st[v]["lo"] = int(r["val"])
            else:
                st[v]["hi"] = int(r["val"])
            f = (st[v]["hi"] << 8) | st[v]["lo"]
            s = rng.uniform(-abs(semis), abs(semis)) if randomize else semis
            f2 = min(65535, max(0, round(f * (2.0 ** (s / 12.0)))))
            out.append({**r, "reg": lo, "val": f2 & 0xFF})
            out.append({**r, "reg": hi, "val": (f2 >> 8) & 0xFF})
        else:
            out.append(dict(r))
    return _retime(out)


def voiceperm_df(df):
    rows = df.to_dict("records")
    out = []
    for r in rows:
        reg = int(r["reg"])
        out.append({**r, "reg": (reg + 7) % 21 if reg <= 20 else reg})
    return _retime(out)


def blockrev_df(df, K=4):
    uniq = sorted(set(int(x) for x in df["irq"].tolist()))
    if len(uniq) < K:
        return df.copy()
    segs = [uniq[i * len(uniq) // K : (i + 1) * len(uniq) // K] for i in range(K)]
    order = [ir for seg in reversed(segs) for ir in seg]
    rank = {ir: i for i, ir in enumerate(order)}
    rows = df.to_dict("records")
    for r in rows:
        r["irq"] = rank.get(int(r["irq"]), int(r["irq"]))
    rows.sort(key=lambda r: (r["irq"], r["clock"]))
    return _retime(rows)


def atoms_of(df, W):
    a = S.encode(ordered_writes(df), verify=False)
    return [x + 1 for x in a][:W]


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
    hid = cap["h"][0].float()
    return (hid.mean(0)).cpu().numpy()


def cos(a, b):
    return 1 - float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def surf(a, b):
    m = min(len(a), len(b))
    return 1 - sum(1 for x, y in zip(a, b) if x == y) / max(m, 1)


def main():
    p = add_args(argparse.ArgumentParser(conflict_handler="resolve"))
    p.add_argument("--dumps-glob", type=str)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--W", type=int, default=1536)
    a = p.parse_args()
    log = get_logger("INFO")
    _ds, model, device, _ = load_model(a, log)
    model = model.to(device)
    model.eval()
    dumps = sorted(glob.glob(a.dumps_glob, recursive=True))
    base = []
    for dp in dumps:
        try:
            df = pd.read_parquet(dp, columns=["clock", "irq", "chipno", "reg", "val"])
            ow = ordered_writes(df)
            if not S.single_speed(ow):
                continue
            ns = atoms_of(df, a.W)
            if len(ns) < 400:
                continue
            base.append((dp, df, ns))
        except Exception:  # pylint: disable=broad-except
            continue
        if len(base) >= a.n:
            break
    log.info("loaded %d base passages", len(base))
    E = {"id": [], "tr": [], "rnd": [], "vp": [], "sh": []}
    surfs = []
    for dp, df, ns in base:
        ns_tr = atoms_of(freq_shift_df(df, 5, False), a.W)
        ns_rnd = atoms_of(freq_shift_df(df, 5, True, seed=1), a.W)
        ns_vp = atoms_of(voiceperm_df(df), a.W)
        ns_sh = atoms_of(blockrev_df(df), a.W)
        E["id"].append(embed(model, device, ns))
        E["tr"].append(embed(model, device, ns_tr))
        E["rnd"].append(embed(model, device, ns_rnd))
        E["vp"].append(embed(model, device, ns_vp))
        E["sh"].append(embed(model, device, ns_sh))
        surfs.append(
            (surf(ns, ns_tr), surf(ns, ns_rnd), surf(ns, ns_vp), surf(ns, ns_sh))
        )
    import statistics as st

    allE = np.stack(E["id"] + E["tr"] + E["rnd"] + E["vp"] + E["sh"])
    mu = allE.mean(0)
    C = {k: [v - mu for v in E[k]] for k in E}
    raw_diff = [
        cos(E["id"][i], E["id"][j])
        for i in range(len(base))
        for j in range(i + 1, len(base))
    ]
    print(
        f"DIAG raw embedding: norm={np.linalg.norm(mu):.3f} per-dim-std(across id)={np.stack(E['id']).std(0).mean():.4f} raw diff-pass cos={st.mean(raw_diff):.4f}"
    )
    rows = []
    for i, (dp, df, ns) in enumerate(base):
        d_tr = cos(C["id"][i], C["tr"][i])
        d_rnd = cos(C["id"][i], C["rnd"][i])
        d_vp = cos(C["id"][i], C["vp"][i])
        d_sh = cos(C["id"][i], C["sh"][i])
        s_tr, s_rnd, s_vp, s_sh = surfs[i]
        rows.append((d_tr, d_rnd, d_vp, d_sh, s_tr, s_rnd, s_vp, s_sh))
        log.info(
            "%s rep[transpose=%.3f rand=%.3f vperm=%.3f blockrev=%.3f] surf[tr=%.2f rnd=%.2f vp=%.2f sh=%.2f]",
            dp.split("/")[-1][:16],
            d_tr,
            d_rnd,
            d_vp,
            d_sh,
            s_tr,
            s_rnd,
            s_vp,
            s_sh,
        )
    embs_id = C["id"]
    diff = [
        cos(embs_id[i], embs_id[j])
        for i in range(len(embs_id))
        for j in range(i + 1, len(embs_id))
    ]
    m = lambda x: st.mean(x) if x else float("nan")
    R = list(zip(*rows))
    print("RESULT n=", len(rows))
    print(
        f"  rep_dist  transpose(PRESERVE)={m(R[0]):.3f}  random(LOCAL-CHANGE)={m(R[1]):.3f}  voiceperm(PRESERVE)={m(R[2]):.3f}  blockrev(STRUCT-CHANGE)={m(R[3]):.3f}  diff-passages={m(diff):.3f}"
    )
    print(
        f"  surf_dist transpose={m(R[4]):.2f}  random={m(R[5]):.2f}  voiceperm={m(R[6]):.2f}  blockrev={m(R[7]):.2f}"
    )
    print(
        f"  sensitivity (rep/surf): transpose={m(R[0])/max(m(R[4]),1e-6):.3f}  random={m(R[1])/max(m(R[5]),1e-6):.3f}  blockrev={m(R[3])/max(m(R[7]),1e-6):.3f}"
    )
    print(
        f"  STRUCTURE sensitivity = rep(blockrev)/rep(transpose) = {m(R[3])/max(m(R[0]),1e-6):.2f}  (>>1 => encodes long-range structure; ~1 => bag-of-local / structure-blind)"
    )
    print(
        f"  blockrev vs diff-passages = {m(R[3])/max(m(diff),1e-6):.2f}  (near 1 => reorder ~ different tune; near 0 => order ignored)"
    )


main()
