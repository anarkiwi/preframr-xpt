"""Held-out next-onset-pitch predictability for per-voice melodic sequences. Used to run
the melody data-gap ladder (design/melody_data_gap_ladder.md): same metric across Bach and
each progressively-simplified mini level, to localise where SID melody becomes predictable."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from preframr.train.model.bodies import get_llama3_2

def load(path, cap):
    d = json.load(open(path))["seqs"]
    return [(s["dump"], s["pitch"][:cap]) for s in d if len(s["pitch"]) >= 12]

def ngram_ceiling(train, test, k=2):
    ctx = defaultdict(Counter)
    for _, s in train:
        for j in range(k, len(s)): ctx[tuple(s[j-k:j])][s[j]] += 1
    model = {c: cnt.most_common(1)[0][0] for c, cnt in ctx.items()}
    h=t=0
    for _, s in test:
        for j in range(k, len(s)):
            t += 1; h += model.get(tuple(s[j-k:j]), -999) == s[j]
    return h / max(t,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    cli = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = load(cli.data, cli.cap)
    dumps = sorted({d for d,_ in seqs}); rng=np.random.default_rng(0)
    held = set(rng.choice(dumps, max(1,len(dumps)//5), replace=False))
    train = [s for s in seqs if s[0] not in held]; test=[s for s in seqs if s[0] in held]
    alpha = {0:0}
    for _,s in seqs:
        for p in s: alpha.setdefault(p, len(alpha))
    V=len(alpha); L=max(len(s) for _,s in seqs)
    def pack(split):
        x=np.zeros((len(split),L),np.int64); m=np.zeros((len(split),L),bool)
        for i,(_,s) in enumerate(split):
            ids=[alpha[p] for p in s]; x[i,:len(ids)]=ids; m[i,:len(ids)]=True
        return x,m
    xtr,mtr=pack(train); xte,mte=pack(test)
    base2=ngram_ceiling(train,test,2)
    a=argparse.Namespace(layers=4,heads=8,kv_heads=4,embed=192,max_seq_len=L,attn_dropout=0.1,
                         norm_eps=1e-5,rope_base=500000,rope_scale=1.0,tie_word_embeddings=False)
    model=get_llama3_2(V,a).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.01)
    def fwd(x):
        o=model(x); return torch.cat(o,1) if isinstance(o,list) else o
    rng2=np.random.default_rng(1)
    for _ in range(cli.epochs):
        model.train()
        for i in range(0,len(xtr),16):
            b=rng2.permutation(len(xtr))[i:i+16]
            xb=torch.from_numpy(xtr[b]).to(dev); lg=fwd(xb)
            loss=F.cross_entropy(lg[:,:-1].reshape(-1,V),xb[:,1:].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    def acc(x,m):
        model.eval(); h=t=0
        with torch.inference_mode():
            for i in range(0,len(x),16):
                xb=torch.from_numpy(x[i:i+16]).to(dev); pr=fwd(xb).argmax(-1)[:,:-1]; tg=xb[:,1:]
                mm=torch.from_numpy(m[i:i+16,1:]).to(dev)
                h+=int(((pr==tg)&mm).sum()); t+=int(mm.sum())
        return h/max(t,1)
    print(f"{cli.data.name}: {len(train)}tr/{len(test)}te seqs, vocab={V}, maxlen={L}")
    print(f"  2-gram ceiling={base2:.3f}  model train={acc(xtr,mtr):.3f}  HELDOUT={acc(xte,mte):.3f}")
    return 0
if __name__=="__main__": sys.exit(main())
