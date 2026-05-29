"""DEMO: train an interval melody model on extracted SID lead sequences and render a
held-out continuation to WAV (triangle voice). Reproduction helper for the melody data-gap
ladder audition; runs in the preframr image (torch + preframr_audio). Paths are demo-fixed
(/scratch/tmp/...); adjust as needed. See design/melody_data_gap_ladder.md."""
import json, numpy as np, pandas as pd, math, argparse, torch, torch.nn.functional as F
from preframr.train.model.bodies import get_llama3_2
from preframr_audio.fidelity import render_df_to_wav
from preframr_tokens.tokenizer_config import named_config
from pathlib import Path
IRQ=19656; CLOCK=985248
def midi_fn(m): return int(round(440.0*2**((m-69)/12.0)*16777216/CLOCK))
tunes=[s['pitch'] for s in json.load(open('/scratch/tmp/drax_L2lead.json'))['seqs'] if len(s['pitch'])>=24]
# intervals (clamped) per tune
def ivs(p): return [max(-24,min(24,p[i]-p[i-1])) for i in range(1,len(p))]
ivt=[ivs(p) for p in tunes]
alpha={}; 
for s in ivt:
    for v in s: alpha.setdefault(v,len(alpha))
inv={k:v for v,k in alpha.items()}; V=len(alpha); L=min(256,max(len(s) for s in ivt))
X=np.zeros((len(ivt),L),np.int64)
for i,s in enumerate(ivt): 
    s=s[:L]; X[i,:len(s)]=[alpha[v] for v in s]
dev='cuda'; a=argparse.Namespace(layers=4,heads=8,kv_heads=4,embed=192,max_seq_len=L,attn_dropout=0.1,norm_eps=1e-5,rope_base=500000,rope_scale=1.0,tie_word_embeddings=False)
model=get_llama3_2(V,a).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
def fwd(x):
    o=model(x); return torch.cat(o,1) if isinstance(o,list) else o
tr=X[:-3]; rng=np.random.default_rng(1)
for _ in range(60):
    model.train()
    for i in range(0,len(tr),8):
        b=rng.permutation(len(tr))[i:i+8]; xb=torch.from_numpy(tr[b]).to(dev); lg=fwd(xb)
        loss=F.cross_entropy(lg[:,:-1].reshape(-1,V),xb[:,1:].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
# held-out tune: reconstruct pitches; prompt=first third intervals, predict rest
tune=tunes[-2][:120]; iv=ivs(tune); nz=len(iv); base=tune[0]
ids=[alpha[v] for v in iv[:nz//3]]
model.eval()
with torch.inference_mode():
    while len(ids)<nz: ids.append(int(fwd(torch.tensor([ids],device=dev))[0,-1].argmax()))
pred_iv=[inv[i] for i in ids]
def to_pitch(intervals,base):
    p=[base]; 
    for d in intervals: p.append(max(36,min(84,p[-1]+d)))
    return p
pred_pitch=to_pitch(pred_iv,base); gt_pitch=tune[:nz+1]
def render(midis,path,NF=10):
    rows=[]
    for m in midis:
        fn=midi_fn(m)
        for f in range(NF):
            rows.append(dict(op=0,reg=-128,subreg=-1,val=0,diff=IRQ,irq=IRQ,description=0))
            if f==0:
                rows+=[dict(op=0,reg=0,subreg=-1,val=fn&0xFF,diff=0,irq=IRQ,description=0),
                       dict(op=0,reg=1,subreg=-1,val=(fn>>8)&0xFF,diff=0,irq=IRQ,description=0),
                       dict(op=0,reg=5,subreg=-1,val=0x00,diff=0,irq=IRQ,description=0),
                       dict(op=0,reg=6,subreg=-1,val=0xFA,diff=0,irq=IRQ,description=0),
                       dict(op=0,reg=4,subreg=-1,val=0x11,diff=0,irq=IRQ,description=0)]  # triangle+gate
            elif f==NF-2:
                rows.append(dict(op=0,reg=4,subreg=-1,val=0x10,diff=0,irq=IRQ,description=0))  # gate off
    return render_df_to_wav(pd.DataFrame(rows),IRQ,named_config('baseline'),Path(path))[0]
import wave
def peak(p):
    w=wave.open(p); d=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16); return int(np.abs(d).max()), w.getnframes()/w.getframerate()
np_=render(pred_pitch,'/scratch/tmp/enc_audition/drax_prediction.wav')
ng=render(gt_pitch,'/scratch/tmp/enc_audition/drax_ground_truth.wav')
cont=np.mean([pred_iv[i]==iv[i] for i in range(nz//3,nz)])
pk,du=peak('/scratch/tmp/enc_audition/drax_prediction.wav')
print(f'DRAX render: cont interval-match={cont:.3f}; pred peak={pk} dur={du:.1f}s; WAVs drax_prediction/ground_truth.wav')
