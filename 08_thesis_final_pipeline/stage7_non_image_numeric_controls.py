"""Non-image baselines: does the SIGNAL live in the picture, or in the numbers?

Two feature sets on the same corrected labels, splits and protocol:

  shape   -- the 120 OHLC values min-max normalised within the window. This is
             exactly the information an autoscaled candlestick image encodes.
  shape+scale -- adds four magnitude features the autoscaled image destroys:
             window log return, realised volatility, mean absolute hourly
             return, and high-low range over close.

shape vs the image models isolates the cost of the image encoding.
shape+scale vs shape isolates the cost of autoscaling.
"""
import json, warnings, numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from phase1_labels import load_ohlc, WIN
from phase3_train_eval import MLP, soft_ce, metrics
from phase4_uncertainty import block_boot

OUT="/home/claude/redo"; SEEDS=[42,43,44,45,46]
SOFT=[f"target_soft_{c}" for c in range(3)]

def build_features(d, starts):
    O=d["open"].to_numpy(); H=d["high"].to_numpy(); L=d["low"].to_numpy(); C=d["close"].to_numpy()
    shape=np.zeros((len(starts),120),dtype=np.float32)
    scale=np.zeros((len(starts),4),dtype=np.float32)
    for k,s in enumerate(starts):
        o,h,l,c = O[s:s+WIN],H[s:s+WIN],L[s:s+WIN],C[s:s+WIN]
        block=np.concatenate([o,h,l,c])
        lo,hi=block.min(),block.max(); rng=max(hi-lo,1e-12)
        shape[k]=((block-lo)/rng).astype(np.float32)          # per-window autoscale
        r=np.diff(np.log(c))
        scale[k]=[np.log(c[-1]/o[0]), r.std(), np.abs(r).mean(), (h.max()-l.min())/c[-1]]
    return shape, scale

def train_eval(Xtr,Str,Ctr,W,Xva,Yva,Xte,Yte,kind,seeds=SEEDS):
    vs,ts,preds=[],[],[]
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m=nn.Linear(Xtr.shape[1],3) if kind=="linear" else MLP(d=Xtr.shape[1])
        opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
        best,state,bad=-1,None,0
        for ep in range(40):
            m.train(); perm=torch.randperm(len(Xtr))
            for i in range(0,len(Xtr),256):
                b=perm[i:i+256]; opt.zero_grad()
                soft_ce(m(Xtr[b]),Str[b],W,Ctr[b]).backward(); opt.step()
            m.eval()
            with torch.inference_mode(): pv=torch.softmax(m(Xva),1).numpy()
            f1=f1_score(Yva,pv.argmax(1),average="macro")
            if f1>best: best,bad,state=f1,0,{k:v.clone() for k,v in m.state_dict().items()}
            else:
                bad+=1
                if bad>=7: break
        m.load_state_dict(state); m.eval()
        with torch.inference_mode(): pt=torch.softmax(m(Xte),1).numpy()
        p=pt.argmax(1); preds.append(p)
        vs.append(best); ts.append(f1_score(Yte,p,average="macro"))
    return vs,ts,preds

def main():
    d=load_ohlc()
    p=pd.read_parquet(f"{OUT}/pairs_30to30_corrected.parquet")
    s30=np.load(f"{OUT}/window_starts.npy")
    summ=json.load(open(f"{OUT}/dataset_summary_corrected.json"))
    W=torch.from_numpy(np.array(summ["class_weights"],dtype=np.float32))

    print("building numeric features for every input window...")
    shape,scale=build_features(d,s30)

    tr=p[p.training_eligible].reset_index(drop=True)
    va=p[p.split=="validation"].reset_index(drop=True)
    te=p[p.split=="test"].reset_index(drop=True)
    Yva=va.target_cluster.to_numpy(); Yte=te.target_cluster.to_numpy()
    pers=te.input_cluster.to_numpy()
    pers_f1=f1_score(Yte,pers,average="macro")
    Str=torch.from_numpy(tr[SOFT].to_numpy(np.float32))
    Ctr=torch.from_numpy(tr.certainty_weight.to_numpy(np.float32))
    print(f"persistence test macro-F1 = {pers_f1:.6f}\n")

    res={"persistence":float(pers_f1)}
    for tag,mat in (("shape", shape), ("shape+scale", np.hstack([shape,scale]))):
        F=lambda df: mat[df.input_row.to_numpy()]
        Xtr_r=F(tr); mu,sd=Xtr_r.mean(0),Xtr_r.std(0)+1e-8
        T=lambda X: torch.from_numpy(((X-mu)/sd).astype(np.float32))
        Xtr,Xva,Xte=T(Xtr_r),T(F(va)),T(F(te))
        print(f"=== {tag}  ({mat.shape[1]} features) ===")
        res[tag]={}
        for kind in ("linear","mlp"):
            vs,ts,preds=train_eval(Xtr,Str,Ctr,W,Xva,Yva,Xte,Yte,kind)
            dd=np.empty(1500); rng=np.random.default_rng(11); n=len(Yte); blk=120
            nb=int(np.ceil(n/blk)); st=np.arange(0,n-blk+1)
            for i in range(1500):
                ss=rng.choice(st,size=nb,replace=True)
                idx=np.concatenate([np.arange(x,x+blk) for x in ss])[:n]
                yy=Yte[idx]
                dd[i]=np.mean([f1_score(yy,q[idx],average="macro") for q in preds])-f1_score(yy,pers[idx],average="macro")
            lo,hi=np.percentile(dd,[2.5,97.5])
            res[tag][kind]={"val_mean":float(np.mean(vs)),"val_sd":float(np.std(vs,ddof=1)),
                "test_mean":float(np.mean(ts)),"test_sd":float(np.std(ts,ddof=1)),
                "vs_persistence":{"point":float(np.mean(ts)-pers_f1),"ci95":[float(lo),float(hi)],
                                  "p_gt_0":float((dd>0).mean())}}
            print(f"  {kind:<7} val {np.mean(vs):.4f}+/-{np.std(vs,ddof=1):.4f}  "
                  f"test {np.mean(ts):.4f}+/-{np.std(ts,ddof=1):.4f}  "
                  f"vs persistence {np.mean(ts)-pers_f1:+.4f} CI [{lo:+.4f},{hi:+.4f}] P(>0)={float((dd>0).mean()):.3f}")
        print()
    json.dump(res,open(f"{OUT}/results_numeric_baseline.json","w"),indent=2)
    print("wrote results_numeric_baseline.json")

if __name__=="__main__":
    main()
