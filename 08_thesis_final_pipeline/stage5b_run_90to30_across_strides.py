"""90-to-30 corrected rerun across primary strides 5, 10, 15, 30.

Mirrors the original design: soft memberships come from the 30-to-30 build
(same target windows, same temperature and outlier limits); class weights are
recomputed per stride from that stride's training soft mass. Persistence is the
cluster of the final 30 candles of the input window, [t+60, t+89], which is how
the original reconstruct_persistence_labels helper defines it.
"""
import json, warnings, numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, \
                            precision_recall_fscore_support, confusion_matrix
from phase1_labels import load_ohlc
from phase3_train_eval import MLP, soft_ce, metrics, NAMES
from phase4_uncertainty import block_boot

def block_boot_multi(y, preds, base, block, B, seed=0):
    """CI for mean-over-seeds macro-F1 minus baseline macro-F1."""
    rng=np.random.default_rng(seed); n=len(y)
    nb=int(np.ceil(n/block)); starts=np.arange(0, max(1,n-block+1))
    out=np.empty(B)
    for i in range(B):
        s=rng.choice(starts,size=nb,replace=True)
        idx=np.concatenate([np.arange(x,min(x+block,n)) for x in s])[:n]
        yy=y[idx]
        out[i]=np.mean([f1_score(yy,p[idx],average="macro") for p in preds]) \
               - f1_score(yy,base[idx],average="macro")
    return out

OUT="/home/claude/redo"; IN_WIN, HORIZON = 90, 30
EPS=1e-12; CERT_FLOOR=0.10
SPLITS={"train":("2020-01-01 00:00:00+00:00","2024-12-17 13:00:00+00:00"),
        "validation":("2024-12-17 14:00:00+00:00","2025-07-06 23:00:00+00:00"),
        "test":("2025-07-07 00:00:00+00:00","2026-07-31 23:00:00+00:00")}
SEEDS=[42,43,44,45,46]

def main():
    d=load_ohlc(); times=d["open_time"].to_numpy()
    s30=np.load(f"{OUT}/window_starts.npy"); lab30=np.load(f"{OUT}/window_labels.npy")
    dist30=np.load(f"{OUT}/window_distances.npy")
    pos30={int(v):i for i,v in enumerate(s30)}
    summ=json.load(open(f"{OUT}/dataset_summary_corrected.json"))
    tau=summ["temperature"]; limits={int(k):v for k,v in summ["outlier_limits"].items()}
    print(f"reusing 30-to-30 temperature tau={tau:.6f} and outlier limits {list(limits.values())}")

    emb=np.concatenate([np.load(f"{OUT}/_emb90_shard{i}.npy") for i in (0,1)])
    idx=np.concatenate([np.load(f"{OUT}/_idx90_shard{i}.npy") for i in (0,1)])
    o=np.argsort(idx); idx, emb = idx[o], emb[o]
    print(f"90-candle inputs embedded: {len(idx):,}")

    rows=[]
    for k,i in enumerate(idx):
        i=int(i); t=pos30.get(i+IN_WIN); c=pos30.get(i+60)
        if t is None or c is None: continue
        rows.append((i,k,t,c))
    P=pd.DataFrame(rows,columns=["input_start","emb_row","tgt_row","cur_row"])
    P["input_start_time"]=times[P.input_start.to_numpy()]
    P["target_end_time"]=times[P.input_start.to_numpy()+IN_WIN+HORIZON-1]
    P["target_cluster"]=lab30[P.tgt_row.to_numpy()]
    P["input_cluster"]=lab30[P.cur_row.to_numpy()]
    D=dist30[P.tgt_row.to_numpy()]
    d1=D.min(1)
    A=np.exp(-(D-d1[:,None])/tau); M=A/A.sum(1,keepdims=True)
    nent=(-(M*np.log(np.maximum(M,EPS))).sum(1))/np.log(3.0)
    P["certainty_weight"]=CERT_FLOOR+(1-CERT_FLOOR)*np.clip(1-nent,0,1)
    for c in range(3): P[f"target_soft_{c}"]=M[:,c]
    P["nearest_distance"]=d1
    P["is_outlier"]=d1>P.target_cluster.map(limits).to_numpy()
    P["split"]=""
    for n,(lo,hi) in SPLITS.items():
        m=(P.input_start_time>=pd.Timestamp(lo))&(P.target_end_time<=pd.Timestamp(hi))
        P.loc[m,"split"]=n
    P=P[P.split!=""].reset_index(drop=True)
    SOFT=[f"target_soft_{c}" for c in range(3)]

    allres={"temperature":tau,"strides":{}}
    for stride in (5,10,15,30):
        sub=P[P.input_start % stride==0].reset_index(drop=True)
        tr=sub[(sub.split=="train")&(~sub.is_outlier)].reset_index(drop=True)
        va=sub[sub.split=="validation"].reset_index(drop=True)
        te=sub[sub.split=="test"].reset_index(drop=True)
        mass=tr[SOFT].sum(0).to_numpy(float); W=torch.from_numpy((len(tr)/(3.0*mass)).astype(np.float32))
        print(f"\n{'='*66}\nSTRIDE {stride}:  train {len(tr):,}  validation {len(va):,}  test {len(te):,}")
        print(f"  thesis Table 3.2 assigned: "+{5:"8,233/943/1,848",10:"4,122/471/924",
              15:"2,749/314/616",30:"1,376/157/308"}[stride])
        cts=np.bincount(te.target_cluster.to_numpy(),minlength=3)
        print(f"  corrected test shares: "+" ".join(f"{v/len(te)*100:5.1f}%" for v in cts))

        f=lambda df: emb[df.emb_row.to_numpy()].astype(np.float32)
        Xtr_r=f(tr); mu,sd=Xtr_r.mean(0),Xtr_r.std(0)+1e-8
        T=lambda X: torch.from_numpy(((X-mu)/sd).astype(np.float32))
        Xtr,Xva,Xte=T(Xtr_r),T(f(va)),T(f(te))
        Str=torch.from_numpy(tr[SOFT].to_numpy(np.float32))
        Ctr=torch.from_numpy(tr.certainty_weight.to_numpy(np.float32))
        Yva=va.target_cluster.to_numpy(); Yte=te.target_cluster.to_numpy()

        maj=int(np.bincount(tr.target_cluster.to_numpy(),minlength=3).argmax())
        tmap={c:(int(np.bincount(tr[tr.input_cluster==c].target_cluster.to_numpy(),minlength=3).argmax())
                 if (tr.input_cluster==c).any() else maj) for c in range(3)}
        base={"majority":metrics(Yte,np.full(len(Yte),maj)),
              "persistence":metrics(Yte,te.input_cluster.to_numpy()),
              "transition":metrics(Yte,te.input_cluster.map(tmap).to_numpy())}
        for k,v in base.items():
            print(f"  baseline {k:<12} test macro-F1 {v['macro_f1']:.6f}  bal-acc {v['balanced_accuracy']:.6f}")

        sres={"n":{"train":len(tr),"validation":len(va),"test":len(te)},"baselines":base,"models":{}}
        for kind in ("linear","mlp"):
            vs,ts,preds=[],[],[]
            for si,s in enumerate(SEEDS):
                torch.manual_seed(s); np.random.seed(s)
                m=nn.Linear(512,3) if kind=="linear" else MLP()
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
                p=pt.argmax(1)
                preds.append(p)
                vs.append(best); ts.append(f1_score(Yte,p,average="macro"))
            blk=max(8,int(np.ceil(2*(IN_WIN+HORIZON)/stride)))
            dd=block_boot_multi(Yte,preds,te.input_cluster.to_numpy(),block=blk,B=1500,seed=7)
            lo,hi=np.percentile(dd,[2.5,97.5])
            sres["models"][kind]={"val_mean":float(np.mean(vs)),"val_sd":float(np.std(vs,ddof=1)),
                "test_mean":float(np.mean(ts)),"test_sd":float(np.std(ts,ddof=1)),
                "vs_persistence":{"point":float(np.mean(ts)-base["persistence"]["macro_f1"]),
                                  "ci95":[float(lo),float(hi)],"p_gt_0":float((dd>0).mean()),
                                  "block_rows":blk,"n_blocks":int(np.ceil(len(Yte)/blk))}}
            print(f"  {kind:<7} val {np.mean(vs):.4f}+/-{np.std(vs,ddof=1):.4f}  "
                  f"test {np.mean(ts):.4f}+/-{np.std(ts,ddof=1):.4f}  "
                  f"vs persistence {sres['models'][kind]['vs_persistence']['point']:+.4f} "
                  f"CI [{lo:+.4f},{hi:+.4f}] P(>0)={float((dd>0).mean()):.3f} "
                  f"(block {blk} rows, {int(np.ceil(len(Yte)/blk))} blocks)")
        allres["strides"][stride]=sres
    json.dump(allres,open(f"{OUT}/results_90to30_corrected.json","w"),indent=2)
    print("\nwrote results_90to30_corrected.json")

if __name__=="__main__":
    main()
