import numpy as np, pandas as pd, joblib, warnings, torch, torch.nn as nn, json
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from phase3_train_eval import MLP, soft_ce
S="/tmp/claude-0/-home-claude/c1a77bb5-5da2-5b31-ac08-9eb7c709a832/scratchpad"
R=f"{S}/res30/experiment_results/frozen_resnet18_embeddings"
pca=joblib.load(f"{S}/k3/k3_frozen_models/pca.joblib"); km=joblib.load(f"{S}/k3/k3_frozen_models/kmeans.joblib")
def l2(a): return (a/np.linalg.norm(a,axis=1,keepdims=True)).astype(np.float32)
p=pd.read_parquet("pairs_30to30_corrected.parquet"); E90=np.load("window_embeddings.npy")
te=p[p.split=="test"].reset_index(drop=True); va=p[p.split=="validation"].reset_index(drop=True)
tr=p[p.training_eligible].reset_index(drop=True)
g=lambda n: np.ascontiguousarray(np.load(f"{R}/{n}_embeddings.npz")["X"],dtype=np.float32)
X100_te,X100_va=g("test"),g("validation")
X90_te=E90[te.input_row.to_numpy()].astype(np.float32); X90_va=E90[va.input_row.to_numpy()].astype(np.float32)
a=km.predict(pca.transform(l2(X90_te))); b=km.predict(pca.transform(l2(X100_te)))
print(f"ALIGNMENT: 90-DPI and 100-DPI input labels agree on {(a==b).mean()*100:.1f}% of the 9,301 test rows")
print("           (~97% is the DPI effect alone, so the rows are the same windows)\n")
Xtr_r=E90[tr.input_row.to_numpy()].astype(np.float32)
mu,sd=Xtr_r.mean(0),Xtr_r.std(0)+1e-8
T=lambda X: torch.from_numpy(((X-mu)/sd).astype(np.float32))
Xtr=T(Xtr_r); Xva90,Xte90,Xva100,Xte100=T(X90_va),T(X90_te),T(X100_va),T(X100_te)
SOFT=[f"target_soft_{c}" for c in range(3)]
Str=torch.from_numpy(tr[SOFT].to_numpy(np.float32)); Ctr=torch.from_numpy(tr.certainty_weight.to_numpy(np.float32))
W=torch.from_numpy(np.array(json.load(open("dataset_summary_corrected.json"))["class_weights"],dtype=np.float32))
Yva=va.target_cluster.to_numpy(); Yte=te.target_cluster.to_numpy()
print("Models trained on 90-DPI inputs, then scored on each render (5 seeds):")
out={}
for kind in ("linear","mlp"):
    r90,r100=[],[]
    for s in (42,43,44,45,46):
        torch.manual_seed(s); np.random.seed(s)
        m=nn.Linear(512,3) if kind=="linear" else MLP()
        opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
        best,state,bad=-1,None,0
        for ep in range(40):
            m.train(); perm=torch.randperm(len(Xtr))
            for i in range(0,len(Xtr),256):
                bb=perm[i:i+256]; opt.zero_grad()
                soft_ce(m(Xtr[bb]),Str[bb],W,Ctr[bb]).backward(); opt.step()
            m.eval()
            with torch.inference_mode(): pv=torch.softmax(m(Xva90),1).numpy()
            f1=f1_score(Yva,pv.argmax(1),average="macro")
            if f1>best: best,bad,state=f1,0,{k:v.clone() for k,v in m.state_dict().items()}
            else:
                bad+=1
                if bad>=7: break
        m.load_state_dict(state); m.eval()
        with torch.inference_mode():
            r90.append(f1_score(Yte,torch.softmax(m(Xte90),1).numpy().argmax(1),average="macro"))
            r100.append(f1_score(Yte,torch.softmax(m(Xte100),1).numpy().argmax(1),average="macro"))
    out[kind]={"dpi90":[float(np.mean(r90)),float(np.std(r90,ddof=1))],
               "dpi100":[float(np.mean(r100)),float(np.std(r100,ddof=1))]}
    print(f"  {kind:<7} on 90-DPI {np.mean(r90):.4f}+/-{np.std(r90,ddof=1):.4f}   "
          f"on 100-DPI {np.mean(r100):.4f}+/-{np.std(r100,ddof=1):.4f}   "
          f"diff {np.mean(r100)-np.mean(r90):+.4f}")
json.dump(out,open("results_dpi_transfer.json","w"),indent=2)
