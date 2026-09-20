"""Phase 4: seed stability + moving-block bootstrap on the corrected results.

The test rows are stride-1 overlapping 60-hour spans, so neighbouring rows share
up to 59 hours. An ordinary i.i.d. bootstrap would badly understate the interval.
A moving-block bootstrap with block length >= the span keeps that dependence.
"""
import json, warnings, numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from phase3_train_eval import MLP, soft_ce, SOFT

OUT = "/home/claude/redo"
SEEDS = [42, 43, 44, 45, 46]
BLOCK = 120          # hours; 2x the 60-hour input+target span
B     = 2000

def load():
    p = pd.read_parquet(f"{OUT}/pairs_30to30_corrected.parquet")
    E = np.load(f"{OUT}/window_embeddings.npy")
    s = json.load(open(f"{OUT}/dataset_summary_corrected.json"))
    tr = p[p.training_eligible].reset_index(drop=True)
    va = p[p.split=="validation"].reset_index(drop=True)
    te = p[p.split=="test"].reset_index(drop=True)
    f  = lambda df: E[df.input_row.to_numpy()].astype(np.float32)
    Xtr_r = f(tr); mu, sd = Xtr_r.mean(0), Xtr_r.std(0)+1e-8
    T = lambda X: torch.from_numpy(((X-mu)/sd).astype(np.float32))
    return (tr, va, te, T(Xtr_r), T(f(va)), T(f(te)),
            torch.from_numpy(tr[SOFT].to_numpy(np.float32)),
            torch.from_numpy(tr.certainty_weight.to_numpy(np.float32)),
            torch.from_numpy(np.array(s["class_weights"], dtype=np.float32)))

def train_once(seed, kind, Xtr, Str, Ctr, W, Xva, Yva, epochs=40, patience=7, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = nn.Linear(512,3) if kind=="linear" else MLP()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad = -1, None, 0
    n = len(Xtr)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i+bs]; opt.zero_grad()
            soft_ce(m(Xtr[b]), Str[b], W, Ctr[b]).backward(); opt.step()
        m.eval()
        with torch.inference_mode(): pv = torch.softmax(m(Xva),1).numpy()
        f1 = f1_score(Yva, pv.argmax(1), average="macro")
        if f1 > best: best, bad, state = f1, 0, {k:v.clone() for k,v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); return m, best

def block_boot(y, a, b, block=BLOCK, B=B, seed=0):
    """CI for macro-F1(a) - macro-F1(b) under a moving-block bootstrap."""
    rng = np.random.default_rng(seed); n = len(y)
    nb = int(np.ceil(n/block)); starts = np.arange(0, n-block+1)
    out = np.empty(B)
    for i in range(B):
        s = rng.choice(starts, size=nb, replace=True)
        idx = np.concatenate([np.arange(x, x+block) for x in s])[:n]
        out[i] = f1_score(y[idx], a[idx], average="macro") - f1_score(y[idx], b[idx], average="macro")
    return out

def main():
    tr, va, te, Xtr, Xva, Xte, Str, Ctr, W = load()
    Yva = va.target_cluster.to_numpy(); Yte = te.target_cluster.to_numpy()
    pers_te = te.input_cluster.to_numpy()
    res = {"seeds": SEEDS, "block_length_hours": BLOCK, "bootstrap_draws": B}

    print("SEED STABILITY (each seed: train, select on validation, score test)")
    for kind in ("linear","mlp"):
        v_, t_, preds = [], [], []
        for s in SEEDS:
            m, vbest = train_once(s, kind, Xtr, Str, Ctr, W, Xva, Yva)
            m.eval()
            with torch.inference_mode(): pt = torch.softmax(m(Xte),1).numpy()
            p = pt.argmax(1); preds.append(p)
            t = f1_score(Yte, p, average="macro")
            v_.append(vbest); t_.append(t)
            print(f"   {kind:<7} seed {s}: val {vbest:.6f}  test {t:.6f}")
        res[kind] = {"val_macro_f1": v_, "test_macro_f1": t_,
                     "val_mean": float(np.mean(v_)), "val_sd": float(np.std(v_, ddof=1)),
                     "test_mean": float(np.mean(t_)), "test_sd": float(np.std(t_, ddof=1))}
        print(f"   {kind:<7} val {np.mean(v_):.4f} +/- {np.std(v_,ddof=1):.4f}   "
              f"test {np.mean(t_):.4f} +/- {np.std(t_,ddof=1):.4f}\n")
        res[kind]["_preds_seed0"] = preds[0].tolist()

    pers_f1 = f1_score(Yte, pers_te, average="macro")
    print(f"persistence test macro-F1 = {pers_f1:.6f}")
    res["persistence_test_macro_f1"] = float(pers_f1)

    print(f"\nMOVING-BLOCK BOOTSTRAP  (block {BLOCK}h, {B} draws)")
    for kind in ("linear","mlp"):
        p = np.array(res[kind].pop("_preds_seed0"))
        d = block_boot(Yte, p, pers_te)
        lo, hi = np.percentile(d, [2.5, 97.5])
        pgt = float((d > 0).mean())
        res[kind]["vs_persistence"] = {"point": float(f1_score(Yte,p,average="macro")-pers_f1),
                                       "ci95": [float(lo), float(hi)],
                                       "p_gt_0": pgt}
        print(f"   {kind:<7} macro-F1 - persistence = {f1_score(Yte,p,average='macro')-pers_f1:+.4f}"
              f"   95% CI [{lo:+.4f}, {hi:+.4f}]   P(>0) = {pgt:.3f}")

    json.dump(res, open(f"{OUT}/uncertainty_corrected.json","w"), indent=2)
    print("\nwrote uncertainty_corrected.json")

if __name__ == "__main__":
    main()
