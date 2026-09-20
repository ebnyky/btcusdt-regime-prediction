"""Phase 3: baselines + frozen-embedding models on the corrected dataset.

Selection is on validation macro-F1 only. The test split is touched once,
after both models are frozen.
"""
import json, warnings, numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)
OUT = "/home/claude/redo"
torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(2)
SOFT = [f"target_soft_{c}" for c in range(3)]
NAMES = ["Bullish", "Non-directional/volatile", "Bearish"]

def metrics(y, p, proba=None, soft=None):
    m = {"accuracy": accuracy_score(y, p),
         "balanced_accuracy": balanced_accuracy_score(y, p),
         "macro_f1": f1_score(y, p, average="macro"),
         "weighted_f1": f1_score(y, p, average="weighted")}
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=[0,1,2], zero_division=0)
    m["per_class"] = {NAMES[c]: {"precision": float(pr[c]), "recall": float(rc[c]),
                                 "f1": float(f1[c]), "support": int(sup[c])} for c in range(3)}
    m["confusion_matrix"] = confusion_matrix(y, p, labels=[0,1,2]).tolist()
    if proba is not None and soft is not None:
        q = np.clip(proba, 1e-12, 1.0)
        m["soft_cross_entropy"] = float(-(soft * np.log(q)).sum(1).mean())
        m["soft_brier"] = float(((proba - soft) ** 2).sum(1).mean())
    return m

class MLP(nn.Module):
    def __init__(s, d=512, h=(256, 64), p=0.30):
        super().__init__()
        L, prev = [], d
        for u in h:
            L += [nn.Linear(prev, u), nn.ReLU(), nn.Dropout(p)]; prev = u
        L += [nn.Linear(prev, 3)]
        s.net = nn.Sequential(*L)
    def forward(s, x): return s.net(x)

def soft_ce(logits, soft, cls_w, cert_w):
    logp = torch.log_softmax(logits, dim=1)
    per = -(cls_w[None, :] * soft * logp).sum(1)
    return (cert_w * per).sum() / cert_w.sum()

def run(model, tr, va, cls_w, lr, epochs=40, patience=7, bs=256, tag=""):
    Xtr, Str, Ctr, Ytr = tr
    Xva, Sva, _,   Yva = va
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, best_ep, bad = -1.0, None, -1, 0
    hist = []
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            opt.zero_grad()
            loss = soft_ce(model(Xtr[b]), Str[b], cls_w, Ctr[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            pv = torch.softmax(model(Xva), 1).numpy()
        f1 = f1_score(Yva, pv.argmax(1), average="macro")
        hist.append({"epoch": ep, "val_macro_f1": float(f1)})
        if f1 > best:
            best, best_ep, bad = f1, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    print(f"   {tag}: best epoch {best_ep}, validation macro-F1 {best:.6f}")
    return model, best, best_ep, hist

def main():
    p = pd.read_parquet(f"{OUT}/pairs_30to30_corrected.parquet")
    E = np.load(f"{OUT}/window_embeddings.npy")
    summ = json.load(open(f"{OUT}/dataset_summary_corrected.json"))
    cls_w = np.array(summ["class_weights"], dtype=np.float32)

    tr = p[p.training_eligible].reset_index(drop=True)
    va = p[p.split == "validation"].reset_index(drop=True)
    te = p[p.split == "test"].reset_index(drop=True)
    print(f"train(eligible) {len(tr):,}  validation {len(va):,}  test {len(te):,}")

    def feats(df): return E[df.input_row.to_numpy()].astype(np.float32)
    Xtr_r, Xva_r, Xte_r = feats(tr), feats(va), feats(te)
    mu, sd = Xtr_r.mean(0), Xtr_r.std(0) + 1e-8          # training-only standardiser
    T = lambda X: torch.from_numpy(((X - mu) / sd).astype(np.float32))
    Xtr, Xva, Xte = T(Xtr_r), T(Xva_r), T(Xte_r)
    S = lambda df: torch.from_numpy(df[SOFT].to_numpy(np.float32))
    Str, Sva, Ste = S(tr), S(va), S(te)
    Ctr = torch.from_numpy(tr.certainty_weight.to_numpy(np.float32))
    Ytr = tr.target_cluster.to_numpy(); Yva = va.target_cluster.to_numpy(); Yte = te.target_cluster.to_numpy()
    W = torch.from_numpy(cls_w)

    results = {}

    # ---------------- baselines ----------------
    print("\nBASELINES (training-derived rules)")
    maj = int(np.bincount(tr.target_cluster.to_numpy(), minlength=3).argmax())
    trans = {}
    for c in range(3):
        sub = tr[tr.input_cluster == c].target_cluster.to_numpy()
        trans[c] = int(np.bincount(sub, minlength=3).argmax()) if len(sub) else maj
    print(f"   majority class = {maj}   transition map = {trans}")
    for split, df, Y in (("validation", va, Yva), ("test", te, Yte)):
        results.setdefault("baselines", {}).setdefault(split, {})
        results["baselines"][split]["majority"]    = metrics(Y, np.full(len(Y), maj))
        results["baselines"][split]["persistence"] = metrics(Y, df.input_cluster.to_numpy())
        results["baselines"][split]["transition"]  = metrics(Y, df.input_cluster.map(trans).to_numpy())
        for k, v in results["baselines"][split].items():
            print(f"   {split:<11} {k:<12} macro-F1 {v['macro_f1']:.6f}  bal-acc {v['balanced_accuracy']:.6f}  acc {v['accuracy']:.6f}")

    # ---------------- models ----------------
    print("\nTRAINING (selection on validation macro-F1 only)")
    lin = nn.Linear(512, 3)
    lin, vlin, eplin, hlin = run(lin, (Xtr,Str,Ctr,Ytr), (Xva,Sva,None,Yva), W, 1e-3, tag="soft logistic")
    mlp = MLP()
    mlp, vmlp, epmlp, hmlp = run(mlp, (Xtr,Str,Ctr,Ytr), (Xva,Sva,None,Yva), W, 1e-3, tag="embedding MLP")

    sel = "soft_logistic" if vlin >= vmlp else "embedding_mlp"
    print(f"\n   validation-selected model: {sel}")

    print("\nTEST EVALUATION (single pass, after selection)")
    for name, mdl in (("soft_logistic", lin), ("embedding_mlp", mlp)):
        mdl.eval()
        with torch.inference_mode():
            pv = torch.softmax(mdl(Xva), 1).numpy()
            pt = torch.softmax(mdl(Xte), 1).numpy()
        results.setdefault("models", {})[name] = {
            "validation": metrics(Yva, pv.argmax(1), pv, Sva.numpy()),
            "test":       metrics(Yte, pt.argmax(1), pt, Ste.numpy()),
            "best_epoch": eplin if name == "soft_logistic" else epmlp,
            "history":    hlin if name == "soft_logistic" else hmlp,
        }
        t = results["models"][name]["test"]
        print(f"   {name:<16} test macro-F1 {t['macro_f1']:.6f}  bal-acc {t['balanced_accuracy']:.6f}  acc {t['accuracy']:.6f}")
        for c in NAMES:
            q = t["per_class"][c]
            print(f"        {c:<26} P {q['precision']:.4f}  R {q['recall']:.4f}  F1 {q['f1']:.4f}  n={q['support']}")

    results["validation_selected_model"] = sel
    results["class_weights"] = cls_w.tolist()
    results["dataset_summary"] = summ
    json.dump(results, open(f"{OUT}/results_corrected.json", "w"), indent=2)
    print("\nwrote results_corrected.json")

if __name__ == "__main__":
    main()
