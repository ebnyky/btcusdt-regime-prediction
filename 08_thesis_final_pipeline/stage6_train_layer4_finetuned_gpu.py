"""Layer-4 fine-tuned ResNet-18 on the CORRECTED K=3 labels.

Needs a GPU. Everything else in the corrected rerun (labels, soft targets,
baselines, soft-logistic and embedding-MLP models) was produced on CPU and is
supplied alongside this script.

Inputs expected in --data-root:
  images/sample_<start>_<stop>.jpg   90-DPI renders (make_images.py builds these)
  manifests/train_samples.csv, validation_samples.csv, test_samples.csv

Protocol: select the checkpoint by validation macro-F1 only; touch the test
split exactly once, after selection.
"""
import argparse, json, os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, \
                            precision_recall_fscore_support, confusion_matrix

SOFT = [f"target_soft_{c}" for c in range(3)]
NAMES = ["Bullish", "Non-directional/volatile", "Bearish"]
TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def build_image_index(root):
    """Map bare filename -> full path, so manifests may use any path convention.

    Uses os.walk(followlinks=True): Path.rglob does not recurse into symlinked
    directories, and pointing a data root at a symlinked image folder is the
    normal way to avoid copying a large read-only dataset.

    Exact paths always win in _resolve; this index is only a fallback. If the
    dataset reuses filenames across subfolders the fallback is ambiguous, so
    that case is reported rather than silently resolved.
    """
    idx, dupes, total = {}, 0, 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for fn in filenames:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                total += 1
                if fn in idx: dupes += 1
                else: idx[fn] = Path(dirpath) / fn
    if dupes:
        print(f"  note: {total:,} image files but {len(idx):,} distinct names "
              f"({dupes:,} repeated across subfolders). Exact manifest paths are "
              f"used first, so this only matters if a path fails to resolve.")
    return idx

class DS(Dataset):
    def __init__(s, df, root, index=None):
        s.df = df.reset_index(drop=True); s.root = Path(root)
        s.index = index if index is not None else build_image_index(root)
    def __len__(s): return len(s.df)
    def _resolve(s, rel):
        rel = str(rel)
        for cand in (Path(rel), s.root / rel, s.root / "images" / rel):
            if cand.is_file(): return cand
        hit = s.index.get(Path(rel).name)
        if hit is None:
            raise FileNotFoundError(
                f"Could not locate image {rel!r} under {s.root}. "
                f"Indexed {len(s.index):,} image files.")
        return hit
    def __getitem__(s, i):
        r = s.df.iloc[i]
        img = Image.open(s._resolve(r["image"])).convert("RGB")
        return (TF(img),
                torch.tensor([r[c] for c in SOFT], dtype=torch.float32),
                torch.tensor(float(r["certainty_weight"]), dtype=torch.float32),
                int(r["target_cluster"]))

def build(backbone_ckpt=None):
    """ResNet-18 with layer 4 and the head trainable.

    Kaggle notebooks default to internet OFF, so downloading ImageNet weights
    fails there. The pristine ImageNet backbone is bundled with this package
    (recovered from the project's own frozen_backbone checkpoint) and is used by
    default; downloading is only a fallback.
    """
    local = Path(backbone_ckpt) if backbone_ckpt else Path(__file__).with_name("resnet18_imagenet_backbone.pt")
    if local.is_file():
        m = models.resnet18(weights=None)
        sd = torch.load(local, map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("fc.")]
        if missing or unexpected:
            sys.exit(f"Backbone checkpoint does not match ResNet-18: missing={missing[:5]} unexpected={unexpected[:5]}")
        print(f"loaded ImageNet backbone from {local.name}")
    else:
        print("bundled backbone not found; downloading ImageNet weights (needs internet)")
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad_(False)
    for p in m.layer4.parameters(): p.requires_grad_(True)
    m.fc = nn.Linear(512, 3)
    return m

def soft_ce(logits, soft, clsw, certw):
    logp = torch.log_softmax(logits, 1)
    per = -(clsw[None, :] * soft * logp).sum(1)
    return (certw * per).sum() / certw.sum()

def evaluate(model, loader, dev):
    model.eval(); P, Y, S = [], [], []
    with torch.inference_mode():
        for x, s, _, y in loader:
            P.append(torch.softmax(model(x.to(dev)), 1).float().cpu().numpy())
            Y.append(np.asarray(y)); S.append(s.numpy())
    P = np.vstack(P); Y = np.concatenate(Y); S = np.vstack(S); p = P.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(Y, p, labels=[0,1,2], zero_division=0)
    q = np.clip(P, 1e-12, 1)
    return {"accuracy": float(accuracy_score(Y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(Y, p)),
            "macro_f1": float(f1_score(Y, p, average="macro")),
            "weighted_f1": float(f1_score(Y, p, average="weighted")),
            "soft_cross_entropy": float(-(S*np.log(q)).sum(1).mean()),
            "soft_brier": float(((P-S)**2).sum(1).mean()),
            "per_class": {NAMES[c]: {"precision": float(pr[c]), "recall": float(rc[c]),
                                     "f1": float(f1[c]), "support": int(sup[c])} for c in range(3)},
            "confusion_matrix": confusion_matrix(Y, p, labels=[0,1,2]).tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="layer4_corrected_results")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--backbone-checkpoint", default=None,
                    help="Override the bundled ImageNet backbone .pt")
    a = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu": print("WARNING: no GPU found. This will be extremely slow.")
    root = Path(a.data_root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(root/"manifests"/"train_samples.csv")
    va = pd.read_csv(root/"manifests"/"validation_samples.csv")
    te = pd.read_csv(root/"manifests"/"test_samples.csv")
    clsw = torch.tensor(json.load(open(root/"manifests"/"class_weights.json"))["class_weights"],
                        dtype=torch.float32, device=dev)
    print(f"train {len(tr):,}  validation {len(va):,}  test {len(te):,}")

    print("indexing image files...")
    index = build_image_index(root)
    print(f"  found {len(index):,} images under {root}")
    if not index: sys.exit(f"No images found under {root}")
    dl = lambda df, bs, sh: DataLoader(DS(df, root, index), batch_size=bs, shuffle=sh,
                                       num_workers=a.workers, pin_memory=(dev=="cuda"))
    tl, vl, sl = dl(tr,a.batch_size,True), dl(va,a.eval_batch_size,False), dl(te,a.eval_batch_size,False)

    model = build(a.backbone_checkpoint).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(enabled=(dev=="cuda"))
    best, best_ep, bad, hist = -1.0, -1, 0, []
    for ep in range(1, a.epochs+1):
        model.train()
        for x, s, c, _ in tl:
            x, s, c = x.to(dev), s.to(dev), c.to(dev)
            opt.zero_grad()
            with torch.amp.autocast(device_type=dev.split(":")[0], enabled=(dev=="cuda")):
                loss = soft_ce(model(x), s, clsw, c)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        m = evaluate(model, vl, dev); sched.step(m["macro_f1"])
        hist.append({"epoch": ep, **{k: m[k] for k in ("macro_f1","balanced_accuracy","accuracy")}})
        print(f"epoch {ep:>2}  val macro-F1 {m['macro_f1']:.6f}")
        if m["macro_f1"] > best:
            best, best_ep, bad = m["macro_f1"], ep, 0
            torch.save({"model_state_dict": model.state_dict(), "best_epoch": ep,
                        "best_validation_macro_f1": best}, out/"best_model.pt")
        else:
            bad += 1
            if bad >= a.patience: print("early stop"); break

    model.load_state_dict(torch.load(out/"best_model.pt", map_location=dev)["model_state_dict"])
    res = {"best_epoch": best_ep, "best_validation_macro_f1": best,
           "validation": evaluate(model, vl, dev), "test": evaluate(model, sl, dev),
           "history": hist}
    json.dump(res, open(out/"layer4_corrected_results.json","w"), indent=2)
    print(f"\nvalidation macro-F1 {res['validation']['macro_f1']:.6f}")
    print(f"test       macro-F1 {res['test']['macro_f1']:.6f}")

if __name__ == "__main__":
    main()
