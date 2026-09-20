"""Phase 1: corrected regime labels for every valid 30-hour window.

Fixes both defects:
  * renders at 90 DPI, matching the corpus the frozen PCA/K-Means were fitted on
  * L2-normalises embeddings before the frozen PCA, as the discovery stage did
"""
import os, sys, io, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch, joblib
import torch.nn as nn, torchvision.models as models
from torchvision import transforms
from PIL import Image

SCRATCH = "/tmp/claude-0/-home-claude/c1a77bb5-5da2-5b31-ac08-9eb7c709a832/scratchpad"
OUT     = "/home/claude/redo"
WIN, DPI, SIZE = 30, 90, 360

def load_ohlc():
    base = f"{SCRATCH}/ohlc/BTCUSDT_k3_forecasting_OHLC_sources"
    a = pd.read_csv(f"{base}/BTCUSDT_1h_2020-01-01_to_2025-07-06.csv", parse_dates=["open_time"])
    b = pd.read_csv(f"{base}/BTCUSDT_1h_2025-07-07_to_2026-07-31.csv", parse_dates=["open_time"])
    d = pd.concat([a, b], ignore_index=True)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    for c in ("open","high","low","close"):
        d[c] = pd.to_numeric(d[c], errors="raise")
    assert (d[["open","high","low","close"]] > 0).all().all()
    return d

def valid_starts(d):
    """Window [i, i+29] is valid when all 29 consecutive gaps are exactly 1h."""
    gap_ok = (d["open_time"].diff().dt.total_seconds().to_numpy()[1:] == 3600.0)
    n = len(d); ok = []
    csum = np.concatenate([[0], np.cumsum(gap_ok)])   # csum[k] = #good gaps among first k
    for i in range(0, n - WIN + 1):
        if csum[i + WIN - 1] - csum[i] == WIN - 1:
            ok.append(i)
    return np.asarray(ok, dtype=np.int64)

def render(o, h, l, c, dpi=DPI, size=SIZE):
    fig, ax = plt.subplots(figsize=(size/dpi, size/dpi), dpi=dpi)
    stats, cols = [], []
    for oo, hh, ll, cc in zip(o, h, l, c):
        stats.append({"q1": min(oo, cc), "q3": max(oo, cc),
                      "whislo": min(ll, hh), "whishi": max(ll, hh),
                      "med": (oo + cc) / 2.0})
        cols.append("green" if oo <= cc else "red")
    a = ax.bxp(bxpstats=stats, showcaps=False, patch_artist=True, showfliers=False)
    for b, col in zip(a["boxes"], cols): b.set_facecolor(col); b.set_edgecolor(col)
    for n, w in enumerate(a["whiskers"]): w.set_color(cols[n // 2])
    for m in a["medians"]: m.set_visible(False)
    ax.axis("off")
    buf = io.BytesIO(); fig.savefig(buf, dpi=dpi, pad_inches=0, format="jpg"); plt.close(fig)
    buf.seek(0); return Image.open(buf).convert("RGB")

def backbone():
    ck = torch.load(f"/mnt/user-data/uploads/m0zim--Downloads/best_model.pt",
                    map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model_state_dict"].items() if not k.startswith("fc.")}
    net = models.resnet18(weights=None); net.fc = nn.Identity()
    miss, unexp = net.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (miss, unexp)
    return net.eval()

TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def main(shard, nshards):
    torch.set_num_threads(1)
    d = load_ohlc(); starts = valid_starts(d)
    mine = starts[shard::nshards]
    O = d["open"].to_numpy(); H = d["high"].to_numpy()
    L = d["low"].to_numpy();  C = d["close"].to_numpy()
    net = backbone()
    embs = np.zeros((len(mine), 512), dtype=np.float32)
    B, buf, idx = 32, [], []
    done = 0
    prog = f"{OUT}/_p1_shard{shard}.progress"
    for j, s in enumerate(mine):
        buf.append(TF(render(O[s:s+WIN], H[s:s+WIN], L[s:s+WIN], C[s:s+WIN])))
        idx.append(j)
        if len(buf) == B or j == len(mine) - 1:
            with torch.inference_mode():
                embs[idx] = net(torch.stack(buf)).numpy().astype(np.float32)
            done += len(buf); buf, idx = [], []
            if done % 1600 < B:
                open(prog, "w").write(f"{done}/{len(mine)}\n")
    np.save(f"{OUT}/_emb_shard{shard}.npy", embs)
    np.save(f"{OUT}/_idx_shard{shard}.npy", mine)
    open(prog, "w").write(f"COMPLETE {done}/{len(mine)}\n")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
