"""90-to-30: render + embed the 90-candle INPUT images.

Anchor is index 0 (1 Jan 2020 00:00 UTC), per the thesis. Strides 10, 15 and 30
are subsets of the stride-5 anchor set, so one pass covers all four primary
conditions. The 30-hour TARGET labels already exist from phase 1.
"""
import os, sys, io, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from phase1_labels import load_ohlc, backbone, TF, DPI, SIZE

OUT = "/home/claude/redo"
IN_WIN, HORIZON, BASE_STRIDE = 90, 30, 5
SPAN = IN_WIN + HORIZON      # 120

def valid_starts_span(d, span, stride):
    gap_ok = (d["open_time"].diff().dt.total_seconds().to_numpy()[1:] == 3600.0)
    csum = np.concatenate([[0], np.cumsum(gap_ok)])
    n = len(d); out = []
    for i in range(0, n - span + 1, stride):
        if csum[i + span - 1] - csum[i] == span - 1:
            out.append(i)
    return np.asarray(out, dtype=np.int64)

def render90(o, h, l, c):
    fig, ax = plt.subplots(figsize=(SIZE/DPI, SIZE/DPI), dpi=DPI)
    stats, cols = [], []
    for oo, hh, ll, cc in zip(o, h, l, c):
        stats.append({"q1": min(oo, cc), "q3": max(oo, cc),
                      "whislo": min(ll, hh), "whishi": max(ll, hh),
                      "med": (oo+cc)/2.0})
        cols.append("green" if oo <= cc else "red")
    a = ax.bxp(bxpstats=stats, showcaps=False, patch_artist=True, showfliers=False)
    for b, col in zip(a["boxes"], cols): b.set_facecolor(col); b.set_edgecolor(col)
    for n, w in enumerate(a["whiskers"]): w.set_color(cols[n//2])
    for m in a["medians"]: m.set_visible(False)
    ax.axis("off")
    buf = io.BytesIO(); fig.savefig(buf, dpi=DPI, pad_inches=0, format="jpg"); plt.close(fig)
    from PIL import Image; buf.seek(0); return Image.open(buf).convert("RGB")

def main(shard, nshards):
    torch.set_num_threads(1)
    d = load_ohlc()
    starts = valid_starts_span(d, SPAN, BASE_STRIDE)
    mine = starts[shard::nshards]
    O=d["open"].to_numpy(); H=d["high"].to_numpy(); L=d["low"].to_numpy(); C=d["close"].to_numpy()
    net = backbone()
    embs = np.zeros((len(mine), 512), dtype=np.float32)
    B, buf, idx, done = 32, [], [], 0
    prog = f"{OUT}/_p6_shard{shard}.progress"
    for j, s in enumerate(mine):
        buf.append(TF(render90(O[s:s+IN_WIN], H[s:s+IN_WIN], L[s:s+IN_WIN], C[s:s+IN_WIN])))
        idx.append(j)
        if len(buf) == B or j == len(mine)-1:
            with torch.inference_mode():
                embs[idx] = net(torch.stack(buf)).numpy().astype(np.float32)
            done += len(buf); buf, idx = [], []
            if done % 800 < B: open(prog,"w").write(f"{done}/{len(mine)}\n")
    np.save(f"{OUT}/_emb90_shard{shard}.npy", embs)
    np.save(f"{OUT}/_idx90_shard{shard}.npy", mine)
    open(prog,"w").write(f"COMPLETE {done}/{len(mine)}\n")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
