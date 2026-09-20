"""Render the 90-DPI candlestick images the corrected manifests refer to.

The renderer is byte-identical to the project's original: rendering the
supplied 30-row fixture at 90 DPI reproduces SHA-256
9551df8b81bd0b63ff071884c446525942e20e7856c70c198e3c423aabe80487, the hash
recorded in reference/REFERENCE_IMAGE.json.

90 DPI matters: the frozen PCA and K-Means were fitted on a 90-DPI corpus.
The original supervised generators rendered at 100 DPI, which is one of the two
defects this rerun fixes.
"""
import argparse, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

WIN, DPI, SIZE = 30, 90, 360

def render(w, dest):
    fig, ax = plt.subplots(figsize=(SIZE/DPI, SIZE/DPI), dpi=DPI)
    stats, cols = [], []
    for r in w.itertuples(index=False):
        stats.append({"q1": min(r.open, r.close), "q3": max(r.open, r.close),
                      "whislo": min(r.low, r.high), "whishi": max(r.low, r.high),
                      "med": (r.open + r.close)/2.0})
        cols.append("green" if r.open <= r.close else "red")
    a = ax.bxp(bxpstats=stats, showcaps=False, patch_artist=True, showfliers=False)
    for b, c in zip(a["boxes"], cols): b.set_facecolor(c); b.set_edgecolor(c)
    for n, wk in enumerate(a["whiskers"]): wk.set_color(cols[n//2])
    for m in a["medians"]: m.set_visible(False)
    ax.axis("off")
    fig.savefig(dest, dpi=DPI, pad_inches=0, format="jpg"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlc", nargs="+", required=True, help="the two source CSVs, in order")
    ap.add_argument("--windows", required=True, help="window_index.csv from the corrected build")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = pd.concat([pd.read_csv(f, parse_dates=["open_time"]) for f in a.ohlc], ignore_index=True)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    idx = pd.read_csv(a.windows)
    outdir = Path(a.out)/"images"; outdir.mkdir(parents=True, exist_ok=True)
    for n, r in enumerate(idx.itertuples(index=False), 1):
        dest = outdir / r.image
        if dest.exists(): continue
        render(d.iloc[r.start_index:r.start_index+WIN], dest)
        if n % 2000 == 0: print(f"  {n:,}/{len(idx):,}", flush=True)
    print(f"done: {len(idx):,} images in {outdir}")

if __name__ == "__main__":
    main()
