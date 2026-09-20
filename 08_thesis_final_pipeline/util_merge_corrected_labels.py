"""Attach the corrected K=3 labels to your EXISTING Kaggle image dataset.

You do not need to regenerate images. Input render resolution was measured to
change held-out macro-F1 by -0.0002 (linear) and -0.0005 (mlp) against a seed
standard deviation of 0.006-0.008 -- immaterial. Resolution mattered only for
label generation, which is already corrected here.

The join is on the input window's start timestamp. Column names are detected
automatically rather than assumed: every column is tested for how well it parses
as a timestamp and overlaps the corrected timestamps, and the best is used.

Usage:
  python merge_corrected_labels.py \
      --original-manifests /kaggle/input/<your-dataset>/manifests \
      --corrected manifests_corrected \
      --out manifests
"""
import argparse, json, sys, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings('ignore')

SPLITS = ("train", "validation", "test")
IMG_EXT = (".jpg", ".jpeg", ".png")

def pick_time_column(df, want):
    """Column whose parsed timestamps best overlap `want` (a set of Timestamps).

    Deliberately dtype-agnostic: pandas 2.x stores text as object, pandas 3.x as
    a dedicated string dtype, so filtering on dtype silently skips columns.
    """
    best, best_hit = None, 0.0
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], utc=True, errors="coerce")
        except Exception:
            continue
        if float(parsed.isna().mean()) > 0.5: continue
        hit = float(parsed.isin(want).mean())
        if hit > best_hit: best, best_hit = c, hit
    return best, best_hit

def pick_image_column(df):
    pat = r"\.(?:jpg|jpeg|png)\s*$"
    best, best_score = None, 0.0
    for c in df.columns:
        try:
            v = df[c].astype(str)
        except Exception:
            continue
        try:
            score = float(v.str.contains(pat, case=False, regex=True, na=False).mean())
        except Exception:
            continue
        if score > best_score: best, best_score = c, score
    return best, best_score

def find_original(src, split):
    pats = [f"{split}_samples_chronological.csv", f"{split}_samples.csv",
            f"*{split}*chronological*.csv", f"*{split}*.csv"]
    for p in pats:
        hits = sorted(src.glob(p))
        if hits: return hits[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-manifests", required=True)
    ap.add_argument("--corrected", required=True)
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--min-match", type=float, default=0.98)
    a = ap.parse_args()
    src, cor, out = Path(a.original_manifests), Path(a.corrected), Path(a.out)
    if not src.is_dir(): sys.exit(f"Not a directory: {src}")
    out.mkdir(parents=True, exist_ok=True)

    keep = ["target_cluster","input_cluster","target_soft_0","target_soft_1",
            "target_soft_2","entropy","normalized_entropy","certainty",
            "certainty_weight","nearest_distance","is_outlier","training_eligible"]
    ok = True
    for split in SPLITS:
        f = find_original(src, split)
        if f is None:
            sys.exit(f"No original manifest for '{split}' under {src}.\n"
                     f"CSV files present: {[p.name for p in sorted(src.glob('*.csv'))]}")
        o = pd.read_csv(f)
        c = pd.read_csv(cor / f"{split}_samples.csv")
        c["_k"] = pd.to_datetime(c["input_start_time"], utc=True)
        want = set(c["_k"])

        tcol, hit = pick_time_column(o, want)
        icol, iscore = pick_image_column(o)
        if tcol is None or hit < 0.5:
            sys.exit(f"[{split}] could not find a timestamp column in {f.name} matching the "
                     f"corrected windows (best '{tcol}' at {hit:.1%}).\n"
                     f"Columns: {list(o.columns)}")
        if icol is None or iscore < 0.5:
            sys.exit(f"[{split}] could not find an image-path column in {f.name} "
                     f"(best '{icol}' at {iscore:.1%}).\nColumns: {list(o.columns)}")

        o["_k"] = pd.to_datetime(o[tcol], utc=True)
        m = (o[["_k", icol]].rename(columns={icol: "image"})
               .merge(c[["_k"] + keep], on="_k", how="inner")
               .sort_values("_k").reset_index(drop=True))
        m["input_start_time"] = m["_k"].astype(str)
        m = m.drop(columns=["_k"])
        if split == "train":
            m = m[m.training_eligible].reset_index(drop=True)
        m.to_csv(out / f"{split}_samples.csv", index=False)

        expect = len(c[c.training_eligible]) if split == "train" else len(c)
        rate = len(m) / expect if expect else 0
        flag = "OK " if rate >= a.min_match else "!! "
        if rate < a.min_match: ok = False
        print(f"{flag}{split:<11} source {f.name}")
        print(f"    time column '{tcol}' ({hit:.1%} overlap)   image column '{icol}'")
        print(f"    original {len(o):>7,}   corrected {expect:>7,}   written {len(m):>7,}  ({rate:.1%})")

    (out / "class_weights.json").write_text((cor / "class_weights.json").read_text())
    print(f"\nwrote {out}/")
    if not ok:
        print("\nMATCH RATE LOW. Do not train on this. Send the lines above for a fix.")
        sys.exit(1)
    print("Join looks good. Next: python check_before_training.py <data-root>")

if __name__ == "__main__":
    main()
