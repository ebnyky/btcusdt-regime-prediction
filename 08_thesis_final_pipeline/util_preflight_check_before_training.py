"""Run this BEFORE training. Verifies the merge worked and every image resolves.

Catches the two things that can go wrong: a bad timestamp join (wrong row counts)
and image paths that do not resolve against your dataset layout.
"""
import sys, json
from pathlib import Path
import pandas as pd
from train_layer4_corrected import build_image_index, SOFT

EXPECT = {"train": 42100, "validation": 4775, "test": 9301}

def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python check_before_training.py <data-root>")
    root = Path(sys.argv[1])
    md = root / "manifests"
    if not md.is_dir(): sys.exit(f"No manifests/ under {root}")

    print("indexing images...")
    index = build_image_index(root)
    print(f"  {len(index):,} image files found\n")
    if not index: sys.exit("No images found. Check --data-root.")

    ok = True
    for split, want in EXPECT.items():
        f = md / f"{split}_samples.csv"
        if not f.is_file(): sys.exit(f"Missing {f}")
        d = pd.read_csv(f)
        flag = "OK " if len(d) == want else "!! "
        if len(d) != want: ok = False
        print(f"{flag}{split:<11} rows {len(d):>6,}   expected {want:>6,}")

        missing = []
        for rel in d["image"].astype(str):
            p = Path(rel)
            if p.is_file() or (root/rel).is_file() or (root/"images"/rel).is_file(): continue
            if p.name in index: continue
            missing.append(rel)
            if len(missing) >= 5: break
        if missing:
            ok = False
            print(f"   !! {len(missing)}+ images did not resolve, e.g. {missing[:3]}")
        else:
            print(f"   all image paths resolve")

        for c in ["target_cluster","certainty_weight",*SOFT]:
            if c not in d.columns:
                ok = False; print(f"   !! missing column {c}")
        s = d[SOFT].sum(axis=1)
        if not ((s - 1.0).abs() < 1e-4).all():
            ok = False; print("   !! soft targets do not sum to 1")
        dist = d.target_cluster.value_counts(normalize=True).sort_index()
        print(f"   class shares " + " ".join(f"{v*100:5.2f}%" for v in dist))

    cw = md / "class_weights.json"
    if cw.is_file():
        print(f"\nclass weights {json.load(open(cw))['class_weights']}")
    else:
        ok = False; print("\n!! class_weights.json missing from manifests/")

    print("\n" + ("READY TO TRAIN" if ok else "NOT READY — fix the items marked !! above"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
