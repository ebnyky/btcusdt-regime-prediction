"""Phase 2: corrected labels -> 30-to-30 supervised dataset with soft targets.

Mirrors the original recipe exactly (temperature = median positive nearest-to-
second gap on training only, multiplier 1.0; certainty floor 0.10; 99th-pct
per-class distance outliers excluded from training only; inverse soft class
mass weights) but computes it from corrected distances.
"""
import json, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/claude/redo")
from phase1_labels import load_ohlc, WIN

OUT = "/home/claude/redo"
SCRATCH = "/tmp/claude-0/-home-claude/c1a77bb5-5da2-5b31-ac08-9eb7c709a832/scratchpad"
HORIZON = 30
EPS = 1e-12
CERT_FLOOR, OUTLIER_PCT = 0.10, 99.0

SPLITS = {  # thesis Table 3.1, inclusive bounds on window start / target end
    "train":      ("2020-01-01 00:00:00+00:00", "2024-12-17 13:00:00+00:00"),
    "validation": ("2024-12-17 14:00:00+00:00", "2025-07-06 23:00:00+00:00"),
    "test":       ("2025-07-07 00:00:00+00:00", "2026-07-31 23:00:00+00:00"),
}

def assemble_windows():
    emb = np.concatenate([np.load(f"{OUT}/_emb_shard{i}.npy") for i in (0, 1)])
    idx = np.concatenate([np.load(f"{OUT}/_idx_shard{i}.npy") for i in (0, 1)])
    order = np.argsort(idx)
    return idx[order], emb[order]

def label_windows(emb):
    pca = joblib.load(f"{SCRATCH}/k3/k3_frozen_models/pca.joblib")
    km  = joblib.load(f"{SCRATCH}/k3/k3_frozen_models/kmeans.joblib")
    z = emb / np.linalg.norm(emb, axis=1, keepdims=True)      # THE FIX
    red = pca.transform(z.astype(np.float32))
    return km.predict(red), km.transform(red)

def main():
    d = load_ohlc()
    starts, emb = assemble_windows()
    labels, dist = label_windows(emb)
    print(f"labelled {len(starts):,} windows; shares="
          + " ".join(f"{v/len(labels)*100:.2f}%" for v in np.bincount(labels, minlength=3)))

    pos = {int(s): k for k, s in enumerate(starts)}            # window start -> row in arrays
    times = d["open_time"].to_numpy()
    gap1h = np.concatenate([[False], d["open_time"].diff().dt.total_seconds().to_numpy()[1:] == 3600.0])

    rows = []
    for k, s in enumerate(starts):
        ts = int(s) + HORIZON
        j = pos.get(ts)
        if j is None:                    continue   # target window not valid
        if not gap1h[int(s) + WIN]:      continue   # gap between input end and target start
        rows.append((int(s), k, ts, j))
    pairs = pd.DataFrame(rows, columns=["input_start", "input_row", "target_start", "target_row"])
    pairs["input_start_time"] = times[pairs.input_start.to_numpy()]
    pairs["target_end_time"]  = times[pairs.target_start.to_numpy() + WIN - 1]
    pairs["input_cluster"]  = labels[pairs.input_row.to_numpy()]
    pairs["target_cluster"] = labels[pairs.target_row.to_numpy()]
    for c in range(3):
        pairs[f"target_d{c}"] = dist[pairs.target_row.to_numpy(), c]
    print(f"candidate pairs: {len(pairs):,}")

    pairs["split"] = ""
    for name, (lo, hi) in SPLITS.items():
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        m = (pairs.input_start_time >= lo) & (pairs.target_end_time <= hi)
        pairs.loc[m, "split"] = name
    purged = int((pairs.split == "").sum())
    pairs = pairs[pairs.split != ""].reset_index(drop=True)
    print(f"purged at partition boundaries: {purged:,};  assigned: {len(pairs):,}")
    for n in SPLITS: print(f"   {n:<11} {int((pairs.split==n).sum()):>7,}")

    D = pairs[[f"target_d{c}" for c in range(3)]].to_numpy(float)
    srt = np.sort(D, axis=1); d1 = srt[:, 0]; gap = srt[:, 1] - srt[:, 0]
    tr = (pairs.split == "train").to_numpy()

    g = gap[tr]; tau = float(np.median(g[g > 0]))
    print(f"\ntemperature tau (training median positive gap) = {tau:.6f}")

    A = np.exp(-(D - d1[:, None]) / tau)
    M = A / A.sum(axis=1, keepdims=True)
    ent = -(M * np.log(np.maximum(M, EPS))).sum(axis=1)
    nent = ent / np.log(3.0)
    cert = np.clip(1.0 - nent, 0.0, 1.0)
    cw = CERT_FLOOR + (1.0 - CERT_FLOOR) * cert
    for c in range(3):
        pairs[f"target_soft_{c}"] = M[:, c]
    pairs["entropy"] = ent; pairs["normalized_entropy"] = nent
    pairs["certainty"] = cert; pairs["certainty_weight"] = cw
    pairs["nearest_distance"] = d1

    limits = {}
    for c in range(3):
        v = d1[tr & (pairs.target_cluster.to_numpy() == c)]
        limits[c] = float(np.percentile(v, OUTLIER_PCT))
    thr = pairs.target_cluster.map(limits).to_numpy()
    pairs["is_outlier"] = (d1 > thr)
    pairs["training_eligible"] = tr & ~pairs.is_outlier.to_numpy()
    n_out = int((tr & pairs.is_outlier.to_numpy()).sum())
    print(f"outlier limits: " + " ".join(f"{limits[c]:.4f}" for c in range(3)))
    print(f"training outliers excluded: {n_out:,}")

    elig = pairs[pairs.training_eligible].copy()
    mass = elig[[f"target_soft_{c}" for c in range(3)]].sum(axis=0).to_numpy(float)
    weights = len(elig) / (3.0 * mass)
    print(f"soft class mass : " + " ".join(f"{m:.3f}" for m in mass))
    print(f"class weights   : " + " ".join(f"{w:.4f}" for w in weights))

    print("\nhard target shares by split (corrected):")
    for n in SPLITS:
        sub = pairs[pairs.split == n]
        c = np.bincount(sub.target_cluster.to_numpy(), minlength=3)
        print(f"   {n:<11} n={len(sub):>7,}  " + " ".join(f"{v/len(sub)*100:6.2f}%" for v in c))
    print("\nsoft-target uncertainty by split:")
    for n in SPLITS:
        sub = pairs[pairs.split == n]
        mm = sub[[f"target_soft_{c}" for c in range(3)]].to_numpy().max(axis=1)
        print(f"   {n:<11} mean max membership {mm.mean():.4f}  mean norm entropy {sub.normalized_entropy.mean():.4f}")

    pairs.to_parquet(f"{OUT}/pairs_30to30_corrected.parquet", index=False)
    np.save(f"{OUT}/window_embeddings.npy", emb)
    np.save(f"{OUT}/window_starts.npy", starts)
    np.save(f"{OUT}/window_labels.npy", labels)
    np.save(f"{OUT}/window_distances.npy", dist)
    json.dump({"temperature": tau, "certainty_floor": CERT_FLOOR,
               "outlier_percentile": OUTLIER_PCT,
               "outlier_limits": limits,
               "soft_class_mass": mass.tolist(),
               "class_weights": weights.tolist(),
               "training_outliers_excluded": n_out,
               "n_pairs": int(len(pairs)),
               "split_counts": {n: int((pairs.split==n).sum()) for n in SPLITS},
               "render_dpi": 90, "l2_normalised": True,
               "frozen_pca": "k3_frozen_models/pca.joblib",
               "frozen_kmeans": "k3_frozen_models/kmeans.joblib"},
              open(f"{OUT}/dataset_summary_corrected.json", "w"), indent=2)
    print("\nwrote pairs_30to30_corrected.parquet and dataset_summary_corrected.json")

if __name__ == "__main__":
    main()
