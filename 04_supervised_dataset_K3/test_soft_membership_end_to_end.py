#!/usr/bin/env python3
"""Synthetic end-to-end test with imbalance, ties, and outliers."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_manifest(partition: str, counts: dict[int, int]) -> pd.DataFrame:
    rows = []
    rank = 0
    for cluster, count in counts.items():
        for i in range(count):
            base = np.array([10.6, 10.6, 10.6], dtype=float)
            base[cluster] = 9.8
            if i == 0:  # almost tied with a competing class
                competitor = (cluster + 1) % 3
                base[cluster] = 10.222
                base[competitor] = 10.223
                base[(cluster + 2) % 3] = 10.2511
            if i == count - 1:  # a clear but distant membership outlier
                base[:] = 30.0
                base[cluster] = 20.0
            rows.append({
                "pair_id": f"{partition}_{cluster}_{i}",
                "chronological_rank": rank,
                "target_cluster_id": cluster,
                "target_distance_cluster_0": base[0],
                "target_distance_cluster_1": base[1],
                "target_distance_cluster_2": base[2],
                "input_image_path": f"images/{partition}/input_{rank}.jpg",
                "target_image_path": f"images/{partition}/target_{rank}.jpg",
            })
            rank += 1
    return pd.DataFrame(rows)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="k3-soft-test-") as temporary:
        root = Path(temporary) / "BTCUSDT_k3_forecasting_dataset"
        manifests = root / "manifests"
        manifests.mkdir(parents=True)

        counts = {
            "train": {0: 20, 1: 45, 2: 25},
            "validation": {0: 8, 1: 15, 2: 9},
            "test": {0: 9, 1: 17, 2: 10},
        }
        for partition, values in counts.items():
            manifest = create_manifest(partition, values)
            if partition == "test":
                # Simulate a numerical near tie where sklearn predict selected
                # Cluster 2 but exported distances put Cluster 0 microscopically ahead.
                row = manifest.index[manifest["target_cluster_id"] == 2][1]
                manifest.loc[row, "target_distance_cluster_0"] = 10.22299
                manifest.loc[row, "target_distance_cluster_2"] = 10.22300
            manifest.to_csv(
                manifests / f"{partition}_samples_chronological.csv", index=False
            )
        summary = {"frozen_models": {
            "pca_input_features": 512, "pca_components": 120, "kmeans_clusters": 3,
            "kmeans_centroid_shape": [3, 120],
            "kmeans_path": "recorded/original/kmeans.joblib",
            "kmeans_sha256": "synthetic-recorded-model-hash",
        }}
        (manifests / "dataset_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        config = {
            "dataset_folder_path": str(root),
            "output_folder_name": "soft_membership", "temperature_multiplier": 1.0,
            "outlier_percentile": 95.0, "certainty_weight_floor": 0.10,
            "random_seed": 42, "overwrite_output": False,
        }
        config_path = Path(temporary) / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        subprocess.run([sys.executable, str(PACKAGE / "build_soft_membership_dataset.py"),
                        "--config", str(config_path)], check=True)
        subprocess.run([sys.executable, str(PACKAGE / "validate_soft_membership_dataset.py"),
                        "--config", str(config_path)], check=True)

        output = root / "soft_membership" / "manifests"
        train = pd.read_csv(output / "train_soft_annotated_chronological.csv")
        soft = train[[f"target_soft_cluster_{k}" for k in range(3)]].to_numpy()
        assert np.allclose(soft.sum(axis=1), 1.0)
        ambiguous = train.loc[train["pair_id"] == "train_0_0"].iloc[0]
        assert ambiguous["training_eligible"]
        assert ambiguous["certainty_weight"] < 0.2
        excluded = pd.read_csv(output / "train_distance_outliers.csv")
        assert len(excluded) >= 3
        test = pd.read_csv(output / "test_soft_annotated_chronological.csv")
        assert test["source_hard_label_near_tie_disagreement"].sum() == 1
        audit = pd.read_csv(
            root / "soft_membership" / "reports" / "hard_label_near_tie_disagreements.csv"
        )
        assert len(audit) == 1
        assert audit.iloc[0]["source_partition"] == "test"

    print("SYNTHETIC SOFT-MEMBERSHIP END-TO-END TEST PASSED")


if __name__ == "__main__":
    main()
