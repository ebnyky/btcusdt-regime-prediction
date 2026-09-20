#!/usr/bin/env python3
"""Small synthetic integration test for the folder-based package."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distances(cluster: int, kind: str) -> tuple[float, float, float]:
    confident = {
        0: (1.0, 9.0, 9.0),
        1: (9.0, 1.0, 13.0),
        2: (9.0, 13.0, 1.0),
    }
    ambiguous = {
        0: (5.0, 5.1, 8.0),
        1: (5.1, 5.0, 8.0),
        2: (8.0, 8.1, 5.0),
    }
    membership_outlier = {
        0: (8.0, 16.0, 16.0),
        1: (16.0, 8.0, 20.0),
        2: (16.0, 20.0, 8.0),
    }
    return {
        "confident": confident,
        "ambiguous": ambiguous,
        "membership_outlier": membership_outlier,
    }[kind][cluster]


def create_manifest(partition: str, counts: dict[int, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rank = 0
    for cluster, count in counts.items():
        for within_cluster in range(count):
            if within_cluster == count - 1:
                kind = "membership_outlier"
            elif within_cluster == count - 2:
                kind = "ambiguous"
            else:
                kind = "confident"
            d0, d1, d2 = distances(cluster, kind)
            rows.append({
                "pair_id": f"{partition}_{cluster}_{within_cluster}",
                "chronological_rank": rank,
                "partition": partition,
                "target_cluster_id": cluster,
                "target_distance_cluster_0": d0,
                "target_distance_cluster_1": d1,
                "target_distance_cluster_2": d2,
                "input_image_path": f"images/{partition}/input_{rank}.jpg",
                "target_image_path": f"images/{partition}/target_{rank}.jpg",
            })
            rank += 1
    return pd.DataFrame(rows).sort_values("chronological_rank").reset_index(drop=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="k3-balance-test-") as temporary:
        root = Path(temporary) / "BTCUSDT_k3_forecasting_dataset"
        manifests = root / "manifests"
        models = root / "models"
        manifests.mkdir(parents=True)
        models.mkdir(parents=True)

        centers = np.zeros((3, 120), dtype=np.float64)
        centers[1, 0] = 10.0
        centers[2, 1] = 10.0
        model = SimpleNamespace(n_clusters=3, cluster_centers_=centers)
        model_path = models / "kmeans.joblib"
        joblib.dump(model, model_path)

        split_counts = {
            "train": {0: 15, 1: 30, 2: 18},
            "validation": {0: 7, 1: 12, 2: 8},
            "test": {0: 8, 1: 14, 2: 9},
        }
        for partition, counts in split_counts.items():
            create_manifest(partition, counts).to_csv(
                manifests / f"{partition}_samples_chronological.csv", index=False
            )

        summary = {
            "frozen_models": {
                "pca_input_features": 512,
                "pca_components": 120,
                "kmeans_clusters": 3,
                "kmeans_centroid_shape": [3, 120],
                "kmeans_path": str(model_path),
                "kmeans_sha256": sha256(model_path),
            }
        }
        (manifests / "dataset_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        config = {
            "dataset_folder_path": str(root),
            "kmeans_model_path": "auto",
            "output_folder_name": "geometric_balance",
            "confidence_threshold": 0.5,
            "membership_percentile": 95.0,
            "random_seed": 42,
            "create_exactly_balanced_training_manifest": True,
            "overwrite_output": False,
        }
        config_path = Path(temporary) / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        subprocess.run(
            [sys.executable, str(PACKAGE / "balance_k3_forecasting_dataset.py"),
             "--config", str(config_path)],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(PACKAGE / "validate_balanced_dataset.py"),
             "--config", str(config_path)],
            check=True,
        )

        output_manifests = root / "geometric_balance" / "manifests"
        original = pd.read_csv(manifests / "train_samples_chronological.csv")
        accepted = pd.read_csv(output_manifests / "train_confident_chronological.csv")
        rejected = pd.read_csv(output_manifests / "train_rejected.csv")
        balanced = pd.read_csv(output_manifests / "train_balanced_chronological.csv")
        assert len(original) == len(accepted) + len(rejected)
        assert balanced["target_cluster_id"].value_counts().nunique() == 1
        assert set(original["pair_id"]) == set(accepted["pair_id"]) | set(rejected["pair_id"])

    print("SYNTHETIC END-TO-END TEST PASSED")


if __name__ == "__main__":
    main()
