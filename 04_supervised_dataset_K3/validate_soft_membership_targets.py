#!/usr/bin/env python3
"""Validate soft K=3 targets, leakage controls, and source preservation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PARTITIONS = ("train", "validation", "test")
CLUSTERS = (0, 1, 2)
DISTANCE_COLUMNS = [f"target_distance_cluster_{k}" for k in CLUSTERS]
SOFT_COLUMNS = [f"target_soft_cluster_{k}" for k in CLUSTERS]
TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing output: {path}")
    return pd.read_csv(path)


def validate_annotated(frame: pd.DataFrame, temperature: float, floor: float) -> None:
    missing = sorted(set(DISTANCE_COLUMNS + SOFT_COLUMNS) - set(frame.columns))
    assert not missing, f"Missing required columns: {missing}"
    probabilities = frame[SOFT_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= -TOLERANCE).all() and (probabilities <= 1 + TOLERANCE).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=TOLERANCE)
    distances = frame[DISTANCE_COLUMNS].to_numpy(dtype=float)
    assert np.array_equal(np.argmin(distances, axis=1), frame["nearest_cluster_id"].to_numpy())
    assert np.array_equal(np.argmax(probabilities, axis=1), frame["nearest_cluster_id"].to_numpy())
    assert np.allclose(frame["temperature"], temperature, atol=TOLERANCE)

    expected_affinity = np.exp(-(distances - distances.min(axis=1, keepdims=True)) / temperature)
    expected = expected_affinity / expected_affinity.sum(axis=1, keepdims=True)
    assert np.allclose(probabilities, expected, atol=TOLERANCE)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
    certainty = 1.0 - entropy / np.log(3.0)
    expected_weight = floor + (1.0 - floor) * certainty
    assert np.allclose(frame["soft_target_certainty"], certainty, atol=TOLERANCE)
    assert np.allclose(frame["certainty_weight"], expected_weight, atol=TOLERANCE)


def main() -> None:
    config = json.loads(parse_args().config.expanduser().resolve().read_text(encoding="utf-8"))
    root = Path(config["dataset_folder_path"]).expanduser().resolve()
    output = root / str(config["output_folder_name"])
    manifests, reports = output / "manifests", output / "reports"
    summary_path = reports / "soft_membership_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["frozen_contract"]["resnet_features"] == 512
    assert summary["frozen_contract"]["pca_components"] == 120
    assert summary["frozen_contract"]["kmeans_clusters"] == 3
    assert summary["frozen_contract"]["centroid_shape"] == [3, 120]
    assert summary["frozen_contract"]["fit_called"] is False
    assert summary["soft_membership"]["hard_label_used_for_training_loss"] is False
    assert summary["soft_membership"]["relabelled_samples"] is False
    assert summary["soft_membership"]["recorded_centroid_distances_authoritative"] is True
    assert summary["hard_label_consistency"]["hard_labels_preserved"] is True
    assert summary["hard_label_consistency"]["hard_labels_used_for_soft_memberships"] is False
    assert summary["certainty"]["ambiguous_samples_discarded"] is False
    controls = summary["leakage_controls"]
    assert controls["temperature_from_training_only"] is True
    assert controls["outlier_limits_from_training_only"] is True
    assert controls["class_weights_from_training_only"] is True
    assert controls["test_used_for_selection_or_tuning"] is False

    for source in summary["source_files"].values():
        path = Path(source["path"])
        assert path.is_file(), f"Original source disappeared: {path}"
        assert sha256_file(path) == source["sha256"], f"Original source changed: {path}"

    temperature = float(summary["soft_membership"]["temperature"])
    floor = float(summary["certainty"]["certainty_weight_floor"])
    originals: dict[str, pd.DataFrame] = {}
    annotated: dict[str, pd.DataFrame] = {}
    for partition in PARTITIONS:
        originals[partition] = pd.read_csv(summary["source_files"][partition]["path"])
        annotated[partition] = read(
            manifests / f"{partition}_soft_annotated_chronological.csv"
        )
        assert originals[partition]["pair_id"].tolist() == annotated[partition]["pair_id"].tolist()
        validate_annotated(annotated[partition], temperature, floor)
        assert annotated[partition]["chronological_rank"].is_monotonic_increasing
        print(f"{partition}: all {len(annotated[partition]):,} rows conserved and annotated")

    retained = read(manifests / "train_soft_chronological.csv")
    shuffled = read(manifests / "train_soft_shuffled.csv")
    excluded = read(manifests / "train_distance_outliers.csv")
    assert set(retained["pair_id"]).isdisjoint(set(excluded["pair_id"]))
    assert set(retained["pair_id"]) | set(excluded["pair_id"]) == set(originals["train"]["pair_id"])
    assert set(retained["pair_id"]) == set(shuffled["pair_id"])
    assert retained["training_eligible"].all()
    assert (~excluded["training_eligible"]).all()
    assert shuffled["soft_shuffled_rank"].tolist() == list(range(len(shuffled)))

    limits = {int(k): float(v) for k, v in summary["outlier_rule"]["limits"].items()}
    retained_limits = retained["nearest_cluster_id"].map(limits)
    excluded_limits = excluded["nearest_cluster_id"].map(limits)
    assert (retained["nearest_distance"] <= retained_limits + TOLERANCE).all()
    if len(excluded):
        assert (excluded["nearest_distance"] > excluded_limits - TOLERANCE).all()

    weights = json.loads((reports / "soft_training_weights.json").read_text(encoding="utf-8"))
    masses = retained[SOFT_COLUMNS].sum(axis=0).to_numpy(dtype=float)
    expected_weights = len(retained) / (3.0 * masses)
    observed_weights = np.asarray([weights["soft_class_weights"][str(k)] for k in CLUSTERS])
    assert np.allclose(expected_weights, observed_weights, atol=TOLERANCE)

    for name in (
        "hard_counts_and_soft_mass.csv", "uncertainty_summary.csv",
        "soft_training_weights.json", "hard_label_near_tie_disagreements.csv",
    ):
        assert (reports / name).is_file(), f"Missing report: {name}"
    assert (output / "DO_NOT_TUNE_ON_TEST.txt").is_file()
    print("\nVALIDATION PASSED")
    print("Soft targets sum to 1, uncertainty weights and training-only class weights are correct,")
    print("all source rows are conserved, and only training distance outliers are excluded.")


if __name__ == "__main__":
    main()
