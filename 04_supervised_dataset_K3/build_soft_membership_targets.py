#!/usr/bin/env python3
"""Create leakage-safe soft K=3 targets from frozen centroid distances."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PARTITIONS = ("train", "validation", "test")
CLUSTERS = (0, 1, 2)
REGIME_NAMES = {0: "Bullish", 1: "Volatile", 2: "Bearish"}
DISTANCE_COLUMNS = tuple(f"target_distance_cluster_{k}" for k in CLUSTERS)
SOFT_COLUMNS = tuple(f"target_soft_cluster_{k}" for k in CLUSTERS)
EPSILON = 1e-12
DEFAULT_HARD_LABEL_NEAR_TIE_RELATIVE_TOLERANCE = 1e-4
DEFAULT_HARD_LABEL_NEAR_TIE_ABSOLUTE_TOLERANCE = 1e-8


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


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset_folder_path", "output_folder_name",
        "temperature_multiplier", "outlier_percentile", "certainty_weight_floor",
        "random_seed", "overwrite_output",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing configuration key(s): {missing}")
    if float(config["temperature_multiplier"]) <= 0:
        raise ValueError("temperature_multiplier must be positive.")
    if not 90.0 <= float(config["outlier_percentile"]) <= 100.0:
        raise ValueError("outlier_percentile must be between 90 and 100.")
    if not 0.0 <= float(config["certainty_weight_floor"]) <= 1.0:
        raise ValueError("certainty_weight_floor must be between 0 and 1.")
    config.setdefault(
        "hard_label_near_tie_relative_tolerance",
        DEFAULT_HARD_LABEL_NEAR_TIE_RELATIVE_TOLERANCE,
    )
    config.setdefault(
        "hard_label_near_tie_absolute_tolerance",
        DEFAULT_HARD_LABEL_NEAR_TIE_ABSOLUTE_TOLERANCE,
    )
    if float(config["hard_label_near_tie_relative_tolerance"]) < 0:
        raise ValueError("hard_label_near_tie_relative_tolerance must be non-negative.")
    if float(config["hard_label_near_tie_absolute_tolerance"]) < 0:
        raise ValueError("hard_label_near_tie_absolute_tolerance must be non-negative.")
    name = str(config["output_folder_name"]).strip()
    if not name or Path(name).is_absolute() or len(Path(name).parts) != 1:
        raise ValueError("output_folder_name must be one relative folder name.")
    return config


def prepare_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(config["dataset_folder_path"]).expanduser().resolve()
    source = root / "manifests"
    if not source.is_dir():
        raise NotADirectoryError(f"Dataset manifests folder not found: {source}")
    output = root / str(config["output_folder_name"])
    if output.exists() and any(output.iterdir()):
        if bool(config["overwrite_output"]):
            shutil.rmtree(output)
        else:
            raise FileExistsError(
                f"Output is not empty: {output}\nChoose another output_folder_name "
                "or set overwrite_output=true."
            )
    return root, source, output


def load_summary(source: Path) -> tuple[dict[str, Any], Path]:
    path = source / "dataset_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    frozen = summary.get("frozen_models", {})
    expected = {
        "pca_input_features": 512,
        "pca_components": 120,
        "kmeans_clusters": 3,
        "kmeans_centroid_shape": [3, 120],
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"Frozen-model contract mismatch for {key}: expected {value}.")
    return summary, path


def load_partition(
    source: Path,
    partition: str,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[pd.DataFrame, Path]:
    path = source / f"{partition}_samples_chronological.csv"
    frame = pd.read_csv(path)
    required = {"pair_id", "chronological_rank", "target_cluster_id", *DISTANCE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if frame.empty or not frame["pair_id"].is_unique:
        raise ValueError(f"{path.name} is empty or has duplicate pair_id values.")
    frame["target_cluster_id"] = pd.to_numeric(frame["target_cluster_id"], errors="raise").astype(int)
    if not set(frame["target_cluster_id"]).issubset(CLUSTERS):
        raise ValueError(f"{path.name} has labels outside 0, 1, 2.")
    for column in DISTANCE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    values = frame[list(DISTANCE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{path.name} contains invalid centroid distances.")
    nearest = np.argmin(values, axis=1)
    hard_labels = frame["target_cluster_id"].to_numpy(dtype=int)
    rows = np.arange(len(frame))
    nearest_distance = values[rows, nearest]
    hard_label_distance = values[rows, hard_labels]
    disagreement = hard_labels != nearest
    disagreement_gap = hard_label_distance - nearest_distance
    allowed_gap = absolute_tolerance + relative_tolerance * np.maximum(
        nearest_distance, EPSILON
    )
    material = disagreement & (disagreement_gap > allowed_gap)
    if material.any():
        example_columns = ["pair_id", "target_cluster_id", *DISTANCE_COLUMNS]
        examples = frame.loc[material, example_columns].head(10).to_dict("records")
        raise ValueError(
            "Hard labels materially disagree with the recorded centroid distances. "
            "This is larger than the configured near-tie tolerance and may indicate "
            f"a damaged or misaligned manifest. Examples: {examples}"
        )

    # sklearn's optimized predict path and its exported/rounded transform distances
    # can choose opposite sides of an extremely close boundary. Preserve the source
    # label for traceability, but use the recorded distances for soft memberships
    # and all geometric calculations.
    frame["source_hard_label_matches_recorded_nearest"] = ~disagreement
    frame["source_hard_label_near_tie_disagreement"] = disagreement
    frame["source_hard_label_distance"] = hard_label_distance
    frame["source_hard_label_distance_gap"] = disagreement_gap
    frame["source_hard_label_distance_gap_percent"] = (
        100.0 * disagreement_gap / np.maximum(nearest_distance, EPSILON)
    )
    return frame, path


def training_temperature(train: pd.DataFrame, multiplier: float) -> tuple[float, float]:
    distances = np.sort(train[list(DISTANCE_COLUMNS)].to_numpy(dtype=float), axis=1)
    positive_gaps = (distances[:, 1] - distances[:, 0])
    positive_gaps = positive_gaps[positive_gaps > EPSILON]
    if len(positive_gaps) == 0:
        raise ValueError("Training distances contain no positive nearest/second-nearest gaps.")
    base = float(np.median(positive_gaps))
    return base * multiplier, base


def training_outlier_limits(train: pd.DataFrame, percentile: float) -> dict[int, float]:
    distances = train[list(DISTANCE_COLUMNS)].to_numpy(dtype=float)
    nearest = distances.min(axis=1)
    assigned = np.argmin(distances, axis=1)
    limits: dict[int, float] = {}
    for cluster in CLUSTERS:
        values = nearest[assigned == cluster]
        if len(values) == 0:
            raise ValueError(f"Training has no hard assignments to Cluster {cluster}.")
        limits[cluster] = float(np.percentile(values, percentile))
    return limits


def annotate(
    frame: pd.DataFrame,
    temperature: float,
    limits: dict[int, float],
    certainty_floor: float,
) -> pd.DataFrame:
    result = frame.copy()
    distances = result[list(DISTANCE_COLUMNS)].to_numpy(dtype=float)
    order = np.argsort(distances, axis=1)
    rows = np.arange(len(result))
    nearest, second, third = order[:, 0], order[:, 1], order[:, 2]
    d1, d2, d3 = distances[rows, nearest], distances[rows, second], distances[rows, third]

    # Subtract the row minimum before exponentiation for numerical stability.
    affinities = np.exp(-(distances - d1[:, None]) / temperature)
    memberships = affinities / affinities.sum(axis=1, keepdims=True)
    entropy = -(memberships * np.log(np.maximum(memberships, EPSILON))).sum(axis=1)
    normalized_entropy = entropy / np.log(3.0)
    certainty = np.clip(1.0 - normalized_entropy, 0.0, 1.0)
    certainty_weight = certainty_floor + (1.0 - certainty_floor) * certainty

    assigned = nearest
    row_limits = np.asarray([limits[int(k)] for k in assigned], dtype=float)
    outlier = d1 > row_limits

    result["nearest_cluster_id"] = nearest
    result["second_nearest_cluster_id"] = second
    result["third_nearest_cluster_id"] = third
    result["nearest_distance"] = d1
    result["second_nearest_distance_recomputed"] = d2
    result["third_nearest_distance"] = d3
    result["nearest_second_gap"] = d2 - d1
    result["nearest_second_gap_percent"] = 100.0 * (d2 - d1) / np.maximum(d1, EPSILON)
    result["nearest_third_gap_percent"] = 100.0 * (d3 - d1) / np.maximum(d1, EPSILON)
    for cluster, column in zip(CLUSTERS, SOFT_COLUMNS):
        result[column] = memberships[:, cluster]
    result["soft_target_entropy"] = entropy
    result["soft_target_normalized_entropy"] = normalized_entropy
    result["soft_target_certainty"] = certainty
    result["certainty_weight"] = certainty_weight
    result["temperature"] = temperature
    result["cluster_outlier_distance_limit"] = row_limits
    result["is_training_distance_outlier"] = outlier
    result["training_eligible"] = ~outlier
    result["exclusion_reason"] = np.where(outlier, "cluster_membership_distance_outlier", "retained")
    return result


def shuffled(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if "soft_shuffled_rank" in result.columns:
        result = result.drop(columns=["soft_shuffled_rank"])
    result.insert(2, "soft_shuffled_rank", np.arange(len(result), dtype=np.int64))
    return result


def soft_mass(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(SOFT_COLUMNS)].sum(axis=0).to_numpy(dtype=float)


def partition_report(partition: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    masses = soft_mass(frame)
    hard = frame["target_cluster_id"].value_counts()
    geometric = frame["nearest_cluster_id"].value_counts()
    total_mass = masses.sum()
    rows: list[dict[str, Any]] = []
    for cluster in CLUSTERS:
        rows.append({
            "partition": partition,
            "cluster_id": cluster,
            "regime_name": REGIME_NAMES[cluster],
            "hard_count": int(hard.get(cluster, 0)),
            "hard_percentage": 100.0 * int(hard.get(cluster, 0)) / len(frame),
            "recorded_distance_nearest_count": int(geometric.get(cluster, 0)),
            "recorded_distance_nearest_percentage": (
                100.0 * int(geometric.get(cluster, 0)) / len(frame)
            ),
            "soft_membership_mass": float(masses[cluster]),
            "soft_mass_percentage": 100.0 * float(masses[cluster]) / total_mass,
            "rows": int(len(frame)),
        })
    return rows


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    root, source, output = prepare_paths(config)
    summary, summary_path = load_summary(source)
    raw: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, Path] = {}
    for partition in PARTITIONS:
        raw[partition], source_paths[partition] = load_partition(
            source,
            partition,
            float(config["hard_label_near_tie_relative_tolerance"]),
            float(config["hard_label_near_tie_absolute_tolerance"]),
        )

    temperature, base_gap = training_temperature(
        raw["train"], float(config["temperature_multiplier"])
    )
    limits = training_outlier_limits(raw["train"], float(config["outlier_percentile"]))
    annotated = {
        partition: annotate(
            frame, temperature, limits, float(config["certainty_weight_floor"])
        )
        for partition, frame in raw.items()
    }
    train_retained = annotated["train"].loc[annotated["train"]["training_eligible"]].copy()
    train_excluded = annotated["train"].loc[~annotated["train"]["training_eligible"]].copy()
    if train_retained.empty:
        raise RuntimeError("No training rows survived the distance-outlier rule.")

    # All preflight and calculations pass before output is created.
    manifests = output / "manifests"
    reports = output / "reports"
    manifests.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    for partition, frame in annotated.items():
        frame.sort_values(["chronological_rank", "pair_id"]).to_csv(
            manifests / f"{partition}_soft_annotated_chronological.csv", index=False
        )
    train_retained = train_retained.sort_values(["chronological_rank", "pair_id"]).reset_index(drop=True)
    train_excluded = train_excluded.sort_values(["chronological_rank", "pair_id"]).reset_index(drop=True)
    train_retained.to_csv(manifests / "train_soft_chronological.csv", index=False)
    shuffled(train_retained, int(config["random_seed"])).to_csv(
        manifests / "train_soft_shuffled.csv", index=False
    )
    train_excluded.to_csv(manifests / "train_distance_outliers.csv", index=False)

    masses = soft_mass(train_retained)
    if np.any(masses <= EPSILON):
        raise RuntimeError("At least one cluster has zero retained soft training mass.")
    class_weights = len(train_retained) / (3.0 * masses)
    weight_report = {
        "calculated_from": "retained training soft targets only",
        "formula": "N / (3 * sum_i soft_membership_i_k)",
        "soft_membership_mass": {str(k): float(masses[k]) for k in CLUSTERS},
        "soft_class_weights": {str(k): float(class_weights[k]) for k in CLUSTERS},
        "certainty_weight_column": "certainty_weight",
        "combined_loss": "-certainty_weight * sum_k(class_weight_k * soft_target_k * log_softmax_k)",
        "use_with": "train_soft_shuffled.csv",
        "do_not_calculate_from": ["validation", "test"],
    }
    (reports / "soft_training_weights.json").write_text(
        json.dumps(weight_report, indent=2), encoding="utf-8"
    )

    distribution_rows: list[dict[str, Any]] = []
    for partition, frame in annotated.items():
        distribution_rows.extend(partition_report(partition, frame))
    pd.DataFrame(distribution_rows).to_csv(
        reports / "hard_counts_and_soft_mass.csv", index=False
    )
    uncertainty_rows: list[dict[str, Any]] = []
    for partition, frame in annotated.items():
        uncertainty_rows.append({
            "partition": partition,
            "rows": int(len(frame)),
            "mean_max_membership": float(frame[list(SOFT_COLUMNS)].max(axis=1).mean()),
            "median_max_membership": float(frame[list(SOFT_COLUMNS)].max(axis=1).median()),
            "mean_normalized_entropy": float(frame["soft_target_normalized_entropy"].mean()),
            "median_normalized_entropy": float(frame["soft_target_normalized_entropy"].median()),
            "near_uniform_rows_max_membership_le_0_40": int(
                (frame[list(SOFT_COLUMNS)].max(axis=1) <= 0.40).sum()
            ),
            "training_distance_outliers": int(frame["is_training_distance_outlier"].sum()),
        })
    pd.DataFrame(uncertainty_rows).to_csv(reports / "uncertainty_summary.csv", index=False)

    label_consistency_rows = []
    for partition, frame in annotated.items():
        mismatches = frame.loc[frame["source_hard_label_near_tie_disagreement"]].copy()
        if not mismatches.empty:
            mismatches.insert(0, "source_partition", partition)
            label_consistency_rows.append(mismatches)
    consistency_columns = [
        "source_partition", "pair_id", "target_cluster_id", "nearest_cluster_id",
        *DISTANCE_COLUMNS, "source_hard_label_distance",
        "source_hard_label_distance_gap", "source_hard_label_distance_gap_percent",
    ]
    if label_consistency_rows:
        pd.concat(label_consistency_rows, ignore_index=True)[consistency_columns].to_csv(
            reports / "hard_label_near_tie_disagreements.csv", index=False
        )
    else:
        pd.DataFrame(columns=consistency_columns).to_csv(
            reports / "hard_label_near_tie_disagreements.csv", index=False
        )

    result_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "method": "training-scaled Gibbs soft memberships from all three frozen K-Means distances",
        "soft_membership": {
            "formula": "p_k = exp(-(d_k-d_min)/T) / sum_j exp(-(d_j-d_min)/T)",
            "temperature_source": "median positive nearest-to-second distance gap in training only",
            "training_median_gap": base_gap,
            "temperature_multiplier": float(config["temperature_multiplier"]),
            "temperature": temperature,
            "hard_label_used_for_training_loss": False,
            "relabelled_samples": False,
            "recorded_centroid_distances_authoritative": True,
        },
        "hard_label_consistency": {
            "policy": "allow only numerically near-tied disagreements; fail on material mismatches",
            "relative_tolerance": float(config["hard_label_near_tie_relative_tolerance"]),
            "absolute_tolerance": float(config["hard_label_near_tie_absolute_tolerance"]),
            "hard_labels_preserved": True,
            "hard_labels_used_for_soft_memberships": False,
            "near_tie_disagreements": {
                p: int(raw[p]["source_hard_label_near_tie_disagreement"].sum())
                for p in PARTITIONS
            },
            "audit_report": "reports/hard_label_near_tie_disagreements.csv",
        },
        "certainty": {
            "formula": "certainty = 1 - entropy(p)/log(3)",
            "weight_formula": "floor + (1-floor)*certainty",
            "certainty_weight_floor": float(config["certainty_weight_floor"]),
            "ambiguous_samples_discarded": False,
        },
        "outlier_rule": {
            "percentile": float(config["outlier_percentile"]),
            "source": "nearest-distance distribution grouped by recorded-distance argmin in training only",
            "limits": {str(k): limits[k] for k in CLUSTERS},
            "applied_to_training": True,
            "validation_and_test_rows_removed": False,
        },
        "frozen_contract": {
            "resnet_features": 512, "pca_components": 120, "kmeans_clusters": 3,
            "centroid_shape": [3, 120], "fit_called": False,
            "kmeans_path_recorded_by_source": summary.get("frozen_models", {}).get("kmeans_path"),
            "kmeans_sha256_recorded_by_source": summary.get("frozen_models", {}).get("kmeans_sha256"),
            "model_artifact_needed_by_processor": False,
        },
        "source_files": {
            "dataset_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            **{p: {"path": str(source_paths[p]), "sha256": sha256_file(source_paths[p]),
                   "rows": int(len(raw[p]))} for p in PARTITIONS},
        },
        "rows": {
            "train_original": int(len(raw["train"])),
            "train_retained": int(len(train_retained)),
            "train_outliers_excluded": int(len(train_excluded)),
            "validation_annotated_full": int(len(annotated["validation"])),
            "test_annotated_full": int(len(annotated["test"])),
        },
        "leakage_controls": {
            "temperature_from_training_only": True,
            "outlier_limits_from_training_only": True,
            "class_weights_from_training_only": True,
            "test_used_for_selection_or_tuning": False,
        },
    }
    (reports / "soft_membership_summary.json").write_text(
        json.dumps(result_summary, indent=2), encoding="utf-8"
    )
    (output / "DO_NOT_TUNE_ON_TEST.txt").write_text(
        "Temperature, outlier limits, and class weights are frozen from training only.\n"
        "Use validation for model selection. Evaluate test once after the classifier is frozen.\n",
        encoding="utf-8",
    )

    print("\nSOFT-MEMBERSHIP DATASET COMPLETE")
    print(f"temperature: {temperature:.10g} (training median gap {base_gap:.10g})")
    print(f"train retained: {len(train_retained):,}/{len(raw['train']):,}")
    print("training soft mass:", {k: round(float(masses[k]), 3) for k in CLUSTERS})
    print("soft class weights:", {k: round(float(class_weights[k]), 5) for k in CLUSTERS})
    print(f"output: {output}")


if __name__ == "__main__":
    main()
