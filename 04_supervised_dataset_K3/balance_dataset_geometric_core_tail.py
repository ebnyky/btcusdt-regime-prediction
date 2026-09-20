#!/usr/bin/env python3
"""Filter ambiguous K=3 future targets and create leakage-safe train manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PARTITIONS = ("train", "validation", "test")
CLUSTERS = (0, 1, 2)
REGIME_NAMES = {
    0: "Bullish",
    1: "Sideways/non-directional",
    2: "Bearish",
}
DISTANCE_COLUMNS = tuple(f"target_distance_cluster_{k}" for k in CLUSTERS)
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="Path to config.json (default: beside this script).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset_folder_path",
        "kmeans_model_path",
        "output_folder_name",
        "confidence_threshold",
        "membership_percentile",
        "random_seed",
        "create_exactly_balanced_training_manifest",
        "overwrite_output",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing configuration key(s): {missing}")

    threshold = float(config["confidence_threshold"])
    percentile = float(config["membership_percentile"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if not 50.0 <= percentile <= 100.0:
        raise ValueError("membership_percentile must be between 50 and 100.")
    output_name = str(config["output_folder_name"]).strip()
    if not output_name or Path(output_name).is_absolute() or len(Path(output_name).parts) != 1:
        raise ValueError("output_folder_name must be one relative folder name.")
    return config


def prepare_paths(config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    root = Path(config["dataset_folder_path"]).expanduser().resolve()
    manifests = root / "manifests"
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset folder not found: {root}")
    if not manifests.is_dir():
        raise NotADirectoryError(f"Dataset manifests folder not found: {manifests}")

    output = root / str(config["output_folder_name"])
    if output.exists() and any(output.iterdir()):
        if bool(config["overwrite_output"]):
            shutil.rmtree(output)
        else:
            raise FileExistsError(
                f"Output is not empty: {output}\n"
                "Choose another output_folder_name or set overwrite_output=true."
            )
    output_manifests = output / "manifests"
    output_reports = output / "reports"
    return root, manifests, output_manifests, output_reports


def load_dataset_summary(manifests: Path) -> tuple[dict[str, Any], Path]:
    path = manifests / "dataset_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    frozen = summary.get("frozen_models", {})
    if frozen.get("pca_input_features") != 512:
        raise ValueError("Dataset does not declare the frozen 512-feature encoder.")
    if frozen.get("pca_components") != 120:
        raise ValueError("Dataset is not the validated 120-component PCA dataset.")
    if frozen.get("kmeans_clusters") != 3:
        raise ValueError("Dataset is not a frozen K=3 dataset.")
    if frozen.get("kmeans_centroid_shape") != [3, 120]:
        raise ValueError("Dataset summary does not declare centroid shape [3, 120].")
    return summary, path


def resolve_kmeans_path(
    config: dict[str, Any], root: Path, summary: dict[str, Any]
) -> Path:
    configured = str(config["kmeans_model_path"]).strip()
    candidates: list[Path] = []
    if configured.lower() != "auto":
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidates.append(candidate)
    else:
        candidates.extend((root / "models" / "kmeans.joblib", root / "kmeans.joblib"))
        recorded = summary.get("frozen_models", {}).get("kmeans_path")
        if recorded:
            candidates.append(Path(recorded).expanduser())

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    searched = "\n- ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Validated K=3 kmeans.joblib was not found. Searched:\n- " + searched
    )


def load_centroids(path: Path, summary: dict[str, Any]) -> np.ndarray:
    model = joblib.load(path)
    centers = np.asarray(getattr(model, "cluster_centers_", None), dtype=np.float64)
    if int(getattr(model, "n_clusters", -1)) != 3 or centers.shape != (3, 120):
        raise ValueError(
            "Wrong K-Means model. Required n_clusters=3 and centers shaped (3, 120); "
            f"found n_clusters={getattr(model, 'n_clusters', None)}, shape={centers.shape}."
        )
    expected_hash = summary.get("frozen_models", {}).get("kmeans_sha256")
    observed_hash = sha256_file(path)
    if expected_hash and expected_hash != observed_hash:
        raise ValueError(
            "K-Means SHA-256 does not match the model recorded by the dataset. "
            "Use the exact frozen model that created these labels."
        )
    return centers


def centroid_distance_matrix(centers: np.ndarray) -> np.ndarray:
    matrix = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    off_diagonal = matrix[~np.eye(3, dtype=bool)]
    if not np.isfinite(matrix).all() or np.any(off_diagonal <= EPSILON):
        raise ValueError("K-Means centroids are invalid or duplicated.")
    return matrix


def load_partition(manifests: Path, partition: str) -> tuple[pd.DataFrame, Path]:
    path = manifests / f"{partition}_samples_chronological.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing original manifest: {path}")
    frame = pd.read_csv(path)
    required = {"pair_id", "chronological_rank", "target_cluster_id", *DISTANCE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {missing}")
    if frame.empty:
        raise ValueError(f"{path.name} is empty.")
    if not frame["pair_id"].is_unique:
        raise ValueError(f"{path.name} contains duplicate pair_id values.")
    frame["target_cluster_id"] = pd.to_numeric(
        frame["target_cluster_id"], errors="raise"
    ).astype(int)
    if not set(frame["target_cluster_id"]).issubset(CLUSTERS):
        raise ValueError(f"{path.name} contains a target outside clusters 0, 1, 2.")
    for column in DISTANCE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    values = frame[list(DISTANCE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{path.name} contains invalid target centroid distances.")
    return frame, path


def add_geometry(frame: pd.DataFrame, centroid_distances: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    distances = result[list(DISTANCE_COLUMNS)].to_numpy(dtype=np.float64)
    order = np.argsort(distances, axis=1)
    nearest_id = order[:, 0]
    assigned = result["target_cluster_id"].to_numpy(dtype=int)
    inconsistent = assigned != nearest_id
    if inconsistent.any():
        examples = result.loc[inconsistent, "pair_id"].head(10).tolist()
        raise ValueError(
            f"{int(inconsistent.sum())} target labels are not their nearest centroid. "
            f"Examples: {examples}"
        )

    rows = np.arange(len(result))
    second_id = order[:, 1]
    third_id = order[:, 2]
    d1 = distances[rows, nearest_id]
    d2 = distances[rows, second_id]
    d3 = distances[rows, third_id]
    pair_second = centroid_distances[nearest_id, second_id]
    pair_third = centroid_distances[nearest_id, third_id]
    q_second = np.clip((d2 - d1) / pair_second, 0.0, 1.0)
    q_third = np.clip((d3 - d1) / pair_third, 0.0, 1.0)

    result["nearest_cluster_id"] = nearest_id
    result["second_nearest_cluster_id"] = second_id
    result["third_nearest_cluster_id"] = third_id
    result["nearest_distance"] = d1
    result["second_nearest_distance_recomputed"] = d2
    result["third_nearest_distance"] = d3
    result["nearest_margin_absolute"] = d2 - d1
    result["nearest_margin_percent"] = 100.0 * (d2 - d1) / np.maximum(d1, EPSILON)
    result["third_separation_percent"] = 100.0 * (d3 - d1) / np.maximum(d1, EPSILON)
    result["assigned_to_second_centroid_distance"] = pair_second
    result["assigned_to_third_centroid_distance"] = pair_third
    result["confidence_vs_second"] = q_second
    result["confidence_vs_third"] = q_third
    result["geometric_confidence"] = np.minimum(q_second, q_third)
    return result


def training_membership_limits(
    train: pd.DataFrame, percentile: float
) -> dict[int, float]:
    limits: dict[int, float] = {}
    for cluster in CLUSTERS:
        values = train.loc[
            train["target_cluster_id"] == cluster, "nearest_distance"
        ].to_numpy(dtype=float)
        if len(values) == 0:
            raise ValueError(f"Training contains no targets in Cluster {cluster}.")
        limits[cluster] = float(np.percentile(values, percentile))
    return limits


def apply_rules(
    frame: pd.DataFrame,
    threshold: float,
    membership_limits: dict[int, float],
) -> pd.DataFrame:
    result = frame.copy()
    assigned = result["target_cluster_id"].to_numpy(dtype=int)
    limits = np.asarray([membership_limits[int(value)] for value in assigned])
    margin_passed = result["geometric_confidence"].to_numpy(dtype=float) >= threshold
    membership_passed = result["nearest_distance"].to_numpy(dtype=float) <= limits
    result["confidence_threshold"] = threshold
    result["cluster_membership_distance_limit"] = limits
    result["passes_centroid_separation"] = margin_passed
    result["passes_cluster_membership"] = membership_passed
    result["confidence_accepted"] = margin_passed & membership_passed
    result["rejection_reason"] = np.select(
        [
            ~margin_passed & ~membership_passed,
            ~margin_passed,
            ~membership_passed,
        ],
        [
            "insufficient_centroid_separation_and_membership_outlier",
            "insufficient_centroid_separation",
            "cluster_membership_outlier",
        ],
        default="accepted",
    )
    return result


def shuffled(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if "balanced_shuffled_rank" in result.columns:
        result = result.drop(columns=["balanced_shuffled_rank"])
    result.insert(2, "balanced_shuffled_rank", np.arange(len(result), dtype=np.int64))
    return result


def temporal_spread_balance(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = frame["target_cluster_id"].value_counts()
    if any(int(counts.get(cluster, 0)) == 0 for cluster in CLUSTERS):
        raise ValueError(
            "At least one cluster has zero confident training targets. Review the "
            "training/validation threshold sweep before changing the frozen threshold."
        )
    target_count = int(min(counts[cluster] for cluster in CLUSTERS))
    selected_indices: list[int] = []

    for cluster in CLUSTERS:
        group = frame.loc[frame["target_cluster_id"] == cluster].sort_values(
            ["chronological_rank", "pair_id"]
        )
        if len(group) == target_count:
            selected_indices.extend(group.index.tolist())
            continue
        # Divide the cluster's full time span into equal index segments and retain
        # the most geometrically confident row from each segment.
        for segment in np.array_split(np.arange(len(group)), target_count):
            candidates = group.iloc[segment]
            chosen = candidates.sort_values(
                ["geometric_confidence", "chronological_rank", "pair_id"],
                ascending=[False, True, True],
            ).index[0]
            selected_indices.append(int(chosen))

    mask = frame.index.isin(selected_indices)
    balanced = frame.loc[mask].sort_values(
        ["chronological_rank", "pair_id"]
    ).reset_index(drop=True)
    excluded = frame.loc[~mask].copy()
    excluded["balance_exclusion_reason"] = (
        "confident_sample_not_selected_by_temporal_spread_undersampling"
    )
    excluded = excluded.sort_values(["chronological_rank", "pair_id"]).reset_index(drop=True)
    return balanced, excluded


def count_rows(stage: str, partition: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    total = len(frame)
    counts = frame["target_cluster_id"].value_counts()
    return [
        {
            "stage": stage,
            "partition": partition,
            "cluster_id": cluster,
            "regime_name": REGIME_NAMES[cluster],
            "count": int(counts.get(cluster, 0)),
            "percentage": 100.0 * int(counts.get(cluster, 0)) / max(total, 1),
            "total_rows": int(total),
        }
        for cluster in CLUSTERS
    ]


def threshold_sweep(
    frames: dict[str, pd.DataFrame], membership_limits: dict[int, float]
) -> pd.DataFrame:
    thresholds = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for partition, frame in frames.items():
            checked = apply_rules(frame, threshold, membership_limits)
            accepted = checked.loc[checked["confidence_accepted"]]
            counts = accepted["target_cluster_id"].value_counts()
            for cluster in CLUSTERS:
                original_count = int((frame["target_cluster_id"] == cluster).sum())
                accepted_count = int(counts.get(cluster, 0))
                rows.append({
                    "threshold": threshold,
                    "partition": partition,
                    "cluster_id": cluster,
                    "regime_name": REGIME_NAMES[cluster],
                    "original_count": original_count,
                    "accepted_count": accepted_count,
                    "cluster_coverage_percent": (
                        100.0 * accepted_count / max(original_count, 1)
                    ),
                    "partition_accepted_total": int(len(accepted)),
                    "partition_coverage_percent": 100.0 * len(accepted) / len(frame),
                })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    root, original_manifests, output_manifests, output_reports = prepare_paths(config)
    output_root = output_manifests.parent

    summary, dataset_summary_path = load_dataset_summary(original_manifests)
    kmeans_path = resolve_kmeans_path(config, root, summary)
    centers = load_centroids(kmeans_path, summary)
    centroid_matrix = centroid_distance_matrix(centers)

    raw_frames: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, Path] = {}
    geometric_frames: dict[str, pd.DataFrame] = {}
    for partition in PARTITIONS:
        raw, source_path = load_partition(original_manifests, partition)
        raw_frames[partition] = raw
        source_paths[partition] = source_path
        geometric_frames[partition] = add_geometry(raw, centroid_matrix)

    # Do not create or overwrite any output until the complete input/model
    # preflight has passed.
    output_manifests.mkdir(parents=True, exist_ok=True)
    output_reports.mkdir(parents=True, exist_ok=True)

    percentile = float(config["membership_percentile"])
    threshold = float(config["confidence_threshold"])
    seed = int(config["random_seed"])
    limits = training_membership_limits(geometric_frames["train"], percentile)
    evaluated = {
        partition: apply_rules(frame, threshold, limits)
        for partition, frame in geometric_frames.items()
    }

    distribution_rows: list[dict[str, Any]] = []
    accepted_frames: dict[str, pd.DataFrame] = {}
    rejected_frames: dict[str, pd.DataFrame] = {}
    for partition in PARTITIONS:
        checked = evaluated[partition]
        accepted = checked.loc[checked["confidence_accepted"]].copy()
        rejected = checked.loc[~checked["confidence_accepted"]].copy()
        accepted = accepted.sort_values(["chronological_rank", "pair_id"]).reset_index(drop=True)
        rejected = rejected.sort_values(["chronological_rank", "pair_id"]).reset_index(drop=True)
        accepted_frames[partition] = accepted
        rejected_frames[partition] = rejected

        accepted.to_csv(
            output_manifests / f"{partition}_confident_chronological.csv", index=False
        )
        rejected.to_csv(output_manifests / f"{partition}_rejected.csv", index=False)
        distribution_rows.extend(count_rows("original", partition, checked))
        distribution_rows.extend(count_rows("confidence_accepted", partition, accepted))

    shuffled(accepted_frames["train"], seed).to_csv(
        output_manifests / "train_confident_shuffled.csv", index=False
    )

    balanced = pd.DataFrame()
    balance_excluded = pd.DataFrame()
    if bool(config["create_exactly_balanced_training_manifest"]):
        balanced, balance_excluded = temporal_spread_balance(accepted_frames["train"])
        balanced.to_csv(
            output_manifests / "train_balanced_chronological.csv", index=False
        )
        shuffled(balanced, seed).to_csv(
            output_manifests / "train_balanced_shuffled.csv", index=False
        )
        balance_excluded.to_csv(
            output_manifests / "train_balance_excluded.csv", index=False
        )
        distribution_rows.extend(count_rows("exactly_balanced", "train", balanced))

    pd.DataFrame(distribution_rows).to_csv(
        output_reports / "class_distribution_before_after.csv", index=False
    )
    # The final test partition is deliberately excluded from threshold analysis.
    # It is processed once only at the locked configured threshold.
    threshold_sweep(
        {name: geometric_frames[name] for name in ("train", "validation")},
        limits,
    ).to_csv(
        output_reports / "confidence_threshold_sweep.csv", index=False
    )

    centroid_table = pd.DataFrame(
        centroid_matrix,
        index=[f"cluster_{k}" for k in CLUSTERS],
        columns=[f"cluster_{k}" for k in CLUSTERS],
    )
    centroid_table.index.name = "assigned_centroid"
    centroid_table.to_csv(output_reports / "centroid_distance_matrix.csv")

    rejection_rows: list[dict[str, Any]] = []
    for partition, rejected in rejected_frames.items():
        counts = rejected["rejection_reason"].value_counts()
        for reason, count in counts.items():
            rejection_rows.append({
                "partition": partition,
                "reason": reason,
                "count": int(count),
                "percentage_of_partition": 100.0 * int(count) / len(evaluated[partition]),
            })
    pd.DataFrame(rejection_rows).to_csv(
        output_reports / "rejection_reasons.csv", index=False
    )

    train_counts = accepted_frames["train"]["target_cluster_id"].value_counts()
    if any(int(train_counts.get(cluster, 0)) == 0 for cluster in CLUSTERS):
        raise RuntimeError(
            "At least one class has no confidence-accepted training targets; "
            "reports were written, but class weights cannot be calculated."
        )
    class_weights = {
        str(cluster): len(accepted_frames["train"])
        / (3.0 * int(train_counts[cluster]))
        for cluster in CLUSTERS
    }
    weight_report = {
        "calculated_from": "confidence-accepted training targets only",
        "formula": "N / (number_of_classes * class_count)",
        "counts": {str(k): int(train_counts[k]) for k in CLUSTERS},
        "balanced_cross_entropy_weights": class_weights,
        "use_with": "train_confident_shuffled.csv",
        "do_not_apply_to": ["validation", "test"],
    }
    (output_reports / "confident_training_class_weights.json").write_text(
        json.dumps(weight_report, indent=2), encoding="utf-8"
    )

    summary_output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "method": "frozen K=3 centroid-normalized target confidence filtering",
        "rules": {
            "geometric_confidence_formula": (
                "min_j((distance_to_competitor_j - nearest_distance) / "
                "distance_between_assigned_and_competitor_centroids)"
            ),
            "confidence_threshold": threshold,
            "membership_percentile": percentile,
            "membership_limits_calculated_from": "training targets only",
            "cluster_membership_distance_limits": {
                str(k): limits[k] for k in CLUSTERS
            },
            "relabel_rejected_samples": False,
        },
        "frozen_contract": {
            "resnet_features": 512,
            "pca_components": 120,
            "kmeans_clusters": 3,
            "centroid_shape": [3, 120],
            "fit_called": False,
            "kmeans_path": str(kmeans_path),
            "kmeans_sha256": sha256_file(kmeans_path),
        },
        "source_files": {
            "dataset_summary": {
                "path": str(dataset_summary_path),
                "sha256": sha256_file(dataset_summary_path),
            },
            **{
                partition: {
                    "path": str(source_paths[partition]),
                    "sha256": sha256_file(source_paths[partition]),
                    "rows": int(len(raw_frames[partition])),
                }
                for partition in PARTITIONS
            },
        },
        "counts": {
            partition: {
                "original": int(len(evaluated[partition])),
                "confidence_accepted": int(len(accepted_frames[partition])),
                "rejected": int(len(rejected_frames[partition])),
                "coverage_percent": 100.0
                * len(accepted_frames[partition])
                / len(evaluated[partition]),
                "original_by_cluster": {
                    str(k): int((evaluated[partition]["target_cluster_id"] == k).sum())
                    for k in CLUSTERS
                },
                "accepted_by_cluster": {
                    str(k): int((accepted_frames[partition]["target_cluster_id"] == k).sum())
                    for k in CLUSTERS
                },
            }
            for partition in PARTITIONS
        },
        "exact_balance": {
            "created": bool(config["create_exactly_balanced_training_manifest"]),
            "selection": (
                "deterministic temporal-spread undersampling with highest-confidence "
                "selection inside each time segment"
            ),
            "rows": int(len(balanced)),
            "count_per_cluster": (
                int(len(balanced) // 3) if len(balanced) else 0
            ),
            "excluded_confident_rows": int(len(balance_excluded)),
            "random_seed_for_shuffled_copy": seed,
        },
        "primary_evaluation": {
            "validation": str(source_paths["validation"]),
            "test": str(source_paths["test"]),
            "natural_distributions_preserved": True,
            "threshold_must_not_be_selected_from_test": True,
        },
    }
    (output_reports / "balance_summary.json").write_text(
        json.dumps(summary_output, indent=2), encoding="utf-8"
    )
    (output_root / "DO_NOT_TUNE_ON_TEST.txt").write_text(
        "Keep the original validation and test manifests as the primary evaluation sets.\n"
        "Use confident validation/test subsets only for secondary selective reporting.\n"
        "Do not change the confidence threshold after inspecting test performance.\n",
        encoding="utf-8",
    )

    print("\nGEOMETRIC BALANCING COMPLETE")
    for partition in PARTITIONS:
        original = len(evaluated[partition])
        accepted = len(accepted_frames[partition])
        counts = accepted_frames[partition]["target_cluster_id"].value_counts().sort_index()
        print(
            f"{partition}: accepted {accepted:,}/{original:,} "
            f"({100.0 * accepted / original:.2f}%) | {counts.to_dict()}"
        )
    if len(balanced):
        print(
            f"exactly balanced training: {len(balanced):,} rows | "
            f"{len(balanced) // 3:,} per cluster"
        )
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
