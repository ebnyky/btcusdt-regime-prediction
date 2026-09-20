#!/usr/bin/env python3
"""Validate a generated K=3 forecasting dataset without changing it."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================================
# USER SETTING — EDIT THIS PATH
# ============================================================================

DATASET_FOLDER_PATH = Path(
    r"C:\path\to\BTCUSDT_k3_forecasting_dataset"
)


TRAIN_END_UTC = pd.Timestamp("2024-12-17 13:00:00", tz="UTC")
VALIDATION_START_UTC = pd.Timestamp("2024-12-17 14:00:00", tz="UTC")
VALIDATION_END_UTC = pd.Timestamp("2025-07-06 23:00:00", tz="UTC")
TEST_START_UTC = pd.Timestamp("2025-07-07 00:00:00", tz="UTC")
TEST_END_UTC = pd.Timestamp("2026-07-31 23:00:00", tz="UTC")


def validate_pair_times(frame: pd.DataFrame, partition: str) -> None:
    input_start = pd.to_datetime(frame["input_start_time"], utc=True)
    input_stop = pd.to_datetime(frame["input_stop_time"], utc=True)
    target_start = pd.to_datetime(frame["target_start_time"], utc=True)
    target_stop = pd.to_datetime(frame["target_stop_time"], utc=True)

    assert ((input_stop - input_start) == pd.Timedelta(hours=29)).all()
    assert ((target_start - input_stop) == pd.Timedelta(hours=1)).all()
    assert ((target_stop - target_start) == pd.Timedelta(hours=29)).all()

    if partition == "train":
        assert (target_stop <= TRAIN_END_UTC).all()
    elif partition == "validation":
        assert (input_start >= VALIDATION_START_UTC).all()
        assert (target_stop <= VALIDATION_END_UTC).all()
    elif partition == "test":
        assert (input_start >= TEST_START_UTC).all()
        assert (target_stop <= TEST_END_UTC).all()


def main() -> None:
    root = DATASET_FOLDER_PATH.expanduser().resolve()
    manifests = root / "manifests"
    summary_path = manifests / "dataset_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing dataset summary: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["frozen_models"]["pca_components"] == 120
    assert summary["frozen_models"]["pca_input_features"] == 512
    assert summary["frozen_models"]["kmeans_clusters"] == 3
    assert summary["frozen_models"]["kmeans_centroid_shape"] == [3, 120]
    assert summary["encoder"]["fit_called"] is False

    frozen_periods = summary["frozen_periods_utc"]
    assert pd.Timestamp(frozen_periods["validation"][1]) == VALIDATION_END_UTC
    assert pd.Timestamp(frozen_periods["test"][0]) == TEST_START_UTC

    boundary = summary["cross_csv_boundary"]
    assert boundary["passed"] is True
    assert boundary["overlapping_timestamp_count"] == 0
    assert boundary["cross_csv_interval_hours"] == 1.0
    assert boundary["forecast_pairs_may_cross_source_csv_boundary"] is False
    assert pd.Timestamp(boundary["train_validation_csv_end_utc"]) == VALIDATION_END_UTC
    assert pd.Timestamp(boundary["test_csv_start_utc"]) == TEST_START_UTC

    pair_ids_by_partition: dict[str, set[str]] = {}
    for partition in ("train", "validation", "test"):
        chronological_path = manifests / f"{partition}_samples_chronological.csv"
        shuffled_path = manifests / f"{partition}_samples_shuffled.csv"
        chronological = pd.read_csv(chronological_path)
        shuffled = pd.read_csv(shuffled_path)

        assert not chronological.empty
        assert chronological["pair_id"].is_unique
        assert shuffled["pair_id"].is_unique
        assert set(chronological["pair_id"]) == set(shuffled["pair_id"])
        assert chronological["input_start_time"].is_monotonic_increasing
        assert set(chronological["target_cluster_id"]) <= {0, 1, 2}
        assert set(chronological["input_cluster_id"]) <= {0, 1, 2}
        validate_pair_times(chronological, partition)

        missing_inputs = sum(
            not (root / value).is_file() for value in chronological["input_image_path"]
        )
        missing_targets = sum(
            not (root / value).is_file() for value in chronological["target_image_path"]
        )
        assert missing_inputs == 0
        assert missing_targets == 0

        pair_ids_by_partition[partition] = set(chronological["pair_id"])
        print(
            f"{partition}: {len(chronological):,} pairs | "
            f"targets={chronological['target_cluster_id'].value_counts().sort_index().to_dict()}"
        )

    assert pair_ids_by_partition["train"].isdisjoint(pair_ids_by_partition["validation"])
    assert pair_ids_by_partition["train"].isdisjoint(pair_ids_by_partition["test"])
    assert pair_ids_by_partition["validation"].isdisjoint(pair_ids_by_partition["test"])

    train = pd.read_csv(manifests / "train_samples_chronological.csv")
    validation = pd.read_csv(manifests / "validation_samples_chronological.csv")
    test = pd.read_csv(manifests / "test_samples_chronological.csv")
    assert pd.to_datetime(train["target_stop_time"], utc=True).max() < pd.to_datetime(
        validation["input_start_time"], utc=True
    ).min()
    assert set(validation["input_source_id"]) == {"train_validation"}
    assert set(validation["target_source_id"]) == {"train_validation"}
    assert set(test["input_source_id"]) == {"test"}
    assert set(test["target_source_id"]) == {"test"}

    required = (
        root / "skipped_images.txt",
        root / "skipped_pairs.txt",
        manifests / "labeled_images_manifest.csv",
        manifests / "duplicate_images.json",
        root / "reports" / "cluster_distribution.csv",
        root / "reports" / "training_class_weights.json",
        root / "reports" / "current_to_future_transition_counts.csv",
    )
    for path in required:
        assert path.is_file(), f"Missing required output: {path}"

    print("\nVALIDATION PASSED")
    print(
        "Exact 30->30 alignment, adjacent CSV boundaries, source isolation, "
        "labels, images, and shuffles are valid."
    )


if __name__ == "__main__":
    main()
