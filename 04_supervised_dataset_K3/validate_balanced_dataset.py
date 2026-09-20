#!/usr/bin/env python3
"""Validate geometric-confidence outputs and prove originals were preserved."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PARTITIONS = ("train", "validation", "test")
CLUSTERS = (0, 1, 2)
TOLERANCE = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing output: {path}")
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    root = Path(config["dataset_folder_path"]).expanduser().resolve()
    output = root / str(config["output_folder_name"])
    manifests = output / "manifests"
    reports = output / "reports"
    summary_path = reports / "balance_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing balance summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    threshold = float(summary["rules"]["confidence_threshold"])
    limits = {
        int(k): float(v)
        for k, v in summary["rules"]["cluster_membership_distance_limits"].items()
    }

    assert summary["frozen_contract"]["resnet_features"] == 512
    assert summary["frozen_contract"]["pca_components"] == 120
    assert summary["frozen_contract"]["kmeans_clusters"] == 3
    assert summary["frozen_contract"]["centroid_shape"] == [3, 120]
    assert summary["frozen_contract"]["fit_called"] is False
    assert summary["rules"]["relabel_rejected_samples"] is False
    assert summary["primary_evaluation"]["natural_distributions_preserved"] is True
    assert summary["primary_evaluation"]["threshold_must_not_be_selected_from_test"] is True

    for name, source in summary["source_files"].items():
        path = Path(source["path"])
        assert path.is_file(), f"Original source disappeared: {path}"
        assert sha256_file(path) == source["sha256"], f"Original source changed: {path}"

    for partition in PARTITIONS:
        original_path = Path(summary["source_files"][partition]["path"])
        original = pd.read_csv(original_path)
        accepted = read_csv(manifests / f"{partition}_confident_chronological.csv")
        rejected = read_csv(manifests / f"{partition}_rejected.csv")

        assert original["pair_id"].is_unique
        assert accepted["pair_id"].is_unique
        assert rejected["pair_id"].is_unique
        original_ids = set(original["pair_id"])
        accepted_ids = set(accepted["pair_id"])
        rejected_ids = set(rejected["pair_id"])
        assert accepted_ids.isdisjoint(rejected_ids)
        assert accepted_ids | rejected_ids == original_ids
        assert set(accepted["target_cluster_id"]).issubset(CLUSTERS)
        assert (accepted["confidence_accepted"] == True).all()  # noqa: E712
        assert (rejected["confidence_accepted"] == False).all()  # noqa: E712
        assert (accepted["geometric_confidence"] + TOLERANCE >= threshold).all()
        accepted_limits = accepted["target_cluster_id"].map(limits)
        assert (
            accepted["nearest_distance"] <= accepted_limits + TOLERANCE
        ).all()
        assert accepted["chronological_rank"].is_monotonic_increasing

        claimed = summary["counts"][partition]
        assert len(original) == claimed["original"]
        assert len(accepted) == claimed["confidence_accepted"]
        assert len(rejected) == claimed["rejected"]
        print(
            f"{partition}: conserved {len(original):,} rows | "
            f"accepted={len(accepted):,}, rejected={len(rejected):,}"
        )

    confident = read_csv(manifests / "train_confident_chronological.csv")
    confident_shuffled = read_csv(manifests / "train_confident_shuffled.csv")
    assert set(confident["pair_id"]) == set(confident_shuffled["pair_id"])
    assert confident_shuffled["balanced_shuffled_rank"].tolist() == list(
        range(len(confident_shuffled))
    )

    if summary["exact_balance"]["created"]:
        balanced = read_csv(manifests / "train_balanced_chronological.csv")
        balanced_shuffled = read_csv(manifests / "train_balanced_shuffled.csv")
        excluded = read_csv(manifests / "train_balance_excluded.csv")
        counts = balanced["target_cluster_id"].value_counts().reindex(CLUSTERS, fill_value=0)
        assert counts.nunique() == 1
        assert set(balanced["pair_id"]).isdisjoint(set(excluded["pair_id"]))
        assert set(balanced["pair_id"]) | set(excluded["pair_id"]) == set(
            confident["pair_id"]
        )
        assert set(balanced["pair_id"]) == set(balanced_shuffled["pair_id"])
        assert balanced["chronological_rank"].is_monotonic_increasing
        assert len(balanced) == summary["exact_balance"]["rows"]
        assert int(counts.iloc[0]) == summary["exact_balance"]["count_per_cluster"]
        print(f"exact balance: {counts.to_dict()}")

    required_reports = (
        "class_distribution_before_after.csv",
        "confidence_threshold_sweep.csv",
        "confident_training_class_weights.json",
        "centroid_distance_matrix.csv",
        "rejection_reasons.csv",
    )
    for name in required_reports:
        assert (reports / name).is_file(), f"Missing report: {name}"
    sweep = pd.read_csv(reports / "confidence_threshold_sweep.csv")
    assert set(sweep["partition"]) == {"train", "validation"}
    assert (output / "DO_NOT_TUNE_ON_TEST.txt").is_file()

    print("\nVALIDATION PASSED")
    print(
        "Original manifests are unchanged; geometric rules, row conservation, "
        "class balance, and shuffled alignment are valid."
    )


if __name__ == "__main__":
    main()
