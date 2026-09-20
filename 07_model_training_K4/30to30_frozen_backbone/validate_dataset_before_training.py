from __future__ import annotations

import json
from pathlib import Path

from shared import (
    count_missing_images,
    load_json,
    load_manifest,
    resolve_dataset_paths,
    verify_split_boundaries,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


def print_partition(name, frame, dataset_dir):
    print(f"\n{name.upper()}")
    print("-" * 72)
    print(f"Rows: {len(frame):,}")
    print(
        "Target counts:",
        frame["target_cluster_id"].value_counts().sort_index().to_dict(),
    )
    print(
        f"Missing input images: "
        f"{count_missing_images(frame, dataset_dir):,}"
    )
    print(
        f"Input range: "
        f"{int(frame['input_start_index'].min())} "
        f"to {int(frame['input_start_index'].max())}"
    )
    print(
        f"Target stop range: "
        f"{int(frame['target_stop_index'].min())} "
        f"to {int(frame['target_stop_index'].max())}"
    )


def main():
    config = load_json(CONFIG_PATH)
    paths = resolve_dataset_paths(config)

    train = load_manifest(paths.train_manifest)
    validation = load_manifest(paths.validation_manifest)
    test = load_manifest(paths.test_manifest)

    print("=" * 72)
    print("SUPERVISED DATASET VALIDATION")
    print("=" * 72)
    print(f"Dataset directory: {paths.dataset_dir}")

    print_partition("train", train, paths.dataset_dir)
    print_partition("validation", validation, paths.dataset_dir)
    print_partition("test", test, paths.dataset_dir)

    verify_split_boundaries(train, validation, test)

    missing_total = sum(
        count_missing_images(frame, paths.dataset_dir)
        for frame in (train, validation, test)
    )

    if missing_total > 0:
        raise FileNotFoundError(
            f"Found {missing_total} missing input images."
        )

    print("\nChronological leakage checks passed.")
    print("All required input images were found.")
    print("Dataset is ready for baselines and model training.")


if __name__ == "__main__":
    main()
