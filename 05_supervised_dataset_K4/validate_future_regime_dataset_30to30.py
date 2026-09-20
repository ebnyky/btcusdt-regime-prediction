from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    output_dir = Path(config["output_dir"]).expanduser().resolve()
    manifests = output_dir / "manifests"

    required = {
        "all": manifests / "all_samples.csv",
        "train": manifests / "train_samples.csv",
        "validation": manifests / "validation_samples.csv",
        "test": manifests / "test_samples.csv",
        "summary": manifests / "dataset_summary.json",
    }

    for name, path in required.items():
        print(f"{name}: {path} -> {path.is_file()}")

    frames = {
        name: pd.read_csv(path)
        for name, path in required.items()
        if name in {"all", "train", "validation", "test"}
    }

    for name, frame in frames.items():
        missing_inputs = sum(
            not (output_dir / value).is_file()
            for value in frame["input_image"]
        )
        missing_targets = sum(
            not (output_dir / value).is_file()
            for value in frame["target_image"]
        )

        print(f"\n{name}")
        print(f"  rows: {len(frame):,}")
        print(f"  missing input images: {missing_inputs:,}")
        print(f"  missing target images: {missing_targets:,}")
        print(
            "  cluster counts:",
            frame["target_cluster_id"].value_counts().sort_index().to_dict(),
        )

    train = frames["train"]
    validation = frames["validation"]
    test = frames["test"]

    assert train["target_stop_index"].max() < validation["input_start_index"].min()
    assert validation["target_stop_index"].max() < test["input_start_index"].min()

    print("\nLeakage checks passed.")
    print("Dataset package appears valid.")


if __name__ == "__main__":
    main()
