from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if int(config.get("window_size", 0)) != 90:
        raise AssertionError("config.json window_size must be 90.")
    if int(config.get("forecast_horizon", 0)) != 30:
        raise AssertionError("config.json forecast_horizon must be 30.")

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
        if not path.is_file():
            raise FileNotFoundError(f"Missing required output: {path}")

    frames = {
        name: pd.read_csv(path)
        for name, path in required.items()
        if name in {"all", "train", "validation", "test"}
    }

    for name, frame in frames.items():
        input_lengths = frame["input_stop_index"] - frame["input_start_index"] + 1
        target_lengths = frame["target_stop_index"] - frame["target_start_index"] + 1

        assert bool((input_lengths == 90).all()), f"{name}: found a non-90-candle input."
        assert bool((target_lengths == 30).all()), f"{name}: found a non-30-candle target."
        assert bool(
            (frame["target_start_index"] == frame["input_stop_index"] + 1).all()
        ), f"{name}: an input overlaps or is separated from its target."

        assert bool(
            frame["input_image"].str.startswith("images/inputs_90/").all()
        ), f"{name}: input_image contains a path outside images/inputs_90/."
        assert bool(
            frame["target_image"].str.startswith("images/targets_30/").all()
        ), f"{name}: target_image contains a path outside images/targets_30/."

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
        print(f"  missing 90-candle input images: {missing_inputs:,}")
        print(f"  missing 30-candle target audit images: {missing_targets:,}")
        print(
            "  cluster counts:",
            frame["target_cluster_id"].value_counts().sort_index().to_dict(),
        )

        if missing_inputs or missing_targets:
            raise FileNotFoundError(f"{name}: one or more referenced images are missing.")

    train = frames["train"]
    validation = frames["validation"]
    test = frames["test"]

    assert train["target_stop_index"].max() < validation["input_start_index"].min()
    assert validation["target_stop_index"].max() < test["input_start_index"].min()

    print("\nAll checks passed.")
    print("Confirmed schema: 90-candle BXP input -> next non-overlapping 30-candle cluster ID.")
    print("Confirmed folders: images/inputs_90 and images/targets_30.")
    print("Confirmed: chronological train/validation/test boundaries share no raw candles.")


if __name__ == "__main__":
    main()
