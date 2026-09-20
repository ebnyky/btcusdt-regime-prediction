from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


def get_metrics(report, split):
    metrics = report[split]
    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    output_dir = Path(config["output_dir"]).expanduser().resolve()
    rows = []

    baseline_path = (
        output_dir
        / "baselines"
        / "baseline_results.json"
    )

    if baseline_path.is_file():
        with baseline_path.open("r", encoding="utf-8") as file:
            baselines = json.load(file)

        for name, report in baselines.items():
            for split in ("validation", "test"):
                rows.append(
                    {
                        "experiment": name,
                        "type": "baseline",
                        "split": split,
                        **get_metrics(report, split),
                    }
                )

    for mode in (
        "frozen_backbone",
        "unfreeze_layer4",
        "full_finetune",
    ):
        report_path = output_dir / mode / "training_report.json"

        if not report_path.is_file():
            continue

        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)

        for split in ("validation", "test"):
            rows.append(
                {
                    "experiment": mode,
                    "type": "resnet18",
                    "split": split,
                    **get_metrics(report, split),
                }
            )

    if not rows:
        raise FileNotFoundError(
            "No baseline or training result files were found."
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["split", "macro_f1"],
        ascending=[True, False],
    )

    destination = output_dir / "experiment_comparison.csv"
    frame.to_csv(destination, index=False)

    print(frame.to_string(index=False))
    print(f"\nSaved comparison to: {destination}")


if __name__ == "__main__":
    main()
