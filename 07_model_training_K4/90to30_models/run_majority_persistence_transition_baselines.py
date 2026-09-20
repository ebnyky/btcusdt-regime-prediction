"""Evaluate majority, persistence, and most-frequent-transition baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, read_manifests, save_json, validate_schema
from metrics_utils import save_evaluation


def metric_subset(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    }


def evaluate(
    output: Path,
    name: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: int,
    class_names: dict,
) -> dict:
    probabilities = np.eye(classes, dtype=float)[y_pred]
    metrics = save_evaluation(
        output / name,
        split,
        y_true,
        y_pred,
        probabilities,
        class_names,
    )
    return metric_subset(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = load_config(parser.parse_args().config)
    frames = read_manifests(config)
    validate_schema(config, frames, check_all_images=False)

    output = Path(config["output_root"]) / "baselines"
    output.mkdir(parents=True, exist_ok=True)
    label_column = str(config["label_column"])
    classes = int(config["num_classes"])
    class_names = config["class_names"]
    training_labels = frames["train"][label_column].astype(int).to_numpy()
    majority = int(np.bincount(training_labels, minlength=classes).argmax())
    summary: dict[str, object] = {"majority_class": majority}

    for split in ("validation", "test"):
        y_true = frames[split][label_column].astype(int).to_numpy()
        y_pred = np.full(len(y_true), majority, dtype=int)
        summary[f"majority_{split}"] = evaluate(
            output, "majority", split, y_true, y_pred, classes, class_names
        )

    persistence_column = next(
        (
            candidate
            for candidate in (
                "recent_input_cluster_id",
                "last_30_cluster_id",
                "input_last_cluster_id",
            )
            if all(candidate in frames[split].columns for split in frames)
        ),
        None,
    )
    if persistence_column is None:
        summary["persistence_status"] = (
            "not run: generate manifests_with_persistence first"
        )
    else:
        summary["persistence_source_column"] = persistence_column
        for split in ("validation", "test"):
            usable = frames[split].dropna(subset=[persistence_column, label_column])
            y_true = usable[label_column].astype(int).to_numpy()
            y_pred = usable[persistence_column].astype(int).to_numpy()
            summary[f"persistence_{split}"] = evaluate(
                output, "persistence", split, y_true, y_pred, classes, class_names
            )

        transition_train = frames["train"].dropna(
            subset=[persistence_column, label_column]
        )
        transition_table = pd.crosstab(
            transition_train[persistence_column].astype(int),
            transition_train[label_column].astype(int),
        ).reindex(index=range(classes), columns=range(classes), fill_value=0)
        transition_map = {
            state: (
                int(transition_table.loc[state].to_numpy().argmax())
                if int(transition_table.loc[state].sum()) > 0
                else majority
            )
            for state in range(classes)
        }
        summary["most_frequent_transition_map"] = {
            str(key): value for key, value in transition_map.items()
        }
        for split in ("validation", "test"):
            usable = frames[split].dropna(subset=[persistence_column, label_column])
            states = usable[persistence_column].astype(int).to_numpy()
            y_true = usable[label_column].astype(int).to_numpy()
            y_pred = np.asarray(
                [transition_map.get(int(state), majority) for state in states],
                dtype=int,
            )
            summary[f"most_frequent_transition_{split}"] = evaluate(
                output,
                "most_frequent_transition",
                split,
                y_true,
                y_pred,
                classes,
                class_names,
            )

    save_json(output / "baseline_summary.json", summary)
    print(f"Majority class: {majority}")
    print("Baseline evaluation complete.")


if __name__ == "__main__":
    main()
