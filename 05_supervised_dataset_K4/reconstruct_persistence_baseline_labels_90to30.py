"""Add the missing recent-input regime needed by the persistence baseline.

The historical 90-to-30 generator labels only the future 30-candle window.
For strides that divide 30 (5, 10, 15 and 30), the final 30 candles of a
90-candle input already occur as the target window of an earlier manifest row.
This script joins that existing label without refitting PCA/K-Means or reading
test outcomes. It writes a separate manifest directory by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def add_labels(dataset_root: Path, output_name: str) -> None:
    source = dataset_root / "manifests"
    destination = dataset_root / output_name
    destination.mkdir(parents=True, exist_ok=True)
    all_rows = pd.read_csv(source / "all_samples.csv")
    required = {
        "input_stop_index",
        "target_start_index",
        "target_stop_index",
        "target_cluster_id",
    }
    missing = required.difference(all_rows.columns)
    if missing:
        raise ValueError(f"all_samples.csv is missing columns: {sorted(missing)}")

    lookup = {
        (int(row.target_start_index), int(row.target_stop_index)): int(row.target_cluster_id)
        for row in all_rows.itertuples(index=False)
    }

    summary: dict[str, object] = {"output_manifest_dir": output_name, "splits": {}}
    for filename in (
        "all_samples.csv",
        "train_samples.csv",
        "validation_samples.csv",
        "test_samples.csv",
    ):
        frame = pd.read_csv(source / filename)
        labels = []
        for row in frame.itertuples(index=False):
            recent_start = int(row.input_stop_index) - 29
            key = (recent_start, int(row.input_stop_index))
            labels.append(lookup.get(key, pd.NA))
        frame["recent_input_cluster_id"] = pd.array(labels, dtype="Int64")
        frame.to_csv(destination / filename, index=False)
        summary["splits"][filename] = {
            "rows": len(frame),
            "labels_recovered": int(frame["recent_input_cluster_id"].notna().sum()),
            "labels_missing": int(frame["recent_input_cluster_id"].isna().sum()),
        }

    (destination / "persistence_reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-name", default="manifests_with_persistence")
    arguments = parser.parse_args()
    add_labels(arguments.dataset_root.expanduser().resolve(), arguments.output_name)


if __name__ == "__main__":
    main()
