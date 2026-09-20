from pathlib import Path

import pandas as pd

from kline_dataset_generator import split_rows


def test_historical_split_boundary_and_embargo() -> None:
    rows = pd.DataFrame(
        {
            "sample_name": [f"sample_{start}_{start + 29}" for start in range(24100)],
            "start_index": list(range(24100)),
            "stop_index": [start + 29 for start in range(24100)],
            "split": ["unassigned"] * 24100,
        }
    )
    config = {
        "split": {
            "validation_start_index": 24047,
            "train_fraction": 0.9,
            "embargo_candles": 30,
        }
    }
    all_rows, train, validation, details = split_rows(rows, config)

    assert details["train_last_start_index"] == 23987
    assert details["train_last_stop_index"] == 24016
    assert details["validation_first_start_index"] == 24047
    assert len(all_rows[all_rows["split"] == "purged"]) == 59
    assert train.iloc[-1]["sample_name"] == "sample_23987_24016"
    assert validation.iloc[0]["sample_name"] == "sample_24047_24076"
