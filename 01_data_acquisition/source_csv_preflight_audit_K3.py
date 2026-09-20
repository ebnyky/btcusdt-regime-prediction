"""Audit the exact BTCUSDT CSV before any image rendering begins."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED = ["open_time", "open", "high", "low", "close"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contiguous_ranges(values: list[int]) -> list[dict[str, int]]:
    if not values:
        return []
    output: list[dict[str, int]] = []
    first = last = values[0]
    for value in values[1:]:
        if value == last + 1:
            last = value
            continue
        output.append({"first": first, "last": last, "count": last - first + 1})
        first = last = value
    output.append({"first": first, "last": last, "count": last - first + 1})
    return output


def audit(csv_path: Path, window_size: int = 30) -> dict:
    frame = pd.read_csv(csv_path)
    if frame.columns.tolist() != REQUIRED:
        raise ValueError(f"Expected columns {REQUIRED}; found {frame.columns.tolist()}")

    timestamps = pd.to_datetime(frame["open_time"], errors="coerce", utc=True)
    numeric = frame[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid = (
        (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1))
        | (numeric <= 0).any(axis=1)
        | numeric.isna().any(axis=1)
    )
    full_index = pd.date_range(timestamps.min(), timestamps.max(), freq="h", tz="UTC")
    missing = full_index.difference(pd.DatetimeIndex(timestamps.dropna()))
    discontinuity_positions = [
        int(index)
        for index, delta in timestamps.diff().items()
        if pd.notna(delta) and delta != pd.Timedelta(hours=1)
    ]

    candidate_count = max(0, len(frame) - window_size + 1)
    skipped_starts: list[int] = []
    for start in range(candidate_count):
        window = timestamps.iloc[start : start + window_size]
        if not (window.diff().iloc[1:] == pd.Timedelta(hours=1)).all():
            skipped_starts.append(start)

    return {
        "source_csv": str(csv_path.resolve()),
        "sha256": sha256_file(csv_path),
        "columns": frame.columns.tolist(),
        "rows": int(len(frame)),
        "date_start": timestamps.min().isoformat(),
        "date_end": timestamps.max().isoformat(),
        "timestamps_unparseable": int(timestamps.isna().sum()),
        "timestamps_monotonic_increasing": bool(timestamps.is_monotonic_increasing),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "missing_hour_count": int(len(missing)),
        "missing_hours": [value.isoformat() for value in missing],
        "gap_event_count": len(discontinuity_positions),
        "gap_row_positions": discontinuity_positions,
        "invalid_ohlc_rows": int(invalid.sum()),
        "window_size": window_size,
        "candidate_windows": candidate_count,
        "valid_windows": candidate_count - len(skipped_starts),
        "skipped_windows": len(skipped_starts),
        "skipped_start_ranges": contiguous_ranges(skipped_starts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    report = audit(args.csv)
    if args.expected_sha256 and report["sha256"] != args.expected_sha256:
        raise RuntimeError(
            f"CSV hash mismatch: expected {args.expected_sha256}, found {report['sha256']}"
        )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
