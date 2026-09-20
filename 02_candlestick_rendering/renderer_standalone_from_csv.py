#!/usr/bin/env python3
"""Generate the experiment's 30-candle K-line image dataset from one CSV."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-kline-cache"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# USER SETTINGS — EDIT ONLY THESE PATHS
# ============================================================

CSV_FILE_PATH = Path(
    r"C:\Users\m0zim\OneDrive\Desktop\BTCUSDT_1h_2025-07-07_to_2026-06-31.csv"
)

OUTPUT_FOLDER_PATH = Path(
    r"C:\Users\m0zim\OneDrive\Desktop\test_30klines"
)


WINDOW_SIZE = 30
EXPECTED_INTERVAL = pd.Timedelta(hours=1)
IMAGE_SIZE = 360
IMAGE_DPI = 100
REQUIRED_COLUMNS = ("open_time", "open", "high", "low", "close")
TIMESTAMP_ALIASES = ("open_time", "timestamp", "datetime", "date", "time")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    raise ValueError(
        f"Missing required column. Expected one of {names}; "
        f"found {frame.columns.tolist()}."
    )


def load_and_validate_csv(csv_path: Path) -> tuple[pd.DataFrame, bool]:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("The CSV file is empty.")

    timestamp_column = find_column(frame, TIMESTAMP_ALIASES)
    selected = pd.DataFrame({
        "open_time": frame[timestamp_column],
        "open": frame[find_column(frame, ("open",))],
        "high": frame[find_column(frame, ("high",))],
        "low": frame[find_column(frame, ("low",))],
        "close": frame[find_column(frame, ("close",))],
    })

    selected["open_time"] = pd.to_datetime(
        selected["open_time"], errors="coerce", utc=True
    )
    for column in REQUIRED_COLUMNS[1:]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    bad_rows = selected[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    if bad_rows.any():
        examples = frame.index[bad_rows].tolist()[:10]
        raise ValueError(
            f"Found {int(bad_rows.sum())} rows with missing, unparseable, or "
            f"non-numeric required values. Example CSV row indices: {examples}"
        )

    duplicate_times = selected["open_time"].duplicated(keep=False)
    if duplicate_times.any():
        examples = (
            selected.loc[duplicate_times, "open_time"]
            .astype(str)
            .drop_duplicates()
            .tolist()[:10]
        )
        raise ValueError(
            f"Found {int(duplicate_times.sum())} rows with duplicate timestamps. "
            f"Examples: {examples}"
        )

    was_sorted = selected["open_time"].is_monotonic_increasing
    selected = selected.sort_values("open_time").reset_index(drop=True)

    prices = selected[["open", "high", "low", "close"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(prices).all():
        raise ValueError("All OHLC values must be finite numbers.")
    if not (prices > 0).all():
        raise ValueError("All OHLC values must be strictly positive.")

    row_max = selected[["open", "close", "low"]].max(axis=1)
    row_min = selected[["open", "close", "high"]].min(axis=1)
    invalid_ohlc = (selected["high"] < row_max) | (selected["low"] > row_min)
    if invalid_ohlc.any():
        examples = selected.index[invalid_ohlc].tolist()[:10]
        raise ValueError(
            f"Found {int(invalid_ohlc.sum())} rows that violate OHLC high/low "
            f"relationships. Example sorted row indices: {examples}"
        )

    if len(selected) < WINDOW_SIZE:
        raise ValueError(
            f"At least {WINDOW_SIZE} rows are required; found {len(selected)}."
        )

    return selected, was_sorted


def render_kline(window: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(
        figsize=(IMAGE_SIZE / IMAGE_DPI, IMAGE_SIZE / IMAGE_DPI),
        dpi=IMAGE_DPI,
    )

    statistics = []
    colours = []
    for row in window.itertuples(index=False):
        statistics.append({
            "q1": min(row.open, row.close),
            "q3": max(row.open, row.close),
            "whislo": min(row.low, row.high),
            "whishi": max(row.low, row.high),
            "med": (row.open + row.close) / 2.0,
        })
        colours.append("green" if row.open <= row.close else "red")

    artists = axis.bxp(
        bxpstats=statistics,
        showcaps=False,
        patch_artist=True,
        showfliers=False,
    )
    for box, colour in zip(artists["boxes"], colours):
        box.set_facecolor(colour)
        box.set_edgecolor(colour)
    for number, whisker in enumerate(artists["whiskers"]):
        whisker.set_color(colours[number // 2])
    for median in artists["medians"]:
        median.set_visible(False)

    axis.axis("off")
    figure.savefig(destination, dpi=IMAGE_DPI, pad_inches=0, format="jpg")
    plt.close(figure)


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError(f"expected JPEG, found {image.format}")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"expected {IMAGE_SIZE}x{IMAGE_SIZE}, found {image.size}"
            )
        image.verify()


def build_dataset(csv_path: Path, output_dir: Path, stride: int) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Choose a new --output-dir so an earlier dataset is not overwritten."
        )

    images_dir = output_dir / "images"
    manifests_dir = output_dir / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    frame, was_sorted = load_and_validate_csv(csv_path)
    source_hash = sha256_file(csv_path)
    total_candidates = math.floor((len(frame) - WINDOW_SIZE) / stride) + 1
    records: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    hashes: dict[str, list[str]] = {}

    for number, start_index in enumerate(
        range(0, len(frame) - WINDOW_SIZE + 1, stride), start=1
    ):
        stop_index = start_index + WINDOW_SIZE - 1
        sample_name = f"sample_{start_index}_{stop_index}"
        window = frame.iloc[start_index : stop_index + 1]
        differences = window["open_time"].diff().iloc[1:]

        if not differences.eq(EXPECTED_INTERVAL).all():
            bad_positions = differences.index[~differences.eq(EXPECTED_INTERVAL)]
            first_bad = int(bad_positions[0])
            skipped.append({
                "sample_name": sample_name,
                "start_index": start_index,
                "stop_index": stop_index,
                "start_time": window.iloc[0]["open_time"].isoformat(),
                "stop_time": window.iloc[-1]["open_time"].isoformat(),
                "reason": "timestamp_gap_inside_window",
                "gap_after_time": frame.iloc[first_bad - 1]["open_time"].isoformat(),
                "next_time": frame.iloc[first_bad]["open_time"].isoformat(),
            })
            continue

        relative_image = Path("images") / f"{sample_name}.jpg"
        image_path = output_dir / relative_image
        try:
            render_kline(window, image_path)
            validate_image(image_path)
        except Exception as error:
            image_path.unlink(missing_ok=True)
            skipped.append({
                "sample_name": sample_name,
                "start_index": start_index,
                "stop_index": stop_index,
                "start_time": window.iloc[0]["open_time"].isoformat(),
                "stop_time": window.iloc[-1]["open_time"].isoformat(),
                "reason": f"image_generation_or_validation_error: {error}",
            })
            continue

        image_hash = sha256_file(image_path)
        hashes.setdefault(image_hash, []).append(relative_image.as_posix())
        records.append({
            "sample_name": sample_name,
            "start_index": start_index,
            "stop_index": stop_index,
            "window_length": WINDOW_SIZE,
            "start_time": window.iloc[0]["open_time"].isoformat(),
            "stop_time": window.iloc[-1]["open_time"].isoformat(),
            "source_csv": csv_path.name,
            "source_rows": f"{start_index}:{stop_index}",
            "exported_image": relative_image.as_posix(),
            "split": "test",
            "sha256": image_hash,
        })

        if number % 500 == 0 or number == total_candidates:
            print(
                f"Processed {number:,}/{total_candidates:,} | "
                f"generated {len(records):,} | skipped {len(skipped):,}",
                flush=True,
            )

    if not records:
        raise RuntimeError("No valid continuous 30-candle windows were generated.")

    manifest_path = manifests_dir / "all_samples.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    duplicate_groups = [
        {"sha256": digest, "images": paths}
        for digest, paths in hashes.items()
        if len(paths) > 1
    ]
    duplicate_report = {
        "image_count": len(records),
        "unique_hash_count": len(hashes),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
    }
    duplicate_path = manifests_dir / "duplicate_images.json"
    duplicate_path.write_text(
        json.dumps(duplicate_report, indent=2), encoding="utf-8"
    )

    skipped_path = output_dir / "skipped_images.txt"
    with skipped_path.open("w", encoding="utf-8") as file:
        if not skipped:
            file.write("No images were skipped.\n")
        else:
            for item in skipped:
                file.write(json.dumps(item, sort_keys=True) + "\n")

    transitions = frame["open_time"].diff().iloc[1:]
    gap_count = int((transitions != EXPECTED_INTERVAL).sum())
    summary = {
        "stage": "test_kline_dataset_generation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path.resolve()),
        "source_csv_sha256": source_hash,
        "source_rows": len(frame),
        "source_was_already_chronological": bool(was_sorted),
        "source_start_utc": frame.iloc[0]["open_time"].isoformat(),
        "source_end_utc": frame.iloc[-1]["open_time"].isoformat(),
        "timestamp_gaps": gap_count,
        "window_size": WINDOW_SIZE,
        "stride": stride,
        "candidate_windows": total_candidates,
        "generated_images": len(records),
        "skipped_images": len(skipped),
        "renderer": {
            "method": "matplotlib.axes.Axes.bxp",
            "image_size_px": [IMAGE_SIZE, IMAGE_SIZE],
            "dpi": IMAGE_DPI,
            "bullish_and_doji_colour": "green",
            "bearish_colour": "red",
            "axes_visible": False,
            "medians_visible": False,
            "caps_visible": False,
            "fliers_visible": False,
            "save_padding_inches": 0,
            "scaling": "local_per_window",
        },
        "outputs": {
            "images": "images/",
            "manifest": "manifests/all_samples.csv",
            "duplicates": "manifests/duplicate_images.json",
            "skipped": "skipped_images.txt",
        },
    }
    summary_path = manifests_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDATASET GENERATION COMPLETED")
    print(f"Source rows: {len(frame):,}")
    print(f"Generated images: {len(records):,}")
    print(f"Skipped images: {len(skipped):,}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    csv_path = CSV_FILE_PATH.expanduser().resolve()
    output_dir = OUTPUT_FOLDER_PATH.expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(
            "CSV_FILE_PATH does not point to an existing file:\n"
            f"{csv_path}\n\n"
            "Edit CSV_FILE_PATH in the USER SETTINGS section."
        )

    if csv_path == output_dir:
        raise ValueError(
            "OUTPUT_FOLDER_PATH must be a folder, not the CSV file itself."
        )

    build_dataset(csv_path, output_dir, stride=1)
