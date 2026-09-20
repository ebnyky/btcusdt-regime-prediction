#!/usr/bin/env python3
"""Build the leakage-safe K=3 current-image -> future-regime dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-k3-forecast-cache"),
)

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


# ============================================================================
# USER SETTINGS — EDIT ONLY THESE FIVE PATHS
# ============================================================================

TRAIN_VALID_CSV_PATH = Path(
    r"C:\Users\m0zim\OneDrive\Desktop\k3_forecasting_dataset_package\datasets\BTCUSDT_1h_2020-01-01_to_2025-07-06.csv"
)

TEST_CSV_PATH = Path(
    r"C:\Users\m0zim\OneDrive\Desktop\k3_forecasting_dataset_package\datasets\BTCUSDT_1h_2025-07-07_to_2026-06-31.csv"
)

PCA_MODEL_PATH = Path(
    r"C:\Users\m0zim\OneDrive\Desktop\k3_forecasting_dataset_package\models\pca.joblib"
)

KMEANS_MODEL_PATH = Path(
    r"C:\Users\m0zim\OneDrive\Desktop\k3_forecasting_dataset_package\models\kmeans.joblib"
)

OUTPUT_FOLDER_PATH = Path(
    r"C:\Users\m0zim\OneDrive\Desktop\BTCUSDT_k3_forecasting_dataset"
)


# Frozen experiment contract. Do not change these values for this experiment.
TRAIN_START_UTC = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
TRAIN_END_UTC = pd.Timestamp("2024-12-17 13:00:00", tz="UTC")
VALIDATION_START_UTC = pd.Timestamp("2024-12-17 14:00:00", tz="UTC")
VALIDATION_END_UTC = pd.Timestamp("2025-07-06 23:00:00", tz="UTC")
TEST_START_UTC = pd.Timestamp("2025-07-07 00:00:00", tz="UTC")
TEST_END_UTC = pd.Timestamp("2026-07-31 23:00:00", tz="UTC")

WINDOW_SIZE = 30
FORECAST_HORIZON = 30
EXPECTED_INTERVAL = pd.Timedelta(hours=1)
STRIDE = 1
IMAGE_SIZE = 360
IMAGE_DPI = 100
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 0
CPU_THREADS = 4
DEVICE = "auto"
OVERWRITE_OUTPUT = False

REGIME_NAMES = {
    0: "Bullish",
    1: "Sideways/non-directional",
    2: "Bearish",
}

TIMESTAMP_ALIASES = ("open_time", "timestamp", "datetime", "date", "time")
PRICE_COLUMNS = ("open", "high", "low", "close")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    images_train_valid: Path
    images_test: Path
    manifests: Path
    reports: Path


class KlineImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform: Any) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as error:
            raise RuntimeError(f"Could not load image: {path}") from error
        return tensor, index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    raise ValueError(
        f"Missing required column. Expected one of {aliases}; "
        f"found {frame.columns.tolist()}."
    )


def load_and_validate_csv(path: Path, source_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{source_id} CSV not found: {path}")

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"{source_id} CSV is empty: {path}")

    timestamp_col = find_column(raw, TIMESTAMP_ALIASES)
    resolved = {name: find_column(raw, (name,)) for name in PRICE_COLUMNS}

    frame = pd.DataFrame({
        "csv_row_index": np.arange(len(raw), dtype=np.int64),
        "open_time": raw[timestamp_col],
        **{name: raw[column] for name, column in resolved.items()},
    })

    frame["open_time"] = pd.to_datetime(frame["open_time"], errors="coerce", utc=True)
    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_required = frame[["open_time", *PRICE_COLUMNS]].isna().any(axis=1)
    if invalid_required.any():
        examples = frame.loc[invalid_required, "csv_row_index"].head(10).tolist()
        raise ValueError(
            f"{source_id} contains {int(invalid_required.sum())} rows with missing, "
            f"unparseable, or non-numeric required values. Example CSV rows: {examples}"
        )

    duplicate_times = frame["open_time"].duplicated(keep=False)
    if duplicate_times.any():
        examples = (
            frame.loc[duplicate_times, "open_time"].astype(str).drop_duplicates().head(10).tolist()
        )
        raise ValueError(
            f"{source_id} contains duplicate hourly timestamps. Examples: {examples}"
        )

    was_sorted = bool(frame["open_time"].is_monotonic_increasing)
    frame = frame.sort_values("open_time").reset_index(drop=True)
    frame["source_index"] = np.arange(len(frame), dtype=np.int64)

    prices = frame[list(PRICE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all():
        raise ValueError(f"{source_id} contains non-finite OHLC values.")
    if not (prices > 0).all():
        raise ValueError(f"{source_id} contains non-positive OHLC values.")

    invalid_ohlc = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    if invalid_ohlc.any():
        examples = frame.loc[invalid_ohlc, "csv_row_index"].head(10).tolist()
        raise ValueError(
            f"{source_id} contains {int(invalid_ohlc.sum())} invalid OHLC rows. "
            f"Example CSV rows: {examples}"
        )

    gaps = frame["open_time"].diff().iloc[1:] != EXPECTED_INTERVAL
    audit = {
        "source_id": source_id,
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": int(len(frame)),
        "was_already_chronological": was_sorted,
        "start_utc": iso(frame.iloc[0]["open_time"]),
        "end_utc": iso(frame.iloc[-1]["open_time"]),
        "irregular_transitions": int(gaps.sum()),
    }
    return frame, audit


def select_exact_period(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    source_id: str,
) -> pd.DataFrame:
    timestamps = set(frame["open_time"].tolist())
    if start not in timestamps:
        raise ValueError(f"{source_id} is missing required start candle {iso(start)}.")
    if end not in timestamps:
        raise ValueError(f"{source_id} is missing required end candle {iso(end)}.")

    selected = frame.loc[
        (frame["open_time"] >= start) & (frame["open_time"] <= end)
    ].copy()
    selected = selected.reset_index(drop=True)
    selected["period_index"] = np.arange(len(selected), dtype=np.int64)
    if len(selected) < WINDOW_SIZE:
        raise ValueError(f"{source_id} has fewer than {WINDOW_SIZE} rows in its fixed period.")
    return selected


def validate_csv_boundaries(
    train_valid: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    """Verify the two source CSVs form the exact frozen chronological sequence."""
    train_valid_start = pd.Timestamp(train_valid.iloc[0]["open_time"])
    train_valid_end = pd.Timestamp(train_valid.iloc[-1]["open_time"])
    test_start = pd.Timestamp(test.iloc[0]["open_time"])
    test_end = pd.Timestamp(test.iloc[-1]["open_time"])

    expected_observed = (
        ("train/validation CSV start", TRAIN_START_UTC, train_valid_start),
        ("train/validation CSV end", VALIDATION_END_UTC, train_valid_end),
        ("test CSV start", TEST_START_UTC, test_start),
        ("test CSV end", TEST_END_UTC, test_end),
    )
    boundary_errors = [
        f"{name} must be {iso(expected)}, but the file contains {iso(observed)}."
        for name, expected, observed in expected_observed
        if observed != expected
    ]

    train_times = pd.Index(train_valid["open_time"])
    test_times = pd.Index(test["open_time"])
    overlap = train_times.intersection(test_times)
    if len(overlap):
        examples = [iso(value) for value in overlap[:10]]
        boundary_errors.append(
            f"The two CSVs overlap at {len(overlap):,} timestamp(s). "
            f"Examples: {examples}."
        )

    cross_csv_interval = test_start - train_valid_end
    if cross_csv_interval != EXPECTED_INTERVAL:
        boundary_errors.append(
            "The first test candle must occur exactly one hour after the final "
            "train/validation candle; observed interval: "
            f"{cross_csv_interval}."
        )

    train_boundary = train_valid.loc[
        train_valid["open_time"].isin([TRAIN_END_UTC, VALIDATION_START_UTC]),
        "open_time",
    ]
    missing_internal = [
        iso(value)
        for value in (TRAIN_END_UTC, VALIDATION_START_UTC)
        if value not in set(train_boundary.tolist())
    ]
    if missing_internal:
        boundary_errors.append(
            "The train/validation CSV is missing frozen split-boundary candle(s): "
            f"{missing_internal}."
        )

    if boundary_errors:
        raise ValueError(
            "CSV boundary validation failed:\n- " + "\n- ".join(boundary_errors)
        )

    return {
        "passed": True,
        "train_validation_csv_start_utc": iso(train_valid_start),
        "train_validation_csv_end_utc": iso(train_valid_end),
        "test_csv_start_utc": iso(test_start),
        "test_csv_end_utc": iso(test_end),
        "cross_csv_interval": str(cross_csv_interval),
        "cross_csv_interval_hours": float(
            cross_csv_interval / pd.Timedelta(hours=1)
        ),
        "overlapping_timestamp_count": int(len(overlap)),
        "train_validation_split_end_utc": iso(TRAIN_END_UTC),
        "train_validation_split_start_utc": iso(VALIDATION_START_UTC),
        "forecast_pairs_may_cross_source_csv_boundary": False,
    }


def prepare_output(path: Path) -> OutputPaths:
    if path.exists() and any(path.iterdir()):
        if OVERWRITE_OUTPUT:
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"Output folder is not empty: {path}\n"
                "Choose a new/empty folder. Existing results were not changed."
            )

    result = OutputPaths(
        root=path,
        images_train_valid=path / "images" / "train_validation",
        images_test=path / "images" / "test",
        manifests=path / "manifests",
        reports=path / "reports",
    )
    for folder in (
        result.root,
        result.images_train_valid,
        result.images_test,
        result.manifests,
        result.reports,
    ):
        folder.mkdir(parents=True, exist_ok=True)
    return result


def render_kline(window: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(
        figsize=(IMAGE_SIZE / IMAGE_DPI, IMAGE_SIZE / IMAGE_DPI),
        dpi=IMAGE_DPI,
    )

    statistics: list[dict[str, float]] = []
    colours: list[str] = []
    for row in window.itertuples(index=False):
        statistics.append({
            "q1": min(float(row.open), float(row.close)),
            "q3": max(float(row.open), float(row.close)),
            "whislo": min(float(row.low), float(row.high)),
            "whishi": max(float(row.low), float(row.high)),
            "med": (float(row.open) + float(row.close)) / 2.0,
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
    for index, whisker in enumerate(artists["whiskers"]):
        whisker.set_color(colours[index // 2])
    for median in artists["medians"]:
        median.set_visible(False)

    axis.axis("off")
    figure.savefig(destination, dpi=IMAGE_DPI, pad_inches=0, format="jpg")
    plt.close(figure)


def validate_image(path: Path) -> None:
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError(f"Expected JPEG, found {image.format}.")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, found {image.size}."
            )
        image.verify()


def generate_windows(
    frame: pd.DataFrame,
    source_id: str,
    image_dir: Path,
    output_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    maximum_start = len(frame) - WINDOW_SIZE
    starts = range(0, maximum_start + 1, STRIDE)
    total = (maximum_start // STRIDE) + 1

    print(f"\nGenerating {source_id} 30-candle images: {total:,} candidates")
    for number, start in enumerate(starts, start=1):
        stop = start + WINDOW_SIZE - 1
        window = frame.iloc[start : stop + 1]
        sample_name = f"sample_{start}_{stop}"
        base = {
            "source_id": source_id,
            "sample_name": sample_name,
            "start_index": int(start),
            "stop_index": int(stop),
            "source_start_index": int(window.iloc[0]["source_index"]),
            "source_stop_index": int(window.iloc[-1]["source_index"]),
            "csv_start_row": int(window.iloc[0]["csv_row_index"]),
            "csv_stop_row": int(window.iloc[-1]["csv_row_index"]),
            "start_time": iso(window.iloc[0]["open_time"]),
            "stop_time": iso(window.iloc[-1]["open_time"]),
        }

        differences = window["open_time"].diff().iloc[1:]
        if not differences.eq(EXPECTED_INTERVAL).all():
            bad_index = int(differences.index[~differences.eq(EXPECTED_INTERVAL)][0])
            skipped.append({
                **base,
                "reason": "timestamp_gap_inside_30_candle_window",
                "gap_after_time": iso(frame.iloc[bad_index - 1]["open_time"]),
                "next_time": iso(frame.iloc[bad_index]["open_time"]),
            })
            continue

        destination = image_dir / f"{sample_name}.jpg"
        try:
            render_kline(window, destination)
            validate_image(destination)
        except Exception as error:
            destination.unlink(missing_ok=True)
            skipped.append({
                **base,
                "reason": f"image_generation_or_validation_error: {error}",
            })
            continue

        records.append({
            **base,
            "window_length": WINDOW_SIZE,
            "image_path": destination.relative_to(output_root).as_posix(),
            "image_sha256": sha256_file(destination),
        })

        if number % 500 == 0 or number == total:
            print(
                f"  processed={number:,}/{total:,} | "
                f"generated={len(records):,} | skipped={len(skipped):,}",
                flush=True,
            )

    if not records:
        raise RuntimeError(f"No valid images were generated for {source_id}.")
    return pd.DataFrame(records), skipped


def resolve_device() -> torch.device:
    requested = DEVICE.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def load_frozen_models() -> tuple[Any, Any, dict[str, Any]]:
    pca_path = PCA_MODEL_PATH.expanduser().resolve()
    kmeans_path = KMEANS_MODEL_PATH.expanduser().resolve()
    if not pca_path.is_file():
        raise FileNotFoundError(f"PCA model not found: {pca_path}")
    if not kmeans_path.is_file():
        raise FileNotFoundError(f"K-Means model not found: {kmeans_path}")

    pca = joblib.load(pca_path)
    kmeans = joblib.load(kmeans_path)

    pca_components = int(getattr(pca, "n_components_", -1))
    pca_input = int(getattr(pca, "n_features_in_", -1))
    cluster_count = int(getattr(kmeans, "n_clusters", -1))
    centers = getattr(kmeans, "cluster_centers_", None)
    center_shape = tuple(centers.shape) if centers is not None else None

    if pca_components != 120 or pca_input != 512:
        raise ValueError(
            "Wrong PCA model. Required n_components_=120 and n_features_in_=512; "
            f"found {pca_components} and {pca_input}."
        )
    if cluster_count != 3 or center_shape != (3, 120):
        raise ValueError(
            "Wrong K-Means model. Required 3 clusters with centers shaped (3, 120); "
            f"found n_clusters={cluster_count}, centers={center_shape}."
        )

    audit = {
        "pca_path": str(pca_path),
        "pca_sha256": sha256_file(pca_path),
        "pca_components": pca_components,
        "pca_input_features": pca_input,
        "kmeans_path": str(kmeans_path),
        "kmeans_sha256": sha256_file(kmeans_path),
        "kmeans_clusters": cluster_count,
        "kmeans_centroid_shape": list(center_shape),
    }
    return pca, kmeans, audit


def build_encoder(device: torch.device) -> tuple[nn.Module, Any]:
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, transform


def label_windows(
    windows: pd.DataFrame,
    output_root: Path,
    pca: Any,
    kmeans: Any,
    device: torch.device,
) -> pd.DataFrame:
    encoder, transform = build_encoder(device)
    relative_paths = windows["image_path"].tolist()
    absolute_paths = [output_root / value for value in relative_paths]
    dataset = KlineImageDataset(absolute_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    results: list[dict[str, Any] | None] = [None] * len(dataset)
    processed = 0
    print(f"\nAssigning frozen K=3 labels to {len(dataset):,} images...")
    with torch.inference_mode():
        for images, indices in loader:
            embeddings = encoder(images.to(device, non_blocking=True))
            embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
            reduced = pca.transform(embeddings_np)
            labels = kmeans.predict(reduced)
            distances = kmeans.transform(reduced)

            for local_index, dataset_index in enumerate(indices.tolist()):
                row_distances = distances[local_index]
                order = np.argsort(row_distances)
                nearest = float(row_distances[order[0]])
                second = float(row_distances[order[1]])
                label = int(labels[local_index])
                results[dataset_index] = {
                    "cluster_id": label,
                    "regime_name": REGIME_NAMES[label],
                    "distance_to_centroid": float(row_distances[label]),
                    "second_nearest_distance": second,
                    "distance_margin": second - nearest,
                    "relative_distance_margin": (second - nearest) / max(second, 1e-12),
                    "nearest_cluster_order": json.dumps([int(value) for value in order]),
                    "distance_cluster_0": float(row_distances[0]),
                    "distance_cluster_1": float(row_distances[1]),
                    "distance_cluster_2": float(row_distances[2]),
                }

            processed += len(indices)
            if processed % 800 == 0 or processed == len(dataset):
                print(f"  labelled={processed:,}/{len(dataset):,}", flush=True)

    if any(item is None for item in results):
        raise RuntimeError("At least one image did not receive a frozen cluster label.")
    return pd.concat(
        [windows.reset_index(drop=True), pd.DataFrame(results)],
        axis=1,
    )


def prefixed(record: pd.Series, prefix: str) -> dict[str, Any]:
    names = (
        "source_id", "sample_name", "start_index", "stop_index",
        "source_start_index", "source_stop_index", "csv_start_row", "csv_stop_row",
        "start_time", "stop_time", "window_length", "image_path", "image_sha256",
        "cluster_id", "regime_name", "distance_to_centroid",
        "second_nearest_distance", "distance_margin", "relative_distance_margin",
        "nearest_cluster_order", "distance_cluster_0", "distance_cluster_1",
        "distance_cluster_2",
    )
    return {f"{prefix}_{name}": record[name] for name in names}


def pair_partition(
    input_row: pd.Series,
    target_row: pd.Series,
) -> tuple[str | None, str | None]:
    input_start = pd.Timestamp(input_row["start_time"])
    target_stop = pd.Timestamp(target_row["stop_time"])
    source_id = str(input_row["source_id"])

    if source_id == "train_validation":
        if input_start >= TRAIN_START_UTC and target_stop <= TRAIN_END_UTC:
            return "train", None
        if input_start >= VALIDATION_START_UTC and target_stop <= VALIDATION_END_UTC:
            return "validation", None
        if input_start < VALIDATION_START_UTC <= target_stop:
            return None, "crosses_train_validation_boundary"
        return None, "outside_fixed_train_or_validation_period"

    if source_id == "test":
        if input_start >= TEST_START_UTC and target_stop <= TEST_END_UTC:
            return "test", None
        return None, "future_target_outside_fixed_test_period"

    return None, f"unknown_source_id: {source_id}"


def build_pairs(
    labelled_windows: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    partitions: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    skipped: list[dict[str, Any]] = []

    for source_id, source_windows in labelled_windows.groupby("source_id", sort=False):
        source_windows = source_windows.sort_values("start_time").reset_index(drop=True)
        by_start = {
            pd.Timestamp(row["start_time"]): row
            for _, row in source_windows.iterrows()
        }

        for _, input_row in source_windows.iterrows():
            input_start = pd.Timestamp(input_row["start_time"])
            required_target_start = input_start + (WINDOW_SIZE * EXPECTED_INTERVAL)
            target_row = by_start.get(required_target_start)

            skip_base = {
                "source_id": source_id,
                "input_sample_name": input_row["sample_name"],
                "input_start_time": input_row["start_time"],
                "required_target_start_time": iso(required_target_start),
            }
            if target_row is None:
                skipped.append({**skip_base, "reason": "exact_future_target_image_unavailable"})
                continue

            input_stop = pd.Timestamp(input_row["stop_time"])
            target_start = pd.Timestamp(target_row["start_time"])
            if target_start - input_stop != EXPECTED_INTERVAL:
                skipped.append({**skip_base, "reason": "input_target_not_adjacent_by_one_hour"})
                continue

            partition, reason = pair_partition(input_row, target_row)
            if partition is None:
                skipped.append({**skip_base, "reason": reason})
                continue

            pair_id = (
                f"{source_id}__{input_row['sample_name']}__to__"
                f"{target_row['sample_name']}"
            )
            record = {
                "pair_id": pair_id,
                "partition": partition,
                "forecast_horizon_candles": FORECAST_HORIZON,
                **prefixed(input_row, "input"),
                **prefixed(target_row, "target"),
                "target_cluster_id": int(target_row["cluster_id"]),
                "target_regime_name": target_row["regime_name"],
            }
            partitions[partition].append(record)

    result: dict[str, pd.DataFrame] = {}
    for name, records in partitions.items():
        if not records:
            raise RuntimeError(f"No valid {name} forecasting pairs were created.")
        frame = pd.DataFrame(records).sort_values("input_start_time").reset_index(drop=True)
        frame.insert(1, "chronological_rank", np.arange(len(frame), dtype=np.int64))
        result[name] = frame
    return result, skipped


def shuffled_copy(frame: pd.DataFrame) -> pd.DataFrame:
    shuffled = frame.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    shuffled.insert(2, "shuffled_rank", np.arange(len(shuffled), dtype=np.int64))
    return shuffled


def cluster_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["target_cluster_id"].value_counts().sort_index()
    return {str(cluster): int(counts.get(cluster, 0)) for cluster in range(3)}


def write_json_lines(path: Path, rows: list[dict[str, Any]], empty_message: str) -> None:
    with path.open("w", encoding="utf-8") as file:
        if not rows:
            file.write(empty_message + "\n")
        else:
            for row in rows:
                file.write(json.dumps(row, sort_keys=True) + "\n")


def write_reports(
    partitions: dict[str, pd.DataFrame],
    paths: OutputPaths,
) -> dict[str, Any]:
    distribution_rows: list[dict[str, Any]] = []
    for partition_name, frame in partitions.items():
        counts = frame["target_cluster_id"].value_counts()
        for cluster_id in range(3):
            count = int(counts.get(cluster_id, 0))
            distribution_rows.append({
                "partition": partition_name,
                "cluster_id": cluster_id,
                "regime_name": REGIME_NAMES[cluster_id],
                "count": count,
                "percentage": (100.0 * count / len(frame)),
            })
    pd.DataFrame(distribution_rows).to_csv(
        paths.reports / "cluster_distribution.csv", index=False
    )

    train = partitions["train"]
    train_counts = train["target_cluster_id"].value_counts()
    if any(int(train_counts.get(cluster_id, 0)) == 0 for cluster_id in range(3)):
        raise RuntimeError("At least one target cluster is absent from training data.")

    class_weights = {
        str(cluster_id): len(train) / (3.0 * int(train_counts[cluster_id]))
        for cluster_id in range(3)
    }
    balance = {
        "calculated_from": "training targets only",
        "formula": "N / (number_of_classes * class_count)",
        "training_target_counts": cluster_counts(train),
        "balanced_cross_entropy_weights": class_weights,
        "instruction": (
            "Use these weights only for training loss. Do not resample or reweight "
            "validation or test data."
        ),
    }
    (paths.reports / "training_class_weights.json").write_text(
        json.dumps(balance, indent=2), encoding="utf-8"
    )

    transition_rows: list[dict[str, Any]] = []
    for partition_name, frame in partitions.items():
        matrix = pd.crosstab(frame["input_cluster_id"], frame["target_cluster_id"])
        matrix = matrix.reindex(index=range(3), columns=range(3), fill_value=0)
        for current_cluster in range(3):
            for future_cluster in range(3):
                transition_rows.append({
                    "partition": partition_name,
                    "current_cluster_id": current_cluster,
                    "future_cluster_id": future_cluster,
                    "count": int(matrix.loc[current_cluster, future_cluster]),
                })
    pd.DataFrame(transition_rows).to_csv(
        paths.reports / "current_to_future_transition_counts.csv", index=False
    )
    return balance


def duplicate_report(windows: pd.DataFrame) -> dict[str, Any]:
    grouped = windows.groupby("image_sha256")["image_path"].apply(list)
    duplicate_groups = [
        {"sha256": digest, "images": images}
        for digest, images in grouped.items()
        if len(images) > 1
    ]
    return {
        "image_count": int(len(windows)),
        "unique_hash_count": int(windows["image_sha256"].nunique()),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
    }


def save_outputs(
    windows: pd.DataFrame,
    partitions: dict[str, pd.DataFrame],
    skipped_images: list[dict[str, Any]],
    skipped_pairs: list[dict[str, Any]],
    csv_audits: list[dict[str, Any]],
    csv_boundary_audit: dict[str, Any],
    model_audit: dict[str, Any],
    paths: OutputPaths,
    device: torch.device,
) -> None:
    windows.sort_values(["source_id", "start_time"]).to_csv(
        paths.manifests / "labeled_images_manifest.csv", index=False
    )

    all_chronological: list[pd.DataFrame] = []
    for partition_name in ("train", "validation", "test"):
        chronological = partitions[partition_name]
        shuffled = shuffled_copy(chronological)
        chronological.to_csv(
            paths.manifests / f"{partition_name}_samples_chronological.csv", index=False
        )
        shuffled.to_csv(
            paths.manifests / f"{partition_name}_samples_shuffled.csv", index=False
        )
        all_chronological.append(chronological)

    pd.concat(all_chronological, ignore_index=True).to_csv(
        paths.manifests / "all_samples_chronological.csv", index=False
    )

    write_json_lines(
        paths.root / "skipped_images.txt",
        skipped_images,
        "No images were skipped.",
    )
    write_json_lines(
        paths.root / "skipped_pairs.txt",
        skipped_pairs,
        "No forecasting pairs were skipped.",
    )

    duplicates = duplicate_report(windows)
    (paths.manifests / "duplicate_images.json").write_text(
        json.dumps(duplicates, indent=2), encoding="utf-8"
    )
    class_balance = write_reports(partitions, paths)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": (
            "Predict the frozen K=3 cluster of candles t+30..t+59 from the "
            "360x360 K-line image of candles t..t+29."
        ),
        "frozen_periods_utc": {
            "train": [iso(TRAIN_START_UTC), iso(TRAIN_END_UTC)],
            "validation": [iso(VALIDATION_START_UTC), iso(VALIDATION_END_UTC)],
            "test": [iso(TEST_START_UTC), iso(TEST_END_UTC)],
        },
        "window_size": WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "stride": STRIDE,
        "random_seed": RANDOM_SEED,
        "shuffling": (
            "Rows were independently shuffled inside each already-frozen partition. "
            "Images were not renamed or moved."
        ),
        "cluster_mapping": {str(key): value for key, value in REGIME_NAMES.items()},
        "counts": {
            "labeled_images": int(len(windows)),
            "skipped_images": len(skipped_images),
            "skipped_pairs": len(skipped_pairs),
            **{f"{name}_pairs": int(len(frame)) for name, frame in partitions.items()},
        },
        "target_cluster_counts": {
            name: cluster_counts(frame) for name, frame in partitions.items()
        },
        "training_class_balance": class_balance,
        "csv_inputs": csv_audits,
        "cross_csv_boundary": csv_boundary_audit,
        "frozen_models": model_audit,
        "encoder": {
            "architecture": "torchvision ResNet18",
            "weights": "ResNet18_Weights.DEFAULT",
            "output_features": 512,
            "device_used": str(device),
            "fit_called": False,
        },
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
        "duplicate_summary": {
            key: value for key, value in duplicates.items() if key != "duplicate_groups"
        },
    }
    (paths.manifests / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if CPU_THREADS > 0:
        torch.set_num_threads(CPU_THREADS)

    train_valid_path = TRAIN_VALID_CSV_PATH.expanduser().resolve()
    test_path = TEST_CSV_PATH.expanduser().resolve()
    output_path = OUTPUT_FOLDER_PATH.expanduser().resolve()
    if train_valid_path == test_path:
        raise ValueError("TRAIN_VALID_CSV_PATH and TEST_CSV_PATH must be different files.")
    if output_path in (train_valid_path, test_path, PCA_MODEL_PATH, KMEANS_MODEL_PATH):
        raise ValueError("OUTPUT_FOLDER_PATH must be a folder, not an input file.")

    device = resolve_device()

    print("=" * 80)
    print("K=3 FUTURE-REGIME FORECASTING DATASET GENERATOR")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Output: {output_path}")

    pca, kmeans, model_audit = load_frozen_models()
    print("Frozen model contract verified: ResNet18 512 -> PCA 120 -> K-Means K=3")

    train_valid_raw, train_valid_audit = load_and_validate_csv(
        train_valid_path, "train_validation"
    )
    test_raw, test_audit = load_and_validate_csv(test_path, "test")

    csv_boundary_audit = validate_csv_boundaries(train_valid_raw, test_raw)
    print(
        "CSV boundary verified: "
        f"{iso(VALIDATION_END_UTC)} -> {iso(TEST_START_UTC)} "
        "(exactly 1 hour, no overlap)"
    )

    train_valid = select_exact_period(
        train_valid_raw, TRAIN_START_UTC, VALIDATION_END_UTC, "train_validation"
    )
    test = select_exact_period(test_raw, TEST_START_UTC, TEST_END_UTC, "test")

    # Create the output only after every input and frozen-model preflight passes.
    paths = prepare_output(output_path)

    train_valid_windows, skipped_tv_images = generate_windows(
        train_valid,
        "train_validation",
        paths.images_train_valid,
        paths.root,
    )
    test_windows, skipped_test_images = generate_windows(
        test,
        "test",
        paths.images_test,
        paths.root,
    )
    windows = pd.concat([train_valid_windows, test_windows], ignore_index=True)
    labelled_windows = label_windows(windows, paths.root, pca, kmeans, device)
    partitions, skipped_pairs = build_pairs(labelled_windows)

    save_outputs(
        labelled_windows,
        partitions,
        [*skipped_tv_images, *skipped_test_images],
        skipped_pairs,
        [train_valid_audit, test_audit],
        csv_boundary_audit,
        model_audit,
        paths,
        device,
    )

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETED")
    print("=" * 80)
    print(f"Training pairs:   {len(partitions['train']):,}")
    print(f"Validation pairs: {len(partitions['validation']):,}")
    print(f"Test pairs:       {len(partitions['test']):,}")
    print(f"Skipped images:   {len(skipped_tv_images) + len(skipped_test_images):,}")
    print(f"Skipped pairs:    {len(skipped_pairs):,}")
    print(f"Output folder:    {paths.root}")
    print("\nUse train_samples_shuffled.csv for training.")
    print("Use chronological validation/test manifests for primary reporting.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nDATASET GENERATION FAILED\n")
        print(traceback.format_exc())
        raise
