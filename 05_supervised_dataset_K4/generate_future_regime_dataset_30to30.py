from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


REGIME_NAMES = {
    0: "High-volatility consolidation",
    1: "Bullish trend and breakout",
    2: "Volatile sideways and reversal",
    3: "Bearish trend and breakdown",
}


@dataclass
class Paths:
    output_dir: Path
    images_dir: Path
    manifests_dir: Path
    reports_dir: Path


class TargetImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform: Any) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Could not read target image: {path}") from exc

        return tensor, index


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def set_reproducibility(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)


def resolve_device(value: str) -> torch.device:
    value = value.strip().lower()

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(value)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return device


def prepare_paths(config: dict[str, Any]) -> Paths:
    output_dir = Path(config["output_dir"]).expanduser().resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        if config.get("overwrite_output", False):
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}\n"
                "Set overwrite_output=true to replace it."
            )

    paths = Paths(
        output_dir=output_dir,
        images_dir=output_dir / "images",
        manifests_dir=output_dir / "manifests",
        reports_dir=output_dir / "reports",
    )

    for folder in (
        paths.output_dir,
        paths.images_dir,
        paths.manifests_dir,
        paths.reports_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    return paths


def find_column(frame: pd.DataFrame, configured: str, aliases: Iterable[str]) -> str:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}

    candidates = [configured, *aliases]

    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in lookup:
            return lookup[key]

    raise ValueError(
        f"Could not find column {configured!r}. "
        f"Available columns: {frame.columns.tolist()}"
    )


def load_ohlc(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    csv_path = Path(config["ohlc_csv"]).expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"OHLC CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)

    if frame.empty:
        raise ValueError(f"OHLC CSV is empty: {csv_path}")

    columns_cfg = config.get("columns", {})

    timestamp = find_column(
        frame,
        columns_cfg.get("timestamp", "timestamp"),
        ["datetime", "date", "time", "open_time", "timestamp"],
    )
    open_col = find_column(frame, columns_cfg.get("open", "open"), ["open"])
    high_col = find_column(frame, columns_cfg.get("high", "high"), ["high"])
    low_col = find_column(frame, columns_cfg.get("low", "low"), ["low"])
    close_col = find_column(frame, columns_cfg.get("close", "close"), ["close"])

    column_map = {
        "timestamp": timestamp,
        "open": open_col,
        "high": high_col,
        "low": low_col,
        "close": close_col,
    }

    frame = frame.copy()
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce", utc=True)

    numeric_columns = [open_col, high_col, low_col, close_col]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    before = len(frame)
    frame = frame.dropna(subset=[timestamp, *numeric_columns]).copy()
    dropped = before - len(frame)

    frame = frame.sort_values(timestamp).drop_duplicates(
        subset=[timestamp],
        keep="first",
    ).reset_index(drop=True)

    invalid_price_rows = (
        (frame[high_col] < frame[[open_col, close_col, low_col]].max(axis=1))
        | (frame[low_col] > frame[[open_col, close_col, high_col]].min(axis=1))
        | (frame[[open_col, high_col, low_col, close_col]] <= 0).any(axis=1)
    )

    if invalid_price_rows.any():
        count = int(invalid_price_rows.sum())
        raise ValueError(
            f"Found {count} invalid OHLC rows. "
            "Check high/low consistency and non-positive prices."
        )

    print(f"Loaded OHLC rows: {len(frame):,}")
    print(f"Dropped rows with missing/unparseable fields: {dropped:,}")
    print(f"Date range: {frame[timestamp].iloc[0]} to {frame[timestamp].iloc[-1]}")

    return frame, column_map


def expected_delta(config: dict[str, Any]) -> pd.Timedelta:
    frequency = str(config.get("expected_frequency", "1h"))
    try:
        return pd.to_timedelta(frequency)
    except Exception as exc:
        raise ValueError(
            f"Invalid expected_frequency={frequency!r}. "
            "Examples: '1h', '30min', '4h'."
        ) from exc


def window_is_continuous(
    timestamps: pd.Series,
    expected: pd.Timedelta,
) -> bool:
    diffs = timestamps.diff().iloc[1:]
    return bool((diffs == expected).all())


def render_candlestick(
    window: pd.DataFrame,
    column_map: dict[str, str],
    destination: Path,
    config: dict[str, Any],
) -> None:
    """
    Generate a candlestick image using the same Matplotlib BXP method
    used for the original unsupervised dataset.
    """

    style = config.get("chart_style", {})

    image_size_px = int(style.get("width_px", 360))
    image_dpi = int(style.get("dpi", 100))

    figure, axis = plt.subplots(
        figsize=(
            image_size_px / image_dpi,
            image_size_px / image_dpi,
        ),
        dpi=image_dpi,
    )

    boxplot_information = []
    candle_colours = []

    for _, row in window.iterrows():
        open_price = float(row[column_map["open"]])
        high_price = float(row[column_map["high"]])
        low_price = float(row[column_map["low"]])
        close_price = float(row[column_map["close"]])

        candle_information = {
            "q1": min(open_price, close_price),
            "q3": max(open_price, close_price),
            "whishi": max(high_price, low_price),
            "whislo": min(high_price, low_price),
            "med": (open_price + close_price) / 2,
        }

        if open_price <= close_price:
            candle_colours.append("green")
        else:
            candle_colours.append("red")

        boxplot_information.append(candle_information)

    boxplots = axis.bxp(
        bxpstats=boxplot_information,
        showcaps=False,
        patch_artist=True,
        showfliers=False,
    )

    for candle_index, box in enumerate(boxplots["boxes"]):
        colour = candle_colours[candle_index]

        box.set_facecolor(colour)
        box.set_edgecolor(colour)

    for whisker_index, whisker in enumerate(boxplots["whiskers"]):
        candle_index = whisker_index // 2
        whisker.set_color(candle_colours[candle_index])

    for median in boxplots["medians"]:
        median.set_visible(False)

    axis.axis("off")

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        destination,
        dpi=image_dpi,
        pad_inches=0,
    )

    plt.close(figure)


def relative_image_path(start: int, stop: int) -> str:
    return f"images/sample_{start}_{stop}.jpg"


def create_image_if_needed(
    frame: pd.DataFrame,
    start: int,
    stop: int,
    paths: Paths,
    column_map: dict[str, str],
    config: dict[str, Any],
) -> Path:
    destination = paths.output_dir / relative_image_path(start, stop)

    if not destination.is_file():
        render_candlestick(
            frame.iloc[start : stop + 1],
            column_map,
            destination,
            config,
        )

    return destination


def generate_candidate_rows(
    frame: pd.DataFrame,
    column_map: dict[str, str],
    paths: Paths,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    window_size = int(config.get("window_size", 30))
    horizon = int(config.get("forecast_horizon", 30))
    stride = int(config.get("stride", 1))
    expected = expected_delta(config)

    if window_size <= 1:
        raise ValueError("window_size must be greater than 1.")
    if horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    total_required = window_size + horizon
    maximum_start = len(frame) - total_required

    if maximum_start < 0:
        raise ValueError(
            f"Not enough rows. Need at least {total_required}, found {len(frame)}."
        )

    timestamp_col = column_map["timestamp"]
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    starts = range(0, maximum_start + 1, stride)
    total_candidates = math.floor(maximum_start / stride) + 1

    print(f"\nGenerating candidate supervised samples: {total_candidates:,}")

    for number, start in enumerate(starts, start=1):
        input_start = start
        input_stop = start + window_size - 1
        target_start = input_stop + 1
        target_stop = target_start + horizon - 1

        combined = frame.iloc[input_start : target_stop + 1]

        if len(combined) != total_required:
            skipped.append(
                {
                    "input_start_index": input_start,
                    "reason": "incomplete_60_candle_span",
                }
            )
            continue

        if not window_is_continuous(combined[timestamp_col], expected):
            skipped.append(
                {
                    "input_start_index": input_start,
                    "reason": "timestamp_gap_within_input_or_target",
                }
            )
            continue

        try:
            input_path = create_image_if_needed(
                frame,
                input_start,
                input_stop,
                paths,
                column_map,
                config,
            )
            target_path = create_image_if_needed(
                frame,
                target_start,
                target_stop,
                paths,
                column_map,
                config,
            )
        except Exception as exc:
            skipped.append(
                {
                    "input_start_index": input_start,
                    "reason": f"image_generation_error: {exc}",
                }
            )
            continue

        rows.append(
            {
                "sample_name": f"sample_{input_start}_{input_stop}",
                "input_start_index": input_start,
                "input_stop_index": input_stop,
                "input_start_time": frame.iloc[input_start][timestamp_col].isoformat(),
                "input_stop_time": frame.iloc[input_stop][timestamp_col].isoformat(),
                "input_image": input_path.relative_to(paths.output_dir).as_posix(),
                "target_sample_name": f"sample_{target_start}_{target_stop}",
                "target_start_index": target_start,
                "target_stop_index": target_stop,
                "target_start_time": frame.iloc[target_start][timestamp_col].isoformat(),
                "target_stop_time": frame.iloc[target_stop][timestamp_col].isoformat(),
                "target_image": target_path.relative_to(paths.output_dir).as_posix(),
            }
        )

        if number % 1000 == 0 or number == total_candidates:
            print(
                f"  processed={number:,}/{total_candidates:,} | "
                f"valid={len(rows):,} | skipped={len(skipped):,}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No valid samples were generated.")

    return pd.DataFrame(rows), skipped


def build_encoder(device: torch.device) -> tuple[nn.Module, Any]:
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, transform


def load_saved_models(
    config: dict[str, Any],
) -> tuple[PCA, KMeans]:
    pca_path = Path(config["pca_model"]).expanduser().resolve()
    kmeans_path = Path(config["kmeans_model"]).expanduser().resolve()

    if not pca_path.is_file():
        raise FileNotFoundError(f"PCA model not found: {pca_path}")
    if not kmeans_path.is_file():
        raise FileNotFoundError(f"K-Means model not found: {kmeans_path}")

    pca = joblib.load(pca_path)
    kmeans = joblib.load(kmeans_path)

    if hasattr(pca, "n_features_in_") and int(pca.n_features_in_) != 512:
        raise ValueError(
            f"PCA expects {pca.n_features_in_} features, not 512."
        )

    if int(kmeans.n_clusters) != 4:
        raise ValueError(
            f"Expected four clusters, found {kmeans.n_clusters}."
        )

    return pca, kmeans


def label_target_images(
    samples: pd.DataFrame,
    paths: Paths,
    config: dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    model, transform = build_encoder(device)
    pca, kmeans = load_saved_models(config)

    unique_target_rel_paths = samples["target_image"].drop_duplicates().tolist()
    target_paths = [paths.output_dir / value for value in unique_target_rel_paths]

    dataset = TargetImageDataset(target_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    result_by_path: dict[str, dict[str, Any]] = {}
    processed = 0

    print(f"\nLabelling {len(target_paths):,} unique future-window images...")

    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            embeddings = model(images).detach().cpu().numpy().astype(np.float32)

            reduced = pca.transform(embeddings)
            labels = kmeans.predict(reduced)
            distances = kmeans.transform(reduced)

            for local_position, dataset_index in enumerate(indices.numpy()):
                rel_path = unique_target_rel_paths[int(dataset_index)]
                row_distances = distances[local_position]
                order = np.argsort(row_distances)
                nearest = float(row_distances[order[0]])
                second = float(row_distances[order[1]])
                margin = second - nearest
                relative_margin = margin / max(second, 1e-8)
                label = int(labels[local_position])

                result_by_path[rel_path] = {
                    "target_cluster_id": label,
                    "target_regime_name": REGIME_NAMES.get(
                        label,
                        f"Cluster {label}",
                    ),
                    "target_distance_to_centroid": float(row_distances[label]),
                    "target_second_nearest_distance": second,
                    "target_distance_margin": margin,
                    "target_relative_margin": relative_margin,
                    "target_nearest_cluster_order": json.dumps(
                        [int(value) for value in order]
                    ),
                    "target_distance_cluster_0": float(row_distances[0]),
                    "target_distance_cluster_1": float(row_distances[1]),
                    "target_distance_cluster_2": float(row_distances[2]),
                    "target_distance_cluster_3": float(row_distances[3]),
                }

            processed += len(indices)
            if processed % 800 == 0 or processed == len(target_paths):
                print(f"  labelled={processed:,}/{len(target_paths):,}")

    labelled = samples.copy()

    label_frame = pd.DataFrame(
        [
            {"target_image": path, **values}
            for path, values in result_by_path.items()
        ]
    )

    labelled = labelled.merge(
        label_frame,
        on="target_image",
        how="left",
        validate="many_to_one",
    )

    if labelled["target_cluster_id"].isna().any():
        raise RuntimeError("Some target images did not receive cluster labels.")

    labelled["target_cluster_id"] = labelled["target_cluster_id"].astype(int)

    return labelled


def split_chronologically(
    samples: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_cfg = config.get("split", {})
    train_fraction = float(split_cfg.get("train_fraction", 0.70))
    validation_fraction = float(split_cfg.get("validation_fraction", 0.15))
    test_fraction = float(split_cfg.get("test_fraction", 0.15))
    embargo = int(split_cfg.get("extra_embargo_candles", 0))

    total_fraction = train_fraction + validation_fraction + test_fraction
    if not math.isclose(total_fraction, 1.0, abs_tol=1e-9):
        raise ValueError(
            "train_fraction + validation_fraction + test_fraction must equal 1."
        )

    ordered = samples.sort_values("input_start_index").reset_index(drop=True)
    n = len(ordered)

    raw_validation_position = min(max(int(n * train_fraction), 1), n - 2)
    raw_test_position = min(
        max(int(n * (train_fraction + validation_fraction)), raw_validation_position + 1),
        n - 1,
    )

    validation_start = int(
        ordered.iloc[raw_validation_position]["input_start_index"]
    )
    test_start = int(
        ordered.iloc[raw_test_position]["input_start_index"]
    )

    train = ordered[
        ordered["target_stop_index"] < (validation_start - embargo)
    ].copy()

    validation = ordered[
        (ordered["input_start_index"] >= validation_start)
        & (ordered["target_stop_index"] < (test_start - embargo))
    ].copy()

    test = ordered[
        ordered["input_start_index"] >= test_start
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise RuntimeError(
            "Chronological split produced an empty partition. "
            "Adjust fractions, stride, or embargo."
        )

    if int(train["target_stop_index"].max()) >= int(validation["input_start_index"].min()):
        raise AssertionError("Train target overlaps validation input.")

    if int(validation["target_stop_index"].max()) >= int(test["input_start_index"].min()):
        raise AssertionError("Validation target overlaps test input.")

    assigned_indices = set(train.index) | set(validation.index) | set(test.index)
    purged_count = n - len(assigned_indices)

    split_details = {
        "raw_validation_boundary_position": raw_validation_position,
        "raw_test_boundary_position": raw_test_position,
        "validation_input_start_index": validation_start,
        "test_input_start_index": test_start,
        "extra_embargo_candles": embargo,
        "purged_sample_count": purged_count,
        "train_last_target_stop_index": int(train["target_stop_index"].max()),
        "validation_first_input_start_index": int(validation["input_start_index"].min()),
        "validation_last_target_stop_index": int(validation["target_stop_index"].max()),
        "test_first_input_start_index": int(test["input_start_index"].min()),
    }

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
        split_details,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_report(
    samples: pd.DataFrame,
    paths: Paths,
) -> dict[str, Any]:
    image_rel_paths = sorted(
        set(samples["input_image"].tolist())
        | set(samples["target_image"].tolist())
    )

    hash_to_paths: dict[str, list[str]] = {}

    print(f"\nHashing {len(image_rel_paths):,} unique images for duplicate checking...")

    for number, rel_path in enumerate(image_rel_paths, start=1):
        digest = sha256_file(paths.output_dir / rel_path)
        hash_to_paths.setdefault(digest, []).append(rel_path)

        if number % 2000 == 0 or number == len(image_rel_paths):
            print(f"  hashed={number:,}/{len(image_rel_paths):,}")

    duplicate_groups = [
        {"sha256": digest, "files": file_paths}
        for digest, file_paths in hash_to_paths.items()
        if len(file_paths) > 1
    ]

    return {
        "unique_image_count": len(image_rel_paths),
        "unique_hash_count": len(hash_to_paths),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
    }


def cluster_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["target_cluster_id"].value_counts().sort_index()
    return {str(int(index)): int(value) for index, value in counts.items()}


def save_outputs(
    all_samples: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    skipped: list[dict[str, Any]],
    duplicate_info: dict[str, Any],
    split_details: dict[str, Any],
    frame: pd.DataFrame,
    column_map: dict[str, str],
    paths: Paths,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    all_samples.to_csv(paths.manifests_dir / "all_samples.csv", index=False)
    train.to_csv(paths.manifests_dir / "train_samples.csv", index=False)
    validation.to_csv(paths.manifests_dir / "validation_samples.csv", index=False)
    test.to_csv(paths.manifests_dir / "test_samples.csv", index=False)

    with (paths.manifests_dir / "duplicate_images.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(duplicate_info, file, indent=2)

    skipped_path = paths.output_dir / "skipped_samples.txt"
    with skipped_path.open("w", encoding="utf-8") as file:
        if not skipped:
            file.write("No samples were skipped.\n")
        else:
            for row in skipped:
                file.write(json.dumps(row) + "\n")

    timestamp_col = column_map["timestamp"]

    summary = {
        "task": (
            "Predict the cluster regime of candles t+30 through t+59 "
            "from the image of candles t through t+29."
        ),
        "source_ohlc_csv": str(Path(config["ohlc_csv"]).expanduser().resolve()),
        "output_dir": str(paths.output_dir),
        "device": str(device),
        "source_row_count_after_cleaning": int(len(frame)),
        "source_date_start": frame[timestamp_col].iloc[0].isoformat(),
        "source_date_end": frame[timestamp_col].iloc[-1].isoformat(),
        "window_size": int(config.get("window_size", 30)),
        "forecast_horizon": int(config.get("forecast_horizon", 30)),
        "stride": int(config.get("stride", 1)),
        "expected_frequency": str(config.get("expected_frequency", "1h")),
        "all_valid_samples_before_split": int(len(all_samples)),
        "training_samples": int(len(train)),
        "validation_samples": int(len(validation)),
        "test_samples": int(len(test)),
        "skipped_samples": int(len(skipped)),
        "cluster_mapping": {str(key): value for key, value in REGIME_NAMES.items()},
        "cluster_counts": {
            "all": cluster_counts(all_samples),
            "train": cluster_counts(train),
            "validation": cluster_counts(validation),
            "test": cluster_counts(test),
        },
        "split_details": split_details,
        "duplicate_summary": {
            key: value
            for key, value in duplicate_info.items()
            if key != "duplicate_groups"
        },
        "models": {
            "pca_model": str(Path(config["pca_model"]).expanduser().resolve()),
            "kmeans_model": str(Path(config["kmeans_model"]).expanduser().resolve()),
            "resnet18_weights": "torchvision.models.ResNet18_Weights.DEFAULT",
        },
        "chart_style": config.get("chart_style", {}),
        "columns": column_map,
        "manifest_files": {
            "all": "manifests/all_samples.csv",
            "train": "manifests/train_samples.csv",
            "validation": "manifests/validation_samples.csv",
            "test": "manifests/test_samples.csv",
            "duplicate_images": "manifests/duplicate_images.json",
            "skipped_samples": "skipped_samples.txt",
        },
    }

    with (paths.manifests_dir / "dataset_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    with (paths.reports_dir / "cluster_distribution.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["partition", "cluster_id", "regime_name", "count"])

        for partition_name, partition in (
            ("all", all_samples),
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            counts = partition["target_cluster_id"].value_counts().sort_index()
            for cluster_id in range(4):
                writer.writerow(
                    [
                        partition_name,
                        cluster_id,
                        REGIME_NAMES[cluster_id],
                        int(counts.get(cluster_id, 0)),
                    ]
                )


def main() -> None:
    config = load_config()
    paths = prepare_paths(config)

    seed = int(config.get("random_seed", 42))
    set_reproducibility(seed, int(config.get("cpu_threads", 4)))
    device = resolve_device(str(config.get("device", "auto")))

    print("=" * 80)
    print("FUTURE REGIME SUPERVISED DATASET GENERATOR")
    print("=" * 80)
    print(f"Config: {CONFIG_PATH}")
    print(f"Output: {paths.output_dir}")
    print(f"Device: {device}")

    frame, column_map = load_ohlc(config)

    candidate_samples, skipped = generate_candidate_rows(
        frame,
        column_map,
        paths,
        config,
    )

    labelled_samples = label_target_images(
        candidate_samples,
        paths,
        config,
        device,
    )

    train, validation, test, split_details = split_chronologically(
        labelled_samples,
        config,
    )

    duplicate_info = duplicate_report(labelled_samples, paths)

    save_outputs(
        labelled_samples,
        train,
        validation,
        test,
        skipped,
        duplicate_info,
        split_details,
        frame,
        column_map,
        paths,
        config,
        device,
    )

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"All valid samples: {len(labelled_samples):,}")
    print(f"Training samples:  {len(train):,}")
    print(f"Validation samples:{len(validation):,}")
    print(f"Test samples:      {len(test):,}")
    print(f"Skipped samples:   {len(skipped):,}")
    print(f"Output directory:  {paths.output_dir}")
    print("\nSupervised task:")
    print("  input_image (t..t+29) -> target_cluster_id of (t+30..t+59)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nDATASET GENERATION FAILED\n")
        print(traceback.format_exc())
        raise
