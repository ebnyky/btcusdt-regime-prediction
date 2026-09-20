"""Generate the original rolling BTCUSDT candlestick-image corpus.

This module reconstructs the first executable stage of the MSc research:

    OHLC CSV -> 30-candle Matplotlib ``Axes.bxp`` JPEGs -> manifests

The renderer intentionally follows the recovered historical implementation.
It uses local (per-window) autoscaling, the named Matplotlib colours ``green``
and ``red``, hidden axes, 360 x 360 JPEG output, and the original BXP body/wick
statistics.  The code contains no model fitting and no future labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "kline_dataset_config.json"


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    images: Path
    manifests: Path


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_path(value: str | Path, config_path: Path) -> Path:
    """Resolve user paths relative to the configuration file."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def find_column(frame: pd.DataFrame, configured: str, aliases: Iterable[str]) -> str:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in (configured, *aliases):
        key = str(candidate).strip().lower()
        if key in lookup:
            return lookup[key]
    raise ValueError(
        f"Could not find column {configured!r}. Available columns: {frame.columns.tolist()}"
    )


def load_ohlc(
    config: dict[str, Any], config_path: Path
) -> tuple[pd.DataFrame, dict[str, str], Path, int]:
    csv_path = resolve_config_path(config["ohlc_csv"], config_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"OHLC CSV not found: {csv_path}\n"
            "Edit ohlc_csv in kline_dataset_config.json."
        )

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"OHLC CSV is empty: {csv_path}")

    configured = config.get("columns", {})
    columns = {
        "timestamp": find_column(
            frame,
            configured.get("timestamp", "open_time"),
            ["timestamp", "datetime", "date", "time", "open_time"],
        ),
        "open": find_column(frame, configured.get("open", "open"), ["open"]),
        "high": find_column(frame, configured.get("high", "high"), ["high"]),
        "low": find_column(frame, configured.get("low", "low"), ["low"]),
        "close": find_column(frame, configured.get("close", "close"), ["close"]),
    }

    frame = frame.copy()
    timestamp = columns["timestamp"]
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce", utc=True)
    for name in ("open", "high", "low", "close"):
        column = columns[name]
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    before = len(frame)
    required = [columns[key] for key in ("timestamp", "open", "high", "low", "close")]
    frame = frame.dropna(subset=required)
    frame = frame.sort_values(timestamp).drop_duplicates(timestamp, keep="first")
    frame = frame.reset_index(drop=True)
    dropped = before - len(frame)

    open_values = frame[columns["open"]]
    high_values = frame[columns["high"]]
    low_values = frame[columns["low"]]
    close_values = frame[columns["close"]]
    invalid = (
        (high_values < frame[[columns["open"], columns["close"], columns["low"]]].max(axis=1))
        | (low_values > frame[[columns["open"], columns["close"], columns["high"]]].min(axis=1))
        | (frame[[columns["open"], columns["high"], columns["low"], columns["close"]]] <= 0).any(axis=1)
    )
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} invalid OHLC rows.")

    print(f"Loaded {len(frame):,} clean OHLC rows from {csv_path}")
    print(f"Dropped missing, unparseable, or duplicate timestamp rows: {dropped:,}")
    return frame, columns, csv_path, dropped


def prepare_output(config: dict[str, Any], config_path: Path) -> OutputPaths:
    root = resolve_config_path(config["output_dir"], config_path)
    policy = str(config.get("existing_output_policy", "error")).lower()
    if policy not in {"error", "resume"}:
        raise ValueError("existing_output_policy must be 'error' or 'resume'.")
    if root.exists() and any(root.iterdir()) and policy == "error":
        raise FileExistsError(
            f"Output directory is not empty: {root}\n"
            "Choose a new output_dir or set existing_output_policy to 'resume'."
        )
    images = root / "images"
    manifests = root / "manifests"
    images.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    return OutputPaths(root=root, images=images, manifests=manifests)


def expected_delta(config: dict[str, Any]) -> pd.Timedelta:
    try:
        return pd.to_timedelta(str(config.get("expected_frequency", "1h")))
    except Exception as exc:
        raise ValueError("expected_frequency must be a pandas duration such as '1h'.") from exc


def window_is_continuous(timestamps: pd.Series, delta: pd.Timedelta) -> bool:
    return bool((timestamps.diff().iloc[1:] == delta).all())


def render_original_bxp(
    window: pd.DataFrame,
    columns: dict[str, str],
    destination: Path,
    chart_style: dict[str, Any] | None = None,
) -> None:
    """Render one window with the recovered original ``Axes.bxp`` schema."""

    style = chart_style or {}
    width_px = int(style.get("width_px", 360))
    height_px = int(style.get("height_px", 360))
    # The recovered script used 100 DPI, whereas the supplied reference JPEG
    # is tagged 90 DPI. The active config requests 90; this historical fallback
    # remains 100 when no style is supplied.
    dpi = int(style.get("dpi", 100))
    if width_px <= 0 or height_px <= 0 or dpi <= 0:
        raise ValueError("Image width, height, and DPI must be positive.")

    figure, axis = plt.subplots(
        figsize=(width_px / dpi, height_px / dpi),
        dpi=dpi,
    )

    stats: list[dict[str, float]] = []
    colours: list[str] = []
    for _, row in window.iterrows():
        open_price = float(row[columns["open"]])
        high_price = float(row[columns["high"]])
        low_price = float(row[columns["low"]])
        close_price = float(row[columns["close"]])
        stats.append(
            {
                "q1": min(open_price, close_price),
                "q3": max(open_price, close_price),
                "whishi": max(high_price, low_price),
                "whislo": min(high_price, low_price),
                "med": (open_price + close_price) / 2.0,
            }
        )
        colours.append("green" if open_price <= close_price else "red")

    artists = axis.bxp(
        bxpstats=stats,
        showcaps=False,
        patch_artist=True,
        showfliers=False,
    )
    for candle_index, box in enumerate(artists["boxes"]):
        colour = colours[candle_index]
        box.set_facecolor(colour)
        box.set_edgecolor(colour)
    for whisker_index, whisker in enumerate(artists["whiskers"]):
        whisker.set_color(colours[whisker_index // 2])
    for median in artists["medians"]:
        median.set_visible(False)

    axis.axis("off")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, pad_inches=0, format="jpg")
    plt.close(figure)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_rows(
    frame: pd.DataFrame,
    columns: dict[str, str],
    source_csv: Path,
    paths: OutputPaths,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    window_size = int(config.get("window_size", 30))
    stride = int(config.get("stride", 1))
    if window_size < 2 or stride < 1:
        raise ValueError("window_size must be >= 2 and stride must be >= 1.")
    maximum_start = len(frame) - window_size
    if maximum_start < 0:
        raise ValueError(f"Need at least {window_size} OHLC rows; found {len(frame)}.")

    timestamp = columns["timestamp"]
    delta = expected_delta(config)
    starts = range(0, maximum_start + 1, stride)
    total = math.floor(maximum_start / stride) + 1
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    hashes: dict[str, list[str]] = {}

    print(f"Generating {total:,} candidate {window_size}-candle windows...")
    for number, start in enumerate(starts, start=1):
        stop = start + window_size - 1
        window = frame.iloc[start : stop + 1]
        reason: str | None = None
        if len(window) != window_size:
            reason = "incomplete_window"
        elif not window_is_continuous(window[timestamp], delta):
            reason = "timestamp_gap_inside_window"
        if reason is not None:
            skipped.append(
                {
                    "sample_name": f"sample_{start}_{stop}",
                    "start_index": start,
                    "stop_index": stop,
                    "reason": reason,
                }
            )
            continue

        sample_name = f"sample_{start}_{stop}"
        relative_image = Path("images") / f"{sample_name}.jpg"
        destination = paths.root / relative_image
        if not destination.is_file():
            try:
                render_original_bxp(window, columns, destination, config.get("chart_style", {}))
            except Exception as exc:
                skipped.append(
                    {
                        "sample_name": sample_name,
                        "start_index": start,
                        "stop_index": stop,
                        "reason": f"image_generation_error: {exc}",
                    }
                )
                continue

        try:
            with Image.open(destination) as image:
                if image.size != (
                    int(config.get("chart_style", {}).get("width_px", 360)),
                    int(config.get("chart_style", {}).get("height_px", 360)),
                ):
                    raise ValueError(f"unexpected image size {image.size}")
                image.verify()
        except Exception as exc:
            skipped.append(
                {
                    "sample_name": sample_name,
                    "start_index": start,
                    "stop_index": stop,
                    "reason": f"image_validation_error: {exc}",
                }
            )
            continue

        digest = sha256_file(destination)
        hashes.setdefault(digest, []).append(relative_image.as_posix())
        rows.append(
            {
                "sample_name": sample_name,
                "start_index": start,
                "stop_index": stop,
                "window_length": window_size,
                "start_time": frame.iloc[start][timestamp].isoformat(),
                "stop_time": frame.iloc[stop][timestamp].isoformat(),
                # Keep manifests portable across Windows, Kaggle, and the
                # submission archive. The source CSV identity is locked by
                # SHA-256 in the Kaggle preflight/provenance report.
                "source_image": f"{source_csv.name}#rows={start}:{stop}",
                "exported_image": relative_image.as_posix(),
                "split": "unassigned",
                "sha256": digest,
            }
        )
        if number % 1000 == 0 or number == total:
            print(
                f"  processed={number:,}/{total:,} valid={len(rows):,} "
                f"skipped={len(skipped):,}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No valid image windows were generated.")
    duplicate_groups = [
        {"sha256": digest, "files": files}
        for digest, files in hashes.items()
        if len(files) > 1
    ]
    duplicate_report = {
        "image_count": len(rows),
        "unique_hash_count": len(hashes),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
    }
    with (paths.manifests / "duplicate_images.json").open("w", encoding="utf-8") as handle:
        json.dump(duplicate_report, handle, indent=2)
    return pd.DataFrame(rows), skipped


def split_rows(
    all_rows: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_config = config.get("split", {})
    fixed_start = split_config.get("validation_start_index")
    embargo = int(split_config.get("embargo_candles", 30))
    if embargo < 0:
        raise ValueError("embargo_candles cannot be negative.")

    ordered = all_rows.sort_values("start_index").reset_index(drop=True)
    if fixed_start is None:
        train_fraction = float(split_config.get("train_fraction", 0.90))
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1.")
        raw_position = min(max(int(len(ordered) * train_fraction), 1), len(ordered) - 1)
        validation_start = int(ordered.iloc[raw_position]["start_index"])
        split_mode = "fraction"
    else:
        validation_start = int(fixed_start)
        split_mode = "fixed_validation_start_index"

    train = ordered[ordered["stop_index"] < validation_start - embargo].copy()
    validation = ordered[ordered["start_index"] >= validation_start].copy()
    purged = ordered[
        ~ordered.index.isin(train.index) & ~ordered.index.isin(validation.index)
    ].copy()
    if train.empty or validation.empty:
        raise RuntimeError(
            "The chronological split produced an empty partition. Check the fixed "
            "validation_start_index or use fraction-based splitting."
        )

    ordered.loc[train.index, "split"] = "train"
    ordered.loc[validation.index, "split"] = "validation"
    ordered.loc[purged.index, "split"] = "purged"
    train = ordered.loc[train.index].reset_index(drop=True)
    validation = ordered.loc[validation.index].reset_index(drop=True)
    purged = ordered.loc[purged.index].reset_index(drop=True)

    details = {
        "mode": split_mode,
        "validation_start_index": validation_start,
        "embargo_candles": embargo,
        "train_last_start_index": int(train["start_index"].max()),
        "train_last_stop_index": int(train["stop_index"].max()),
        "validation_first_start_index": int(validation["start_index"].min()),
        "purged_sample_count": int(len(purged)),
    }
    if details["train_last_stop_index"] >= validation_start - embargo:
        raise AssertionError("The training partition violates the configured embargo.")
    return ordered, train, validation, details


def save_outputs(
    all_rows: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    skipped: list[dict[str, Any]],
    split_details: dict[str, Any],
    frame: pd.DataFrame,
    columns: dict[str, str],
    source_csv: Path,
    dropped_rows: int,
    paths: OutputPaths,
    config: dict[str, Any],
) -> None:
    all_rows.to_csv(paths.manifests / "all_samples.csv", index=False)
    train.to_csv(paths.manifests / "train_samples.csv", index=False)
    validation.to_csv(paths.manifests / "validation_samples.csv", index=False)

    with (paths.root / "skipped_samples.txt").open("w", encoding="utf-8") as handle:
        if not skipped:
            handle.write("No samples were skipped.\n")
        else:
            for item in skipped:
                handle.write(json.dumps(item, sort_keys=True) + "\n")

    timestamp = columns["timestamp"]
    summary = {
        "stage": "original_unsupervised_kline_dataset_generation",
        "source_ohlc_csv": source_csv.name,
        "source_rows_after_cleaning": int(len(frame)),
        "source_rows_dropped": int(dropped_rows),
        "source_date_start": frame.iloc[0][timestamp].isoformat(),
        "source_date_end": frame.iloc[-1][timestamp].isoformat(),
        "window_size": int(config.get("window_size", 30)),
        "stride": int(config.get("stride", 1)),
        "expected_frequency": str(config.get("expected_frequency", "1h")),
        "valid_samples": int(len(all_rows)),
        "training_samples": int(len(train)),
        "validation_samples": int(len(validation)),
        "skipped_samples": int(len(skipped)),
        "split": split_details,
        "renderer": {
            "method": "matplotlib.axes.Axes.bxp",
            "profile": "historical_original_autoscaled_bxp",
            "chart_style": config.get("chart_style", {}),
            "bullish_colour": "green",
            "bearish_colour": "red",
            "axes_visible": False,
        },
        "manifest_files": {
            "all": "manifests/all_samples.csv",
            "train": "manifests/train_samples.csv",
            "validation": "manifests/validation_samples.csv",
            "duplicates": "manifests/duplicate_images.json",
            "skipped": "skipped_samples.txt",
        },
    }
    with (paths.manifests / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def generate(config_path: Path) -> OutputPaths:
    config_path = config_path.expanduser().resolve()
    config = read_json(config_path)
    paths = prepare_output(config, config_path)
    frame, columns, source_csv, dropped_rows = load_ohlc(config, config_path)
    rows, skipped = generate_rows(frame, columns, source_csv, paths, config)
    all_rows, train, validation, split_details = split_rows(rows, config)
    save_outputs(
        all_rows,
        train,
        validation,
        skipped,
        split_details,
        frame,
        columns,
        source_csv,
        dropped_rows,
        paths,
        config,
    )
    print("\nK-line dataset generation completed.")
    print(f"Images: {paths.images}")
    print(f"Manifests: {paths.manifests}")
    print(f"Skipped-sample log: {paths.root / 'skipped_samples.txt'}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Configuration JSON (default: {DEFAULT_CONFIG.name})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args().config)
