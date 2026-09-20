"""
reconstructed_autoscaled_dataset_pipeline.py

Reconstructed from the supplied dataset artifacts:
- dataset_summary.json
- all_samples.csv
- train_samples.csv
- validation_samples.csv
- duplicate_images.json

Purpose
-------
Reconstruct the export, manifest, duplicate-checking, and chronological split stages for the autoscaled experimental dataset. The source sample images are assumed to have already been rendered with Matplotlib per-window autoscaling. Export the 01_kline.jpg image from each generated sample folder into one
flat image directory, create a chronological train/validation split, purge
boundary samples whose target horizon would overlap validation, calculate
SHA-256 hashes, detect exact duplicate images, and write manifest files.

Important
---------
This is a reconstruction, not the missing original autoscaled generator/exporter. It does not recreate the exact candlestick rendering code; it assumes autoscaled source images already exist. Its behaviour
matches the supplied summary fields and observed split boundary:
- validation_fraction = 0.10
- purge_candles = 30
- last training sample = sample_23987_24016
- first validation sample = sample_24047_24076
- 59 boundary samples purged

Review paths and settings before running.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SOURCE_DATASET_DIR = Path(
    r"C:\Users\m0zim\OneDrive\Desktop\final_proj\btc_1h_30klines_dataset"
)
OUTPUT_DATASET_DIR = Path(
    r"C:\Users\m0zim\OneDrive\Desktop\final_year_project_2_datasets\autoscaled"
)

CANDLESTICK_FILENAME = "01_kline.jpg"
EXPORT_MODE = "copy"  # "copy" or "symlink"
VALIDATION_FRACTION = 0.10
PURGE_CANDLES = 30

SAMPLE_PATTERN = re.compile(r"^sample_(\d+)_(\d+)$")


@dataclass(frozen=True)
class Sample:
    sample_name: str
    start_index: int
    stop_index: int
    window_length: int
    source_image: Path


def parse_sample_folder(folder: Path) -> Sample | None:
    match = SAMPLE_PATTERN.match(folder.name)
    if match is None:
        return None

    start_index = int(match.group(1))
    stop_index = int(match.group(2))
    source_image = folder / "images" / CANDLESTICK_FILENAME

    if not source_image.is_file():
        return None

    return Sample(
        sample_name=folder.name,
        start_index=start_index,
        stop_index=stop_index,
        window_length=stop_index - start_index + 1,
        source_image=source_image,
    )


def discover_samples(source_dir: Path) -> list[Sample]:
    samples: list[Sample] = []

    for folder in source_dir.glob("sample_*"):
        if not folder.is_dir():
            continue
        parsed = parse_sample_folder(folder)
        if parsed is not None:
            samples.append(parsed)

    samples.sort(key=lambda item: (item.start_index, item.stop_index))

    if not samples:
        raise RuntimeError(f"No valid sample folders found in {source_dir}")

    return samples


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def export_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        raise ValueError("EXPORT_MODE must be 'copy' or 'symlink'.")


def chronological_split_with_purge(
    samples: list[Sample],
    validation_fraction: float,
    purge_candles: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Create the split that is consistent with the supplied artifacts.

    1. Reserve the final validation_fraction of chronologically sorted samples.
    2. Keep the earlier samples as training.
    3. Purge early validation candidates until their input start lies strictly
       after the last training target horizon.

    The generator creates a target window immediately after each input window.
    For a 30-candle input, the last training target ends at:
        last_train.stop_index + purge_candles
    Therefore validation starts must be greater than that index.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")

    validation_count_before_purge = int(round(len(samples) * validation_fraction))
    validation_count_before_purge = max(1, min(validation_count_before_purge, len(samples) - 1))
    split_position = len(samples) - validation_count_before_purge

    train = list(samples[:split_position])
    validation_candidates = list(samples[split_position:])

    last_train_target_stop = train[-1].stop_index + purge_candles
    purged: list[Sample] = []
    validation: list[Sample] = []

    for sample in validation_candidates:
        if sample.start_index <= last_train_target_stop:
            purged.append(sample)
        else:
            validation.append(sample)

    if not validation:
        raise RuntimeError("Purging removed the entire validation partition.")

    return train, validation, purged


def image_metadata(path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        dimensions = f"{image.width}x{image.height}"
        mode = image.mode
    return dimensions, mode


def main() -> None:
    source_dir = SOURCE_DATASET_DIR.expanduser().resolve()
    output_dir = OUTPUT_DATASET_DIR.expanduser().resolve()
    images_dir = output_dir / "images"
    manifests_dir = output_dir / "manifests"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")

    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(source_dir)
    train, validation, purged = chronological_split_with_purge(
        samples,
        VALIDATION_FRACTION,
        PURGE_CANDLES,
    )

    split_lookup = {sample.sample_name: "train" for sample in train}
    split_lookup.update({sample.sample_name: "validation" for sample in validation})

    retained = train + validation
    retained.sort(key=lambda item: (item.start_index, item.stop_index))

    records: list[dict[str, object]] = []
    hashes: dict[str, list[str]] = {}
    dimension_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}

    for number, sample in enumerate(retained, start=1):
        destination = images_dir / f"{sample.sample_name}.jpg"
        export_image(sample.source_image, destination, EXPORT_MODE)
        digest = sha256_file(destination)
        dimensions, image_mode = image_metadata(destination)

        dimension_counts[dimensions] = dimension_counts.get(dimensions, 0) + 1
        mode_counts[image_mode] = mode_counts.get(image_mode, 0) + 1
        hashes.setdefault(digest, []).append(sample.sample_name)

        records.append(
            {
                "sample_name": sample.sample_name,
                "start_index": sample.start_index,
                "stop_index": sample.stop_index,
                "window_length": sample.window_length,
                "source_image": str(sample.source_image),
                "exported_image": str(destination),
                "split": split_lookup[sample.sample_name],
                "sha256": digest,
            }
        )

        if number % 1000 == 0 or number == len(retained):
            print(f"Exported {number:,}/{len(retained):,}")

    frame = pd.DataFrame(records).sort_values(["start_index", "stop_index"])
    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    validation_frame = frame[frame["split"] == "validation"].reset_index(drop=True)

    frame.to_csv(manifests_dir / "all_samples.csv", index=False)
    train_frame.to_csv(manifests_dir / "train_samples.csv", index=False)
    validation_frame.to_csv(manifests_dir / "validation_samples.csv", index=False)

    duplicate_groups = [names for names in hashes.values() if len(names) > 1]
    with (manifests_dir / "duplicate_images.json").open("w", encoding="utf-8") as file:
        json.dump(duplicate_groups, file, indent=2)

    with (output_dir / "purged_samples.txt").open("w", encoding="utf-8") as file:
        for sample in purged:
            file.write(f"{sample.sample_name}\n")

    summary = {
        "source_dataset_dir": str(source_dir),
        "output_dataset_dir": str(output_dir),
        "candlestick_filename": CANDLESTICK_FILENAME,
        "export_mode": EXPORT_MODE,
        "validation_fraction": VALIDATION_FRACTION,
        "purge_candles": PURGE_CANDLES,
        "source_sample_folders": len(samples),
        "exported_samples": len(frame),
        "training_samples": len(train_frame),
        "validation_samples": len(validation_frame),
        "purged_samples": len(purged),
        "skipped_samples": 0,
        "duplicate_image_groups": len(duplicate_groups),
        "image_dimensions": dimension_counts,
        "image_modes": mode_counts,
        "first_training_sample": train_frame.iloc[0]["sample_name"],
        "last_training_sample": train_frame.iloc[-1]["sample_name"],
        "first_validation_sample": validation_frame.iloc[0]["sample_name"],
        "last_validation_sample": validation_frame.iloc[-1]["sample_name"],
        "clustering_rule": (
            "Fit CNN embedding preprocessing, PCA, and clustering on "
            "train_samples.csv only. Use validation_samples.csv only for "
            "out-of-sample transformation and cluster assignment."
        ),
        "reconstruction_notice": (
            "This exporter was reconstructed from supplied artifacts and is "
            "not claimed to be the missing original source file."
        ),
    }

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
