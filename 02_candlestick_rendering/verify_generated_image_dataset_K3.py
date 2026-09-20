"""Strict verification for the full-range Kaggle kline corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


EXPECTED = {
    "source_sha256": "c938439e45cce76aee7234c62be1ba5aaa6b80c6f3598924715d5d0c9563ce8d",
    "source_rows": 48281,
    "valid_samples": 47817,
    "training_samples": 42976,
    "validation_samples": 4782,
    "purged_samples": 59,
    "skipped_samples": 435,
    "validation_start_index": 43470,
    "first_image_sha256": "9551df8b81bd0b63ff071884c446525942e20e7856c70c198e3c423aabe80487",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument(
        "--all-image-dimensions",
        action="store_true",
        help="Open all 47,817 JPEGs; slower but recommended before archiving.",
    )
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifests = root / "manifests"
    summary = json.loads((manifests / "dataset_summary.json").read_text("utf-8"))
    all_rows = pd.read_csv(manifests / "all_samples.csv")
    train = pd.read_csv(manifests / "train_samples.csv")
    validation = pd.read_csv(manifests / "validation_samples.csv")
    images = sorted((root / "images").glob("*.jpg"))
    skipped_lines = [
        line
        for line in (root / "skipped_samples.txt").read_text("utf-8").splitlines()
        if line.strip() and line.strip() != "No samples were skipped."
    ]

    require(digest(args.source_csv) == EXPECTED["source_sha256"], "Source CSV hash differs.")
    require(summary["source_rows_after_cleaning"] == EXPECTED["source_rows"], "Wrong source row count.")
    require(len(all_rows) == EXPECTED["valid_samples"], "Wrong all-samples count.")
    require(len(images) == EXPECTED["valid_samples"], "Image/manifest count mismatch.")
    require(len(train) == EXPECTED["training_samples"], "Wrong training count.")
    require(len(validation) == EXPECTED["validation_samples"], "Wrong validation count.")
    require((all_rows["split"] == "purged").sum() == EXPECTED["purged_samples"], "Wrong purge count.")
    require(len(skipped_lines) == EXPECTED["skipped_samples"], "Wrong skipped-window count.")
    require(summary["split"]["validation_start_index"] == EXPECTED["validation_start_index"], "Wrong split boundary.")
    require(train["stop_index"].max() < validation["start_index"].min() - 30, "Embargo violation.")
    require(all_rows["sample_name"].is_unique, "Duplicate sample names.")
    require(all_rows["sha256"].str.fullmatch(r"[0-9a-f]{64}").all(), "Malformed image hashes.")

    first = root / "images" / "sample_0_29.jpg"
    require(first.is_file(), "First reference image is missing.")
    require(digest(first) == EXPECTED["first_image_sha256"], "First image is not byte-identical to the historical reference.")

    to_check = images if args.all_image_dimensions else [images[0], images[len(images) // 2], images[-1]]
    for path in to_check:
        with Image.open(path) as image:
            require(image.size == (360, 360), f"Wrong image size: {path}")
            require(image.mode == "RGB", f"Wrong image mode: {path}")
            require(image.format == "JPEG", f"Wrong image format: {path}")
            image.verify()

    print("VERIFICATION PASSED")
    print(json.dumps({
        "source_csv_sha256": EXPECTED["source_sha256"],
        "images": len(images),
        "training": len(train),
        "validation": len(validation),
        "purged": int((all_rows["split"] == "purged").sum()),
        "skipped": len(skipped_lines),
        "first_image_sha256": digest(first),
        "all_image_dimensions_checked": args.all_image_dimensions,
    }, indent=2))


if __name__ == "__main__":
    main()
