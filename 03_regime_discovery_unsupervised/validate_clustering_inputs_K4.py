"""Validate the kline corpus before extracting ResNet-18 embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


def resolve(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def main(config_path: Path) -> None:
    config_path = config_path.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = resolve(config["dataset_dir"], config_path)
    image_column = str(config.get("image_column", "exported_image"))
    summaries: dict[str, dict[str, object]] = {}
    previous_stop: int | None = None

    for split, key in (("train", "train_manifest"), ("validation", "validation_manifest")):
        manifest = dataset / str(config[key])
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing {split} manifest: {manifest}")
        frame = pd.read_csv(manifest)
        required = {"sample_name", "start_index", "stop_index", image_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{split} manifest is missing {sorted(missing)}")
        if frame.empty or not frame["start_index"].is_monotonic_increasing:
            raise ValueError(f"{split} manifest is empty or not chronological.")
        if previous_stop is not None and int(frame["start_index"].min()) <= previous_stop:
            raise ValueError("Training and validation raw candle ranges overlap.")

        bad_images: list[str] = []
        bad_size: list[str] = []
        for value in frame[image_column].astype(str):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = dataset / path
            if not path.is_file():
                bad_images.append(value)
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.size != (360, 360) or image.convert("RGB").mode != "RGB":
                        bad_size.append(value)
            except Exception:
                bad_images.append(value)
        if bad_images or bad_size:
            raise ValueError(
                f"{split}: missing/corrupt={len(bad_images)}, wrong-size={len(bad_size)}"
            )
        previous_stop = int(frame["stop_index"].max())
        summaries[split] = {
            "rows": int(len(frame)),
            "first_start_index": int(frame["start_index"].min()),
            "last_stop_index": int(frame["stop_index"].max()),
            "images_checked": int(len(frame)),
        }

    print(json.dumps(summaries, indent=2))
    print("CLUSTER INPUT VALIDATION PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    main(parser.parse_args().config)
