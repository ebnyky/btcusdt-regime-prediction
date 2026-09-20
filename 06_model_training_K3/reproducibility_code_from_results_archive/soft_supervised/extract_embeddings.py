from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.auto import tqdm

from shared import (
    SOFT_TARGET_COLUMNS,
    SoftFutureRegimeImageDataset,
    load_json,
    load_soft_manifest,
    resolve_dataset_paths,
    resolve_device,
    save_json,
    set_seed,
    verify_split_boundaries,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def loader_kwargs(config, device):
    workers = int(config.get("num_workers", 2))
    result = {
        "num_workers": workers,
        "pin_memory": bool(config.get("pin_memory", True)) and device.type == "cuda",
        "persistent_workers": bool(config.get("persistent_workers", True)) and workers > 0,
    }
    if workers > 0:
        result["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return result


def frozen_resnet18_extractor():
    weights = models.ResNet18_Weights.DEFAULT
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Identity()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    return backbone, weights.transforms()


@torch.inference_mode()
def extract_split(model, dataset, config, device, description):
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("embedding_batch_size", 128)),
        shuffle=False,
        **loader_kwargs(config, device),
    )
    features, soft, certainty, hard, row_indices = [], [], [], [], []
    amp_enabled = device.type == "cuda" and bool(config.get("amp_enabled", True))
    for images, batch_soft, batch_certainty, batch_hard, indices in tqdm(
        loader, desc=description, unit="batch"
    ):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            batch_features = model(images)
        features.append(batch_features.float().cpu().numpy())
        soft.append(batch_soft.numpy())
        certainty.append(batch_certainty.numpy())
        hard.append(batch_hard.numpy())
        row_indices.append(indices.numpy())
    return {
        "X": np.concatenate(features).astype(np.float32),
        "soft_targets": np.concatenate(soft).astype(np.float32),
        "certainty_weights": np.concatenate(certainty).astype(np.float32),
        "hard_targets": np.concatenate(hard).astype(np.int64),
        "row_indices": np.concatenate(row_indices).astype(np.int64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Extract the held-out test embeddings. Use only during the locked final evaluation.",
    )
    args = parser.parse_args()
    config = load_json(PACKAGE_ROOT / "configs" / "training_config.json")
    set_seed(int(config.get("random_seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    paths = resolve_dataset_paths(config)
    all_frames = {
        "train": load_soft_manifest(paths.train_manifest, "train"),
        "validation": load_soft_manifest(paths.validation_manifest, "validation"),
        "test": load_soft_manifest(paths.test_manifest, "test"),
    }
    verify_split_boundaries(all_frames["train"], all_frames["validation"], all_frames["test"])
    splits = ("train", "validation", "test") if args.include_test else ("train", "validation")
    frames = {split: all_frames[split] for split in splits}
    output = Path(config["output_dir"]).expanduser().resolve() / "frozen_resnet18_embeddings"
    output.mkdir(parents=True, exist_ok=True)
    model, transform = frozen_resnet18_extractor()
    model.to(device)
    for split, frame in frames.items():
        destination = output / f"{split}_embeddings.npz"
        if destination.is_file() and not args.force:
            with np.load(destination) as cached:
                valid = cached["X"].shape == (len(frame), 512)
            if valid:
                print(f"Using cached {split} embeddings: {destination}")
                continue
            raise ValueError(f"Cached embedding shape disagrees with manifest: {destination}")
        dataset = SoftFutureRegimeImageDataset(frame, paths.dataset_dir, transform)
        arrays = extract_split(model, dataset, config, device, f"Extracting {split} embeddings")
        np.savez_compressed(destination, **arrays)
        print(f"Saved {split}: {arrays['X'].shape} -> {destination}")
    standardizer_path = output / "training_fitted_standardizer.npz"
    train_path = output / "train_embeddings.npz"
    with np.load(train_path) as train_cache:
        train_x = train_cache["X"].astype(np.float32)
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-7] = 1.0
    np.savez(standardizer_path, mean=mean, scale=scale)
    save_json({
        "backbone": "torchvision ResNet18_Weights.DEFAULT",
        "embedding_dimension": 512,
        "backbone_frozen": True,
        "input_column": "input_image",
        "target_image_used_as_input": False,
        "soft_target_columns": list(SOFT_TARGET_COLUMNS),
        "extracted_splits": list(splits),
        "test_extraction_requested_only_for_final_evaluation": bool(args.include_test),
        "standardizer": "training_fitted_standardizer.npz",
        "standardizer_fitted_on_training_only": True,
    }, output / "embedding_metadata.json")
    if not args.include_test:
        print("Held-out test embeddings were not extracted during training/validation.")


if __name__ == "__main__":
    main()
