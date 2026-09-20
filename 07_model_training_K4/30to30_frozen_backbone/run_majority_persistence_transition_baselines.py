from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from shared import (
    REGIME_NAMES,
    compute_metrics,
    load_json,
    load_manifest,
    resolve_dataset_paths,
    resolve_device,
    save_json,
    set_seed,
    verify_split_boundaries,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


class CurrentImageDataset(Dataset):
    def __init__(self, frame, dataset_dir, transform):
        self.frame = frame.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        image_path = self.dataset_dir / str(
            self.frame.iloc[index]["input_image"]
        )
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, index


def build_encoder(device):
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model, transform


def assign_current_clusters(
    frame: pd.DataFrame,
    dataset_dir: Path,
    pca: Any,
    kmeans: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    encoder, transform = build_encoder(device)
    dataset = CurrentImageDataset(frame, dataset_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    result = np.empty(len(frame), dtype=int)

    with torch.inference_mode():
        for images, indices in loader:
            embeddings = (
                encoder(images.to(device, non_blocking=True))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            reduced = pca.transform(embeddings)
            labels = kmeans.predict(reduced).astype(int)
            result[indices.numpy()] = labels

    return result


def majority_predictions(
    train_targets: np.ndarray,
    size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    counts = np.bincount(train_targets, minlength=4)
    majority_class = int(np.argmax(counts))
    predictions = np.full(size, majority_class, dtype=int)

    return predictions, {
        "majority_class": majority_class,
        "majority_regime_name": REGIME_NAMES[majority_class],
        "training_counts": counts.tolist(),
    }


def transition_mapping(
    current_train: np.ndarray,
    future_train: np.ndarray,
) -> tuple[dict[int, int], np.ndarray]:
    matrix = np.zeros((4, 4), dtype=int)

    for current, future in zip(current_train, future_train):
        matrix[int(current), int(future)] += 1

    global_majority = int(
        np.argmax(np.bincount(future_train, minlength=4))
    )

    mapping = {}
    for current_class in range(4):
        row = matrix[current_class]
        mapping[current_class] = (
            int(np.argmax(row))
            if row.sum() > 0
            else global_majority
        )

    return mapping, matrix


def apply_mapping(
    current_labels: np.ndarray,
    mapping: dict[int, int],
) -> np.ndarray:
    return np.asarray(
        [mapping[int(value)] for value in current_labels],
        dtype=int,
    )


def main():
    config = load_json(CONFIG_PATH)
    set_seed(int(config.get("random_seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    paths = resolve_dataset_paths(config)

    train = load_manifest(paths.train_manifest)
    validation = load_manifest(paths.validation_manifest)
    test = load_manifest(paths.test_manifest)
    verify_split_boundaries(train, validation, test)

    output_dir = Path(config["output_dir"]).expanduser().resolve()
    baseline_dir = output_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    train_targets = train["target_cluster_id"].to_numpy(int)
    validation_targets = validation["target_cluster_id"].to_numpy(int)
    test_targets = test["target_cluster_id"].to_numpy(int)

    all_results = {}

    # Majority baseline
    validation_majority, majority_info = majority_predictions(
        train_targets,
        len(validation),
    )
    test_majority = np.full(
        len(test),
        majority_info["majority_class"],
        dtype=int,
    )

    all_results["majority"] = {
        "details": majority_info,
        "validation": compute_metrics(
            validation_targets,
            validation_majority,
        ),
        "test": compute_metrics(test_targets, test_majority),
    }

    # Current-cluster baselines require saved PCA/K-Means.
    pca_path = Path(config["pca_model"]).expanduser().resolve()
    kmeans_path = Path(config["kmeans_model"]).expanduser().resolve()

    if not pca_path.is_file():
        raise FileNotFoundError(f"PCA model not found: {pca_path}")
    if not kmeans_path.is_file():
        raise FileNotFoundError(f"K-Means model not found: {kmeans_path}")

    pca = joblib.load(pca_path)
    kmeans = joblib.load(kmeans_path)

    batch_size = int(config.get("baseline_batch_size", 64))
    num_workers = int(config.get("num_workers", 0))

    print("Assigning current-window clusters for training...")
    current_train = assign_current_clusters(
        train,
        paths.dataset_dir,
        pca,
        kmeans,
        device,
        batch_size,
        num_workers,
    )

    print("Assigning current-window clusters for validation...")
    current_validation = assign_current_clusters(
        validation,
        paths.dataset_dir,
        pca,
        kmeans,
        device,
        batch_size,
        num_workers,
    )

    print("Assigning current-window clusters for testing...")
    current_test = assign_current_clusters(
        test,
        paths.dataset_dir,
        pca,
        kmeans,
        device,
        batch_size,
        num_workers,
    )

    # Persistence baseline
    all_results["persistence"] = {
        "validation": compute_metrics(
            validation_targets,
            current_validation,
        ),
        "test": compute_metrics(
            test_targets,
            current_test,
        ),
    }

    # Transition baseline
    mapping, matrix = transition_mapping(
        current_train,
        train_targets,
    )

    validation_transition = apply_mapping(
        current_validation,
        mapping,
    )
    test_transition = apply_mapping(
        current_test,
        mapping,
    )

    all_results["most_frequent_transition"] = {
        "mapping": {
            str(key): int(value)
            for key, value in mapping.items()
        },
        "transition_matrix_training": matrix.tolist(),
        "validation": compute_metrics(
            validation_targets,
            validation_transition,
        ),
        "test": compute_metrics(
            test_targets,
            test_transition,
        ),
    }

    save_json(
        all_results,
        baseline_dir / "baseline_results.json",
    )

    summary_rows = []
    for baseline_name, values in all_results.items():
        for split_name in ("validation", "test"):
            metrics = values[split_name]
            summary_rows.append(
                {
                    "baseline": baseline_name,
                    "split": split_name,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics[
                        "balanced_accuracy"
                    ],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        baseline_dir / "baseline_summary.csv",
        index=False,
    )

    print("\nBaseline results")
    print("-" * 72)
    for row in summary_rows:
        print(
            f"{row['baseline']:28s} "
            f"{row['split']:10s} "
            f"macro-F1={row['macro_f1']:.4f} "
            f"balanced-acc={row['balanced_accuracy']:.4f} "
            f"accuracy={row['accuracy']:.4f}"
        )

    print(f"\nSaved results to: {baseline_dir}")


if __name__ == "__main__":
    main()
