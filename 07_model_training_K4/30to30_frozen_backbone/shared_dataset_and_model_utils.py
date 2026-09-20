from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import models


REGIME_NAMES = {
    0: "High-volatility consolidation",
    1: "Bullish trend and breakout",
    2: "Volatile sideways and reversal",
    3: "Bearish trend and breakdown",
}


@dataclass
class DatasetPaths:
    dataset_dir: Path
    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(value: str) -> torch.device:
    value = value.strip().lower()

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(value)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return device


def resolve_dataset_paths(config: dict[str, Any]) -> DatasetPaths:
    dataset_dir = Path(config["dataset_dir"]).expanduser().resolve()
    manifests_dir = dataset_dir / "manifests"

    return DatasetPaths(
        dataset_dir=dataset_dir,
        train_manifest=manifests_dir / "train_samples.csv",
        validation_manifest=manifests_dir / "validation_samples.csv",
        test_manifest=manifests_dir / "test_samples.csv",
    )


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    frame = pd.read_csv(path)

    required_columns = {
        "input_image",
        "target_cluster_id",
        "input_start_index",
        "target_stop_index",
    }

    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"Manifest {path.name} is missing columns: {sorted(missing)}"
        )

    frame = frame.copy()
    frame["target_cluster_id"] = frame["target_cluster_id"].astype(int)

    invalid_labels = sorted(
        set(frame["target_cluster_id"].unique()) - {0, 1, 2, 3}
    )
    if invalid_labels:
        raise ValueError(
            f"Unexpected target labels in {path.name}: {invalid_labels}"
        )

    return frame


def verify_split_boundaries(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    train_target_stop = int(train["target_stop_index"].max())
    validation_input_start = int(validation["input_start_index"].min())
    validation_target_stop = int(validation["target_stop_index"].max())
    test_input_start = int(test["input_start_index"].min())

    if train_target_stop >= validation_input_start:
        raise ValueError(
            "Leakage detected: training target windows overlap validation inputs."
        )

    if validation_target_stop >= test_input_start:
        raise ValueError(
            "Leakage detected: validation target windows overlap test inputs."
        )


class FutureRegimeImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        dataset_dir: Path,
        transform: Any,
        image_column: str = "input_image",
        target_column: str = "target_cluster_id",
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.image_column = image_column
        self.target_column = target_column

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        row = self.frame.iloc[index]
        image_path = self.dataset_dir / str(row[self.image_column])

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Could not read image: {image_path}") from exc

        target = int(row[self.target_column])

        return image_tensor, target, index


def build_resnet18_classifier(
    num_classes: int = 4,
    freeze_backbone: bool = True,
    unfreeze_layer4: bool = False,
) -> tuple[nn.Module, Any]:
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    if unfreeze_layer4:
        for parameter in model.layer4.parameters():
            parameter.requires_grad_(True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)

    return model, transform


def calculate_class_weights(
    targets: pd.Series,
    num_classes: int = 4,
) -> torch.Tensor:
    counts = np.bincount(targets.to_numpy(int), minlength=num_classes)
    if (counts == 0).any():
        raise ValueError(
            f"At least one class has zero training samples: {counts.tolist()}"
        )

    total = counts.sum()
    weights = total / (num_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=[REGIME_NAMES[i] for i in range(4)],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": {
            str(i): {
                "regime_name": REGIME_NAMES[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(4)
        },
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3],
        ).tolist(),
        "classification_report": report,
    }


def count_missing_images(
    frame: pd.DataFrame,
    dataset_dir: Path,
    image_column: str = "input_image",
) -> int:
    return sum(
        not (dataset_dir / str(value)).is_file()
        for value in frame[image_column]
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def total_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
