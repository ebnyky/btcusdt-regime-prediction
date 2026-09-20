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

@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
    manifests = dataset_dir / "manifests"
    return DatasetPaths(
        dataset_dir=dataset_dir,
        train_manifest=manifests / "train_samples.csv",
        validation_manifest=manifests / "validation_samples.csv",
        test_manifest=manifests / "test_samples.csv",
    )


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = pd.read_csv(path)
    required = {"input_image", "target_cluster_id", "input_start_index", "target_stop_index"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest {path.name} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["target_cluster_id"] = frame["target_cluster_id"].astype(int)
    invalid = sorted(set(frame["target_cluster_id"].unique()) - {0, 1, 2, 3})
    if invalid:
        raise ValueError(f"Unexpected target labels in {path.name}: {invalid}")
    return frame


def verify_split_boundaries(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    if int(train["target_stop_index"].max()) >= int(validation["input_start_index"].min()):
        raise ValueError("Leakage detected: training targets overlap validation inputs.")
    if int(validation["target_stop_index"].max()) >= int(test["input_start_index"].min()):
        raise ValueError("Leakage detected: validation targets overlap test inputs.")


class FutureRegimeImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, dataset_dir: Path, transform: Any) -> None:
        self.frame = frame.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        row = self.frame.iloc[index]
        image_path = self.dataset_dir / str(row["input_image"])
        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Could not read image: {image_path}") from exc
        return tensor, int(row["target_cluster_id"]), index


def build_unfreeze_layer4_model(num_classes: int = 4) -> tuple[nn.Module, Any]:
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    model = models.resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.layer4.parameters():
        parameter.requires_grad_(True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)
    return model, transform


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Keep BatchNorm in frozen blocks fixed while allowing layer4 BatchNorm to adapt."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            trainable = any(parameter.requires_grad for parameter in module.parameters())
            if not trainable:
                module.eval()


def calculate_class_weights(targets: pd.Series, num_classes: int = 4) -> torch.Tensor:
    counts = np.bincount(targets.to_numpy(int), minlength=num_classes)
    if (counts == 0).any():
        raise ValueError(f"At least one class has zero training samples: {counts.tolist()}")
    weights = counts.sum() / (num_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    y_true_np = np.asarray(y_true, dtype=int)
    y_pred_np = np.asarray(y_pred, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_np, y_pred_np, labels=[0, 1, 2, 3], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
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
        "confusion_matrix": confusion_matrix(y_true_np, y_pred_np, labels=[0, 1, 2, 3]).tolist(),
        "classification_report": classification_report(
            y_true_np,
            y_pred_np,
            labels=[0, 1, 2, 3],
            target_names=[REGIME_NAMES[i] for i in range(4)],
            output_dict=True,
            zero_division=0,
        ),
    }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
