from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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

from manifest_schema import (
    canonicalize_input_image_column,
    verify_source_aware_split_boundaries,
)

NUM_CLASSES = 3
SOFT_TARGET_COLUMNS = tuple(f"target_soft_cluster_{i}" for i in range(NUM_CLASSES))
REGIME_NAMES = {0: "Bullish", 1: "Volatile", 2: "Bearish"}
TARGET_KIND = "k3_soft_membership_v1"


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    soft_membership_dir: Path
    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path
    weights_report: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


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
    configured_soft_dir = config.get("soft_membership_dir")
    soft_dir = (
        Path(configured_soft_dir).expanduser().resolve()
        if configured_soft_dir
        else dataset_dir / "soft_membership"
    )
    manifests = soft_dir / "manifests"
    return DatasetPaths(
        dataset_dir=dataset_dir,
        soft_membership_dir=soft_dir,
        train_manifest=manifests / "train_soft_shuffled.csv",
        validation_manifest=manifests / "validation_soft_annotated_chronological.csv",
        test_manifest=manifests / "test_soft_annotated_chronological.csv",
        weights_report=soft_dir / "reports" / "soft_training_weights.json",
    )


def _to_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    values = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not values.isin(mapping).all():
        bad = sorted(values.loc[~values.isin(mapping)].unique().tolist())
        raise ValueError(f"Column {column} contains invalid Boolean values: {bad[:5]}")
    return values.map(mapping).astype(bool)


def load_soft_manifest(path: Path, partition: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = canonicalize_input_image_column(pd.read_csv(path), path.name)

    required = {
        "input_image", "target_cluster_id", "input_start_index", "target_stop_index",
        "certainty_weight", *SOFT_TARGET_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest {path.name} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["target_cluster_id"] = frame["target_cluster_id"].astype(int)
    invalid = sorted(set(frame["target_cluster_id"].unique()) - set(range(NUM_CLASSES)))
    if invalid:
        raise ValueError(f"Unexpected target labels in {path.name}: {invalid}")

    soft = frame[list(SOFT_TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(soft).all() or (soft < 0).any():
        raise ValueError(f"Invalid soft targets in {path.name}.")
    if not np.allclose(soft.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"Soft targets do not sum to 1 in {path.name}.")
    certainty = frame["certainty_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(certainty).all() or (certainty <= 0).any() or (certainty > 1).any():
        raise ValueError(f"certainty_weight must be in (0, 1] in {path.name}.")

    if "training_eligible" in frame.columns:
        frame["training_eligible"] = _to_bool(frame["training_eligible"], "training_eligible")
        if partition == "train" and not frame["training_eligible"].all():
            raise ValueError("The training manifest includes ineligible distance outliers.")
    return frame


def load_soft_class_weights(path: Path) -> Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Soft class-weight report not found: {path}")
    report = load_json(path)
    values = report.get("soft_class_weights")
    if not isinstance(values, dict):
        raise ValueError("soft_training_weights.json lacks soft_class_weights.")
    weights = torch.tensor([float(values[str(i)]) for i in range(NUM_CLASSES)], dtype=torch.float32)
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Soft class weights must be positive finite values.")
    return weights


def verify_split_boundaries(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> list[dict[str, object]]:
    return verify_source_aware_split_boundaries(train, validation, test)


class SoftFutureRegimeImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, dataset_dir: Path, transform: Any) -> None:
        self.frame = frame.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, int, int]:
        row = self.frame.iloc[index]
        image_path = Path(str(row["input_image"]))
        if not image_path.is_absolute():
            image_path = self.dataset_dir / image_path
        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Could not read image: {image_path}") from exc
        soft_target = torch.tensor(
            [float(row[column]) for column in SOFT_TARGET_COLUMNS], dtype=torch.float32
        )
        certainty = torch.tensor(float(row["certainty_weight"]), dtype=torch.float32)
        return tensor, soft_target, certainty, int(row["target_cluster_id"]), index


def build_unfreeze_layer4_model(num_classes: int = NUM_CLASSES) -> tuple[nn.Module, Any]:
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
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()


def soft_weighted_cross_entropy(
    logits: Tensor,
    soft_targets: Tensor,
    class_weights: Tensor | None = None,
    certainty_weights: Tensor | None = None,
) -> Tensor:
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"Expected logits shaped [batch, {NUM_CLASSES}], got {tuple(logits.shape)}")
    if soft_targets.shape != logits.shape:
        raise ValueError("soft_targets must have the same shape as logits.")
    if not torch.allclose(
        soft_targets.sum(dim=1), torch.ones(logits.shape[0], device=logits.device), atol=1e-5
    ):
        raise ValueError("Every soft target must sum to 1.")
    terms = -soft_targets * F.log_softmax(logits, dim=1)
    if class_weights is not None:
        if class_weights.shape != (NUM_CLASSES,):
            raise ValueError(f"class_weights must have shape [{NUM_CLASSES}].")
        terms = terms * class_weights.unsqueeze(0)
    per_sample = terms.sum(dim=1)
    if certainty_weights is None:
        return per_sample.mean()
    if certainty_weights.ndim != 1 or len(certainty_weights) != len(per_sample):
        raise ValueError("certainty_weights must have shape [batch].")
    denominator = certainty_weights.sum().clamp_min(torch.finfo(per_sample.dtype).eps)
    return (per_sample * certainty_weights).sum() / denominator


def soft_brier_score(probabilities: np.ndarray, soft_targets: np.ndarray) -> float:
    return float(np.mean(np.sum((probabilities - soft_targets) ** 2, axis=1)))


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    labels = list(range(NUM_CLASSES))
    y_true_np = np.asarray(y_true, dtype=int)
    y_pred_np = np.asarray(y_pred, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_np, y_pred_np, labels=labels, zero_division=0
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
            for i in labels
        },
        "confusion_matrix": confusion_matrix(y_true_np, y_pred_np, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true_np,
            y_pred_np,
            labels=labels,
            target_names=[REGIME_NAMES[i] for i in labels],
            output_dict=True,
            zero_division=0,
        ),
    }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
