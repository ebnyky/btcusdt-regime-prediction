"""PyTorch helpers for the manifests produced by this package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor


SOFT_TARGET_COLUMNS = (
    "target_soft_cluster_0",
    "target_soft_cluster_1",
    "target_soft_cluster_2",
)


def load_soft_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(SOFT_TARGET_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Soft-target columns are missing: {missing}")
    if "certainty_weight" not in frame.columns:
        raise ValueError("certainty_weight is missing from the manifest.")
    return frame


def targets_from_rows(frame: pd.DataFrame, device: str | torch.device) -> Tensor:
    return torch.as_tensor(
        frame[list(SOFT_TARGET_COLUMNS)].to_numpy(),
        dtype=torch.float32,
        device=device,
    )


def certainty_from_rows(frame: pd.DataFrame, device: str | torch.device) -> Tensor:
    return torch.as_tensor(
        frame["certainty_weight"].to_numpy(), dtype=torch.float32, device=device
    )


def load_class_weights(path: str | Path, device: str | torch.device) -> Tensor:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = report["soft_class_weights"]
    return torch.tensor([weights[str(k)] for k in range(3)], dtype=torch.float32, device=device)


def soft_weighted_cross_entropy(
    logits: Tensor,
    soft_targets: Tensor,
    class_weights: Tensor | None = None,
    certainty_weights: Tensor | None = None,
) -> Tensor:
    """Cross-entropy for three-class soft targets and optional two-level weights."""
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError(f"Expected logits shaped [batch, 3], got {tuple(logits.shape)}")
    if soft_targets.shape != logits.shape:
        raise ValueError("soft_targets must have the same shape as logits.")
    if not torch.allclose(
        soft_targets.sum(dim=1),
        torch.ones(logits.shape[0], device=logits.device),
        atol=1e-5,
    ):
        raise ValueError("Every soft target must sum to 1.")

    log_probabilities = torch.log_softmax(logits, dim=1)
    terms = -soft_targets * log_probabilities
    if class_weights is not None:
        if class_weights.shape != (3,):
            raise ValueError("class_weights must have shape [3].")
        terms = terms * class_weights.unsqueeze(0)
    per_sample = terms.sum(dim=1)

    if certainty_weights is None:
        return per_sample.mean()
    if certainty_weights.ndim != 1 or len(certainty_weights) != len(per_sample):
        raise ValueError("certainty_weights must have shape [batch].")
    denominator = certainty_weights.sum().clamp_min(torch.finfo(per_sample.dtype).eps)
    return (per_sample * certainty_weights).sum() / denominator
