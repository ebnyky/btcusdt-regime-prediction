from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader

from shared import (
    FutureRegimeImageDataset,
    build_resnet18_classifier,
    compute_metrics,
    load_json,
    load_manifest,
    resolve_dataset_paths,
    resolve_device,
    save_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    targets = []
    predictions = []
    probabilities = []
    indices = []

    for images, batch_targets, batch_indices in loader:
        logits = model(images.to(device, non_blocking=True))
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        targets.extend(batch_targets.tolist())
        predictions.extend(preds.cpu().tolist())
        probabilities.extend(probs.cpu().tolist())
        indices.extend(batch_indices.tolist())

    return compute_metrics(targets, predictions), {
        "targets": targets,
        "predictions": predictions,
        "probabilities": probabilities,
        "indices": indices,
    }


def main():
    config = load_json(CONFIG_PATH)
    device = resolve_device(config.get("device", "auto"))
    paths = resolve_dataset_paths(config)

    mode = config.get(
        "evaluation_training_mode",
        config.get("training_mode", "frozen_backbone"),
    )

    checkpoint_path = (
        Path(config["output_dir"]).expanduser().resolve()
        / mode
        / "best_model.pt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    mode = checkpoint["training_mode"]
    freeze_backbone = mode != "full_finetune"
    unfreeze_layer4 = mode == "unfreeze_layer4"

    model, transform = build_resnet18_classifier(
        num_classes=4,
        freeze_backbone=freeze_backbone,
        unfreeze_layer4=unfreeze_layer4,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_frame = load_manifest(paths.test_manifest)
    dataset = FutureRegimeImageDataset(
        test_frame,
        paths.dataset_dir,
        transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    metrics, raw = evaluate(model, loader, device)

    output_dir = checkpoint_path.parent
    save_json(
        metrics,
        output_dir / "independent_test_evaluation.json",
    )

    predictions = test_frame.iloc[raw["indices"]].copy()
    predictions["actual_target_cluster_id"] = raw["targets"]
    predictions["predicted_target_cluster_id"] = (
        raw["predictions"]
    )

    probabilities = np.asarray(raw["probabilities"])
    for cluster_id in range(4):
        predictions[f"probability_cluster_{cluster_id}"] = (
            probabilities[:, cluster_id]
        )

    predictions.to_csv(
        output_dir / "independent_test_predictions.csv",
        index=False,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(
            metrics["confusion_matrix"],
            dtype=int,
        ),
        display_labels=["0", "1", "2", "3"],
    )
    display.plot(values_format="d")
    plt.title("Independent test confusion matrix")
    plt.tight_layout()
    plt.savefig(
        output_dir
        / "independent_test_confusion_matrix.png",
        dpi=180,
    )
    plt.close()

    print("=" * 72)
    print("INDEPENDENT CHECKPOINT EVALUATION")
    print("=" * 72)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
