from __future__ import annotations

import copy
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from shared import (
    FutureRegimeImageDataset,
    REGIME_NAMES,
    build_resnet18_classifier,
    calculate_class_weights,
    compute_metrics,
    load_json,
    load_manifest,
    resolve_dataset_paths,
    resolve_device,
    save_json,
    set_seed,
    total_parameter_count,
    trainable_parameter_count,
    verify_split_boundaries,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


def create_loaders(
    train_frame,
    validation_frame,
    test_frame,
    dataset_dir,
    transform,
    config,
):
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = (
        resolve_device(config.get("device", "auto")).type == "cuda"
    )

    train_dataset = FutureRegimeImageDataset(
        train_frame,
        dataset_dir,
        transform,
    )
    validation_dataset = FutureRegimeImageDataset(
        validation_frame,
        dataset_dir,
        transform,
    )
    test_dataset = FutureRegimeImageDataset(
        test_frame,
        dataset_dir,
        transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, validation_loader, test_loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    grad_clip_norm,
):
    model.train()

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip_norm,
            )

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)

        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

    metrics = compute_metrics(all_targets, all_predictions)
    metrics["loss"] = running_loss / len(loader.dataset)

    return metrics


@torch.inference_mode()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_indices = []

    for images, targets, indices in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)

        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())
        all_indices.extend(indices.tolist())

    metrics = compute_metrics(all_targets, all_predictions)
    metrics["loss"] = running_loss / len(loader.dataset)

    return metrics, {
        "targets": all_targets,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "indices": all_indices,
    }


def plot_training_history(history, destination):
    epochs = range(1, len(history) + 1)

    train_loss = [row["train"]["loss"] for row in history]
    validation_loss = [
        row["validation"]["loss"] for row in history
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Training loss")
    plt.plot(epochs, validation_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination / "loss_curve.png", dpi=180)
    plt.close()

    train_f1 = [row["train"]["macro_f1"] for row in history]
    validation_f1 = [
        row["validation"]["macro_f1"] for row in history
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_f1, label="Training macro-F1")
    plt.plot(
        epochs,
        validation_f1,
        label="Validation macro-F1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training and validation macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination / "macro_f1_curve.png", dpi=180)
    plt.close()


def save_confusion_matrix(metrics, destination, title):
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[str(i) for i in range(4)],
    )

    display.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()


def predictions_frame(
    manifest,
    raw_predictions,
):
    result = manifest.iloc[
        raw_predictions["indices"]
    ].copy().reset_index(drop=True)

    result["actual_target_cluster_id"] = (
        raw_predictions["targets"]
    )
    result["predicted_target_cluster_id"] = (
        raw_predictions["predictions"]
    )

    probabilities = np.asarray(
        raw_predictions["probabilities"],
        dtype=float,
    )

    for cluster_id in range(4):
        result[f"probability_cluster_{cluster_id}"] = (
            probabilities[:, cluster_id]
        )

    result["prediction_correct"] = (
        result["actual_target_cluster_id"]
        == result["predicted_target_cluster_id"]
    )

    return result


def main():
    config = load_json(CONFIG_PATH)

    seed = int(config.get("random_seed", 42))
    set_seed(seed)

    device = resolve_device(config.get("device", "auto"))
    paths = resolve_dataset_paths(config)

    train_frame = load_manifest(paths.train_manifest)
    validation_frame = load_manifest(
        paths.validation_manifest
    )
    test_frame = load_manifest(paths.test_manifest)

    verify_split_boundaries(
        train_frame,
        validation_frame,
        test_frame,
    )

    training_mode = config.get(
        "training_mode",
        "frozen_backbone",
    )

    if training_mode not in {
        "frozen_backbone",
        "unfreeze_layer4",
        "full_finetune",
    }:
        raise ValueError(
            "training_mode must be frozen_backbone, "
            "unfreeze_layer4 or full_finetune."
        )

    freeze_backbone = training_mode != "full_finetune"
    unfreeze_layer4 = training_mode == "unfreeze_layer4"

    model, transform = build_resnet18_classifier(
        num_classes=4,
        freeze_backbone=freeze_backbone,
        unfreeze_layer4=unfreeze_layer4,
    )
    model.to(device)

    train_loader, validation_loader, test_loader = (
        create_loaders(
            train_frame,
            validation_frame,
            test_frame,
            paths.dataset_dir,
            transform,
            config,
        )
    )

    output_dir = (
        Path(config["output_dir"]).expanduser().resolve()
        / training_mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    use_class_weights = bool(
        config.get("use_class_weights", True)
    )

    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(
            train_frame["target_cluster_id"]
        ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(
            config.get("label_smoothing", 0.0)
        ),
    )

    optimizer = AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=float(config.get("learning_rate", 0.001)),
        weight_decay=float(
            config.get("weight_decay", 0.0001)
        ),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config.get("lr_factor", 0.5)),
        patience=int(config.get("lr_patience", 2)),
    )

    max_epochs = int(config.get("max_epochs", 20))
    early_stopping_patience = int(
        config.get("early_stopping_patience", 5)
    )
    grad_clip_norm = float(
        config.get("gradient_clip_norm", 1.0)
    )

    print("=" * 80)
    print("SUPERVISED FUTURE-REGIME CLASSIFIER")
    print("=" * 80)
    print(f"Training mode: {training_mode}")
    print(f"Device: {device}")
    print(f"Training rows: {len(train_frame):,}")
    print(f"Validation rows: {len(validation_frame):,}")
    print(f"Test rows: {len(test_frame):,}")
    print(
        f"Trainable parameters: "
        f"{trainable_parameter_count(model):,}"
    )
    print(
        f"Total parameters: "
        f"{total_parameter_count(model):,}"
    )

    if class_weights is not None:
        print(
            "Class weights:",
            [round(float(x), 4) for x in class_weights],
        )

    best_validation_f1 = -math.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm,
        )

        validation_metrics, _ = evaluate_model(
            model,
            validation_loader,
            criterion,
            device,
        )

        scheduler.step(validation_metrics["macro_f1"])

        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )

        print(
            f"Epoch {epoch:02d}/{max_epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"macro-F1={train_metrics['macro_f1']:.4f} | "
            f"val loss={validation_metrics['loss']:.4f} "
            f"macro-F1={validation_metrics['macro_f1']:.4f} "
            f"balanced-acc="
            f"{validation_metrics['balanced_accuracy']:.4f}"
        )

        current_f1 = validation_metrics["macro_f1"]

        if current_f1 > best_validation_f1 + 1e-6:
            best_validation_f1 = current_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": best_state,
                    "training_mode": training_mode,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_validation_f1,
                    "config": config,
                    "regime_names": REGIME_NAMES,
                },
                output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= early_stopping_patience
        ):
            print(
                "Early stopping triggered after "
                f"{epoch} epochs."
            )
            break

    if best_state is None:
        raise RuntimeError("No model checkpoint was created.")

    model.load_state_dict(best_state)

    validation_metrics, validation_raw = evaluate_model(
        model,
        validation_loader,
        criterion,
        device,
    )

    # Test is evaluated only after training and validation selection.
    test_metrics, test_raw = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
    )

    elapsed_seconds = time.time() - start_time

    final_report = {
        "training_mode": training_mode,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_f1,
        "elapsed_seconds": elapsed_seconds,
        "trainable_parameters": trainable_parameter_count(
            model
        ),
        "total_parameters": total_parameter_count(model),
        "class_weights": (
            [float(value) for value in class_weights.cpu()]
            if class_weights is not None
            else None
        ),
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
    }

    save_json(
        final_report,
        output_dir / "training_report.json",
    )

    pd.DataFrame(
        [
            {
                "epoch": row["epoch"],
                "learning_rate": row["learning_rate"],
                "train_loss": row["train"]["loss"],
                "train_macro_f1": row["train"]["macro_f1"],
                "train_balanced_accuracy": row["train"][
                    "balanced_accuracy"
                ],
                "validation_loss": row["validation"]["loss"],
                "validation_macro_f1": row["validation"][
                    "macro_f1"
                ],
                "validation_balanced_accuracy": row[
                    "validation"
                ]["balanced_accuracy"],
            }
            for row in history
        ]
    ).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    predictions_frame(
        validation_frame,
        validation_raw,
    ).to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
    )

    predictions_frame(
        test_frame,
        test_raw,
    ).to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    plot_training_history(history, output_dir)

    save_confusion_matrix(
        validation_metrics,
        output_dir / "validation_confusion_matrix.png",
        "Validation confusion matrix",
    )
    save_confusion_matrix(
        test_metrics,
        output_dir / "test_confusion_matrix.png",
        "Test confusion matrix",
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
    print(
        f"Validation macro-F1: "
        f"{validation_metrics['macro_f1']:.4f}"
    )
    print(
        f"Test macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )
    print(
        f"Test balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nTRAINING FAILED\n")
        print(traceback.format_exc())
        raise
