from __future__ import annotations

import copy
import math
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

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
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from shared import (
    NUM_CLASSES,
    REGIME_NAMES,
    SoftFutureRegimeImageDataset,
    TARGET_KIND,
    build_unfreeze_layer4_model,
    compute_metrics,
    load_json,
    load_soft_class_weights,
    load_soft_manifest,
    resolve_dataset_paths,
    resolve_device,
    save_json,
    set_frozen_batchnorm_eval,
    set_seed,
    soft_brier_score,
    soft_weighted_cross_entropy,
    total_parameter_count,
    trainable_parameter_count,
    verify_split_boundaries,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PACKAGE_ROOT / "configs" / "training_config.json"


class FixedBatchSampler(Sampler[list[int]]):
    def __init__(self, order: list[int], batch_size: int, start_batch: int = 0) -> None:
        self.batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        self.start_batch = start_batch

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches[self.start_batch:]

    def __len__(self) -> int:
        return max(0, len(self.batches) - self.start_batch)


def epoch_order(dataset_size: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return torch.randperm(dataset_size, generator=generator).tolist()


def loader_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    workers = int(config.get("num_workers", 2))
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(config.get("pin_memory", True)) and device.type == "cuda",
        "persistent_workers": bool(config.get("persistent_workers", True)) and workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return kwargs


def make_eval_loader(dataset, config, device):
    return DataLoader(
        dataset,
        batch_size=int(config.get("evaluation_batch_size", config.get("batch_size", 16))),
        shuffle=False,
        **loader_kwargs(config, device),
    )


def save_atomic_torch(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_intra_epoch_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    next_batch: int,
    order: list[int],
    running_loss: float,
    processed_samples: int,
    targets: list[int],
    predictions: list[int],
    history: list[dict[str, Any]],
    best_validation_f1: float,
    best_epoch: int,
    best_state: dict[str, Any] | None,
    epochs_without_improvement: int,
    config: dict[str, Any],
) -> None:
    save_atomic_torch(
        {
            "kind": "intra_epoch",
            "training_mode": "unfreeze_layer4",
            "target_kind": TARGET_KIND,
            "num_classes": NUM_CLASSES,
            "epoch": epoch,
            "next_batch": next_batch,
            "order": order,
            "running_loss": running_loss,
            "processed_samples": processed_samples,
            "targets": targets,
            "predictions": predictions,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "best_validation_f1": best_validation_f1,
            "best_epoch": best_epoch,
            "best_state_dict": best_state,
            "epochs_without_improvement": epochs_without_improvement,
            "rng_state": capture_rng_state(),
            "config": config,
        },
        path,
    )


def train_one_epoch(
    model,
    dataset,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    config,
    epoch,
    max_epochs,
    order,
    start_batch,
    running_loss,
    processed_samples,
    all_targets,
    all_predictions,
    intra_path,
    history,
    best_validation_f1,
    best_epoch,
    best_state,
    epochs_without_improvement,
):
    model.train()
    if bool(config.get("keep_frozen_batchnorm_eval", True)):
        set_frozen_batchnorm_eval(model)

    batch_size = int(config.get("batch_size", 16))
    sampler = FixedBatchSampler(order, batch_size, start_batch)
    loader = DataLoader(dataset, batch_sampler=sampler, **loader_kwargs(config, device))
    checkpoint_every = max(1, int(config.get("checkpoint_every_batches", 100)))
    grad_clip = float(config.get("gradient_clip_norm", 1.0))
    amp_enabled = bool(config.get("amp_enabled", True)) and device.type == "cuda"

    total_batches = math.ceil(len(order) / batch_size)
    progress = tqdm(
        loader,
        total=total_batches - start_batch,
        initial=0,
        desc=f"Epoch {epoch:02d}/{max_epochs} training (resume batch {start_batch})",
        unit="batch",
    )

    for local_batch, (images, soft_targets, certainty_weights, hard_targets, _) in enumerate(progress):
        absolute_batch = start_batch + local_batch
        images = images.to(device, non_blocking=True)
        soft_targets = soft_targets.to(device, non_blocking=True)
        certainty_weights = certainty_weights.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, soft_targets, certainty_weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        predictions = logits.argmax(dim=1)
        batch_n = images.size(0)
        running_loss += loss.item() * batch_n
        processed_samples += batch_n
        all_targets.extend(hard_targets.tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())
        progress.set_postfix(loss=f"{running_loss / max(1, processed_samples):.4f}")

        next_batch = absolute_batch + 1
        if next_batch % checkpoint_every == 0:
            save_intra_epoch_checkpoint(
                intra_path, model, optimizer, scheduler, scaler, epoch, next_batch,
                order, running_loss, processed_samples, all_targets, all_predictions,
                history, best_validation_f1, best_epoch, best_state,
                epochs_without_improvement, config,
            )

    metrics = compute_metrics(all_targets, all_predictions)
    metrics["loss"] = running_loss / len(dataset)
    return metrics


@torch.inference_mode()
def evaluate_model(model, loader, device, description):
    model.eval()
    running_loss = 0.0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    probabilities_all: list[list[float]] = []
    soft_targets_all: list[list[float]] = []
    certainty_all: list[float] = []
    indices_all: list[int] = []
    amp_enabled = device.type == "cuda"
    for images, soft_targets, certainty_weights, hard_targets, indices in tqdm(loader, desc=description, unit="batch"):
        images = images.to(device, non_blocking=True)
        soft_targets = soft_targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = soft_weighted_cross_entropy(logits, soft_targets)
        # AMP may return float16 logits. Compute reported probabilities in
        # float32 so saved rows sum to one without half-precision tie rounding.
        logits_float = logits.float()
        probabilities = torch.softmax(logits_float, dim=1)
        predictions = logits_float.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        targets_all.extend(hard_targets.tolist())
        predictions_all.extend(predictions.cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())
        soft_targets_all.extend(soft_targets.cpu().tolist())
        certainty_all.extend(certainty_weights.tolist())
        indices_all.extend(indices.tolist())
    metrics = compute_metrics(targets_all, predictions_all)
    metrics["loss"] = running_loss / len(loader.dataset)
    metrics["soft_cross_entropy"] = metrics["loss"]
    metrics["soft_brier_score"] = soft_brier_score(
        np.asarray(probabilities_all, dtype=float), np.asarray(soft_targets_all, dtype=float)
    )
    return metrics, {
        "targets": targets_all,
        "predictions": predictions_all,
        "probabilities": probabilities_all,
        "soft_targets": soft_targets_all,
        "certainty_weights": certainty_all,
        "indices": indices_all,
    }


def save_confusion(metrics, path, title):
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["confusion_matrix"], dtype=int),
        display_labels=[f"{i}: {REGIME_NAMES[i]}" for i in range(NUM_CLASSES)],
    )
    display.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_history(history, output_dir):
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train"]["loss"] for row in history], label="Training loss")
    plt.plot(epochs, [row["validation"]["loss"] for row in history], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=180); plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train"]["macro_f1"] for row in history], label="Training macro-F1")
    plt.plot(epochs, [row["validation"]["macro_f1"] for row in history], label="Validation macro-F1")
    plt.xlabel("Epoch"); plt.ylabel("Macro-F1"); plt.legend(); plt.tight_layout()
    plt.savefig(output_dir / "macro_f1_curve.png", dpi=180); plt.close()


def prediction_frame(manifest, raw):
    result = manifest.iloc[raw["indices"]].copy().reset_index(drop=True)
    result["actual_target_cluster_id"] = raw["targets"]
    result["predicted_target_cluster_id"] = raw["predictions"]
    probabilities = np.asarray(raw["probabilities"], dtype=float)
    for cluster in range(NUM_CLASSES):
        result[f"probability_cluster_{cluster}"] = probabilities[:, cluster]
    result["prediction_correct"] = result["actual_target_cluster_id"] == result["predicted_target_cluster_id"]
    return result


def main():
    config = load_json(CONFIG_PATH)
    if config.get("training_mode") != "unfreeze_layer4":
        raise ValueError("This package is fixed to training_mode='unfreeze_layer4'.")

    seed = int(config.get("random_seed", 42))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    paths = resolve_dataset_paths(config)
    train_frame = load_soft_manifest(paths.train_manifest, "train")
    validation_frame = load_soft_manifest(paths.validation_manifest, "validation")
    test_frame = load_soft_manifest(paths.test_manifest, "test")
    verify_split_boundaries(train_frame, validation_frame, test_frame)

    model, transform = build_unfreeze_layer4_model(num_classes=NUM_CLASSES)
    model.to(device)
    train_dataset = SoftFutureRegimeImageDataset(train_frame, paths.dataset_dir, transform)
    validation_dataset = SoftFutureRegimeImageDataset(validation_frame, paths.dataset_dir, transform)
    validation_loader = make_eval_loader(validation_dataset, config, device)

    output_dir = Path(config["output_dir"]).expanduser().resolve() / "soft_membership_unfreeze_layer4"
    output_dir.mkdir(parents=True, exist_ok=True)
    intra_path = output_dir / "intra_epoch_checkpoint.pt"
    last_path = output_dir / "last_checkpoint.pt"

    class_weights = load_soft_class_weights(paths.weights_report).to(device)
    def criterion(logits, soft_targets, certainty_weights):
        return soft_weighted_cross_entropy(
            logits, soft_targets, class_weights=class_weights,
            certainty_weights=certainty_weights,
        )
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config.get("learning_rate", 0.0001)),
        weight_decay=float(config.get("weight_decay", 0.0001)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config.get("lr_factor", 0.5)),
        patience=int(config.get("lr_patience", 2)),
    )
    amp_enabled = bool(config.get("amp_enabled", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    max_epochs = int(config.get("max_epochs", 20))
    patience = int(config.get("early_stopping_patience", 5))
    best_validation_f1 = -math.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    resume_intra = None

    if bool(config.get("resume_training", True)) and intra_path.is_file():
        resume_intra = torch.load(intra_path, map_location=device)
        if resume_intra.get("target_kind") != TARGET_KIND or resume_intra.get("num_classes") != NUM_CLASSES:
            raise ValueError("Existing intra-epoch checkpoint is not a K=3 soft-membership checkpoint.")
        model.load_state_dict(resume_intra["model_state_dict"])
        optimizer.load_state_dict(resume_intra["optimizer_state_dict"])
        scheduler.load_state_dict(resume_intra["scheduler_state_dict"])
        scaler.load_state_dict(resume_intra["scaler_state_dict"])
        history = resume_intra.get("history", [])
        best_validation_f1 = float(resume_intra.get("best_validation_f1", -math.inf))
        best_epoch = int(resume_intra.get("best_epoch", 0))
        best_state = resume_intra.get("best_state_dict")
        epochs_without_improvement = int(resume_intra.get("epochs_without_improvement", 0))
        restore_rng_state(resume_intra.get("rng_state"))
        start_epoch = int(resume_intra["epoch"])
        print(f"Resuming inside epoch {start_epoch}, batch {resume_intra['next_batch']}.", flush=True)
    elif bool(config.get("resume_training", True)) and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device)
        if checkpoint.get("target_kind") != TARGET_KIND or checkpoint.get("num_classes") != NUM_CLASSES:
            raise ValueError("Existing last checkpoint is not a K=3 soft-membership checkpoint.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        history = checkpoint.get("history", [])
        best_validation_f1 = float(checkpoint.get("best_validation_f1", -math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_state = checkpoint.get("best_state_dict")
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming at epoch {start_epoch}.", flush=True)

    print("=" * 80)
    print("KAGGLE K=3 SOFT-MEMBERSHIP UNFREEZE_LAYER4 CLASSIFIER")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Training rows: {len(train_frame):,}")
    print(f"Validation rows: {len(validation_frame):,}")
    print(f"Test rows held out: {len(test_frame):,}")
    print(f"Trainable parameters: {trainable_parameter_count(model):,}")
    print(f"Total parameters: {total_parameter_count(model):,}")
    print(f"AMP enabled: {amp_enabled}")
    print("Soft class-mass weights:", [round(float(x), 5) for x in class_weights])

    start_time = time.time()
    for epoch in range(start_epoch, max_epochs + 1):
        if resume_intra is not None and int(resume_intra["epoch"]) == epoch:
            order = list(resume_intra["order"])
            start_batch = int(resume_intra["next_batch"])
            running_loss = float(resume_intra["running_loss"])
            processed_samples = int(resume_intra["processed_samples"])
            targets_acc = list(resume_intra["targets"])
            predictions_acc = list(resume_intra["predictions"])
            resume_intra = None
        else:
            order = epoch_order(len(train_dataset), seed, epoch)
            start_batch = 0
            running_loss = 0.0
            processed_samples = 0
            targets_acc = []
            predictions_acc = []

        train_metrics = train_one_epoch(
            model, train_dataset, criterion, optimizer, scheduler, scaler, device,
            config, epoch, max_epochs, order, start_batch, running_loss,
            processed_samples, targets_acc, predictions_acc, intra_path, history,
            best_validation_f1, best_epoch, best_state, epochs_without_improvement,
        )
        if intra_path.exists():
            intra_path.unlink()

        validation_metrics, validation_raw = evaluate_model(
            model, validation_loader, device,
            f"Epoch {epoch:02d}/{max_epochs} validation",
        )
        scheduler.step(validation_metrics["macro_f1"])
        history.append({
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        })
        print(
            f"Epoch {epoch:02d}/{max_epochs} | train loss={train_metrics['loss']:.4f} "
            f"macro-F1={train_metrics['macro_f1']:.4f} | val loss={validation_metrics['loss']:.4f} "
            f"macro-F1={validation_metrics['macro_f1']:.4f} balanced-acc={validation_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

        current_f1 = validation_metrics["macro_f1"]
        if current_f1 > best_validation_f1 + 1e-6:
            best_validation_f1 = current_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            save_atomic_torch({
                "model_state_dict": best_state,
                "training_mode": "unfreeze_layer4",
                "target_kind": TARGET_KIND,
                "num_classes": NUM_CLASSES,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_validation_f1,
                "config": config,
                "regime_names": REGIME_NAMES,
            }, output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        save_json({"history": history, "last_completed_epoch": epoch}, output_dir / "training_progress.json")
        save_atomic_torch({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "history": history,
            "training_mode": "unfreeze_layer4",
            "target_kind": TARGET_KIND,
            "num_classes": NUM_CLASSES,
            "config": config,
            "best_validation_f1": best_validation_f1,
            "best_epoch": best_epoch,
            "best_state_dict": best_state,
            "epochs_without_improvement": epochs_without_improvement,
        }, last_path)

        pd.DataFrame([{
            "epoch": row["epoch"],
            "learning_rate": row["learning_rate"],
            "train_loss": row["train"]["loss"],
            "train_accuracy": row["train"]["accuracy"],
            "train_macro_f1": row["train"]["macro_f1"],
            "train_balanced_accuracy": row["train"]["balanced_accuracy"],
            "validation_loss": row["validation"]["loss"],
            "validation_soft_brier_score": row["validation"]["soft_brier_score"],
            "validation_accuracy": row["validation"]["accuracy"],
            "validation_macro_f1": row["validation"]["macro_f1"],
            "validation_balanced_accuracy": row["validation"]["balanced_accuracy"],
        } for row in history]).to_csv(output_dir / "training_history.csv", index=False)
        plot_history(history, output_dir)
        prediction_frame(validation_frame, validation_raw).to_csv(
            output_dir / "latest_validation_predictions.csv", index=False
        )
        save_confusion(validation_metrics, output_dir / "latest_validation_confusion_matrix.png", "Latest validation confusion matrix")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after epoch {epoch}.", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No best model was created.")
    model.load_state_dict(best_state)
    best_validation_metrics, best_validation_raw = evaluate_model(
        model, validation_loader, device, "Best-checkpoint validation evaluation"
    )
    save_json({
        "training_mode": "unfreeze_layer4",
        "target_kind": TARGET_KIND,
        "num_classes": NUM_CLASSES,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_f1,
        "elapsed_seconds": time.time() - start_time,
        "trainable_parameters": trainable_parameter_count(model),
        "total_parameters": total_parameter_count(model),
        "soft_class_weights": [float(x) for x in class_weights.cpu()],
        "validation": best_validation_metrics,
        "history": history,
        "test_evaluated": False,
    }, output_dir / "training_report.json")
    prediction_frame(validation_frame, best_validation_raw).to_csv(
        output_dir / "validation_predictions.csv", index=False
    )
    save_confusion(best_validation_metrics, output_dir / "validation_confusion_matrix.png", "Best-checkpoint validation confusion matrix")
    print("\nTraining and validation selection complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_validation_metrics['macro_f1']:.4f}")
    print("The test set has not been evaluated. Run evaluate_checkpoint.py once.")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nTRAINING FAILED\n")
        print(traceback.format_exc())
        raise
