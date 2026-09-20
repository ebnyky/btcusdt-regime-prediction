from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from embedding_models import build_embedding_model
from shared import (
    NUM_CLASSES,
    REGIME_NAMES,
    TARGET_KIND,
    compute_metrics,
    load_json,
    load_soft_class_weights,
    load_soft_manifest,
    resolve_dataset_paths,
    resolve_device,
    save_json,
    set_seed,
    soft_brier_score,
    soft_weighted_cross_entropy,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def load_embeddings(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Embeddings missing: {path}. Run extract_embeddings.py first.")
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def make_loader(data, mean, scale, batch_size, shuffle, seed):
    x = ((data["X"] - mean) / scale).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(data["soft_targets"].astype(np.float32)),
        torch.from_numpy(data["certainty_weights"].astype(np.float32)),
        torch.from_numpy(data["hard_targets"].astype(np.int64)),
        torch.from_numpy(data["row_indices"].astype(np.int64)),
    )
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def run_epoch(model, loader, device, class_weights, optimizer=None, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    total_weighted_loss, total_rows = 0.0, 0
    hard_all, pred_all, probability_all, soft_all, certainty_all, index_all = [], [], [], [], [], []
    for x, soft, certainty, hard, indices in loader:
        x, soft, certainty = x.to(device), soft.to(device), certainty.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = soft_weighted_cross_entropy(
            logits,
            soft,
            class_weights=class_weights if training else None,
            certainty_weights=certainty if training else None,
        )
        if training:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        probabilities = torch.softmax(logits.detach(), dim=1)
        count = x.shape[0]
        total_weighted_loss += float(loss.detach().cpu()) * count
        total_rows += count
        hard_all.extend(hard.tolist())
        pred_all.extend(probabilities.argmax(dim=1).cpu().tolist())
        probability_all.extend(probabilities.cpu().tolist())
        soft_all.extend(soft.detach().cpu().tolist())
        certainty_all.extend(certainty.detach().cpu().tolist())
        index_all.extend(indices.tolist())
    metrics = compute_metrics(hard_all, pred_all)
    probabilities_np = np.asarray(probability_all, dtype=float)
    soft_np = np.asarray(soft_all, dtype=float)
    metrics["loss"] = total_weighted_loss / max(1, total_rows)
    metrics["soft_brier_score"] = soft_brier_score(probabilities_np, soft_np)
    return metrics, {
        "hard_targets": hard_all,
        "predictions": pred_all,
        "probabilities": probabilities_np,
        "soft_targets": soft_np,
        "certainty_weights": certainty_all,
        "indices": index_all,
    }


def save_validation_outputs(frame, raw, metrics, output):
    rows = frame.iloc[raw["indices"]].copy().reset_index(drop=True)
    rows["actual_target_cluster_id"] = raw["hard_targets"]
    rows["predicted_target_cluster_id"] = raw["predictions"]
    for cluster in range(NUM_CLASSES):
        rows[f"predicted_probability_cluster_{cluster}"] = raw["probabilities"][:, cluster]
    rows.to_csv(output / "validation_predictions.csv", index=False)
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["confusion_matrix"]),
        display_labels=[f"{i}: {REGIME_NAMES[i]}" for i in range(NUM_CLASSES)],
    )
    display.plot(values_format="d")
    plt.title("Validation confusion matrix")
    plt.tight_layout()
    plt.savefig(output / "validation_confusion_matrix.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("soft_logistic_regression", "embedding_mlp"), required=True)
    args = parser.parse_args()
    config = load_json(PACKAGE_ROOT / "configs" / "training_config.json")
    seed = int(config.get("random_seed", 42))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    paths = resolve_dataset_paths(config)
    validation_frame = load_soft_manifest(paths.validation_manifest, "validation")
    embedding_dir = Path(config["output_dir"]).expanduser().resolve() / "frozen_resnet18_embeddings"
    data = {
        split: load_embeddings(embedding_dir / f"{split}_embeddings.npz")
        for split in ("train", "validation")
    }
    standardizer_path = embedding_dir / "training_fitted_standardizer.npz"
    if not standardizer_path.is_file():
        raise FileNotFoundError(f"Shared training-fitted standardizer missing: {standardizer_path}")
    with np.load(standardizer_path) as scaler:
        mean, scale = scaler["mean"].copy(), scaler["scale"].copy()
    output = Path(config["output_dir"]).expanduser().resolve() / args.model
    output.mkdir(parents=True, exist_ok=True)
    batch_size = int(config.get("embedding_training_batch_size", 256))
    loaders = {
        "train": make_loader(data["train"], mean, scale, batch_size, True, seed),
        "validation": make_loader(data["validation"], mean, scale, batch_size, False, seed),
    }
    model = build_embedding_model(args.model, config).to(device)
    class_weights = load_soft_class_weights(paths.weights_report).to(device)
    learning_rate = float(config.get(
        "logistic_learning_rate" if args.model == "soft_logistic_regression" else "mlp_learning_rate",
        0.001,
    ))
    weight_decay = float(config.get(
        "logistic_weight_decay" if args.model == "soft_logistic_regression" else "mlp_weight_decay",
        0.0001,
    ))
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=float(config.get("lr_factor", 0.5)),
        patience=int(config.get("lr_patience", 2)),
    )
    max_epochs = int(config.get("embedding_max_epochs", 40))
    patience = int(config.get("embedding_early_stopping_patience", 7))
    best_f1, best_epoch, best_state, wait = -math.inf, 0, None, 0
    history = []
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        train_metrics, _ = run_epoch(
            model, loaders["train"], device, class_weights, optimizer,
            float(config.get("gradient_clip_norm", 1.0)),
        )
        with torch.inference_mode():
            validation_metrics, validation_raw = run_epoch(
                model, loaders["validation"], device, class_weights=None
            )
        scheduler.step(validation_metrics["macro_f1"])
        history.append({
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        })
        print(
            f"{args.model} epoch {epoch:02d}/{max_epochs} | "
            f"train loss={train_metrics['loss']:.4f} macro-F1={train_metrics['macro_f1']:.4f} | "
            f"val loss={validation_metrics['loss']:.4f} macro-F1={validation_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if validation_metrics["macro_f1"] > best_f1 + 1e-6:
            best_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save({
                "model_state_dict": best_state,
                "model_name": args.model,
                "target_kind": TARGET_KIND,
                "num_classes": NUM_CLASSES,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_f1,
                "config": config,
            }, output / "best_model.pt")
        else:
            wait += 1
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "model_name": args.model,
            "target_kind": TARGET_KIND,
            "num_classes": NUM_CLASSES,
        }, output / "last_checkpoint.pt")
        pd.DataFrame([{
            "epoch": row["epoch"],
            "learning_rate": row["learning_rate"],
            "train_loss": row["train"]["loss"],
            "train_macro_f1": row["train"]["macro_f1"],
            "validation_loss": row["validation"]["loss"],
            "validation_macro_f1": row["validation"]["macro_f1"],
            "validation_balanced_accuracy": row["validation"]["balanced_accuracy"],
            "validation_soft_brier_score": row["validation"]["soft_brier_score"],
        } for row in history]).to_csv(output / "training_history.csv", index=False)
        if wait >= patience:
            print(f"Early stopping after epoch {epoch}.")
            break
    if best_state is None:
        raise RuntimeError("No best checkpoint was created.")
    model.load_state_dict(best_state)
    with torch.inference_mode():
        best_metrics, best_raw = run_epoch(model, loaders["validation"], device, None)
    save_json({
        "model": args.model,
        "target_kind": TARGET_KIND,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "validation": best_metrics,
        "soft_class_weights": [float(x) for x in class_weights.cpu()],
        "standardizer_fitted_on_training_only": True,
        "shared_standardizer_path": str(standardizer_path),
        "elapsed_seconds": time.time() - start,
        "test_evaluated": False,
    }, output / "training_report.json")
    save_validation_outputs(validation_frame, best_raw, best_metrics, output)
    print(f"Completed {args.model}. Test has not been evaluated. Outputs: {output}")


if __name__ == "__main__":
    main()
