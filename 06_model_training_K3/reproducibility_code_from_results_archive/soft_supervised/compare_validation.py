from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared import load_json

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODELS = (
    ("soft_logistic_regression", "Soft Logistic Regression"),
    ("embedding_mlp", "Embedding MLP"),
    ("soft_membership_unfreeze_layer4", "ResNet18 Layer 4"),
)


def main():
    config = load_json(PACKAGE_ROOT / "configs" / "training_config.json")
    output_root = Path(config["output_dir"]).expanduser().resolve()
    rows = []
    for directory, display_name in MODELS:
        report_path = output_root / directory / "training_report.json"
        report = load_json(report_path)
        metrics = report["validation"]
        rows.append({
            "model": display_name,
            "best_epoch": report["best_epoch"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
            "validation_accuracy": metrics["accuracy"],
            "validation_soft_cross_entropy": metrics.get("soft_cross_entropy", metrics.get("loss")),
            "validation_soft_brier_score": metrics["soft_brier_score"],
            "test_evaluated": bool(report.get("test_evaluated", False)),
        })
    comparison = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    comparison.to_csv(output_root / "validation_model_comparison.csv", index=False)
    print(comparison.to_string(index=False))
    print("\nRanking is based only on validation macro-F1. Test remains untouched.")


if __name__ == "__main__":
    main()
