from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "soft_supervised"


def read(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_python_files_parse() -> None:
    for path in sorted(SCRIPTS.glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_three_soft_models_and_no_hard_loss() -> None:
    models = read("embedding_models.py")
    workflow = read("run_all_models.py")
    shared = read("shared.py")
    assert "SoftLogisticRegression" in models
    assert "EmbeddingMLP" in models
    assert 'run(SCRIPT_DIR / "train_classifier.py")' in workflow
    assert "soft_weighted_cross_entropy" in shared
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in SCRIPTS.glob("*.py"))
    assert "CrossEntropyLoss(" not in corpus
    assert "num_classes=4" not in corpus


def test_test_embeddings_are_final_stage_only() -> None:
    extractor = read("extract_embeddings.py")
    training = read("run_all_models.py")
    final_evaluation = read("evaluate_all_models.py")
    assert '"--include-test"' in extractor
    assert 'run(SCRIPT_DIR / "extract_embeddings.py")' in training
    assert '"--include-test"' not in training
    assert '"--include-test"' in final_evaluation
    assert "TEST_EVALUATION_IN_PROGRESS.json" in final_evaluation
    assert "TEST_EVALUATION_COMPLETE.json" in final_evaluation


def test_config_declares_training_only_soft_mass_weighting() -> None:
    config = json.loads((ROOT / "configs" / "training_config.json").read_text(encoding="utf-8"))
    assert config["package_version"] == "1.3"
    assert config["class_imbalance_strategy"] == "training_only_inverse_soft_membership_mass"
    assert config["training_mode"] == "unfreeze_layer4"


def test_forecasting_generator_image_path_schema_is_canonicalized() -> None:
    import sys

    sys.path.insert(0, str(SCRIPTS))
    from manifest_schema import canonicalize_input_image_column

    source = pd.DataFrame({"input_image_path": ["images/train/example.jpg"]})
    result = canonicalize_input_image_column(source, "train_soft_shuffled.csv")
    assert result.columns.tolist() == ["input_image"]
    assert result.loc[0, "input_image"] == "images/train/example.jpg"
    assert source.columns.tolist() == ["input_image_path"]

    conflict = pd.DataFrame({
        "input_image": ["images/train/a.jpg"],
        "input_image_path": ["images/train/b.jpg"],
    })
    try:
        canonicalize_input_image_column(conflict, "conflict.csv")
    except ValueError as exc:
        assert "conflicting" in str(exc)
    else:
        raise AssertionError("Conflicting aliases must be rejected.")


def _boundary_frame(source, pair, input_index, target_index, input_time, target_time):
    return pd.DataFrame({
        "pair_id": [pair],
        "input_source_id": [source],
        "target_source_id": [source],
        "input_start_index": [input_index],
        "target_stop_index": [target_index],
        "input_start_time": [input_time],
        "target_stop_time": [target_time],
    })


def test_source_aware_boundaries_allow_independent_test_index_restart() -> None:
    import sys

    sys.path.insert(0, str(SCRIPTS))
    from manifest_schema import verify_source_aware_split_boundaries

    train = _boundary_frame(
        "train_validation", "train-pair", 0, 43469,
        "2020-01-01T00:00:00Z", "2024-12-17T13:00:00Z",
    )
    validation = _boundary_frame(
        "train_validation", "validation-pair", 43470, 48303,
        "2024-12-17T14:00:00Z", "2025-07-06T23:00:00Z",
    )
    test = _boundary_frame(
        "test", "test-pair", 0, 59,
        "2025-07-07T00:00:00Z", "2025-07-09T11:00:00Z",
    )
    audit = verify_source_aware_split_boundaries(train, validation, test)
    assert audit[0]["index_check_applied"] is True
    assert audit[1]["index_check_applied"] is False
    assert str(audit[1]["gap"]) == "0 days 01:00:00"


def test_source_aware_boundaries_reject_timestamp_overlap() -> None:
    import sys

    sys.path.insert(0, str(SCRIPTS))
    from manifest_schema import verify_source_aware_split_boundaries

    train = _boundary_frame(
        "train_validation", "train-pair", 0, 10,
        "2024-01-01T00:00:00Z", "2024-01-01T10:00:00Z",
    )
    validation = _boundary_frame(
        "train_validation", "validation-pair", 11, 20,
        "2024-01-01T11:00:00Z", "2024-01-01T20:00:00Z",
    )
    test = _boundary_frame(
        "test", "test-pair", 0, 9,
        "2024-01-01T20:00:00Z", "2024-01-02T05:00:00Z",
    )
    try:
        verify_source_aware_split_boundaries(train, validation, test)
    except ValueError as exc:
        assert "final validation target" in str(exc)
    else:
        raise AssertionError("Timestamp overlap must be rejected.")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
