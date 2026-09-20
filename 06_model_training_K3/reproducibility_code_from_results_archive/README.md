# K=3 soft-membership three-model experiment — v1.3

Version 1.3 verifies split separation using UTC timestamps and source identity.
Local indices are compared only when adjacent partitions share a source CSV;
indices that restart in the independent test CSV are never compared directly.

Version 1.2 accepts the forecasting generator's `input_image_path` field and
canonicalizes it internally to `input_image`. Source manifests are unchanged.

This Kaggle package preserves the clean structure of the original 90-to-30
supervised-learning workflow while using the validated K=3 soft-membership
method throughout.

## Models

1. Soft Logistic Regression: one linear softmax layer on frozen 512-D ResNet18 embeddings.
2. Embedding MLP: a nonlinear classifier on the identical cached embeddings.
3. ResNet18 Layer 4: `layer4` and the three-output classifier head are trainable;
   earlier layers remain frozen.

The output mapping is `0 = Bullish`, `1 = Volatile`, `2 = Bearish`.

## Class imbalance and uncertain labels

Every trainer minimizes the same training-only weighted soft-label loss:

```text
loss_i = certainty_i * [-sum_k(class_weight_k * soft_target_ik * log(probability_ik))]
```

The processor-derived class weights are inverse soft-membership-mass weights.
Preflight recomputes them from `train_soft_shuffled.csv` and stops if the saved
weights disagree. Do not undersample, oversample, use a weighted sampler, or
add ordinary hard-label class weights on top of this loss.

## Folder structure

```text
k3_soft_membership_kaggle_three_model_v1_3/
├── OPEN_ME_FIRST.txt
├── README.md
├── requirements.txt
├── configs/
│   └── training_config.json
├── notebooks/
│   └── KAGGLE_K3_SOFT_THREE_MODELS.ipynb
├── soft_supervised/
│   ├── check_setup.py
│   ├── extract_embeddings.py
│   ├── train_embedding_model.py
│   ├── train_classifier.py
│   ├── compare_validation.py
│   ├── evaluate_all_models.py
│   └── ...
└── tests/
    └── test_static_contract.py
```

## Required dataset layout

Set `dataset_dir` in `configs/training_config.json` to the directory containing:

```text
BTCUSDT_k3_forecasting_dataset/
├── images/
└── soft_membership/
    ├── manifests/
    │   ├── train_soft_shuffled.csv
    │   ├── validation_soft_annotated_chronological.csv
    │   └── test_soft_annotated_chronological.csv
    └── reports/
        └── soft_training_weights.json
```

## Kaggle workflow

Attach the processed dataset and this package, enable a GPU, open
`notebooks/KAGGLE_K3_SOFT_THREE_MODELS.ipynb`, and run cells sequentially.

The training stage performs:

```text
preflight -> train/validation embedding cache -> Soft Logistic Regression
                                             -> Embedding MLP
images --------------------------------------> Layer-4 ResNet18
                                             -> validation comparison
```

During this stage, test embeddings are not created. Checkpoints are selected
only by validation macro-F1. After the experimental choices are frozen, the
final notebook cell creates test embeddings, evaluates all three validation-best
checkpoints, writes the comparison, and creates an evaluation lock.

Command-line equivalents from the extracted package are:

```bash
python soft_supervised/check_setup.py
python -u soft_supervised/run_all_models.py
python soft_supervised/compare_validation.py
python -u soft_supervised/evaluate_all_models.py
```

Run the final command once. A second run is refused unless `--force` is used for
a deliberately documented recovery. After test evaluation, use a new
`output_dir` for any fresh experiment; do not resume tuning in the same output.

## Outputs

```text
output_dir/
├── frozen_resnet18_embeddings/
├── soft_logistic_regression/
├── embedding_mlp/
├── soft_membership_unfreeze_layer4/
├── validation_model_comparison.csv
├── independent_test_model_comparison.csv
└── TEST_EVALUATION_COMPLETE.json
```

Reports include macro-F1, balanced accuracy, accuracy, per-class precision,
recall and F1, confusion matrices, soft cross-entropy, and soft Brier score.

Soft memberships are geometric affinities to the frozen centroids; they should
not be described as calibrated probabilities of future market regimes.
