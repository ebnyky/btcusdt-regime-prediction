# K=3 Soft-Membership Dataset Package

This package replaces the earlier hard-label confidence-filtering package. It
post-processes the complete `BTCUSDT_k3_forecasting_dataset` folder and uses the
three frozen centroid distances already stored for every **future target**.
It does not regenerate images and never refits ResNet18, PCA, or K-Means.

If an original hard label and the smallest recorded distance disagree only by
the configured numerical near-tie tolerance, the row is retained and audited.
The source hard label is preserved, while the recorded distances remain
authoritative for soft memberships. A larger disagreement still stops the run
because it may indicate a damaged or misaligned manifest.

## What changes

The original K-Means label remains in `target_cluster_id` only for traceability
and ordinary hard-label evaluation. The training target becomes three values:

```text
target_soft_cluster_0
target_soft_cluster_1
target_soft_cluster_2
```

They sum to 1. A target almost tied between Bullish and Bearish may therefore
look like `[0.48, 0.03, 0.49]` instead of being forced to `[0, 0, 1]`.

The conversion uses all three distances:

```text
p_k = exp(-(d_k - d_min) / T) / sum_j exp(-(d_j - d_min) / T)
```

`T` is the median positive nearest-to-second-nearest distance gap in the
training partition, multiplied by `temperature_multiplier`. It is never fitted
from validation or test. These are distance-derived **soft memberships**, not
claims of calibrated market probabilities.

## How ambiguity and imbalance are handled

Ambiguous boundary targets are retained. Their uncertainty is measured by
normalized entropy:

```text
certainty = 1 - entropy(p) / log(3)
certainty_weight = floor + (1 - floor) * certainty
```

The default floor is `0.10`, so a near-uniform target contributes gently rather
than being silently deleted. Only genuine distance outliers are excluded from
training, using the configured training-only percentile inside each geometric
nearest-centroid cluster (99th percentile by default).

Class imbalance is calculated from **soft membership mass**, not forced hard
counts:

```text
class_weight_k = N / (3 * sum_i p_i,k)
```

The included PyTorch loss combines soft targets, soft class-mass weights, and
the certainty weight. Exact undersampling is deliberately not performed.

## Run

1. Install Python 3.11 or 3.12.
2. Run `python -m pip install -r requirements.txt`.
3. Edit `dataset_folder_path` in `config.json`.
4. Run `run_soft_membership.bat`.
5. Run `run_validator.bat`.

No model file is required at this stage: the source dataset already records all
three distances and the identity of the frozen model that produced them.

## Output

```text
<dataset>/soft_membership/
├── manifests/
│   ├── train_soft_chronological.csv
│   ├── train_soft_shuffled.csv
│   ├── train_distance_outliers.csv
│   ├── train_soft_annotated_chronological.csv
│   ├── validation_soft_annotated_chronological.csv
│   └── test_soft_annotated_chronological.csv
├── reports/
│   ├── soft_membership_summary.json
│   ├── soft_training_weights.json
│   ├── hard_counts_and_soft_mass.csv
│   ├── uncertainty_summary.csv
│   └── hard_label_near_tie_disagreements.csv
└── DO_NOT_TUNE_ON_TEST.txt
```

No image or original manifest is copied, deleted, or edited. Output image paths
continue to point to the original dataset.

## Training contract

Use `train_soft_shuffled.csv` and the function
`soft_weighted_cross_entropy` in `soft_label_training.py`. An ordinary hard
`CrossEntropyLoss(target_cluster_id)` would ignore this package's central idea.

Primary validation and test reporting should still include hard metrics against
the original nearest-centroid labels: macro-F1, balanced accuracy, per-class
precision/recall/F1, and confusion matrix. Also report soft cross-entropy,
Brier score, and uncertainty-stratified performance. Choose the model with
validation only and evaluate the test partition once after freezing it.

## Scientific limitation

K-Means is not probabilistic, so the memberships are geometric affinities. The
temperature is a transparent training-only scale, not a guarantee of calibrated
probabilities. This approach honestly preserves centroid ambiguity, but it does
not prove that the semantic names Bullish, Volatile, and Bearish
are perfect.
