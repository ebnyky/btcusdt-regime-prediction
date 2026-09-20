# K=3 Geometric Confidence and Training-Balance Package

This package post-processes the complete `BTCUSDT_k3_forecasting_dataset`
folder. It does not regenerate images, refit a model, relabel an ambiguous
sample, or alter any original file.

The processor applies two target-quality rules to every forecasting pair:

1. **Centroid-separation rule.** For an assigned target cluster `a` and each
   competing cluster `j`, it calculates

   ```text
   q_j = (distance_to_j - distance_to_a) / distance_between_centroids_a_and_j
   geometric_confidence = minimum q_j over both competing clusters
   ```

   A target passes when `geometric_confidence >= confidence_threshold`.
   The default threshold is `0.5`, the experimental candidate agreed for this
   project. At an assigned centroid the score is 1; near a decision boundary it
   approaches 0.

2. **Cluster-membership rule.** The nearest distance must not exceed the
   assigned cluster's training-only 95th-percentile distance. The same frozen
   training-derived limit is then applied to validation and test.

A rejected target is marked ambiguous/outlying. It is never changed to another
cluster. Because the future image supplies the supervised target, a failed
future target rejects the whole training pair.

## Run

1. Install Python 3.11 or 3.12.
2. Run `python -m pip install -r requirements.txt`.
3. Edit only `dataset_folder_path` in `config.json`.
4. Double-click `run_balance.bat`, or run:

   ```powershell
   python balance_k3_forecasting_dataset.py --config config.json
   ```

The script first searches for K-Means in these locations:

1. `<dataset>/models/kmeans.joblib`
2. `<dataset>/kmeans.joblib`
3. the original `kmeans_path` recorded in `manifests/dataset_summary.json`

If none exists, copy the validated K=3 `kmeans.joblib` into
`<dataset>/models/`. PCA and ResNet18 are not required because the original
manifests already contain the three frozen centroid distances.

## Output

The script creates `<dataset>/geometric_balance/`:

```text
geometric_balance/
├── manifests/
│   ├── train_confident_chronological.csv
│   ├── train_confident_shuffled.csv
│   ├── train_rejected.csv
│   ├── train_balanced_chronological.csv
│   ├── train_balanced_shuffled.csv
│   ├── train_balance_excluded.csv
│   ├── validation_confident_chronological.csv
│   ├── validation_rejected.csv
│   ├── test_confident_chronological.csv
│   └── test_rejected.csv
├── reports/
│   ├── balance_summary.json
│   ├── class_distribution_before_after.csv
│   ├── confidence_threshold_sweep.csv
│   ├── confident_training_class_weights.json
│   ├── centroid_distance_matrix.csv
│   └── rejection_reasons.csv
└── DO_NOT_TUNE_ON_TEST.txt
```

No image is copied or deleted. Every output manifest retains the original image
paths relative to the dataset root.

## Which training manifest should be used?

The recommended primary experiment is:

- train: `train_confident_shuffled.csv`
- loss: weights in `confident_training_class_weights.json`
- validation: the original `validation_samples_chronological.csv`
- final test: the original `test_samples_chronological.csv`

This retains every geometrically reliable training target. For a second
experiment, `train_balanced_shuffled.csv` contains exactly the same number of
targets from clusters 0, 1, and 2. It uses deterministic temporal-spread
undersampling, selecting the highest-confidence row inside each time segment,
so the dominant class is not taken from only one market period.

The original validation and test manifests remain the primary evaluation sets.
Their confident subsets are secondary selective-evaluation views only. The
threshold-sweep report contains training and validation only; test is processed
once at the configured locked threshold. Never choose the threshold from test
results.

## Validate

After processing, run:

```powershell
python validate_balanced_dataset.py --config config.json
```

The validator checks dataset identity, manifest conservation, disjoint accepted
and rejected rows, exact class balance, confidence rules, training-only distance
limits, shuffle alignment, and preservation of the original manifests.
