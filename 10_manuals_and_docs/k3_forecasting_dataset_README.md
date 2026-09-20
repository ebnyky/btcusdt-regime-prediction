# K=3 Future-Regime Forecasting Dataset Package

This package creates the supervised dataset:

```text
current 30-hour K-line image -> frozen cluster of the next 30-hour window
```

Cluster meanings are fixed as `0 = bullish`, `1 = sideways/non-directional`, and
`2 = bearish`.

## Run

1. Install Python 3.11 or 3.12.
2. Run `python -m pip install -r requirements.txt`.
3. Open `generate_k3_forecasting_dataset.py`.
4. Edit the five paths in the `USER SETTINGS` block.
5. Run `python generate_k3_forecasting_dataset.py`.

The script requires the validated frozen models:

- PCA: 120 components, 512 input features.
- K-Means: 3 clusters, centroid shape `(3, 120)`.

It rejects the historical K=4/50-component models and never calls `fit` or
`fit_transform`.

## Fixed chronological periods

- Training: `2020-01-01 00:00 UTC` through `2024-12-17 13:00 UTC`.
- Validation: `2024-12-17 14:00 UTC` through `2025-07-06 23:00 UTC`.
- Test: `2025-07-07 00:00 UTC` through `2026-07-31 23:00 UTC`.

A pair is kept only if its complete current and future windows belong to one
partition. The exact target begins one hour after the input ends. Missing or
discontinuous images are never replaced with the next available image.

Before creating the output folder, the generator verifies that the
train/validation CSV ends at `2025-07-06 23:00 UTC` and the test CSV begins at
`2025-07-07 00:00 UTC`. It rejects incorrect endpoints, overlap, a duplicated
boundary candle, or a boundary gap. The CSVs remain isolated: no forecasting
pair can use an input from one source file and a target from the other.

## Main outputs

```text
BTCUSDT_k3_forecasting_dataset/
├── images/
│   ├── train_validation/
│   └── test/
├── manifests/
│   ├── labeled_images_manifest.csv
│   ├── train_samples_chronological.csv
│   ├── train_samples_shuffled.csv
│   ├── validation_samples_chronological.csv
│   ├── validation_samples_shuffled.csv
│   ├── test_samples_chronological.csv
│   ├── test_samples_shuffled.csv
│   ├── all_samples_chronological.csv
│   ├── dataset_summary.json
│   └── duplicate_images.json
├── reports/
│   ├── cluster_distribution.csv
│   ├── training_class_weights.json
│   └── current_to_future_transition_counts.csv
├── skipped_images.txt
└── skipped_pairs.txt
```

Only manifest rows are shuffled, independently inside each already-frozen
partition, with seed 42. Image filenames and label alignment are unchanged.

Train from `train_samples_shuffled.csv`. Use the chronological validation and
test manifests for primary reporting. The validation and test distributions are
not balanced or resampled. Training-only class weights are calculated and saved
to address a dominant Cluster 1 without deleting data.

## Validate

After generation, open `validate_k3_forecasting_dataset.py`, edit
`DATASET_FOLDER_PATH`, and run:

```powershell
python validate_k3_forecasting_dataset.py
```
