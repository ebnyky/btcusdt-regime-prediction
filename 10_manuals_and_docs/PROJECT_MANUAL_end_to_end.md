# Project manual

## 1. Environment

Create a virtual environment and install the root requirements:

```bash
python -m pip install -r requirements.txt
```

PyTorch/torchvision may require a platform-specific installation command.
The first pretrained ResNet-18 run may download ImageNet weights.

Run the fast reconstruction checks before generating the full dataset:

```bash
python verify_reconstruction.py
```

## 2. Original kline dataset

Place the hourly CSV at `data/BTCUSDT_1h.csv`, or edit
`kline_dataset_config.json`. Then run from the project root:

```bash
python kline_dataset_generator.py
```

The configured output contains exactly the required top-level artefacts:

```text
outputs/01_kline_dataset/
├── images/
├── manifests/
└── skipped_samples.txt
```

The manifests folder contains `all_samples.csv`, `train_samples.csv`,
`validation_samples.csv`, `dataset_summary.json`, and
`duplicate_images.json`. Image names follow `sample_<start>_<stop>.jpg`.

The renderer is the recovered original implementation. The attached reference
JPEG records 90 DPI, so the active configuration uses 90 DPI while retaining
360×360 output. This setting was verified byte-for-byte using the first 30
official Binance BTCUSDT hourly candles from 1 January 2020. The exact generated
SHA-256 equals the supplied reference hash. The verbatim historical source with its 100-DPI constant is
kept under `historical_source/`. Do not replace this with the later explicit-
axis 90-to-30 renderer when reproducing the original clustering experiment.

## 3. Unsupervised cluster discovery

Edit `cluster_regime_package/config.json`, particularly `dataset_dir` and
`output_dir`, then run:

```bash
cd cluster_regime_package
python run_unsupervised_regime_discovery.py
```

The pipeline reports silhouette score, Davies-Bouldin index,
Calinski-Harabasz score, inertia, cluster balance, and stability. It saves the
PCA and K-Means models plus cluster labels, plots, representative images and
montages. PCA and K-Means are fitted on training data only.

## 4. Supervised dataset creation

Two historically distinct generators are retained:

- `supervised_learning_package/dataset_generation/30_to_30/`
- `supervised_learning_package/dataset_generation/90_to_30/`

Edit the relevant `config.json`, including the CSV, PCA model, K-Means model,
and output directory, then run `generate_future_regime_dataset.py`. Validate
the output using `validate_generated_dataset.py` before training.

The 30-to-30 task is:

```text
[t, t+29] -> cluster([t+30, t+59])
```

The corrected 90-to-30 task is:

```text
[t, t+89] -> cluster([t+90, t+119])
```

The 90-to-30 renderer is intentionally retained as executed, even though its
explicit candle width and plot padding differ from the earliest renderer.

## 5. Supervised modelling and evaluation

The training packages are stored in historical order:

1. `training/30_to_30_frozen/`
2. `training/30_to_30_unfreeze_layer4/`
3. `training/90_to_30_models/`

Run dataset preflight/validation first, then baselines, then training. Select
models only by validation macro-F1. Test macro-F1 is the primary final metric;
balanced accuracy, accuracy, per-class precision/recall/F1, confusion matrices
and row-level predictions are also saved by the supplied evaluation scripts.

For the 90-to-30 package, the default dataset and training configs both begin
with historical stride 30. The named stride-15, stride-10 and stride-5 configs
then reproduce the later order. `run_all_models.py` executes the majority/persistence
baseline logic available in the original package, frozen ResNet head,
embedding extraction, logistic regression, and embedding MLP. The historical
generator did not write a recent-input regime column, so persistence must be
generated explicitly or reconstructed from the target window ending at
`t+89` before treating the comparison as complete. Run
`dataset_generation/90_to_30/reconstruct_persistence_labels.py` to create a
non-destructive `manifests_with_persistence/` directory. The active training
configs already point to that directory; the preserved `_historical_windows`
configs retain the original `manifests/` setting. The baseline script then
reports majority, persistence, and most-frequent-transition metrics using the
same accuracy, balanced-accuracy, macro-F1, weighted-F1, per-class report, and
confusion-matrix machinery used for learned models.

## 6. Reproducibility rules

- Keep chronological splits; never randomly mix overlapping windows.
- Fit PCA, K-Means, scalers and classifiers using training data only.
- Keep validation and test timestamps fixed when comparing training strides.
- Pin scikit-learn to the version used to create persisted models when exact
  historical labels are required.
- Never treat post-hoc cluster merging or confidence filtering as a new
  confirmatory model result.
- Preserve the four-cluster experiments as Phase 1. A future K=3 study is a
  fresh Phase 2 experiment with a new untouched holdout.
