# Cluster regime package manual

This is stage 2 of the reconstructed research pipeline. It reads the chronological
30-candle image corpus, extracts fixed 512-dimensional `ResNet18_Weights.DEFAULT`
embeddings, fits PCA on training embeddings only, evaluates candidate K-Means
solutions, and fits the selected K-Means model on training PCA features only.
Validation observations are transformed and assigned without refitting.

## Historical K=4 configuration

The supplied `config.json` reproduces the recorded settings:

- PCA components: 50
- candidate K: 4, 6, 8, 10, 12
- selected K: 4
- K-Means `n_init`: 20
- seed: 42
- embedding batch size: 16
- workers: 0
- nearest and random examples per cluster: 16 each

The verified historical `pca.joblib` and `kmeans.joblib` are preserved under
`historical_k4_models/`. They are evidence from the completed run; the pipeline
does not overwrite them.

## Run order

From `msc_project_files`:

```bash
python -m pip install -r cluster_regime_package/requirements.txt
python cluster_regime_package/validate_cluster_inputs.py \
  --config cluster_regime_package/config.json
python cluster_regime_package/run_unsupervised_regime_discovery.py \
  --config cluster_regime_package/config.json
```

The discoverable output is `outputs/02_cluster_regime_k4/`, containing:

- `embeddings/`: raw ResNet embeddings and PCA features;
- `models/`: fitted PCA and K-Means models;
- `reports/`: candidate-K scores, labels, cluster summaries, and run summary;
- `plots/`: selection metrics;
- `clusters/`: nearest-member and random-member montages.

Inspect the nearest and random montages before naming any cluster. Cluster IDs
are mathematical labels, not semantic market regimes by themselves.

## Planned K=3 rerun

Do not merge old labels. Copy `config.json`, set `candidate_k` to include 3,
set `selected_k` to 3, and choose a new output directory. This refits PCA and
K-Means as a distinct experiment and preserves the K=4 evidence.
