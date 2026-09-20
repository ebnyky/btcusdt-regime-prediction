# File inventory

| Location | Purpose |
|---|---|
| `kline_dataset_generator.py` | Reconstructed original rolling kline generator and `Axes.bxp` renderer |
| `kline_dataset_config.json` | CSV, output, renderer, window, stride and chronological split configuration |
| `verify_reconstruction.py` | Fast syntax, JSON, reference-renderer and historical-model identity checks |
| `reference/sample_0_29.jpg` | User-supplied 360×360 reference image |
| `reference/BTCUSDT_1h_2020-01-01_first30.csv` | Official 30-row OHLC fixture for byte-exact renderer regression testing |
| `historical_source/candlestick_reconstructed_code.py` | Verbatim recovered standalone renderer source |
| `cluster_regime_package/` | Frozen ResNet-18 embedding, PCA, candidate-K K-Means, metrics, plots and montages |
| `cluster_regime_package/historical_k4_models/` | Saved scikit-learn 1.6.1 PCA and four-cluster K-Means objects |
| `supervised_learning_package/dataset_generation/30_to_30/` | Original current-30 to future-30 dataset generator and validator |
| `supervised_learning_package/dataset_generation/90_to_30/` | Corrected current-90 to future-30 generator, validator and persistence helper |
| `supervised_learning_package/training/30_to_30_frozen/` | Frozen-backbone classifier, baselines and checkpoint evaluation |
| `supervised_learning_package/training/30_to_30_unfreeze_layer4/` | Layer-4 fine-tuning experiment and independent test evaluation |
| `supervised_learning_package/training/90_to_30_models/` | Majority/persistence baselines, frozen head, logistic regression and embedding MLP |
| `docs/RESEARCH_SEQUENCE.md` | Actual chronological experiment sequence |
| `docs/KNOWN_HISTORICAL_ISSUES.md` | Reproducibility, evidence and comparability warnings |
| `PROJECT_MANUAL.md` | End-to-end operating instructions |

Historical machine-specific configs are retained with filenames ending in
`_historical_windows.json` or `_historical_kaggle.json`; active configs use
portable, discoverable relative paths.
