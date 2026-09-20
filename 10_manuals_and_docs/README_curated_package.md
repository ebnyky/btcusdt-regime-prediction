# MSc Cryptocurrency Visual-Regime Project

This reconstructed project preserves the code in the order in which the
research was conducted:

1. Generate 30-candle kline images with the original pandas/Matplotlib
   `Axes.bxp` renderer.
2. Extract frozen ResNet-18 embeddings, fit PCA on training data, compare
   K-Means candidates, and save the selected clustering pipeline.
3. Generate supervised current-window to future-regime datasets.
4. Run majority, persistence and transition baselines.
5. Train and evaluate the frozen ResNet head, the layer-4 fine-tuned model,
   logistic regression on embeddings, and the embedding MLP where applicable.

Start with [PROJECT_MANUAL.md](PROJECT_MANUAL.md). Every active package has a
`config.json` or named configuration file containing discoverable input and
output paths. Original PDFs and source scripts are retained under the relevant
package; reconstructed orchestration files are clearly labelled.

The historical four-cluster experiment is preserved. A future three-cluster
rerun must refit PCA/K-Means and regenerate all labels rather than hard-merging
old cluster IDs.

Run `python verify_reconstruction.py` before a research run. It checks every
Python/JSON file, reproduces `sample_0_29.jpg` byte-for-byte from the supplied
OHLC fixture, and verifies the identities of the historical K=4 models.
