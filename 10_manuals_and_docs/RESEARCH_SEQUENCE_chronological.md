# Preserved experimental sequence

1. Original 30-candle kline dataset generation with pandas and `Axes.bxp`.
2. Frozen ResNet-18 image embeddings.
3. Training-only PCA and candidate-K K-Means discovery.
4. Selected four-cluster regime assignment and qualitative montage review.
5. 30-to-30 future-regime dataset, baselines, and frozen ResNet head.
6. 30-to-30 `layer4` fine-tuning experiment.
7. Corrected 90-to-30 dataset, initially stride 30.
8. 90-to-30 frozen head, embedding logistic regression and embedding MLP.
9. Repeated 90-to-30 generation/training at strides 15, 10 and 5.
10. Exploratory post-hoc Cluster 0+2 and confidence-margin diagnostics.

The post-hoc diagnostics are not training hyperparameters. Cluster merging
changes the target ontology; confidence thresholds change evaluated coverage.
