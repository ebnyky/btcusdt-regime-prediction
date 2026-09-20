# Known historical issues preserved with the source

- The image named `future_regime_dataset_creation_source_code_running_evidence.png`
  inside the corrected 90-to-30 package shows the earlier 30-to-30 mapping and
  is not execution evidence for the 90-to-30 generator.
- The historical 90-to-30 generator configuration was edited over time. The
  initial run used stride 30; later datasets used strides 15, 10 and 5.
- The saved historical PCA/K-Means objects were created with scikit-learn
  1.6.1. Exact reproduction should use that version.
- The original 90-to-30 schema omitted `recent_input_cluster_id`. The included
  reconstruction helper creates it for persistence evaluation without changing
  the original manifests.
- The 30-to-30 and corrected 90-to-30 generators use different renderer
  details. Their historical results are therefore not a controlled causal test
  of input-window length alone.
- The earliest unsupervised manifest recorded 23,568 training rows, but 752
  referenced images were missing from the execution environment; the executed
  embedding run therefore used 22,816 training images plus 2,560 validation
  images.
- The four-cluster silhouette score was weak (0.1143). A future K=3 study must
  be a fresh clustering and supervised-learning phase, not a hard merge of old
  predictions.
