# MSc Thesis — Project Code

Every code file used to produce **`MSc_Thesis_Nyamekye_Emmanuel_Barima_2026_3.docx`**
(*Predicting Future Bitcoin Market Regimes from Candlestick Images*), gathered from both
Windows accounts on 2026-09-20 and renamed by what each file actually does.

The originals were left untouched — these are copies.

The thesis has two phases: **K = 3 is the study proper**, **K = 4 the preliminary phase**
(Section 4.4). Folder names say which is which. Folders run in pipeline order.

---

## `01_data_acquisition/`
| File | What it does |
|---|---|
| `binance_hourly_ohlc_downloader.py` | Pulls BTCUSDT one-hour OHLC from Binance |
| `source_csv_preflight_audit_K3.py` | The integrity checks of Section 3.2.2 — duplicates, one-hour continuity, high/low relations |
| `source_csv_audit_record_K3.json` | The audit's recorded output (feeds Table 3.2) |
| `execution_environment_K3.json` | Package versions at execution time |
| `crypto_price_api_extractor.py` | Earlier standalone price-API extractor |

## `02_candlestick_rendering/`
Section 3.3. The renderer is the part most sensitive to reproduction, so every variant is kept.

| File | What it does |
|---|---|
| `kline_image_generator_K3_as_executed.py` | **The renderer as actually run for the K=3 study**, with its executed config beside it |
| `kline_image_generator_K4_canonical.py` | The curated K=4 generator |
| `renderer_original_recovered_100dpi.py` | Verbatim recovered original, retaining the 100-DPI constant — kept because the 90 vs 100 DPI difference is measured in Section 4.18 |
| `renderer_standalone_from_csv.py` | Standalone CSV-to-images renderer |
| `kline_generation_notebook_K3_kaggle.ipynb` | The Kaggle generation notebook |
| `verify_renderer_byte_exact.py` | Byte-exact regression check against the reference fixture |
| `reference_render_sample_0_29__APPENDIX_E_VERIFIED.jpg` | The Appendix E reference render — **SHA-256 verified** |
| `reference_fixture_first30_candles.csv` | The 30-row fixture the reference render is built from |
| `test_*.py` | Renderer, reference-image and dataset-schema tests |

## `03_regime_discovery_unsupervised/`
Sections 3.4–3.5, the K=4 phase. `run_unsupervised_regime_discovery_K4.py` is the main
pipeline: ResNet-18 embeddings, PCA, K-means sweep, silhouette / Davies–Bouldin /
Calinski–Harabasz diagnostics, montages. `frozen_pca_K4.joblib` and `frozen_kmeans_K4.joblib`
are the fitted objects.

## `04_supervised_dataset_K3/` — **the thesis target construction**
| File | What it does |
|---|---|
| `generate_k3_forecasting_dataset_30to30.py` | Builds the `[t, t+29] -> cluster([t+30, t+59])` pairs |
| `build_soft_membership_targets.py` | Distance-derived soft memberships, entropy, certainty weights (Section 3.6) |
| `soft_label_training_reference_impl.py` | Reference implementation of the soft-target loss |
| `balance_dataset_geometric_core_tail.py` | The core-and-tail balancing explored in Section 4.14 |
| `frozen_pca_K3.joblib`, `frozen_kmeans_K3.joblib` | **The frozen regime definition** — the objects that define what a regime *is* |
| `validate_*.py`, `test_*.py` | Validators and end-to-end tests for each step |
| `k3_90to30_dataset_generation_*.ipynb` | Ninety-hour dataset generation, stride 1 and strides 5–30 |

## `05_supervised_dataset_K4/`
The same construction for the preliminary phase, at both 30→30 and 90→30.
`reconstruct_persistence_baseline_labels_90to30.py` matters: the historical generator did not
write a current-regime column, so the persistence baseline had to be reconstructed before
the comparison was complete.

## `06_model_training_K3/` — **the models the thesis reports**
The executed Kaggle notebooks, versions 1.1 → 1.3. `..._v1_3_EXECUTED_with_outputs.ipynb`
and `k3_90to30_five_strides_EXECUTED_with_outputs.ipynb` retain their output cells, so the
reported numbers can be read without re-running anything.
`reproducibility_code_from_results_archive/` holds the 14 Python files lifted out of the
hash-verified thirty-to-thirty results archive — the exact code that produced the final numbers.

## `07_model_training_K4/`
Preliminary-phase training, in three sub-folders matching the three historical experiments:
`30to30_frozen_backbone/`, `30to30_unfreeze_layer4/`, `90to30_models/`, plus the
Colab and Google-Drive runner variants. Generic names like `shared.py` and `train_classifier.py`
have been renamed to say which experiment they belong to and what they train.

## `08_thesis_final_pipeline/` — **Appendix D, stages 1–8**
The eight reproduction stages of Appendix D, renamed from `phase*.py` to say what each stage does:

| File | Appendix D stage |
|---|---|
| `stage1_render_window_images.py`, `stage1_extract_frozen_resnet18_embeddings.py`, `stage1_apply_frozen_pca_kmeans_labels.py` | 1 — render and embed every window |
| `stage2_build_supervised_pairs_and_soft_targets.py` | 2 — pair manifests, hard labels, soft memberships, certainty weights |
| `stage3_train_and_evaluate_baselines_and_models.py` | 3 — baselines and the two frozen-feature models |
| `stage4_seed_replication_and_block_bootstrap.py` | 4 — five seeds, moving-block bootstrap intervals |
| `stage5a_build_90hour_input_windows.py`, `stage5b_run_90to30_across_strides.py` | 5 — ninety-hour inputs at each stride |
| `stage6_train_layer4_finetuned_gpu.py` | 6 — the partially fine-tuned image model |
| `stage7_non_image_numeric_controls.py` | 7 — the non-image controls of Section 3.14.1 |
| `stage8_rendering_resolution_sensitivity_test.py` | 8 — 90 vs 100 DPI sensitivity |

## `09_inference_single_image/`
Assign a regime to one new candlestick image, for K=3 and K=4.

## `10_manuals_and_docs/`
`PROJECT_MANUAL_end_to_end.md` is the best single starting point — it gives the run order and the
reproducibility rules. `KNOWN_HISTORICAL_ISSUES.md` records the comparability warnings.
`PROJECT_CONTINUITY_REPORT.pdf` narrates how the project developed.

## `11_environment/`
The `requirements.txt` from each package, kept separate because the packages were run at
different times against different versions. Section 3.15 notes that scikit-learn must be pinned
to the version that created the persisted models when exact historical labels are needed.

---

## Gaps worth knowing about

**Analyses in the thesis with no separate script found here.** Several Chapter Four analyses appear
to have been run inside the executed notebooks rather than as standalone files: the rolling-origin
evaluation (Section 4.22), the hidden Markov and econometric comparators (Section 4.21),
the Diebold–Mariano tests (Table 4.27), the equivalence and power calculations (Table 4.28),
and the adjusted Rand index stability check (Section 4.20.2). Look inside
`06_model_training_K3\*_EXECUTED_with_outputs.ipynb` and the `reproducibility_code/` folder first.

**K=2 control.** Section 4.20.1 reports a two-regime control; searching for it was explicitly
descoped, so no K=2 code is included.

**Excluded as unrelated.** `past_research/` on the other account (multi-view OHLC return
regression, experiments v2–v6) is an earlier, different project and is not part of this thesis.
Coursework, DBMS and deep-learning assignment code was likewise left out.
