# Predicting Future Bitcoin Market Regimes from Candlestick Images

A supervised learning approach using cluster-derived labels generated through unsupervised learning.

MSc Data Science thesis — Department of Computer Science, University of Ghana, September 2026.
Author: Nyamekye Emmanuel Barima.

---

## What this is

Hourly BTCUSDT observations from January 2020 to July 2026 are rendered as standardised
candlestick images. A frozen ImageNet-pretrained ResNet-18 produces 512-dimensional embeddings,
unit-normalised and reduced by PCA to 120 components. K-means over those components defines
three market regimes — bullish, non-directional and bearish — and each window carries both a
hard label and a distance-derived soft membership.

The forecasting question: **can the regime of the next thirty hours be predicted from an
image of an earlier window, better than a rule that simply assumes nothing changes?**

Three supervised models are compared across two input contexts (30 and 90 hours) and four
sampling strides, against reference baselines, non-image numerical controls, and econometric
and hidden Markov comparators.

## The finding

No. The validation-selected model reached a held-out macro-F1 of **0.3350** against a
persistence baseline of **0.3281**, and every 95% interval on the difference contained zero.
Against a transition rule using only the current regime, the strongest model's advantage was
**+0.0048**, interval [−0.0203, +0.0279] — again containing zero.

Three different encodings of the same price window fail alike, which locates the limitation in
**the target definition rather than the visual encoding**. This is an absence of detectable
predictability at this sample size, not proof that the target is unpredictable: the design
could not have detected an improvement below 0.051 in macro-F1.

## Why the uncertainty analysis is the point

Neighbouring candlestick windows share almost all of their candles, so a large row count is not
a large amount of independent evidence. On a held-out partition of 9,301 rows the effective
number of independent blocks is **seventy-eight**. Differences that look consistent across
architectures are entirely consistent with no difference at all. Every interval here comes from
a moving-block bootstrap that respects that dependence.

## Repository layout

Folders run in pipeline order. Every file is named for what it does.

| Folder | Contents |
|---|---|
| `01_data_acquisition/` | Binance OHLC download, source-CSV integrity audit |
| `02_candlestick_rendering/` | The renderer and its variants, reference fixture, byte-exact tests |
| `03_regime_discovery_unsupervised/` | Embedding, PCA, K-means sweep and diagnostics (K=4 phase) |
| `04_supervised_dataset_K3/` | **Target construction for the main study** — pairs, soft memberships, certainty weights, frozen PCA/K-means |
| `05_supervised_dataset_K4/` | The same for the preliminary phase, at 30→30 and 90→30 |
| `06_model_training_K3/` | **The models the thesis reports** — executed notebooks with outputs retained |
| `07_model_training_K4/` | Preliminary-phase training: frozen backbone, layer-4 fine-tuning, 90→30 models |
| `08_thesis_final_pipeline/` | The eight reproduction stages of Appendix D, one file per stage |
| `09_inference_single_image/` | Assign a regime to one new candlestick image |
| `10_manuals_and_docs/` | Start with `PROJECT_MANUAL_end_to_end.md` |
| `11_environment/` | Per-package requirements |

`00_CODE_MAP.md` is the full file-by-file index, including known gaps.

**K = 3 is the study proper. K = 4 is the preliminary phase** written up in Section 4.4.

## Reproducing

See `10_manuals_and_docs/PROJECT_MANUAL_end_to_end.md` for the run order. In short:

```bash
python -m pip install -r 11_environment/requirements_root.txt
python 02_candlestick_rendering/verify_renderer_byte_exact.py
```

The renderer is the part most sensitive to reproduction. Images are rendered at **90 DPI**,
the resolution at which the regime definition was fitted; rendering the same 30 candles at
100 DPI changes 19.1% of pixels. The JPEG output format is a documented defect rather than a
choice — rendering losslessly moves about one cluster assignment in ten, roughly five times the
effect of the resolution change, without altering any verdict in the thesis.

Pin scikit-learn to the version that created the persisted `.joblib` models when exact
historical labels are required.

## Results and data

Not in this repository. The result archives run to 838 MB, including model checkpoints that
exceed GitHub's per-file limit, and the rendered image corpora are roughly 3 GB and
regenerable from this code.

All eight artefacts listed in Appendix E of the thesis have been located and verified against
their recorded SHA-256 digests, including the thirty-to-thirty and ninety-to-thirty results
archives and the frozen regime definition.

## Status

Unpublished MSc thesis work. The programme archive and frozen objects are lodged with the
Department of Computer Science, University of Ghana, and are available from the author on
request. This repository is the specification and the code, not a published artefact with a
persistent identifier.

## A note on scope

This study evaluates **classification** under chronological conditions. It does not evaluate a
trading strategy and draws no inference about profitability. No entry rule, exit rule,
transaction cost, spread, slippage, position size, leverage or drawdown control was tested.
Classification performance is not trading performance, and nothing here is investment advice.
