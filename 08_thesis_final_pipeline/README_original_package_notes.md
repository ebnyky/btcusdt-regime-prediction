# Layer-4 rerun on corrected labels

Everything else in the corrected rerun is done. This one model needs a GPU:
fine-tuning ResNet-18 layer 4 on ~42k images is about 19 hours of CPU and
minutes on a T4.

## You do NOT need to regenerate the images

Input render resolution was measured, not assumed. Models trained on 90-DPI
inputs and scored on both renders differ by -0.0002 (linear) and -0.0005 (mlp),
against a seed standard deviation of 0.006 to 0.008 — an order of magnitude
below noise, and that was the harsher train-90 / test-100 direction.

Resolution mattered only for *label generation*, where the frozen PCA and
K-Means were applied to images from a different corpus than the one they were
fitted on. Layer-4 trains end to end and never touches those frozen models.

**So: keep your existing `BTCUSDT_k3_forecasting_dataset` images and swap in the
corrected labels.** That is route A below and it is the one to use.

## ImageNet weights — do this first

Kaggle notebooks default to internet OFF, which makes
`ResNet18_Weights.DEFAULT` fail. Pick one:

**Easiest:** turn Kaggle's internet toggle ON in notebook settings. Nothing else
to do.

**Offline:** rebuild the backbone from a checkpoint you already have. Any
frozen-backbone checkpoint carries conv1 through layer4 unmodified from
ImageNet, and your `best_model.pt` is one (training_mode "frozen_backbone",
best epoch 16, validation macro-F1 0.24564):

```bash
python extract_backbone.py --checkpoint /kaggle/input/<...>/best_model.pt \
    --out resnet18_imagenet_backbone.pt
```

The training script picks that file up automatically if it sits beside it.

## Route A — reuse your existing dataset (recommended, ~5 minutes of setup)

1. Attach your existing K=3 forecasting dataset as a Kaggle input.
2. Join the corrected labels onto its manifests by input-window timestamp:

```bash
python merge_corrected_labels.py \
  --original-manifests /kaggle/input/<your-dataset>/manifests \
  --corrected manifests_corrected \
  --out manifests
```

   It prints a matched-row count per split. Expect 42,100 train, 4,775
   validation, 9,301 test. A low match rate means the timestamp column was
   guessed wrong — the warning names the column it used, so tell me and I will
   adjust the join.

3. Arrange a data root with the existing `images/` and the new `manifests/`.
   A symlink to the read-only Kaggle input is fine — image indexing follows
   symlinks deliberately.

4. Verify before spending GPU time:

```bash
python check_before_training.py /kaggle/working/data
```

   Expect `READY TO TRAIN` with 42,100 / 4,775 / 9,301 rows and class shares
   near 22/53/24 on train. Shares of 17/66/17 mean the old labels leaked
   through and the merge did not take.

5. Train:

```bash
python train_layer4_corrected.py --data-root /kaggle/working/data \
    --out /kaggle/working/layer4_corrected_results
```

## Route B — regenerate at 90 DPI (only if the join fails)

```bash
python make_images.py \
  --ohlc BTCUSDT_1h_2020-01-01_to_2025-07-06.csv BTCUSDT_1h_2025-07-07_to_2026-07-31.csv \
  --windows window_index.csv --out data
cp -r manifests_corrected data/manifests
python train_layer4_corrected.py --data-root data --out layer4_corrected_results
```

Roughly 50 minutes of rendering. The renderer reproduces the project's recorded
reference hash `9551df8b81bd0b63...` byte-for-byte at 90 DPI.

## Protocol

Select on validation macro-F1 only. The script saves `best_model.pt` at the best
validation epoch and evaluates the test split once, at the end. Do not re-run
the test evaluation after seeing it.

## What to send back

`layer4_corrected_results/layer4_corrected_results.json`. It carries the
validation and test metrics, per-class figures, confusion matrices and the epoch
history. That fills the four **[LAYER-4 PENDING]** gaps in the revision pack.

## What is already in this package

| File | Purpose |
|---|---|
| `manifests_corrected/` | corrected labels, soft targets, certainty weights, class weights |
| `merge_corrected_labels.py` | joins those onto your existing manifests by timestamp |
| `make_images.py` | 90-DPI renderer, only needed for route B |
| `window_index.csv` | window list for route B |
| `train_layer4_corrected.py` | the training script |
| `check_before_training.py` | preflight: row counts, image resolution, label sanity |
| `extract_backbone.py` | rebuilds ImageNet weights offline from your own checkpoint |
| `phase1..phase8*.py` | the full corrected pipeline as run on CPU |
| `results_corrected.json` | corrected 30-to-30 baselines and model results |
| `results_90to30_corrected.json` | corrected 90-to-30, all four primary strides |
| `results_numeric_baseline.json` | non-image controls |
| `uncertainty_corrected.json` / `uncertainty_original.json` | block-bootstrap intervals |
| `results_dpi_transfer.json` | the measurement behind "do not regenerate" |
