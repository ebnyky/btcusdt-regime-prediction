"""Recover the pristine ImageNet ResNet-18 backbone from a frozen_backbone checkpoint.

Kaggle notebooks default to internet OFF, which breaks
`ResNet18_Weights.DEFAULT`. Rather than ship 43 MB of weights, this rebuilds
them from a checkpoint you already have: any checkpoint trained with a frozen
backbone carries conv1 through layer4 unmodified from ImageNet.

Your project's `best_model.pt` (the K=4 30-to-30 frozen-head run, recorded
training_mode "frozen_backbone", best epoch 16, validation macro-F1 0.24564)
is exactly such a checkpoint.

  python extract_backbone.py --checkpoint best_model.pt \
      --out resnet18_imagenet_backbone.pt

Simplest alternative: turn Kaggle's internet toggle ON and skip this entirely.
"""
import argparse, sys, torch
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="resnet18_imagenet_backbone.pt")
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    mode = ck.get("training_mode") if isinstance(ck, dict) else None
    if mode and mode != "frozen_backbone":
        print(f"WARNING: training_mode is '{mode}', not 'frozen_backbone'. "
              f"layer4 may have been fine-tuned and would not be pristine ImageNet.")
    body = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    if len(body) != 120:
        sys.exit(f"Expected 120 backbone tensors for ResNet-18, found {len(body)}.")
    torch.save(body, a.out)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size/1e6:.1f} MB, {len(body)} tensors)")
    if mode: print(f"source checkpoint training_mode: {mode}")

if __name__ == "__main__":
    main()
