"""Command-line prediction.

Usage:  python -m scripts.predict_cli path/to/leaf.jpg [--save-mask out.png]
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

from PIL import Image

from app.inference import LeafLensPredictor, artifact_exists


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict leaf mask + disease.")
    parser.add_argument("image", type=Path, help="Path to a leaf image.")
    parser.add_argument("--save-mask", type=Path, help="Write the predicted mask PNG here.")
    parser.add_argument("--save-overlay", type=Path, help="Write the mask overlay PNG here.")
    args = parser.parse_args()

    if not artifact_exists():
        print("No trained model found. Run `python -m app.train` first.", file=sys.stderr)
        return 1
    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    predictor = LeafLensPredictor()
    result = predictor.predict(Image.open(args.image))

    print(f"Predicted: {result['predicted_class']}  "
          f"(confidence {result['confidence']:.3f})")
    print(f"Leaf coverage: {result['leaf_coverage']:.1%}")
    print("Top-k:")
    for item in result["top_k"]:
        print(f"  {item['label']:<35} {item['confidence']:.3f}")

    if args.save_mask:
        args.save_mask.write_bytes(base64.b64decode(result["mask_png_base64"]))
        print(f"Mask written to {args.save_mask}")
    if args.save_overlay:
        args.save_overlay.write_bytes(base64.b64decode(result["overlay_png_base64"]))
        print(f"Overlay written to {args.save_overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
