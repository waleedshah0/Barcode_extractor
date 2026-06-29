"""
process_images.py
==================
CLI for the barcode-tag OCR POC.

Usage:
    python process_images.py photo.jpg --out results.csv
    python process_images.py photo1.jpg photo2.jpg --out results.csv --save-crops crops/
    python process_images.py ./my_photos_folder --out results.csv

First run will download PaddleOCR's detection/recognition/orientation
models (~50-100MB) - requires outbound network access to
paddle-model-ecology.bj.bcebos.com or huggingface.co.
"""
import argparse
import glob
import os
import sys

from tag_ocr_core import process_images_to_csv

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def resolve_inputs(paths):
    resolved = []
    for p in paths:
        if os.path.isdir(p):
            for ext in IMG_EXTS:
                resolved.extend(glob.glob(os.path.join(p, f"*{ext}")))
                resolved.extend(glob.glob(os.path.join(p, f"*{ext.upper()}")))
        else:
            resolved.append(p)
    return sorted(set(resolved))


def main():
    parser = argparse.ArgumentParser(description="Detect, deskew, and OCR-decode barcode tags.")
    parser.add_argument("inputs", nargs="+", help="Image file(s) or a folder of images")
    parser.add_argument("--out", default="results.csv", help="Output CSV path")
    parser.add_argument("--save-crops", default=None, help="Optional folder to save upright tag crops")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                         help="Drop candidate regions with OCR confidence below this (filters false positives)")
    args = parser.parse_args()

    image_paths = resolve_inputs(args.inputs)
    if not image_paths:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(image_paths)} image(s)...")
    rows = process_images_to_csv(image_paths, args.out,
                                  save_crops_dir=args.save_crops,
                                  min_confidence=args.min_confidence)

    print(f"\nDecoded {len(rows)} tag(s) -> {args.out}")
    for r in rows:
        print(f"  {r['source_image']:35s} tag{r['tag_index']}  ->  {r['decoded_value']}  "
              f"(conf={r['confidence']}, rotation={r['rotation_deg']}°)")


if __name__ == "__main__":
    main()
