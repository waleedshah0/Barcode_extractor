<<<<<<< HEAD
# Barcode Tag OCR — POC

Upload a photo of one or more wrapped barcode tags (at any angle/rotation) →
each tag is automatically located, deskewed to upright, read with PaddleOCR,
and written to a CSV.

## Setup

```bash
pip install -r requirements.txt
```

PaddleOCR downloads its detection/recognition/orientation model weights on
first run (~50-100MB) from `paddle-model-ecology.bj.bcebos.com` or
`huggingface.co`. **Outbound network access to one of those hosts is
required.** (This is the one thing I could not fully test in my sandbox here
— it's locked to a small domain allowlist that doesn't include either host —
but everything up through the OCR call is tested and verified working on
your sample images. If you run this somewhere with normal internet access,
the OCR call will just work.)

## Usage

CLI:
```bash
python process_images.py photo.jpg --out results.csv
python process_images.py photo1.jpg photo2.jpg folder_of_photos/ --out results.csv --save-crops crops/
```

Upload UI:
```bash
streamlit run app.py
```

## How it works

1. **`find_tag_centers`** — classical computer vision (Sobel gradient
   difference + contour analysis), *not* a trained detector. It proposes
   candidate tag regions. It's tuned to favor recall, so it occasionally
   flags a non-tag region (shadow, glare) too — that's fine, see step 4.
2. **`auto_crop_and_deskew`** — crops a generous window around each
   candidate and straightens it. This works at *any* starting rotation by
   finding the barcode block's own long axis via `minAreaRect`, the same
   approach validated earlier across all 6 of your sample photos (16/17
   tags auto-decoded correctly).
3. **`read_tag_value`** — PaddleOCR reads the printed digit string under
   the bars. `use_textline_orientation=True` handles any leftover 0°/180°
   ambiguity natively, so there's no separate "is this upside-down" check
   needed (this replaces the hand-rolled Tesseract-based check from the
   earlier prototype).
4. **Confidence filtering** — any candidate region with no recognized text
   or low OCR confidence is dropped before it reaches the CSV. This is what
   keeps step 1's occasional false positives out of your results, without
   needing to perfect the detector itself.

## Output CSV columns

| column | meaning |
|---|---|
| `source_image` | input filename |
| `tag_index` | index of the tag within that image |
| `center_x`, `center_y` | pixel location in the source image |
| `rotation_deg` | rotation applied to make it upright |
| `decoded_value` | the OCR-read tag value |
| `confidence` | PaddleOCR recognition confidence (0–1) |
| `raw_ocr_text` | all text PaddleOCR found in that crop, for debugging |
| `crop_path` | path to the saved upright crop, if `--save-crops` was used |

## Known limitations (POC scope)

- The detector (`find_tag_centers`) is classical CV, tuned on your 6 sample
  photos — very different lighting/backgrounds may need its thresholds
  re-tuned (`min_area_frac`, `max_aspect`, etc. in `tag_ocr_core.py`).
- If two tags sit very close together, the detector may merge or miss one.
- `min_confidence` (default 0.3) is the main knob if you're getting too few
  or too many results.
=======
# Barcode_extractor
>>>>>>> e4136cceaa1e75e2675c4e3f8faacee037cb5725
