"""
app.py
======
Streamlit POC: upload an image -> detect/deskew/OCR each tag -> view + download CSV.

Run with:
    streamlit run app.py

First run will download PaddleOCR models (~50-100MB) - requires outbound
network access to paddle-model-ecology.bj.bcebos.com or huggingface.co.
"""
import os
import tempfile

import cv2
import pandas as pd
import streamlit as st

from tag_ocr_core import process_image, get_ocr_engine

st.set_page_config(page_title="Barcode Tag OCR POC", layout="wide")
st.title("Barcode Tag OCR — POC")
st.caption(
    "Upload a photo containing one or more wrapped barcode tags. "
    "Each tag is auto-detected, deskewed to upright, and read with PaddleOCR."
)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
min_conf = st.slider(
    "Minimum OCR confidence to keep a result",
    0.0, 1.0, 0.3, 0.05,
    help="Lower this if real tags are being dropped; raise it if junk regions are showing up.",
)

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, uploaded.name)
        with open(img_path, "wb") as f:
            f.write(uploaded.getbuffer())

        crops_dir = os.path.join(tmpdir, "crops")

        with st.spinner("Loading OCR model (first run downloads weights)..."):
            ocr_engine = get_ocr_engine()

        with st.spinner("Detecting and reading tags..."):
            rows = process_image(img_path, ocr_engine=ocr_engine,
                                  save_crops_dir=crops_dir, min_confidence=min_conf)

        if not rows:
            st.warning("No tags were confidently decoded. Try lowering the confidence slider.")
        else:
            st.success(f"Decoded {len(rows)} tag(s).")

            df = pd.DataFrame(rows)
            st.dataframe(df.drop(columns=["crop_path"]), use_container_width=True)

            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False),
                file_name="decoded_tags.csv",
                mime="text/csv",
            )

            st.subheader("Upright tag crops")
            cols = st.columns(min(4, len(rows)))
            for i, row in enumerate(rows):
                if row["crop_path"] and os.path.exists(row["crop_path"]):
                    img_rgb = cv2.cvtColor(cv2.imread(row["crop_path"]), cv2.COLOR_BGR2RGB)
                    with cols[i % len(cols)]:
                        st.image(img_rgb, caption=f"{row['decoded_value']} (conf {row['confidence']})")
else:
    st.info("Upload an image to get started.")
