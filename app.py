import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import numpy as np
import torch

from models import load_dinov2_model, load_sam_model
from detection_utils import (
    run_sam_on_bboxes, overlay_mask_on_image, extract_object_region,
    get_dinov2_embedding, cosine_similarity, deduplicate_boxes, nms,
    get_multiscale_region_proposals, sliding_window_supplement
)

# Streamlit 
st.set_page_config(page_title="Pattern-Based Detection", layout="wide")
st.title("Pattern-Based Object Detection Prototype")

# Upload 
st.sidebar.header("Step 1: Upload Images")
pattern_file = st.sidebar.file_uploader("Upload Pattern Image (Support)", type=["png", "jpg", "jpeg"])
query_file = st.sidebar.file_uploader("Upload Query Image (Scene)", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
pattern_img, query_img = None, None
if pattern_file:
    pattern_img = Image.open(pattern_file).convert("RGB")
    with col1: st.image(pattern_img, caption="Pattern Image")
if query_file:
    query_img = Image.open(query_file).convert("RGB")
    with col2: st.image(query_img, caption="Query Image")

# Pattern
if pattern_img:
    st.header("Step 2: Annotate Pattern Image")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        background_image=pattern_img,
        update_streamlit=True,
        height=pattern_img.height,
        width=pattern_img.width,
        drawing_mode="rect",
        key="canvas"
    )
    if canvas_result.json_data:
        shapes = canvas_result.json_data.get("objects", [])
        bboxes = [{"x": int(o["left"]), "y": int(o["top"]), "width": int(o["width"]), "height": int(o["height"])} for o in shapes if o["type"] == "rect"]
        if bboxes and st.button("Run SAM Segmentation"):
            predictor = load_sam_model()
            masks = run_sam_on_bboxes(pattern_img, bboxes, predictor)
            if masks:
                major_idx = int(np.argmax([np.sum(m) for m in masks]))
                major_mask = masks[major_idx]
                overlay = overlay_mask_on_image(pattern_img, major_mask)
                st.image(overlay, caption="SAM Mask")
                object_img = extract_object_region(pattern_img, major_mask)
                st.session_state["object_img"] = object_img
                st.image(object_img, caption="Segmented Object")
if "object_img" in st.session_state and st.button("Extract DINOv2 Embedding"):
    model, processor = load_dinov2_model()
    emb = get_dinov2_embedding(st.session_state["object_img"], model, processor)
    st.session_state["pattern_embedding"] = emb
    st.success("Embedding Extracted")

# Detection 
if query_img and "pattern_embedding" in st.session_state:
    st.header("Step 3: Detect in Query Image")
    proposals = get_multiscale_region_proposals(query_img)
    proposals += sliding_window_supplement(query_img)
    proposals = deduplicate_boxes([(x, y, x+w, y+h) for (x, y, w, h) in proposals])
    proposals = proposals[:500]

    model, processor = load_dinov2_model()
    matches, scores = [], []
    for (x0, y0, x1, y1) in proposals:
        patch = query_img.crop((x0, y0, x1, y1))
        emb = get_dinov2_embedding(patch, model, processor)
        dist = 1 - cosine_similarity(st.session_state["pattern_embedding"], emb)
        if dist < 0.3:
            matches.append((x0, y0, x1, y1))
            scores.append(dist)

    keep = nms(matches, scores)
    bboxes = [matches[i] for i in keep]
    st.image(query_img, caption="Detected Matches")
