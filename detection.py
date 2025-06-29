import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
import clip
import numpy as np, cv2
from streamlit_drawable_canvas import st_canvas

# 1. Load CLIP Model & Preprocessor
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# 2. Region Proposals with Selective Search
@st.cache_resource
def get_proposals(img_pil, max_regions=200):
    arr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(arr)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()[:max_regions]
    return [(x, y, x+w, y+h) for (x, y, w, h) in rects if w*h >= 500]

# 3. Non-Maximum Suppression
def iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union else 0

def nms(boxes, scores, iou_thresh=0.3):
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        order = [j for j in order[1:] if iou(boxes[i], boxes[j]) < iou_thresh]
    return keep

# 4. Detection Logic with Single Proposal Embedding (no batch)
def detect_clip(support_crops, query_img, model, preprocess, device, threshold=0.3, iou_thresh=0.3):
    embs = []
    for crop in support_crops:
        img_t = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            e = model.encode_image(img_t)
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e)
    proto = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)

    boxes = get_proposals(query_img)
    results = []

    for box in boxes:
        patch = query_img.crop(box).resize((224, 224))
        tensor = preprocess(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            qemb = model.encode_image(tensor)
            qemb = qemb / qemb.norm(dim=-1, keepdim=True)
            sim = (proto @ qemb.T).item()
            if sim >= threshold:
                results.append((box, sim))

    if not results:
        return []

    boxes_, scores = zip(*results)
    keep_idxs = nms(boxes_, scores, iou_thresh)
    return [(boxes_[i], scores[i]) for i in keep_idxs]

# 5. Streamlit UI
st.title("🎯 Few-Shot Detection with CLIP (Safe Mode)")

support = st.file_uploader("Upload Support Image", type=["jpg", "png", "jpeg"])
query = st.file_uploader("Upload Query Image", type=["jpg", "png", "jpeg"])
threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3, 0.01)
iou_thresh = st.slider("NMS IoU Threshold", 0.0, 1.0, 0.3, 0.01)

if support and query:
    support_img = Image.open(support).convert("RGB")
    query_img = Image.open(query).convert("RGB")

    st.write("🖊 Draw bounding boxes around object(s) in support image:")
    canvas = st_canvas(
        fill_color="rgba(0,255,0,0.3)", stroke_color="green",
        stroke_width=2, background_image=support_img,
        update_streamlit=True, height=support_img.height,
        width=support_img.width, drawing_mode="rect", key="canvas"
    )

    crops = []
    if canvas.json_data and canvas.json_data.get("objects"):
        for obj in canvas.json_data["objects"]:
            x, y = obj["left"], obj["top"]
            w, h = obj["width"], obj["height"]
            crops.append(support_img.crop((int(x), int(y), int(x+w), int(y+h))))
        st.image(crops, caption=[f"Crop {i+1}" for i in range(len(crops))], width=150)

    if st.button("🔍 Detect Object in Query Image") and crops:
        model, preprocess, device = load_clip()
        results = detect_clip(crops, query_img, model, preprocess, device, threshold, iou_thresh)

        st.write(f"✅ Found {len(results)} match(es)")
        draw = ImageDraw.Draw(query_img)
        for box, score in results:
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f"{score:.2f}", fill="red")

        st.image(query_img, caption="Detected Regions", use_column_width=True)

        if st.button("📥 Download Result Image"):
            query_img.save("clip_detection_result.png")
            st.success("Saved: clip_detection_result.png")