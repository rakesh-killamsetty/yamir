import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

# ==== 1. Siamese Network ====
class SiameseNetwork(nn.Module):
    def _init_(self):
        super()._init_()
        base = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # Remove final FC

    def forward_once(self, x):
        return self.encoder(x).view(x.size(0), -1)

# ==== 2. Image Preprocessing ====
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
def preprocess(img):
    return transform(img).unsqueeze(0)

# ==== 3. Region Proposals via Selective Search ====
def get_proposals(img_pil, max_regions=200):
    arr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(arr)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    boxes = []
    for (x, y, w, h) in rects[:max_regions]:
        if w * h >= 500:
            boxes.append((x, y, x + w, y + h))
    return boxes

# ==== 4. Detection ====
def detect_manual(support_crop, query_img, threshold=0.8):
    model = SiameseNetwork().eval()
    support_tensor = preprocess(support_crop)
    support_emb = model.forward_once(support_tensor)

    boxes = get_proposals(query_img)
    matches = []
    for (x1, y1, x2, y2) in boxes:
        region = query_img.crop((x1, y1, x2, y2)).resize((224, 224))
        region_tensor = preprocess(region)
        with torch.no_grad():
            region_emb = model.forward_once(region_tensor)
            sim = F.cosine_similarity(support_emb, region_emb).item()
        if sim >= threshold:
            matches.append(((x1, y1, x2, y2), sim))
    return matches

# ==== 5. Streamlit UI ====
st.title("🎯 Few-Shot Object Detection (Manual Box)")

support_file = st.file_uploader("Upload Support Image", type=["png", "jpg", "jpeg"])
query_file = st.file_uploader("Upload Query Image", type=["png", "jpg", "jpeg"])
threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.01)

if support_file and query_file:
    support_img = Image.open(support_file).convert("RGB")
    query_img = Image.open(query_file).convert("RGB")

    st.write("### 🖊 Draw bounding box on the support image (just one box)")
    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_color="red",
        stroke_width=2,
        background_image=support_img,
        update_streamlit=True,
        height=support_img.height,
        width=support_img.width,
        drawing_mode="rect",
        key="canvas"
    )

    if canvas.json_data and len(canvas.json_data["objects"]) > 0:
        obj = canvas.json_data["objects"][0]
        x, y = obj["left"], obj["top"]
        w, h = obj["width"], obj["height"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        support_crop = support_img.crop((x1, y1, x2, y2))

        st.image(support_crop, caption="Cropped Support Region")

        if st.button("🔍 Detect Similar Regions"):
            results = detect_manual(support_crop, query_img, threshold)
            st.write(f"✅ {len(results)} region(s) found above threshold {threshold}.")

            draw = ImageDraw.Draw(query_img)
            for (box, sim) in results:
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), f"{sim:.2f}", fill="red")

            st.image(query_img, caption="Query Image with Matches")