# Few-Shot Object Detection Pipeline with Bounding Box Highlighting

# Step 1: Imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Step 2: Siamese Network Definition
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer

    def forward_once(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2

# Step 3: Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Step 4: Load Image and Model
def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# def load_image(path):
#     img = cv2.imread(str(path))
#     if img is None:
#         raise FileNotFoundError(f"Image not found at path: {path}")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img


def get_embedding(model, image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_once(image_tensor)
    return embedding

# Step 5: Sliding Window

def sliding_window(image, step=32, window_size=128):
    boxes = []
    H, W, _ = image.shape
    for y in range(0, H - window_size, step):
        for x in range(0, W - window_size, step):
            crop = image[y:y+window_size, x:x+window_size]
            boxes.append(((x, y, x+window_size, y+window_size), crop))
    return boxes

# Step 6: Detection Loop
def detect_largest_match(model, query_img, support_embed):
    best_sim = -1
    best_box = None
    best_area = 0

    for (coords, crop) in sliding_window(query_img):
        try:
            crop_embed = get_embedding(model, crop)
            sim = torch.nn.functional.cosine_similarity(support_embed, crop_embed).item()
            area = (coords[2] - coords[0]) * (coords[3] - coords[1])

            if sim > best_sim:
                best_sim = sim
                best_box = coords
                best_area = area
        except:
            continue

    return best_box, best_sim, best_area

# Step 7: Visualize

def draw_box(image, box):
    x1, y1, x2, y2 = box
    out_img = image.copy()
    cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out_img

# === MAIN ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SiameseNetwork().to(device)
model.eval()  # Not loading weights here for simplicity

support_image_path = Path(r"C:\Users\sures\OneDrive\Desktop\yamir\data\image1.png")  # Change path accordingly
query_image_path = Path(r"C:\Users\sures\OneDrive\Desktop\yamir\data\Screenshot 2025-06-19 172917.png")

support_img = load_image(support_image_path)
query_img = load_image(query_image_path)

support_embed = get_embedding(model, support_img)

box, score, area = detect_largest_match(model, query_img, support_embed)
print(f"Best box: {box}, Score: {score:.4f}, Area: {area} pixels")

result_img = draw_box(query_img, box)
plt.imshow(result_img)
plt.title(f"Detected Object\nScore: {score:.2f}, Area: {area}px")
plt.axis('off')
plt.show()
