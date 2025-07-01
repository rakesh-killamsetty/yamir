from transformers import AutoImageProcessor, AutoModel
from segment_anything import sam_model_registry, SamPredictor

def load_dinov2_model():
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    return model, processor

def load_sam_model(checkpoint_path="./sam_vit_h.pth"):
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    return SamPredictor(sam)
