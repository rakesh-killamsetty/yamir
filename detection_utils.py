import numpy as np
from PIL import Image

def run_sam_on_bboxes(image_pil, bboxes, predictor):
    image = np.array(image_pil)
    predictor.set_image(image[:, :, :3])
    masks = []
    for bbox in bboxes:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        box = np.array([x, y, x+w, y+h])
        mask, _, _ = predictor.predict(box[None, :], multimask_output=True)
        idx = int(np.argmax([np.sum(m) for m in mask]))
        masks.append(mask[idx])
    return masks

def overlay_mask_on_image(image_pil, mask, color=(255, 0, 0), alpha=0.4):
    image = image_pil.convert("RGBA")
    mask = (mask > 0).astype(np.uint8)
    mask_img = Image.fromarray(mask*255).resize(image.size)
    color_img = Image.new("RGBA", image.size, color + (0,))
    alpha_img = Image.fromarray((mask * int(255 * alpha)).astype(np.uint8)).resize(image.size)
    color_img.putalpha(alpha_img)
    return Image.alpha_composite(image, color_img)

def extract_object_region(image_pil, mask):
    mask = (mask > 0).astype(np.uint8)
    image_np = np.array(image_pil)
    coords = np.argwhere(mask)
    if coords.size == 0: return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = image_np[y0:y1, x0:x1]
    return Image.fromarray(cropped)

def get_dinov2_embedding(image_pil, model, processor):
    inputs = processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

def cosine_similarity(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.dot(a, b))

def nms(boxes, scores, iou_threshold=0.3):
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep

def deduplicate_boxes(boxes, iou_thresh=0.7):
    return nms(boxes, [-i for i in range(len(boxes))], iou_thresh)

def get_multiscale_region_proposals(image_pil, scales=[1.0, 0.85, 0.7]):
    import cv2
    image = np.array(image_pil)
    H, W = image.shape[:2]
    all_rects = []
    for scale in scales:
        scaled = cv2.resize(image, (int(W*scale), int(H*scale)))
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(scaled)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        for (x, y, w, h) in rects[:200]:
            all_rects.append((int(x/scale), int(y/scale), int(w/scale), int(h/scale)))
    return all_rects

def sliding_window_supplement(image_pil, sizes=[64, 96, 128], stride_ratio=0.5):
    image = np.array(image_pil)
    H, W = image.shape[:2]
    rects = []
    for size in sizes:
        stride = int(size * stride_ratio)
        for y in range(0, H - size + 1, stride):
            for x in range(0, W - size + 1, stride):
                rects.append((x, y, size, size))
    return rects
