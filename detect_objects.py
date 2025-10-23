import torch
import torch.nn.functional as F
import json
import clip
import cv2
from torchvision.ops import roi_align
import sys
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import List, Tuple
import tkinter as tk
from tkinter import filedialog

yolo_model = YOLO(r"D:\DocCument\DATN\test\yolov5\yolov5xu.pt")

# Backbone feature capture for RoIAlign descriptors
_BACKBONE_LAYER_INDEX = 9  # SPPF layer index inside YOLO backbone
_BACKBONE_STRIDE = int(yolo_model.model.model[-1].stride[-1].item())
_feature_map_store = {}


def _compute_global_context(feature_map: torch.Tensor) -> List[float]:
    """Return an L2-normalized global context vector pooled from the backbone feature map."""
    if feature_map is None or feature_map.numel() == 0:
        return []
    with torch.no_grad():
        if feature_map.dim() == 4:
            pooled = feature_map.mean(dim=(2, 3), keepdim=False)
        else:
            pooled = feature_map
        pooled = pooled.flatten(start_dim=1)
        pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.squeeze(0).cpu().tolist()

def _capture_backbone_feature(module, inputs, output):
    """Store latest backbone feature map for ROI extraction."""
    _feature_map_store["backbone"] = output.detach().cpu()

# Register hook once so every inference populates the shared store
yolo_model.model.model[_BACKBONE_LAYER_INDEX].register_forward_hook(_capture_backbone_feature)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Danh sách từ vựng mở rộng (có thể tùy chỉnh)
animals = [
    "cat", "dog", "bird", "horse", "cow", " sheep", "lion",
    "tiger", "elephant", "bear", "deer", "monkey", "zebra",
    "giraffe", "kangaroo", "dolphin", "shark", "snake", "turtle",
    "rabbit", "fox", "wolf", "panda", "crocodile", "peacock","person"
]
vehicles = [
    "car", "motorcycle", "bicycle", " bus", " truck", " train",
    "airplane", " helicopter", "boat", " yacht", " submarine",
    " scooter", " skateboard", "tram", " taxi", " police car",
    " ambulance", " fire truck", " forklift", " van"
]
household_items = [
    " chair", " table", " sofa", " bed", " lamp", " television",
    " laptop", " smartphone", " refrigerator", " microwave", " washing machine",
    "vacuum cleaner", " mirror", " bookshelf", " fan", " clock",
    " pillow", " blanket", " rug", " cupboard", " kettle", " toaster","key"
]
food_drinks = [
    " pizza", " hamburger", " sandwich", " hotdog", " steak", " fish",
    "bowl of soup", " plate of spaghetti", " salad", " cake", " donut",
    " ice cream", " cup of coffee", " bottle of water", " soda can",
    " glass of milk", " loaf of bread", " croissant", " chocolate bar"
]
clothing = [
    "t-shirt", " shirt", " pair of jeans", " dress", "skirt", " jacket",
    " coat", " pair of shorts", " hat", " cap", " pair of sunglasses",
    "pair of gloves", " belt", " scarf", " backpack", " handbag",
    " pair of shoes", " pair of sandals", " pair of boots"
]
electronics = [
    " smartphone", " laptop", " desktop computer", " keyboard", " mouse",
    " printer", " projector", " television", " camera", " drone",
    " game console", " tablet", " smartwatch", " microphone",
    " speaker", " headphone", " charger", " USB drive", " hard drive"
]
buildings = [
    "house", " skyscraper", " bridge", " tower", " lighthouse",
    " castle", " temple", "church", " mosque", " stadium",
    " factory", " warehouse", " hospital", "school", " shopping mall",
    " hotel", " police station", " fire station", " library", "a museum"
]
nature = [
    "mountain", " river", " lake", " ocean", " beach", " desert",
    "forest", " waterfall", " volcano", "cave", " rainbow",
    " sunset", " sunrise", " thunderstorm", "snow-covered mountain",
    " flower", " tree", " bush", " meadow", " glacier"
]
sports = [
    "soccer ball", " basketball", " baseball bat", " tennis racket",
    " golf club", " hockey stick", " snowboard", " skateboard",
    " pair of ice skates", " bicycle helmet", " football", " volleyball",
    " badminton racket", " boxing glove", " jump rope"
]
school_supplies = [
    " book", " notebook", " pen", " pencil", " eraser", " ruler",
    " calculator", " protractor", " compass", " highlighter", 
    " stapler", " pair of scissors", " glue stick", " backpack",
    " whiteboard", " blackboard", " piece of chalk", " marker",
    " set of colored pencils", "paintbrush", " watercolor palette",
    " binder", " paper clip", " sticky note", " file folder",
    " document scanner", " desk lamp", "tablet", " laptop", 
    " printer", " USB flash drive"
]

# Loại bỏ dấu cách thừa và tránh trùng nhãn
label_texts = list(set([label.strip() for label in (
    animals + vehicles + household_items +
    food_drinks + clothing + electronics +
    buildings + nature + sports + school_supplies
)]))

def _consume_feature_map():
    """Retrieve and remove the cached backbone feature map."""
    feature_map = _feature_map_store.pop("backbone", None)
    if feature_map is None:
        raise RuntimeError("Backbone feature map not captured from the latest YOLO forward pass.")
    return feature_map

def _compute_resize_params(feature_map: torch.Tensor, image_shape: Tuple[int, int, int]):
    """Compute scale and padding applied during YOLO letterboxing."""
    feat_h, feat_w = feature_map.shape[-2:]
    input_h, input_w = feat_h * _BACKBONE_STRIDE, feat_w * _BACKBONE_STRIDE
    orig_h, orig_w = image_shape[:2]
    scale = min(input_h / max(orig_h, 1), input_w / max(orig_w, 1))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_w = max(input_w - new_w, 0) / 2.0
    pad_h = max(input_h - new_h, 0) / 2.0
    return scale, pad_w, pad_h, feat_w, feat_h

def extract_roi_features(feature_map: torch.Tensor, boxes: List[Tuple[int, int, int, int]], image_shape: Tuple[int, int, int]):
    """Return L2-normalized ROI pooled feature vectors for each bounding box."""
    if feature_map is None or not boxes:
        return []
    scale, pad_w, pad_h, feat_w, feat_h = _compute_resize_params(feature_map, image_shape)
    device = feature_map.device
    dtype = feature_map.dtype
    aligned_boxes = []
    eps = 1e-3
    for x1, y1, x2, y2 in boxes:
        x1_s = (x1 * scale + pad_w) / _BACKBONE_STRIDE
        y1_s = (y1 * scale + pad_h) / _BACKBONE_STRIDE
        x2_s = (x2 * scale + pad_w) / _BACKBONE_STRIDE
        y2_s = (y2 * scale + pad_h) / _BACKBONE_STRIDE
        x1_s = min(max(x1_s, 0.0), feat_w - eps)
        y1_s = min(max(y1_s, 0.0), feat_h - eps)
        x2_s = min(max(x2_s, x1_s + eps), feat_w - eps)
        y2_s = min(max(y2_s, y1_s + eps), feat_h - eps)
        aligned_boxes.append([0.0, x1_s, y1_s, x2_s, y2_s])
    if not aligned_boxes:
        return []
    rois = torch.tensor(aligned_boxes, device=device, dtype=dtype)
    pooled = roi_align(feature_map, rois, output_size=(7, 7), spatial_scale=1.0, aligned=True)
    pooled = F.adaptive_avg_pool2d(pooled, (1, 1)).flatten(1)
    pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.cpu().tolist()

text_inputs = clip.tokenize(label_texts).to(device)

def add_padding(image, bbox, padding=10):
    x1, y1, x2, y2 = bbox
    img_height, img_width = image.shape[:2]
    return image[max(0, y1-padding):min(y2+padding, img_height), max(0, x1-padding):min(x2+padding, img_width)]

# Detect objects with YOLO
def detect_objects(image_source):
    if isinstance(image_source, str):
        image = cv2.imread(image_source)
        if image is None:
            raise FileNotFoundError(f"Unable to load image from path: {image_source}")
        inference_input = image_source
    elif isinstance(image_source, Image.Image):
        image = cv2.cvtColor(np.array(image_source.convert("RGB")), cv2.COLOR_RGB2BGR)
        inference_input = image
    elif isinstance(image_source, np.ndarray):
        image = image_source.copy()
        inference_input = image
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    results = yolo_model(inference_input)
    feature_map = _consume_feature_map()
    global_context = _compute_global_context(feature_map)
    detected_objects, yolo_labels = [], []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            if (x2 - x1 < 20) or (y2 - y1 < 20):
                print(f"⚠️ Bỏ qua đối tượng nhỏ quá [{x1}, {y1}, {x2}, {y2}]")
                continue
            cropped_pil = Image.fromarray(cv2.cvtColor(add_padding(image, (x1, y1, x2, y2)), cv2.COLOR_BGR2RGB))
            detected_objects.append((cropped_pil, (x1, y1, x2, y2)))
            yolo_labels.append(results[0].names[int(cls)])
    return detected_objects, yolo_labels, image, feature_map, global_context

# Phân loại với CLIP, fallback về YOLO nếu confidence thấp
def classify_with_clip(detected_objects, yolo_labels):
    results = []
    text_features = clip_model.encode_text(text_inputs)

    # 🌟 Lấy toàn bộ nhãn từ YOLO làm nhãn quan trọng
    important_labels = list(yolo_model.names.values())
    print(f"🔹 Danh sách nhãn quan trọng từ YOLO: {important_labels}")

    for idx, (cropped_pil, bbox) in enumerate(detected_objects):
        try:
            if not isinstance(cropped_pil, Image.Image):
                print(f"⚠️ Đối tượng {idx+1} không phải ảnh PIL! Giữ nhãn YOLO: '{yolo_labels[idx]}'")
                results.append((yolo_labels[idx], bbox))
                continue

            # Đưa ảnh vào CLIP để phân loại
            image_input = preprocess(cropped_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                similarities = (clip_model.encode_image(image_input) @ text_features.T).softmax(dim=-1)
                best_label = label_texts[similarities.argmax().item()]
                confidence = similarities.max().item()

            # 🎯 Logic thông minh giữ nhãn YOLO nếu CLIP nhận sai
            if yolo_labels[idx] in important_labels and best_label != yolo_labels[idx]:
                print(f"🔹 Giữ nhãn YOLO (quan trọng): '{yolo_labels[idx]}' dù CLIP báo '{best_label}' (conf: {confidence:.2f})")
                results.append((yolo_labels[idx], bbox))
            elif confidence < 0.3:
                print(f"⚠️ Độ tự tin thấp ({confidence:.2f}) → Giữ nhãn YOLO: '{yolo_labels[idx]}'")
                results.append((yolo_labels[idx], bbox))
            else:
                print(f"✅ Đổi nhãn CLIP: '{yolo_labels[idx]}' ➜ '{best_label}' (confidence: {confidence:.2f})")
                results.append((best_label.strip(), bbox))

        except Exception as e:
            print(f"⚠️ Lỗi CLIP: {e} → Giữ nhãn YOLO: '{yolo_labels[idx]}'")
            results.append((yolo_labels[idx], bbox))

    return results

# Full pipeline
def run_pipeline(image_path=None):
    if image_path is None:
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(title="Chọn file ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not image_path:
            print("❌ Không có file nào được chọn!")
            return

    detected_objects, yolo_labels, original_image, feature_map, _ = detect_objects(image_path)
    classified_results = classify_with_clip(detected_objects, yolo_labels)

    boxes = [bbox for _, bbox in classified_results]
    roi_features = extract_roi_features(feature_map, boxes, original_image.shape)
    if boxes:
        feature_dim = feature_map.shape[1]
        if len(roi_features) != len(boxes):
            fallback = [0.0] * feature_dim
            while len(roi_features) < len(boxes):
                roi_features.append(fallback.copy())
            if len(roi_features) > len(boxes):
                roi_features = roi_features[:len(boxes)]
    else:
        roi_features = []
    if roi_features:
        print(f"Do. Extracted {len(roi_features)} ROIAlign feature vectors (dim {len(roi_features[0]) if roi_features else 0}).")
    results_json = []
    for idx, (label, (x1, y1, x2, y2)) in enumerate(classified_results):
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        feature_vector = roi_features[idx] if idx < len(roi_features) else []
        if feature_vector:
            feature_vector = [float(v) for v in feature_vector]
        results_json.append({"label": label, "bbox": [int(x1), int(y1), int(x2), int(y2)], "feature": feature_vector})
        print(f"✅ Đối tượng {idx+1} | Class: {label} | BBox: [{x1}, {y1}, {x2}, {y2}]")

    with open("result.json", "w") as json_file:
        json.dump(results_json, json_file, indent=4)

    cv2.imwrite("result.jpg", original_image)
    print("✅ Nhận diện hoàn tất! Kết quả đã lưu.")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(image_path)
