import json
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from PIL import Image
import torchvision.transforms as T
from models import build_model
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple

# ============= MODEL CACHING FOR SPEED =============
# Cache RelTR model to avoid reloading every inference
_cached_reltr_model = None
_cached_reltr_device = None
_cached_reltr_checkpoint_path = None

# Pre-compute transform once
_RELTR_TRANSFORM = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Relationship classes (constant)
REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
               'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
               'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
               'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
               'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
               'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']


# ============= SPATIAL VALIDATION FOR RELATIONSHIPS =============
VEHICLE_CLASSES = {
    "car", "vehicle", "truck", "trunk", "bus", "van", "automobile", "taxi", 
    "motorcycle", "motorbike", "bicycle", "bike"
}
PERSON_CLASSES = {
    "person", "human", "man", "woman", "child", "boy", "girl", 
    "people", "pedestrian", "worker"
}
ABOVE_RELATIONS = {"on", "above", "over", "riding", "sitting on", "standing on", "on back of"}
BELOW_RELATIONS = {"under", "below", "beneath", "lying on", "laying on"}

# Additional class sets for semantic validation
ANIMAL_CLASSES = {
    "dog", "cat", "horse", "cow", "sheep", "bird", "elephant", "bear",
    "zebra", "giraffe", "animal", "pet"
}
TRANSPORT_CLASSES = {
    "skateboard", "surfboard", "snowboard", "bicycle", "bike", "motorcycle",
    "horse", "elephant", "scooter", "skis", "sled"
}

# Semantic relationship corrections
SEMANTIC_CORRECTIONS = [
    {
        "subject": ANIMAL_CLASSES | PERSON_CLASSES,
        "object": TRANSPORT_CLASSES,
        "wrong_relations": {"wearing", "wears", "has", "holding", "carrying"},
        "correct_relation": "riding",
    },
    {
        "subject": ANIMAL_CLASSES | PERSON_CLASSES,
        "object": {"skateboard", "surfboard", "snowboard", "bicycle", "car", "truck"},
        "wrong_relations": {"wearing", "wears"},
        "correct_relation": "on",
    },
]


def validate_semantic_relationship_bbox(
    subject_class: str, object_class: str,
    predicted_relation: str, confidence: float = 1.0
):
    """Validate and correct semantically invalid relationships."""
    subject_lower = subject_class.lower().strip()
    object_lower = object_class.lower().strip()
    relation_lower = predicted_relation.lower().strip()
    
    for rule in SEMANTIC_CORRECTIONS:
        subject_match = any(s in subject_lower for s in rule["subject"])
        object_match = any(o in object_lower for o in rule["object"])
        relation_wrong = relation_lower in rule["wrong_relations"]
        
        if subject_match and object_match and relation_wrong:
            correct = rule["correct_relation"]
            print(f"🔧 [Semantic] Correcting: '{subject_class} {predicted_relation} {object_class}' → '{correct}'")
            return correct, max(confidence * 0.85, 0.6)
    
    return predicted_relation, confidence


def validate_spatial_relationship_bbox(
    subject_bbox, object_bbox, subject_class: str, object_class: str, 
    predicted_relation: str, confidence: float = 1.0
):
    """Validate and correct predicted spatial relationships based on bounding box positions."""
    if len(subject_bbox) < 4 or len(object_bbox) < 4:
        return predicted_relation, confidence
    
    subject_lower = subject_class.lower().strip()
    object_lower = object_class.lower().strip()
    relation_lower = predicted_relation.lower().strip()
    
    subj_x1, subj_y1, subj_x2, subj_y2 = subject_bbox[:4]
    obj_x1, obj_y1, obj_x2, obj_y2 = object_bbox[:4]
    
    subj_center_y = (subj_y1 + subj_y2) / 2
    obj_center_y = (obj_y1 + obj_y2) / 2
    obj_height = max(obj_y2 - obj_y1, 1)
    
    vertical_diff_ratio = (subj_center_y - obj_center_y) / obj_height
    
    horizontal_overlap = (
        max(0, min(subj_x2, obj_x2) - max(subj_x1, obj_x1)) / 
        max(min(subj_x2 - subj_x1, obj_x2 - obj_x1), 1)
    )
    
    is_person_subject = any(p in subject_lower for p in PERSON_CLASSES)
    is_vehicle_object = any(v in object_lower for v in VEHICLE_CLASSES)
    
    if is_person_subject and is_vehicle_object and horizontal_overlap > 0.3:
        if vertical_diff_ratio > 0.5:
            if relation_lower in ABOVE_RELATIONS or relation_lower == "near":
                return "under", max(confidence * 0.9, 0.6)
        elif vertical_diff_ratio < -0.3:
            if relation_lower in BELOW_RELATIONS:
                return "on", max(confidence * 0.9, 0.6)
    
    return predicted_relation, confidence


def generate_heuristic_person_vehicle_relations(objects, existing_relations):
    """Generate heuristic relationships between person and vehicle when RelTR misses them."""
    heuristic_relations = []
    
    if len(objects) < 2:
        return heuristic_relations
    
    existing_pairs = set()
    for rel in existing_relations:
        subj = rel.get('subject', '').lower()
        obj = rel.get('object', '').lower()
        existing_pairs.add((subj, obj))
    
    # Check both 'class' and 'label' keys since different pipelines use different keys
    def get_class_name(o):
        return (o.get('class') or o.get('label') or '').lower().strip()
    
    persons = [(i, o) for i, o in enumerate(objects) 
               if any(p in get_class_name(o) for p in PERSON_CLASSES)]
    vehicles = [(i, o) for i, o in enumerate(objects) 
                if any(v in get_class_name(o) for v in VEHICLE_CLASSES)]
    
    # Debug: print detected classes
    if objects:
        print(f"🔍 [Heuristic Debug] Object keys: {list(objects[0].keys())}, first class: '{get_class_name(objects[0])}'")
    print(f"🔍 [Heuristic] Found {len(persons)} persons, {len(vehicles)} vehicles")
    
    for person_idx, person_obj in persons:
        person_class = person_obj.get('label', 'person')
        person_bbox = person_obj.get('bbox', [])
        if len(person_bbox) < 4:
            continue
        
        px1, py1, px2, py2 = [float(x) for x in person_bbox[:4]]
        person_center_x = (px1 + px2) / 2
        person_center_y = (py1 + py2) / 2
        person_width = px2 - px1
        person_height = py2 - py1
        person_area = person_width * person_height
        
        for vehicle_idx, vehicle_obj in vehicles:
            vehicle_class = vehicle_obj.get('label', 'car')
            if (person_class.lower(), vehicle_class.lower()) in existing_pairs:
                continue
            
            vehicle_bbox = vehicle_obj.get('bbox', [])
            if len(vehicle_bbox) < 4:
                continue
            
            vx1, vy1, vx2, vy2 = [float(x) for x in vehicle_bbox[:4]]
            vehicle_center_x = (vx1 + vx2) / 2
            vehicle_center_y = (vy1 + vy2) / 2
            vehicle_top = vy1
            vehicle_bottom = vy2
            vehicle_width = vx2 - vx1
            vehicle_height = max(vy2 - vy1, 1)
            
            # Calculate overlaps
            overlap_x = max(0, min(px2, vx2) - max(px1, vx1))
            overlap_y = max(0, min(py2, vy2) - max(py1, vy1))
            overlap_area = overlap_x * overlap_y
            
            horizontal_overlap_ratio = overlap_x / max(min(person_width, vehicle_width), 1)
            vertical_overlap_ratio = overlap_y / max(min(person_height, vehicle_height), 1)
            
            # How much of person is inside vehicle bbox?
            person_inside_vehicle_ratio = overlap_area / max(person_area, 1)
            
            relation = None
            confidence = 0.0
            
            print(f"🔍 [Heuristic] {person_class} vs {vehicle_class}: "
                  f"h_overlap={horizontal_overlap_ratio:.2f}, v_overlap={vertical_overlap_ratio:.2f}, "
                  f"person_inside={person_inside_vehicle_ratio:.2f}")
            
            # === CASE 1: Person UNDER vehicle ===
            # Scenario A: Person working under elevated vehicle (bbox overlaps)
            if person_inside_vehicle_ratio > 0.3:
                # Person is significantly inside vehicle bbox
                # Check if person is in LOWER part of vehicle (under it)
                person_relative_y = (person_center_y - vehicle_top) / vehicle_height
                if person_relative_y > 0.5:  # Person is in lower half of vehicle bbox
                    relation = "under"
                    confidence = 0.80
                    print(f"  → Detected: {person_class} UNDER {vehicle_class} (inside vehicle bbox)")
            
            # Scenario B: Person is below vehicle (traditional case)
            if relation is None and horizontal_overlap_ratio > 0.2:
                if person_center_y > vehicle_bottom - vehicle_height * 0.3:
                    relation = "under"
                    confidence = 0.75
                    print(f"  → Detected: {person_class} UNDER {vehicle_class} (below vehicle)")
                elif py2 > vehicle_bottom:
                    relation = "under"
                    confidence = 0.65
                    print(f"  → Detected: {person_class} UNDER {vehicle_class} (feet below)")
            
            # === CASE 2: Person BEHIND/IN FRONT OF vehicle ===
            if relation is None and horizontal_overlap_ratio > 0.1:
                vertical_distance = abs(person_center_y - vehicle_center_y)
                if vertical_distance < vehicle_height * 0.7:
                    if person_center_x > vehicle_center_x + vehicle_width * 0.2:
                        relation = "behind"
                        confidence = 0.55
                    elif person_center_x < vehicle_center_x - vehicle_width * 0.2:
                        relation = "in front of"
                        confidence = 0.55
            
            # === CASE 3: Person NEAR vehicle (fallback) ===
            if relation is None:
                # Calculate distance between bbox edges
                dist_x = max(0, max(px1 - vx2, vx1 - px2))
                dist_y = max(0, max(py1 - vy2, vy1 - py2))
                edge_distance = (dist_x**2 + dist_y**2)**0.5
                
                # "Near" if within reasonable distance
                proximity_threshold = max(vehicle_width, vehicle_height) * 0.3
                if edge_distance < proximity_threshold:
                    relation = "near"
                    confidence = 0.50
            
            if relation is not None:
                heuristic_relations.append({
                    'subject': person_class,
                    'relation': relation,
                    'object': vehicle_class,
                    'confidence': confidence,
                    'source': 'heuristic_spatial'
                })
                existing_pairs.add((person_class.lower(), vehicle_class.lower()))
                print(f"✅ [Heuristic] Generated: {person_class} {relation} {vehicle_class}")
    
    return heuristic_relations

def _get_cached_reltr_model(args, device):
    """Get or create cached RelTR model - avoids reloading checkpoint every call."""
    global _cached_reltr_model, _cached_reltr_device, _cached_reltr_checkpoint_path
    
    checkpoint_path = getattr(args, 'resume', None)
    
    # Return cached model if available and checkpoint matches
    if (_cached_reltr_model is not None and 
        _cached_reltr_device == device and 
        _cached_reltr_checkpoint_path == checkpoint_path):
        return _cached_reltr_model
    
    print(f"[RelTR] Loading model (first time or checkpoint changed)...")
    model, _, _ = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Enable inference optimizations
    if device.type == 'cuda':
        model = model.half()  # FP16 for faster inference
        torch.backends.cudnn.benchmark = True
    
    # Cache the model
    _cached_reltr_model = model
    _cached_reltr_device = device
    _cached_reltr_checkpoint_path = checkpoint_path
    
    print(f"[RelTR] Model cached successfully on {device}")
    return model


def _resolve_device(device_arg: str) -> torch.device:
    """Return the torch.device requested by CLI, falling back safely when unavailable."""
    if device_arg:
        device_arg = device_arg.lower()
        if device_arg.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(device_arg)
            print(f"[RelTR] Requested CUDA device '{device_arg}' unavailable. Falling back to CPU.")
            return torch.device("cpu")
        if device_arg in ("cpu", "mps"):
            return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_yolo_output(json_path):
    """Load object detection results from YOLO JSON output, preserving metadata."""
    with open(json_path, 'r', encoding='utf-8') as f:
        yolo_data = json.load(f)

    entries = []
    for image_entry in yolo_data:
        image_id = image_entry.get("image_id", "unknown")
        objects_list = image_entry.get("objects", [])
        image_path = image_entry.get("image_path", "")
        global_context = image_entry.get("global_context", [])

        print(f"Processing image: {image_id}")
        if not objects_list:
            print(f"LI: Khng c objects trong nh {image_id}")
            continue

        entries.append({
            "image_id": image_id,
            "image_path": image_path,
            "objects": objects_list,
            "global_context": global_context,
        })

    return entries

def convert_yolo_to_reltr(objects_list, img_size):
    """Convert YOLO bounding boxes to RelTR format (normalized cx, cy, w, h)."""
    img_w, img_h = img_size
    objects = []

    for obj in objects_list:
        print("DEBUG:", obj)  # Ch in tng object  trnh li in c danh sch ln

        if "bbox" not in obj or "class" not in obj:
            continue
        
        x_min, y_min, x_max, y_max = obj["bbox"]
        x_c = (x_min + x_max) / 2 / img_w
        y_c = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        feature_vec = obj.get("feature", [])
        if feature_vec:
            feature_vec = [float(f) for f in feature_vec]
        objects.append({
            "label": obj["class"],
            "bbox": [x_c, y_c, w, h],
            "group_id": hash(obj["class"]), 
            "feature": feature_vec
        })
    return objects

def run_reltr_inference(objects, img_path, args, global_context=None, output_json="relationships.json"):
    """Run RelTR to infer relationships between detected objects and save results to a JSON file.
    
    OPTIMIZED: Uses cached model and FP16 inference for 3-5x speedup.
    """
    if len(objects) < 2:
        print("Lưu ý: Không đủ vật thể để dự đoán quan hệ!")
        return []

    device = _resolve_device(getattr(args, "device", None))
    
    # Use cached model instead of rebuilding every time
    model = _get_cached_reltr_model(args, device)
    use_fp16 = device.type == 'cuda'

    # Prepare ROI features
    feature_dim = next((len(obj.get("feature", [])) for obj in objects if obj.get("feature")), 0)
    roi_feature_tensor = None
    if feature_dim:
        feature_matrix = []
        for obj in objects:
            feat = obj.get("feature", [])
            if len(feat) != feature_dim:
                feat = [0.0] * feature_dim
            feature_matrix.append(feat)
        dtype = torch.float16 if use_fp16 else torch.float32
        roi_feature_tensor = torch.tensor(feature_matrix, device=device, dtype=dtype)
        roi_feature_tensor = F.normalize(roi_feature_tensor, p=2, dim=1)

    # Load and transform image
    img = Image.open(img_path)
    img_tensor = _RELTR_TRANSFORM(img).unsqueeze(0).to(device)
    if use_fp16:
        img_tensor = img_tensor.half()

    context_tensor = None
    if global_context:
        context_tensor = torch.tensor(global_context, dtype=torch.float32, device=device)
        if context_tensor.ndim == 1:
            context_tensor = context_tensor.unsqueeze(0)

    with torch.no_grad():
        if context_tensor is not None:
            outputs = model(img_tensor, global_context=context_tensor)
        else:
            outputs = model(img_tensor)

    rel_logits = outputs["rel_logits"].softmax(-1)[0, :, :-1]
    keep = rel_logits.max(-1).values > 0.4
    rel_scores = rel_logits[keep]
    if rel_scores.numel() == 0:
        rel_scores = rel_logits
    rel_scores = rel_scores.detach()
    num_rel_queries = rel_scores.shape[0]

    relationships = []
    pair_cursor = 0
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            subj = objects[i]["label"]
            obj = objects[j]["label"]
            if num_rel_queries:
                base_vector = rel_scores[pair_cursor % num_rel_queries]
            elif rel_logits.shape[0] > 0:
                base_vector = rel_logits.mean(dim=0)
            else:
                base_vector = torch.zeros(rel_logits.shape[-1], device=rel_logits.device)
            rel_idx = int(base_vector.argmax().item())
            similarity = None
            if roi_feature_tensor is not None and roi_feature_tensor.shape[0] == len(objects):
                sim = float(F.cosine_similarity(roi_feature_tensor[i], roi_feature_tensor[j], dim=0))
                adjusted_vector = base_vector * (1 + 0.25 * sim)
                rel_idx = int(adjusted_vector.argmax().item())
                similarity = sim
            relation = REL_CLASSES[rel_idx % len(REL_CLASSES)]
            
            # Apply spatial validation to correct mispredicted relationships
            subj_bbox = objects[i].get("bbox", [])
            obj_bbox = objects[j].get("bbox", [])
            validated_relation, rel_confidence = validate_spatial_relationship_bbox(
                subj_bbox, obj_bbox, subj, obj, relation
            )
            
            # Apply semantic validation to fix nonsensical relations
            final_relation, final_confidence = validate_semantic_relationship_bbox(
                subj, obj, validated_relation, rel_confidence
            )
            
            # Determine source based on corrections
            if final_relation != relation:
                source = "model_semantic_corrected" if validated_relation == relation else "model_spatial_corrected"
            else:
                source = "model"
            
            relation_entry = {
                "subject": subj,
                "relation": final_relation,
                "object": obj,
                "confidence": final_confidence,
                "source": source
            }
            if similarity is not None:
                relation_entry["visual_similarity"] = similarity
            relationships.append(relation_entry)
            pair_cursor += 1
    
    # === HEURISTIC FALLBACK: Generate person-vehicle relationships if RelTR missed them ===
    heuristic_rels = generate_heuristic_person_vehicle_relations(objects, relationships)
    if heuristic_rels:
        relationships.extend(heuristic_rels)
        print(f"🔍 Added {len(heuristic_rels)} heuristic person-vehicle relationships")
    
    threshold = 0.2
    filtered_relationships = []
    for rel in relationships:
        similarity = rel.get("visual_similarity")
        # Always keep heuristic_spatial relations (critical safety)
        if rel.get("source") == "heuristic_spatial":
            filtered_relationships.append(rel)
        elif similarity is None or similarity >= threshold:
            filtered_relationships.append(rel)
    
    print(f"📊 Lọc mối quan hệ: {len(filtered_relationships)}/{len(relationships)} đạt ngưỡng {threshold}")
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(filtered_relationships, f, indent=4, ensure_ascii=False)

    print(f"Kết quả được lưu vào {output_json}")

    return filtered_relationships

def draw_relationships(image_path, objects_list, relationships, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Khng th m hnh nh: {image_path}")
        return

    # V bounding box cho cc i tng
    for obj in objects_list:
        bbox = obj["bbox"]
        label = obj.get("class", "Unknown")
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # V cc mi quan h gia cc i tng
    for rel in relationships:
        subject_name = rel["subject"]
        object_name = rel["object"]
        relation = rel["relation"]

        subject_obj = next((obj for obj in objects_list if obj["class"] == subject_name), None)
        object_obj = next((obj for obj in objects_list if obj["class"] == object_name), None)

        if subject_obj and object_obj:
            x1, y1, _, _ = subject_obj["bbox"]
            x2, y2, _, _ = object_obj["bbox"]
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(image, relation, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            print(f"Li: Khng tm thy subject ({subject_name}) hoc object ({object_name})")

    #  m bo nh c lu cng th mc nh gc
    if output_path is None:
        output_dir = os.path.dirname(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"output_{image_name}.jpg")

    cv2.imwrite(output_path, image)
    print(f" nh  lu ti {output_path}")

    image = cv2.imread(image_path)

def main():
    parser = argparse.ArgumentParser('YOLO-RelTR Pipeline')
    parser.add_argument('--yolo_json', type=str, default='bboxes_for_reltr.json', help='Path to YOLO JSON output')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg', type=str, help='Dataset type (vg or oi)')
    parser.add_argument('--img_path', type=str, required=False, help='Fallback path to input image')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='ckpt/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")

    args = parser.parse_args()

    if not os.path.exists(args.yolo_json):
        print(f" Khng tm thy file JSON: {args.yolo_json}")
        return

    if not os.path.exists(args.resume):
        print(f" Khng tm thy checkpoint: {args.resume}")
        return

    if args.img_path and not os.path.exists(args.img_path):
        print(f" Khng tm thy nh fallback: {args.img_path}")
        return

    yolo_data = load_yolo_output(args.yolo_json)

    if not yolo_data:
        print("⚠️ Không có dữ liệu hợp lệ trong JSON đầu vào.")
        return

    all_relationships = {}

    for entry in yolo_data:
        image_id = entry['image_id']
        objects_list = entry['objects']
        image_path = entry.get('image_path') or args.img_path

        if not image_path:
            print(f"⚠️ Bỏ qua {image_id}: thiếu đường dẫn ảnh.")
            continue
        if not os.path.exists(image_path):
            print(f"⚠️ Bỏ qua {image_id}: đường dẫn ảnh không tồn tại ({image_path}).")
            continue

        img = Image.open(image_path)
        objects = convert_yolo_to_reltr(objects_list, img.size)

        if len(objects) < 2:
            print(f"Lu : nh {image_id} c {len(objects)} vt th, b qua!")
            continue

        relationships = run_reltr_inference(
            objects,
            image_path,
            args,
            global_context=entry.get('global_context'),
        )
        all_relationships[image_id] = relationships

        output_path = f"output_{Path(image_path).stem}.jpg"
        draw_relationships(image_path, objects_list, relationships, output_path)

    print(json.dumps(all_relationships, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
