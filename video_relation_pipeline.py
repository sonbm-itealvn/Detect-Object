import inspect
import json
import time
from collections import Counter
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.ops import nms as torchvision_nms
from PIL import Image, ImageDraw, ImageFont

SAFE_ZONE_CONFIG_PATH = Path("safe_zone_config.json")

import detect_objects as detection_pipeline
from RL.reinforcement_learning import RELATION_CLASSES
from RL.safety_classifier import SafetyClassifier, SafetyLevel
from models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list


FrameCallback = Optional[Callable[[np.ndarray], None]]
RelationCallback = Optional[Callable[[Dict[str, List[Dict[str, object]]]], None]]
_LOAD_STATE_HAS_ASSIGN = "assign" in inspect.signature(torch.nn.Module.load_state_dict).parameters


# ============= SPATIAL VALIDATION FOR RELATIONSHIPS =============
# Classes that should be considered for spatial relationship validation
VEHICLE_CLASSES = {
    "car", "vehicle", "truck", "trunk", "bus", "van", "automobile", "taxi", 
    "motorcycle", "motorbike", "bicycle", "bike"
}
PERSON_CLASSES = {
    "person", "human", "man", "woman", "child", "boy", "girl", 
    "people", "pedestrian", "worker"
}

# Additional class sets for semantic validation
ANIMAL_CLASSES = {
    "dog", "cat", "horse", "cow", "sheep", "bird", "elephant", "bear",
    "zebra", "giraffe", "animal", "pet"
}
TRANSPORT_CLASSES = {
    "skateboard", "surfboard", "snowboard", "bicycle", "bike", "motorcycle",
    "horse", "elephant", "scooter", "skis", "sled"
}
WEARABLE_CLASSES = {
    "hat", "cap", "shirt", "jacket", "coat", "pants", "shoes", "glasses",
    "tie", "dress", "helmet", "gloves", "bag", "backpack", "watch", "scarf"
}

# Spatial relationship mappings
ABOVE_RELATIONS = {"on", "above", "over", "riding", "sitting on", "standing on", "on back of"}
BELOW_RELATIONS = {"under", "below", "beneath", "lying on", "laying on"}

# ============= SEMANTIC RELATIONSHIP CORRECTIONS =============
# Rules: (subject_pattern, object_pattern, wrong_relations) -> correct_relation
SEMANTIC_CORRECTIONS = [
    # Animals on transport → should be "riding" not "wearing"
    {
        "subject": ANIMAL_CLASSES | PERSON_CLASSES,
        "object": TRANSPORT_CLASSES,
        "wrong_relations": {"wearing", "wears", "has", "holding", "carrying"},
        "correct_relation": "riding",
        "description": "Living beings ride transport, not wear them"
    },
    # Person/animal cannot "wear" large objects
    {
        "subject": ANIMAL_CLASSES | PERSON_CLASSES,
        "object": {"skateboard", "surfboard", "snowboard", "bicycle", "car", "truck", "table", "chair", "bed"},
        "wrong_relations": {"wearing", "wears"},
        "correct_relation": "on",
        "description": "Cannot wear large objects"
    },
    # Objects cannot "ride" things - only living beings can
    {
        "subject": {"table", "chair", "bottle", "cup", "book", "phone", "laptop"},
        "object": TRANSPORT_CLASSES | ANIMAL_CLASSES,
        "wrong_relations": {"riding", "sitting on", "standing on"},
        "correct_relation": "on",
        "description": "Inanimate objects are 'on' not 'riding'"
    },
]


def validate_semantic_relationship(
    subject_class: str,
    object_class: str,
    predicted_relation: str,
    confidence: float,
) -> Tuple[str, float]:
    """
    Validate and correct semantically invalid relationships.
    
    For example: "dog wearing skateboard" → "dog riding skateboard"
    """
    subject_lower = subject_class.lower().strip()
    object_lower = object_class.lower().strip()
    relation_lower = predicted_relation.lower().strip()
    
    for rule in SEMANTIC_CORRECTIONS:
        # Check if subject matches
        subject_match = any(s in subject_lower for s in rule["subject"])
        # Check if object matches
        object_match = any(o in object_lower for o in rule["object"])
        # Check if relation is in wrong_relations
        relation_wrong = relation_lower in rule["wrong_relations"]
        
        if subject_match and object_match and relation_wrong:
            correct = rule["correct_relation"]
            print(f"🔧 [Semantic] Correcting: '{subject_class} {predicted_relation} {object_class}' → '{correct}'")
            print(f"   Reason: {rule['description']}")
            return correct, max(confidence * 0.85, 0.6)
    
    return predicted_relation, confidence



def validate_spatial_relationship(
    subject_bbox: List[float],
    object_bbox: List[float],
    subject_class: str,
    object_class: str,
    predicted_relation: str,
    confidence: float,
) -> Tuple[str, float]:
    """
    Validate and correct predicted spatial relationships based on bounding box positions.
    
    This function checks if the predicted relation matches the actual geometric
    relationship between subject and object. For example, if RelTR predicts
    "person on car" but the person's bounding box is actually BELOW the car's
    bounding box (lower y-coordinate = higher in image), it should be "person under car".
    
    Args:
        subject_bbox: [x1, y1, x2, y2] bounding box of subject
        object_bbox: [x1, y1, x2, y2] bounding box of object
        subject_class: class name of subject (e.g., "person")
        object_class: class name of object (e.g., "car")
        predicted_relation: relation predicted by RelTR
        confidence: confidence score of prediction
        
    Returns:
        Tuple of (corrected_relation, adjusted_confidence)
    """
    if len(subject_bbox) < 4 or len(object_bbox) < 4:
        return predicted_relation, confidence
    
    # Normalize class names
    subject_lower = subject_class.lower().strip()
    object_lower = object_class.lower().strip()
    relation_lower = predicted_relation.lower().strip()
    
    # Calculate bounding box centers and positions
    subj_x1, subj_y1, subj_x2, subj_y2 = subject_bbox[:4]
    obj_x1, obj_y1, obj_x2, obj_y2 = object_bbox[:4]
    
    subj_center_y = (subj_y1 + subj_y2) / 2
    obj_center_y = (obj_y1 + obj_y2) / 2
    
    subj_bottom = subj_y2  # Bottom edge (higher y = lower in image)
    subj_top = subj_y1     # Top edge
    obj_bottom = obj_y2
    obj_top = obj_y1
    
    obj_height = max(obj_y2 - obj_y1, 1)
    subj_height = max(subj_y2 - subj_y1, 1)
    
    # Calculate vertical position difference ratio
    # Positive = subject is below object (in image coordinates where y increases downward)
    vertical_diff_ratio = (subj_center_y - obj_center_y) / obj_height
    
    # Check overlap in horizontal direction
    horizontal_overlap = (
        max(0, min(subj_x2, obj_x2) - max(subj_x1, obj_x1)) / 
        max(min(subj_x2 - subj_x1, obj_x2 - obj_x1), 1)
    )
    
    # Special case: Person and Vehicle interaction
    is_person_subject = any(p in subject_lower for p in PERSON_CLASSES)
    is_vehicle_object = any(v in object_lower for v in VEHICLE_CLASSES)
    
    if is_person_subject and is_vehicle_object and horizontal_overlap > 0.3:
        # Person is significantly BELOW the vehicle (center of person is lower in image)
        # This likely means person is UNDER the vehicle
        if vertical_diff_ratio > 0.5:  # Subject center is below object center by > 50% of object height
            if relation_lower in ABOVE_RELATIONS or relation_lower == "near":
                # Correct to "under" - this is likely a dangerous situation
                return "under", max(confidence * 0.9, 0.6)
        
        # Person is significantly ABOVE the vehicle
        # This likely means person is ON the vehicle
        elif vertical_diff_ratio < -0.3:  # Subject center is above object center
            if relation_lower in BELOW_RELATIONS:
                # Correct to "on"
                return "on", max(confidence * 0.9, 0.6)
    
    # Check if subject's bottom is below object's bottom (subject is lower in image)
    # This is a strong indicator that subject is physically UNDER the object
    if subj_bottom > obj_bottom + obj_height * 0.3:
        if relation_lower in ABOVE_RELATIONS:
            # Person detected as "on" car but they're actually below it
            if is_person_subject and is_vehicle_object:
                return "under", max(confidence * 0.85, 0.55)
    
    # Check if subject's top is above object's top (subject is higher in image)
    # This indicates subject is physically ABOVE/ON the object
    if subj_top < obj_top - obj_height * 0.2:
        if relation_lower in BELOW_RELATIONS:
            if is_person_subject and is_vehicle_object:
                return "on", max(confidence * 0.85, 0.55)
    
    # Additional check: very small subject compared to object, positioned at bottom
    size_ratio = (subj_height * (subj_x2 - subj_x1)) / max((obj_height * (obj_x2 - obj_x1)), 1)
    if size_ratio < 0.3 and subj_center_y > obj_center_y:
        # Small subject below large object - likely "under" or occluded
        if relation_lower in ABOVE_RELATIONS and is_person_subject and is_vehicle_object:
            return "under", max(confidence * 0.8, 0.5)
    
    return predicted_relation, confidence


def generate_heuristic_relationships(
    objects: List[Dict[str, object]],
    existing_relations: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    Generate heuristic-based relationships between person and vehicle when RelTR 
    doesn't detect them. This is a FALLBACK mechanism for critical safety scenarios.
    
    This function analyzes the geometric positions of detected objects and generates
    spatial relationships like "person under car", "child near vehicle", etc.
    
    Args:
        objects: List of detected objects with 'class' and 'bbox' keys
        existing_relations: List of relationships already detected by RelTR
        
    Returns:
        List of additional heuristic-generated relationships
    """
    heuristic_relations: List[Dict[str, object]] = []
    
    if len(objects) < 2:
        return heuristic_relations
    
    # Build dict of existing relationship pairs -> their relations
    # This allows us to check if we should override with a better spatial relation
    CRITICAL_SPATIAL_RELATIONS = {"under", "below", "beneath", "on", "above", "behind", "in front of"}
    
    existing_pairs_relations = {}  # (subj, obj) -> relation
    for rel in existing_relations:
        subj = rel.get('subject', '').lower()
        obj = rel.get('object', '').lower()
        relation = rel.get('relation', '').lower()
        existing_pairs_relations[(subj, obj)] = relation
        existing_pairs_relations[(obj, subj)] = relation  # Also check reverse
    
    # Find all persons and vehicles
    persons = []
    vehicles = []
    
    for idx, obj in enumerate(objects):
        # Check both 'class' and 'label' keys since different pipelines use different keys
        class_name = (obj.get('class') or obj.get('label') or '').lower().strip()
        bbox = obj.get('bbox', [])
        if len(bbox) < 4:
            continue
        
        # Debug: print detected classes
        if idx == 0:
            print(f"🔍 [Heuristic Debug] Object keys: {list(obj.keys())}, class_name: '{class_name}'")
            
        if any(p in class_name for p in PERSON_CLASSES):
            persons.append((idx, obj))
        elif any(v in class_name for v in VEHICLE_CLASSES):
            vehicles.append((idx, obj))
    
    print(f"🔍 [Heuristic] Found {len(persons)} persons, {len(vehicles)} vehicles")
    
    # Generate relationships for each person-vehicle pair
    for person_idx, person_obj in persons:
        person_class = person_obj.get('class', 'person')
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
            vehicle_class = vehicle_obj.get('class') or vehicle_obj.get('label') or 'car'
            vehicle_bbox = vehicle_obj.get('bbox', [])
            
            if len(vehicle_bbox) < 4:
                continue
            
            # Check if relationship already exists and if we should override
            pair_key = (person_class.lower(), vehicle_class.lower())
            existing_relation = existing_pairs_relations.get(pair_key, None)
            
            # Only skip if existing relation is already a critical spatial relation
            if existing_relation and existing_relation in CRITICAL_SPATIAL_RELATIONS:
                print(f"⏭️ [Heuristic] Skipping {pair_key} - has critical relation '{existing_relation}'")
                continue
            
            if existing_relation:
                print(f"🔄 [Heuristic] Will override {pair_key} - has weak relation '{existing_relation}'")
            
            print(f"📍 [Heuristic] Checking: {person_class} vs {vehicle_class}")
            print(f"   Person bbox: {person_bbox[:4]}, Vehicle bbox: {vehicle_bbox[:4]}")
                
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
            person_inside_vehicle_ratio = overlap_area / max(person_area, 1)
            
            print(f"   overlap_x={overlap_x:.1f}, overlap_y={overlap_y:.1f}")
            print(f"   h_overlap_ratio={horizontal_overlap_ratio:.2f}, person_inside={person_inside_vehicle_ratio:.2f}")
            
            # Determine relationship based on geometry
            relation = None
            confidence = 0.0
            description = ""
            
            # === CASE 1: Person UNDER vehicle ===
            # Scenario A: Person working under elevated vehicle (bbox overlaps)
            if person_inside_vehicle_ratio > 0.2:  # Relaxed from 0.3
                person_relative_y = (person_center_y - vehicle_top) / vehicle_height
                print(f"   person_relative_y={person_relative_y:.2f} (need > 0.4)")
                if person_relative_y > 0.4:  # Relaxed from 0.5
                    relation = "under"
                    confidence = 0.80
                    description = "Person inside vehicle bbox - working under"
            
            # Scenario B: Person is below vehicle (traditional case)
            if relation is None and horizontal_overlap_ratio > 0.2:
                if person_center_y > vehicle_bottom - vehicle_height * 0.3:
                    relation = "under"
                    confidence = 0.75
                    description = "Person below vehicle"
                elif py2 > vehicle_bottom:
                    relation = "under"
                    confidence = 0.65
                    description = "Person partially under vehicle"
            
            # === CASE 2: Person BEHIND/IN FRONT OF vehicle ===
            if relation is None and horizontal_overlap_ratio > 0.1:
                vertical_distance = abs(person_center_y - vehicle_center_y)
                if vertical_distance < vehicle_height * 0.7:
                    if person_center_x > vehicle_center_x + vehicle_width * 0.2:
                        relation = "behind"
                        confidence = 0.55
                        description = "Person in vehicle blind spot"
                    elif person_center_x < vehicle_center_x - vehicle_width * 0.2:
                        relation = "in front of"
                        confidence = 0.55
                        description = "Person in front of vehicle"
            
            # === CASE 3: Person NEAR vehicle (fallback) ===
            if relation is None:
                dist_x = max(0, max(px1 - vx2, vx1 - px2))
                dist_y = max(0, max(py1 - vy2, vy1 - py2))
                edge_distance = (dist_x**2 + dist_y**2)**0.5
                proximity_threshold = max(vehicle_width, vehicle_height) * 0.3
                if edge_distance < proximity_threshold:
                    relation = "near"
                    confidence = 0.50
                    description = "Person in proximity to vehicle"
            
            # Add the generated relationship (only if it's a critical spatial relation)
            if relation is not None and relation in CRITICAL_SPATIAL_RELATIONS:
                heuristic_relations.append({
                    'subject': person_class,
                    'relation': relation,
                    'object': vehicle_class,
                    'confidence': confidence,
                    'subject_track_id': person_obj.get('track_id'),
                    'object_track_id': vehicle_obj.get('track_id'),
                    'source': 'heuristic_spatial',
                    'heuristic_description': description,
                })
                existing_pairs_relations[(person_class.lower(), vehicle_class.lower())] = relation
                print(f"✅ [Heuristic] Generated: {person_class} {relation} {vehicle_class}")
    
    return heuristic_relations


class SafeZoneMonitor:
    """Monitor and render the 2m safety zone in front of the ego vehicle."""

    DEFAULT_NORMALIZED_POLYGON: Sequence[Tuple[float, float]] = (
        (0.35, 0.55),
        (0.65, 0.55),
        (0.85, 0.98),
        (0.15, 0.98),
    )
    DEFAULT_INTRUSION_CLASSES = {
        "person",
        "bicycle",
        "motorbike",
        "motorcycle",
        "car",
        "truck",
        "bus",
        "animal",
    }

    def __init__(self, config_path: Path = SAFE_ZONE_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.depth_m = float(self.config.get("safe_zone_depth_m", 2.0))
        self.width_m = float(self.config.get("safe_zone_width_m", 3.0))
        self.forward_offset_m = float(self.config.get("forward_offset_m", 0.0))
        self.hysteresis_frames = int(self.config.get("hysteresis_frames", 3))
        classes = self.config.get("intrusion_classes")
        if classes is None:
            self.monitor_classes = set(self.DEFAULT_INTRUSION_CLASSES)
        else:
            normalized = [cls for cls in classes if isinstance(cls, str)]
            self.monitor_classes = {cls.lower() for cls in normalized}
        self.alpha_normal = float(self.config.get("polygon_alpha_normal", 0.15))
        self.alpha_alert = float(self.config.get("polygon_alpha_alert", 0.3))
        self._homography = None
        self._homography_inv = None
        self._cached_polygons: Dict[Tuple[int, int], np.ndarray] = {}
        self._last_intrusion_frame = -999
        self._init_homography(self.config.get("homography"))

    def _load_config(self) -> Dict[str, object]:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _init_homography(self, homography_cfg: Optional[Dict[str, Sequence[Sequence[float]]]]):
        if not homography_cfg:
            return
        image_points = homography_cfg.get("image_points") or []
        world_points = homography_cfg.get("world_points") or []
        if len(image_points) < 4 or len(world_points) < 4:
            return
        try:
            src = np.array(world_points[:4], dtype=np.float32)
            dst = np.array(image_points[:4], dtype=np.float32)
            self._homography = cv2.getPerspectiveTransform(src, dst)
            self._homography_inv = np.linalg.inv(self._homography)
        except Exception:
            self._homography = None
            self._homography_inv = None

    def evaluate(self, frame_shape: Tuple[int, int, int], objects: List[Dict[str, object]], frame_idx: int):
        polygon = self._get_polygon(frame_shape)
        if polygon is None or len(polygon) < 3:
            return None, [], False
        contour = polygon.reshape((-1, 1, 2)).astype(np.int32)
        intrusions: List[Dict[str, object]] = []
        for idx, obj in enumerate(objects):
            bbox = obj.get("bbox", [0, 0, 0, 0])
            if not bbox or len(bbox) < 4:
                continue
            cls_name = str(obj.get("class", "")).lower()
            if self.monitor_classes and cls_name and cls_name not in self.monitor_classes:
                continue
            x1, y1, x2, y2 = bbox[:4]
            foot_point = np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)
            inside = cv2.pointPolygonTest(contour, tuple(float(v) for v in foot_point), False)
            if inside >= 0:
                distance_m = self._estimate_distance(foot_point, frame_shape)
                intrusion_entry = {
                    "object_index": idx,
                    "track_id": obj.get("track_id"),
                    "class": obj.get("class"),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "foot_point": [float(foot_point[0]), float(foot_point[1])],
                    "confidence": float(obj.get("confidence", 0.0)),
                    "distance_m": distance_m,
                }
                intrusions.append(intrusion_entry)
        danger_active = bool(intrusions)
        if danger_active:
            self._last_intrusion_frame = frame_idx
        elif frame_idx - self._last_intrusion_frame <= self.hysteresis_frames:
            danger_active = True
        return polygon, intrusions, danger_active

    def _get_polygon(self, frame_shape: Tuple[int, int, int]):
        height, width = frame_shape[:2]
        cache_key = (width, height)
        if cache_key in self._cached_polygons:
            return self._cached_polygons[cache_key]
        polygon = self._project_world_polygon(frame_shape)
        if polygon is None:
            polygon = self._polygon_from_normalized(frame_shape)
        if polygon is not None:
            self._cached_polygons[cache_key] = polygon
        return polygon

    def _polygon_from_normalized(self, frame_shape: Tuple[int, int, int]):
        height, width = frame_shape[:2]
        points = self.config.get("normalized_polygon") or self.DEFAULT_NORMALIZED_POLYGON
        if not points:
            return None
        polygon = np.array([[p[0] * width, p[1] * height] for p in points], dtype=np.float32)
        return polygon

    def _project_world_polygon(self, frame_shape: Tuple[int, int, int]):
        if self._homography is None:
            return None
        world_polygon = self.config.get("safe_zone_world")
        if not world_polygon:
            half_width = self.width_m / 2.0
            world_polygon = [
                [self.forward_offset_m, -half_width],
                [self.forward_offset_m + self.depth_m, -half_width],
                [self.forward_offset_m + self.depth_m, half_width],
                [self.forward_offset_m, half_width],
            ]
        try:
            pts = np.array(world_polygon, dtype=np.float32).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(pts, self._homography).reshape(-1, 2)
            return projected.astype(np.float32)
        except Exception:
            return None

    def _estimate_distance(self, image_point: np.ndarray, frame_shape: Tuple[int, int, int]) -> Optional[float]:
        if self._homography_inv is not None:
            try:
                pts = np.array(image_point, dtype=np.float32).reshape(-1, 1, 2)
                world_point = cv2.perspectiveTransform(pts, self._homography_inv).reshape(-1, 2)[0]
                return float(max(0.0, world_point[0]))
            except Exception:
                pass
        # Fallback: approximate using relative vertical position in frame
        height = max(1, frame_shape[0])
        rel = 1.0 - max(0.0, min(1.0, image_point[1] / height))
        approx_distance = max(0.0, self.depth_m * (rel ** 0.5))
        return float(round(approx_distance, 2))

    def get_alpha_values(self):
        return self.alpha_normal, self.alpha_alert


class RelTRInferenceEngine:
    """Lightweight helper that loads RelTR once and reuses decoding helpers."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._load_state_kwargs = {"strict": False}

    def _build_args(self):
        return dict(
            dataset='vg',
            lr_backbone=1e-5,
            backbone='resnet50',
            dilation=False,
            position_embedding='sine',
            enc_layers=6,
            dec_layers=6,
            dim_feedforward=2048,
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            num_entities=100,
            num_triplets=200,
            pre_norm=False,
            aux_loss=True,
            device=str(self.device),
            set_cost_class=1,
            set_cost_bbox=5,
            set_cost_giou=2,
            set_iou_threshold=0.7,
            bbox_loss_coef=5,
            giou_loss_coef=2,
            rel_loss_coef=1,
            eos_coef=0.1,
            return_interm_layers=False,
        )

    def _ensure_model(self):
        if self.model is not None:
            return self.model
        args = self._build_args()
        namespace_args = SimpleNamespace(**args)
        model, _, _ = build_model(namespace_args)
        self._materialize_model(model)
        if _LOAD_STATE_HAS_ASSIGN and "assign" in inspect.signature(model.load_state_dict).parameters:
            self._load_state_kwargs["assign"] = True
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
        if state:
            model.load_state_dict(state, **self._load_state_kwargs)
        model.to(self.device)
        model.eval()
        self.model = model
        return self.model

    def _materialize_model(self, model: torch.nn.Module):
        to_empty = getattr(model, "to_empty", None)
        if callable(to_empty):
            try:
                model.to_empty(device=self.device)
                return
            except RuntimeError:
                pass
        model.to(self.device)

    def _prepare_context(self, global_context: Optional[List[float]]):
        if not global_context:
            return None
        context_tensor = torch.tensor(global_context, dtype=torch.float32, device=self.device)
        if context_tensor.ndim == 1:
            context_tensor = context_tensor.unsqueeze(0)
        return context_tensor

    def infer(self, frame_bgr, objects, global_context):
        if not objects:
            return []
        model = self._ensure_model()
        pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_image).to(self.device)
        samples = nested_tensor_from_tensor_list([tensor])
        context_tensor = self._prepare_context(global_context)
        with torch.no_grad():
            outputs = model(samples, global_context=context_tensor) if context_tensor is not None else model(samples)
        return self._decode_relationships(outputs, objects, pil_image.size)

    def _decode_relationships(self, outputs, objects, image_size):
        rel_logits = outputs.get("rel_logits")
        if rel_logits is None or rel_logits.numel() == 0:
            return []
        try:
            rel_scores = rel_logits.softmax(-1)[0, :, :-1].detach().cpu()
        except Exception:
            return []
        if rel_scores.numel() == 0:
            return []

        width, height = image_size
        relationships: List[Dict[str, object]] = []
        use_geometric = (
            width is not None
            and height is not None
            and outputs.get("sub_boxes") is not None
            and outputs.get("obj_boxes") is not None
            and len(objects) >= 2
        )

        if use_geometric:
            try:
                object_boxes = torch.tensor([obj.get('bbox', [0.0, 0.0, 0.0, 0.0]) for obj in objects], dtype=torch.float32)
                if object_boxes.numel() > 0:
                    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
                    sub_boxes = outputs['sub_boxes'][0].detach().cpu()
                    obj_boxes = outputs['obj_boxes'][0].detach().cpu()
                    sub_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(sub_boxes) * scale
                    obj_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(obj_boxes) * scale
                    min_iou = 0.05
                    for idx in range(rel_scores.shape[0]):
                        rel_vector = rel_scores[idx]
                        rel_conf, rel_idx = rel_vector.max(dim=0)
                        if rel_conf.item() <= 0.0:
                            continue
                        subj_iou_vals = box_ops.box_iou(sub_boxes_xyxy[idx].unsqueeze(0), object_boxes)[0]
                        obj_iou_vals = box_ops.box_iou(obj_boxes_xyxy[idx].unsqueeze(0), object_boxes)[0]
                        subj_iou, subj_idx = subj_iou_vals.max(dim=0)
                        obj_iou, obj_idx = obj_iou_vals.max(dim=0)
                        if subj_iou.item() < min_iou or obj_iou.item() < min_iou:
                            continue
                        relation_name = RELATION_CLASSES[int(rel_idx) % len(RELATION_CLASSES)]
                        confidence = float(rel_conf.item() * max(subj_iou.item(), min_iou) * max(obj_iou.item(), min_iou))
                        
                        # Apply spatial validation to correct mispredicted relationships
                        subject_class = objects[int(subj_idx)].get('class', 'unknown')
                        object_class = objects[int(obj_idx)].get('class', 'unknown')
                        subject_bbox = objects[int(subj_idx)].get('bbox', [])
                        object_bbox = objects[int(obj_idx)].get('bbox', [])
                        
                        validated_relation, validated_confidence = validate_spatial_relationship(
                            subject_bbox, object_bbox,
                            subject_class, object_class,
                            relation_name, confidence
                        )
                        
                        # Apply semantic validation
                        final_relation, final_confidence = validate_semantic_relationship(
                            subject_class, object_class,
                            validated_relation, validated_confidence
                        )
                        
                        # Determine source
                        if final_relation != relation_name:
                            source = 'model_semantic_corrected' if validated_relation == relation_name else 'model_spatial_corrected'
                        else:
                            source = 'model'
                        
                        relationships.append({
                            'subject': subject_class,
                            'relation': final_relation,
                            'object': object_class,
                            'confidence': min(final_confidence, 1.0),
                            'subject_track_id': objects[int(subj_idx)].get('track_id'),
                            'object_track_id': objects[int(obj_idx)].get('track_id'),
                            'source': source,
                        })
                    if relationships:
                        return relationships
            except Exception:
                pass

        keep = rel_scores.max(-1).values > 0.4
        filtered = rel_scores[keep] if keep.any() else rel_scores
        num_queries = filtered.shape[0] or rel_scores.shape[0]
        pair_cursor = 0
        total_objects = len(objects)
        for i in range(total_objects):
            for j in range(i + 1, total_objects):
                if num_queries == 0:
                    continue
                vector = filtered[pair_cursor % num_queries]
                rel_idx = int(vector.argmax().item())
                confidence = float(vector.max().item())
                relation_name = RELATION_CLASSES[rel_idx % len(RELATION_CLASSES)]
                
                # Apply spatial validation to correct mispredicted relationships
                subject_class = objects[i].get('class', 'unknown')
                object_class = objects[j].get('class', 'unknown')
                subject_bbox = objects[i].get('bbox', [])
                object_bbox = objects[j].get('bbox', [])
                
                validated_relation, validated_confidence = validate_spatial_relationship(
                    subject_bbox, object_bbox,
                    subject_class, object_class,
                    relation_name, confidence
                )
                
                # Apply semantic validation to fix nonsensical relations
                final_relation, final_confidence = validate_semantic_relationship(
                    subject_class, object_class,
                    validated_relation, validated_confidence
                )
                
                # Determine source based on what corrections were made
                if final_relation != relation_name:
                    if validated_relation != relation_name:
                        source = 'fallback_spatial_corrected'
                    else:
                        source = 'fallback_semantic_corrected'
                else:
                    source = 'fallback'
                
                relationships.append({
                    'subject': subject_class,
                    'relation': final_relation,
                    'object': object_class,
                    'confidence': final_confidence,
                    'subject_track_id': objects[i].get('track_id'),
                    'object_track_id': objects[j].get('track_id'),
                    'source': source,
                })
                pair_cursor += 1
        
        # === HEURISTIC FALLBACK: Generate person-vehicle relationships if RelTR missed them ===
        heuristic_rels = generate_heuristic_relationships(objects, relationships)
        if heuristic_rels:
            relationships.extend(heuristic_rels)
        
        return relationships


class VideoRelationPipeline:
    """Process an entire video: detect -> track -> infer relationships -> voice announce."""

    def __init__(
        self,
        reltr_checkpoint: str = "reltr_finetuned.pth",
        tracker_config: Optional[str] = "bytetrack.yaml",
        min_confidence: float = 0.55,
        announce_min_confidence: float = 0.6,
        safe_zone_config: Path = SAFE_ZONE_CONFIG_PATH,
        safety_classifier_enabled: bool = True,
    ):
        self.rel_engine = RelTRInferenceEngine(reltr_checkpoint)
        self.yolo_model = detection_pipeline.yolo_model  # COCO model
        self.fire_model = detection_pipeline.fire_model  # Fire model (optional)
        self.tracker_config = tracker_config
        self.min_confidence = min_confidence
        self.announce_threshold = announce_min_confidence
        self.safe_zone = SafeZoneMonitor(safe_zone_config)
        
        # Khởi tạo Safety Classifier (3 tầng: White/Black/Gray list + LLM + Local Rules)
        self.safety_classifier = None
        if safety_classifier_enabled:
            try:
                self.safety_classifier = SafetyClassifier()
                print("✅ Safety Classifier initialized (3-tier system enabled)")
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize Safety Classifier: {e}")
        
        # Tracking cảnh báo
        self.alert_history = []
        self.max_alert_history = 100
        
        # Log dual model status
        if self.fire_model is not None:
            print(f"🔥 Video pipeline: Fire model loaded ({len(self.fire_model.names)} classes)")
            print(f"   Dual model detection enabled: COCO + Fire")
        else:
            print(f"⚠️  Video pipeline: Fire model not available, using COCO model only")

    def process_video(
        self,
        video_path: str,
        output_dir: str = "video_outputs",
        frame_stride: int = 2,
        on_frame: FrameCallback = None,
        on_relations: RelationCallback = None,
        stop_event: Optional[Event] = None,
    ) -> Dict[str, str]:
        video_path = str(video_path)
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        annotated_path = output_root / f"{Path(video_path).stem}_relations.avi"
        stats_path = output_root / f"{Path(video_path).stem}_summary.json"

        fps = self._read_video_fps(video_path)
        writer = None
        object_counter: Counter[str] = Counter()
        relation_counter: Counter[str] = Counter()
        stream = self.yolo_model.track(
            source=video_path,
            tracker=self.tracker_config,
            stream=True,
            persist=True,
            verbose=False,
        )
        try:
            for frame_idx, result in enumerate(stream):
                if stop_event and stop_event.is_set():
                    break
                if frame_stride > 1 and frame_idx % frame_stride != 0:
                    continue

                frame = result.orig_img.copy()
                feature_map = self._safe_consume_feature_map()
                global_context = detection_pipeline._compute_global_context(feature_map) if feature_map is not None else []
                
                # 🔥 Run fire model on this frame and merge with COCO results
                fire_result = None
                if self.fire_model is not None:
                    fire_result = self.fire_model(frame, verbose=False)
                
                # Merge detections from both models
                objects = self._prepare_objects(result, frame, feature_map, fire_result)
                for obj in objects:
                    label = obj.get("class")
                    if label:
                        object_counter[label] += 1
                relations = [
                    rel for rel in self.rel_engine.infer(frame, objects, global_context)
                    if rel.get("confidence", 0.0) >= self.min_confidence
                ]
                
                # Phân loại relationships với Safety Classifier (3 tầng)
                safety_alerts = []
                if self.safety_classifier and relations:
                    classified_relations = self.safety_classifier.classify_batch(relations)
                    for rel in classified_relations:
                        safety_level = rel.get("safety_level", "safe")
                        if safety_level in ["dangerous", "suspicious"]:
                            safety_alerts.append({
                                "subject": rel.get("subject", ""),
                                "relation": rel.get("relation", ""),
                                "object": rel.get("object", ""),
                                "level": safety_level,
                                "confidence": rel.get("safety_confidence", 0.0),
                                "explanation": rel.get("safety_explanation", ""),
                                "frame": frame_idx
                            })
                    relations = classified_relations
                
                for rel in relations:
                    key = f"{rel.get('subject','unknown')}|{rel.get('relation','')}|{rel.get('object','unknown')}"
                    relation_counter[key] += 1
                
                # Bỏ phần safe zone - chỉ chạy video bình thường
                polygon, intrusions, danger_active = None, [], False
                
                # Kiểm tra cảnh báo an toàn
                has_danger = any(alert.get("level") == "dangerous" for alert in safety_alerts)
                has_suspicious = any(alert.get("level") == "suspicious" for alert in safety_alerts)
                
                # Lưu vào lịch sử cảnh báo
                if safety_alerts:
                    self.alert_history.extend(safety_alerts)
                    if len(self.alert_history) > self.max_alert_history:
                        self.alert_history = self.alert_history[-self.max_alert_history:]
                
                annotated = self._draw_annotations(
                    frame.copy(),
                    objects,
                    relations,
                    polygon,
                    intrusions,
                    danger_active,
                    safety_alerts=safety_alerts
                )
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(annotated_path),
                        cv2.VideoWriter_fourcc(*"XVID"),
                        fps,
                        (width, height),
                    )
                writer.write(annotated)
                if on_frame:
                    on_frame(annotated.copy())
                if on_relations:
                    on_relations({
                        "frame": frame_idx,
                        "relations": relations,
                        "objects": objects,
                        "intrusions": intrusions,
                        "safe_zone": polygon.tolist() if polygon is not None else [],
                        "danger": danger_active,
                        "safety_alerts": safety_alerts,
                        "has_danger": has_danger,
                        "has_suspicious": has_suspicious,
                    })
        finally:
            if writer is not None:
                writer.release()
            # Thống kê cảnh báo
            alert_stats = {
                "total_alerts": len(self.alert_history),
                "dangerous_count": sum(1 for a in self.alert_history if a.get("level") == "dangerous"),
                "suspicious_count": sum(1 for a in self.alert_history if a.get("level") == "suspicious"),
                "alerts": self.alert_history[-20:] if len(self.alert_history) > 20 else self.alert_history
            }
            
            stats_payload = {
                "objects": [
                    {"label": label, "count": count}
                    for label, count in object_counter.most_common()
                ],
                "relations": [
                    {
                        "subject": parts[0],
                        "relation": parts[1] if len(parts) > 1 else "",
                        "object": parts[2] if len(parts) > 2 else "",
                        "count": count,
                    }
                    for key, count in relation_counter.most_common()
                    for parts in [key.split("|")]
                ],
                "safety_alerts": alert_stats,
            }
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats_payload, f, ensure_ascii=False, indent=2)
        return {"video": str(annotated_path), "summary": str(stats_path)}

    def _safe_consume_feature_map(self):
        try:
            return detection_pipeline._consume_feature_map()
        except Exception:
            return None

    def _prepare_objects(self, result, frame, feature_map, fire_result=None):
        """
        Prepare objects from COCO model (with tracking) and optionally merge with fire model detections.
        
        Args:
            result: YOLO result from COCO model (with tracking)
            frame: Current frame image
            feature_map: Feature map from COCO model backbone
            fire_result: Optional YOLO result from fire model
        """
        # Collect COCO detections (with tracking)
        coco_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        coco_classes = result.boxes.cls.cpu().numpy().astype(int)
        coco_confs = result.boxes.conf.cpu().numpy().tolist() if result.boxes.conf is not None else [0.0] * len(coco_boxes)
        track_ids = []
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(coco_boxes)
        
        # Collect fire detections (no tracking)
        fire_boxes = []
        fire_classes = []
        fire_confs = []
        if fire_result is not None and len(fire_result) > 0:
            fire_boxes = fire_result[0].boxes.xyxy.cpu().numpy().astype(int)
            fire_classes = fire_result[0].boxes.cls.cpu().numpy().astype(int)
            fire_confs = fire_result[0].boxes.conf.cpu().numpy().tolist() if fire_result[0].boxes.conf is not None else [0.0] * len(fire_boxes)
        
        # Merge detections using NMS (same as detect_objects.py)
        all_detections = []
        
        # Add COCO detections
        coco_names = self.yolo_model.model.names if hasattr(self.yolo_model, "model") and hasattr(self.yolo_model.model, "names") else self.yolo_model.names
        for idx, (bbox, cls_id, conf) in enumerate(zip(coco_boxes, coco_classes, coco_confs)):
            all_detections.append({
                'box': bbox,
                'class': coco_names[int(cls_id)],
                'confidence': float(conf),
                'source': 'coco',
                'track_id': track_ids[idx] if idx < len(track_ids) else None,
            })
        
        # Add fire detections
        if fire_result is not None and len(fire_result) > 0:
            fire_names = self.fire_model.model.names if hasattr(self.fire_model, "model") and hasattr(self.fire_model.model, "names") else self.fire_model.names
            for bbox, cls_id, conf in zip(fire_boxes, fire_classes, fire_confs):
                all_detections.append({
                    'box': bbox,
                    'class': fire_names[int(cls_id)],
                    'confidence': float(conf),
                    'source': 'fire',
                    'track_id': None,  # Fire detections don't have tracking
                })
        
        if not all_detections:
            return []
        
        # Apply NMS to merge overlapping detections
        # Convert to numpy array first to avoid warning
        boxes_array = np.array([det['box'] for det in all_detections], dtype=np.float32)
        boxes_tensor = torch.from_numpy(boxes_array)
        scores_tensor = torch.tensor([det['confidence'] for det in all_detections], dtype=torch.float32)
        keep_indices = torchvision_nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        
        # Process merged detections
        detected_objects = []
        yolo_labels = []
        metadata = []
        
        for idx in keep_indices:
            det = all_detections[idx]
            bbox = det['box']
            x1, y1, x2, y2 = map(int, bbox)
            
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
            
            cropped = detection_pipeline.add_padding(frame, (x1, y1, x2, y2))
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            detected_objects.append((cropped_pil, (x1, y1, x2, y2)))
            yolo_labels.append(det['class'])
            metadata.append({
                "track_id": det.get('track_id'),
                "confidence": det['confidence'],
                "source": det['source'],
            })
        
        if not detected_objects:
            return []
        
        classified = detection_pipeline.classify_with_clip(detected_objects, yolo_labels)
        box_list = [bbox for _, bbox in classified]
        roi_features = detection_pipeline.extract_roi_features(feature_map, box_list, frame.shape) if feature_map is not None else []
        
        # Initialize objects list
        objects = []
        for idx, (label, bbox) in enumerate(classified):
            x1, y1, x2, y2 = map(int, bbox)
            obj = {
                "class": label.strip(),
                "bbox": [x1, y1, x2, y2],
                "confidence": metadata[idx]["confidence"] if idx < len(metadata) else 0.0,
                "track_id": metadata[idx]["track_id"] if idx < len(metadata) else None,
            }
            if idx < len(roi_features) and roi_features[idx]:
                obj["feature"] = [float(v) for v in roi_features[idx]]
            objects.append(obj)
        return objects

    def _draw_annotations(self, frame, objects, relations, safe_zone=None, intrusions=None, danger_active=False, safety_alerts=None):
        intrusions = intrusions or []
        intrusion_lookup = {}
        for alert in intrusions:
            key = ("id", alert.get("track_id"))
            if alert.get("track_id") is None:
                key = ("bbox", tuple(alert.get("bbox", [])))
            intrusion_lookup[key] = alert
            intrusion_lookup[("idx", alert.get("object_index"))] = alert

        if safe_zone is not None and len(safe_zone) >= 3:
            contour = safe_zone.reshape((-1, 1, 2)).astype(np.int32)
            overlay = frame.copy()
            color = (0, 0, 255) if danger_active else (0, 255, 0)
            alpha_normal, alpha_alert = self.safe_zone.get_alpha_values()
            alpha = alpha_alert if danger_active else alpha_normal
            cv2.fillPoly(overlay, [contour], color)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.polylines(frame, [contour], True, color, 3)

        for idx, obj in enumerate(objects):
            x1, y1, x2, y2 = map(int, obj["bbox"])
            track_id = obj.get("track_id")
            label = obj.get("class", "obj")
            alert = None
            if track_id is not None and ("id", track_id) in intrusion_lookup:
                alert = intrusion_lookup[("id", track_id)]
            else:
                bbox_key = ("bbox", (x1, y1, x2, y2))
                if bbox_key in intrusion_lookup:
                    alert = intrusion_lookup[bbox_key]
                elif ("idx", idx) in intrusion_lookup:
                    alert = intrusion_lookup[("idx", idx)]
            color = (0, 0, 255) if alert else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            caption = f"{label}"
            if track_id is not None:
                caption = f"ID {track_id}: {label}"
            if alert and alert.get("distance_m") is not None:
                caption += f" | {alert['distance_m']:.1f}m"
            cv2.putText(frame, caption, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if danger_active:
            warning_text = "WARNING: object inside 2m safety zone"
            cv2.putText(
                frame,
                warning_text,
                (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3,
            )
            for idx, alert in enumerate(intrusions[:2]):
                txt = f"- {alert.get('class', 'object')} @ {alert.get('distance_m', 0):.1f}m"
                cv2.putText(frame, txt, (25, frame.shape[0] - 60 - idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Vẽ relationships với đường nối giữa subject và object
        self._draw_relationship_lines(frame, objects, relations)
        
        # Vẽ cảnh báo an toàn (nếu có)
        if safety_alerts:
            self._draw_safety_alerts(frame, safety_alerts)
        
        return frame

    def _draw_relationship_lines(self, frame, objects, relations):
        """Vẽ đường nối giữa subject và object với relation text trên đường"""
        if not relations or not objects:
            return
        
        # Tạo lookup dictionary cho objects
        objects_by_track_id = {}
        objects_by_class = {}
        for obj in objects:
            track_id = obj.get('track_id')
            class_name = obj.get('class', '').lower()
            if track_id is not None:
                objects_by_track_id[track_id] = obj
            if class_name:
                if class_name not in objects_by_class:
                    objects_by_class[class_name] = []
                objects_by_class[class_name].append(obj)
        
        # Màu sắc cho các relationships (BGR format)
        colors = [
            (0, 255, 255),    # Cyan
            (255, 0, 255),    # Magenta
            (255, 255, 0),    # Yellow
            (0, 165, 255),    # Orange
            (255, 0, 0),      # Blue
            (0, 255, 0),      # Green
            (128, 0, 128),    # Purple
            (255, 192, 203),  # Pink
        ]
        
        # Vẽ từng relationship
        for idx, rel in enumerate(relations[:15]):  # Giới hạn 15 relationships để tránh quá tải
            subject_name = rel.get('subject', '').lower()
            object_name = rel.get('object', '').lower()
            relation_text = rel.get('relation', '')
            confidence = rel.get('confidence', 0.0)
            
            # Tìm subject object
            subject_obj = None
            subject_track_id = rel.get('subject_track_id')
            if subject_track_id is not None and subject_track_id in objects_by_track_id:
                subject_obj = objects_by_track_id[subject_track_id]
            elif subject_name in objects_by_class and objects_by_class[subject_name]:
                subject_obj = objects_by_class[subject_name][0]
            
            # Tìm object object
            object_obj = None
            object_track_id = rel.get('object_track_id')
            if object_track_id is not None and object_track_id in objects_by_track_id:
                object_obj = objects_by_track_id[object_track_id]
            elif object_name in objects_by_class and objects_by_class[object_name]:
                object_obj = objects_by_class[object_name][0]
            
            # Nếu không tìm thấy cả hai, bỏ qua
            if not subject_obj or not object_obj:
                continue
            
            # Lấy bounding boxes
            sub_bbox = subject_obj.get('bbox', [])
            obj_bbox = object_obj.get('bbox', [])
            if len(sub_bbox) < 4 or len(obj_bbox) < 4:
                continue
            
            sub_x1, sub_y1, sub_x2, sub_y2 = map(int, sub_bbox[:4])
            obj_x1, obj_y1, obj_x2, obj_y2 = map(int, obj_bbox[:4])
            
            # Tính center points
            sub_center = ((sub_x1 + sub_x2) // 2, (sub_y1 + sub_y2) // 2)
            obj_center = ((obj_x1 + obj_x2) // 2, (obj_y1 + obj_y2) // 2)
            
            # Chọn màu
            color = colors[idx % len(colors)]
            
            # Vẽ mũi tên từ subject đến object (đậm, dễ nhìn)
            line_thickness = 3
            cv2.arrowedLine(
                frame,
                sub_center,
                obj_center,
                color,
                line_thickness,
                tipLength=0.15,
                line_type=cv2.LINE_AA
            )
            
            # Tính điểm giữa để đặt text
            mid_x = (sub_center[0] + obj_center[0]) // 2
            mid_y = (sub_center[1] + obj_center[1]) // 2
            
            # Chuẩn bị text để hiển thị
            display_text = f"{relation_text}"
            if confidence > 0:
                display_text += f" ({confidence:.2f})"
            
            # Tính kích thước text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)
            
            # Vẽ background cho text (để dễ đọc)
            padding = 5
            text_x = mid_x - text_width // 2
            text_y = mid_y - text_height // 2 - 10
            
            # Đảm bảo text không ra ngoài frame
            text_x = max(padding, min(text_x, frame.shape[1] - text_width - padding))
            text_y = max(text_height + padding, min(text_y, frame.shape[0] - padding))
            
            # Vẽ background rectangle
            bg_color = (0, 0, 0)  # Đen
            cv2.rectangle(
                frame,
                (text_x - padding, text_y - text_height - padding),
                (text_x + text_width + padding, text_y + baseline + padding),
                bg_color,
                -1
            )
            
            # Vẽ text với màu sáng
            text_color = (255, 255, 255)  # Trắng
            cv2.putText(
                frame,
                display_text,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA
            )

    def _draw_safety_alerts(self, frame, safety_alerts):
        """Vẽ cảnh báo an toàn trên frame với font hỗ trợ tiếng Việt"""
        if not safety_alerts or not self.safety_classifier:
            return
        
        # Phân loại cảnh báo theo mức độ
        dangerous_alerts = [a for a in safety_alerts if a.get("level") == "dangerous"]
        suspicious_alerts = [a for a in safety_alerts if a.get("level") == "suspicious"]
        
        # Chuyển frame sang PIL để vẽ text tiếng Việt
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Load font hỗ trợ tiếng Việt
        try:
            # Thử các font phổ biến trên Windows
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/tahoma.ttf",
                "C:/Windows/Fonts/calibri.ttf",
            ]
            font_large = None
            font_medium = None
            font_small = None
            
            for font_path in font_paths:
                if Path(font_path).exists():
                    try:
                        font_large = ImageFont.truetype(font_path, 36)  # Font lớn cho tiêu đề
                        font_medium = ImageFont.truetype(font_path, 24)  # Font vừa cho chi tiết
                        font_small = ImageFont.truetype(font_path, 20)   # Font nhỏ
                        break
                    except:
                        continue
            
            if font_large is None:
                # Fallback về font mặc định
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Vẽ cảnh báo nguy hiểm (ưu tiên cao nhất)
        if dangerous_alerts:
            alert_info = self.safety_classifier.get_alert_info(SafetyLevel.DANGEROUS)
            # Chuyển đổi màu từ RGB sang BGR (OpenCV dùng BGR)
            color_rgb = alert_info["color"]
            color = (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB -> BGR
            message = alert_info["message"]
            
            # Vẽ banner cảnh báo ở trên cùng (lớn hơn)
            banner_height = 100
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_height), color, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Chuyển lại sang PIL sau khi vẽ banner
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # Vẽ text cảnh báo chính (lớn, đậm)
            bbox = draw.textbbox((0, 0), message, font=font_large)
            text_width = bbox[2] - bbox[0]
            text_x = (frame.shape[1] - text_width) // 2
            text_y = 20
            
            # Vẽ outline (để text nổi bật)
            for adj in range(-2, 3):
                for adj2 in range(-2, 3):
                    draw.text((text_x + adj, text_y + adj2), message, font=font_large, fill=(0, 0, 0))
            draw.text((text_x, text_y), message, font=font_large, fill=(255, 255, 255))
            
            # Liệt kê các cảnh báo nguy hiểm
            y_offset = 110
            for idx, alert in enumerate(dangerous_alerts[:3]):  # Tối đa 3 cảnh báo
                subject = alert.get('subject', '?')
                relation = alert.get('relation', '?')
                obj = alert.get('object', '?')
                text = f"⚠️ {subject} {relation} {obj}"
                
                # Vẽ outline
                for adj in range(-1, 2):
                    for adj2 in range(-1, 2):
                        draw.text((22 + adj, y_offset + idx * 35 + adj2), text, font=font_medium, fill=(0, 0, 0))
                draw.text((22, y_offset + idx * 35), text, font=font_medium, fill=(255, 255, 255))
        
        # Vẽ cảnh báo nghi ngờ (mức độ thấp hơn)
        elif suspicious_alerts:
            alert_info = self.safety_classifier.get_alert_info(SafetyLevel.SUSPICIOUS)
            # Chuyển đổi màu từ RGB sang BGR (OpenCV dùng BGR)
            color_rgb = alert_info["color"]
            color = (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB -> BGR: [255, 165, 0] -> [0, 165, 255] (cam)
            message = alert_info["message"]
            
            # Vẽ banner cảnh báo nhẹ (lớn hơn)
            banner_height = 80
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_height), color, -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Chuyển lại sang PIL sau khi vẽ banner
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # Vẽ text cảnh báo chính
            bbox = draw.textbbox((0, 0), message, font=font_medium)
            text_width = bbox[2] - bbox[0]
            text_x = (frame.shape[1] - text_width) // 2
            text_y = 20
            
            # Vẽ outline
            for adj in range(-1, 2):
                for adj2 in range(-1, 2):
                    draw.text((text_x + adj, text_y + adj2), message, font=font_medium, fill=(0, 0, 0))
            draw.text((text_x, text_y), message, font=font_medium, fill=(255, 255, 255))
            
            # Liệt kê các cảnh báo nghi ngờ
            y_offset = 90
            for idx, alert in enumerate(suspicious_alerts[:2]):  # Tối đa 2 cảnh báo
                subject = alert.get('subject', '?')
                relation = alert.get('relation', '?')
                obj = alert.get('object', '?')
                text = f"⚠️ {subject} {relation} {obj}"
                
                # Vẽ outline
                for adj in range(-1, 2):
                    for adj2 in range(-1, 2):
                        draw.text((22 + adj, y_offset + idx * 30 + adj2), text, font=font_small, fill=(0, 0, 0))
                draw.text((22, y_offset + idx * 30), text, font=font_small, fill=(255, 255, 255))
        
        # Chuyển lại sang OpenCV format
        frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        frame[:] = frame_bgr[:]
    
    @staticmethod
    def _read_video_fps(video_path: str) -> float:
        capture = cv2.VideoCapture(video_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
        capture.release()
        return fps
