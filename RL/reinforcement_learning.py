# File: reinforcement_learning.py
import copy
import torch
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, Counter
import random
import math
import os
import json
import shutil
import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO

import detect_objects as detection_pipeline

from util import box_ops
from util.misc import nested_tensor_from_tensor_list
from RL.model_manager import ModelManager
from models import build_model


RELATION_CLASSES = [
    "__background__", "above", "across", "against", "along", "and", "at", "attached to",
    "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating",
    "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in",
    "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near",
    "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing",
    "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in",
    "walking on", "watching", "wearing", "wears", "with"
]

class RelationshipReinforcementLearning:
    def __init__(
        self,
        detection_model,
        relationship_model,
        generator,
        experiment_dir: Optional[str] = None,
        data_paths: Optional[Dict[str, Optional[str]]] = None,
    ):
        self.detection_model = detection_model  # Will be lazily loaded if None
        self.relationship_model = relationship_model  # Will be lazily loaded if None
        self.generator = generator
        self.data_paths = data_paths or {}

        self.memory = deque(maxlen=10000)
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.detection_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reltr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reltr_args: Optional[SimpleNamespace] = None
        self.reltr_criterion = None
        self.reltr_postprocessors = None
        self.reltr_optimizer: Optional[AdamW] = None
        self.reltr_transform = self._build_reltr_transform()

        self.entity_label_to_idx: Dict[str, int] = {}
        self.relation_label_to_idx: Dict[str, int] = {
            self._normalize_label(name): idx for idx, name in enumerate(RELATION_CLASSES)
        }

        self.detection_dataset_dir: Optional[Path] = None
        self.dataset_samples: List[Dict[str, Any]] = []

        # Model management
        self.model_manager = ModelManager()
        if experiment_dir:
            self.model_manager.set_experiment_dir(experiment_dir)

        # Training history
        self.training_history = {
            'epochs': [],
            'best_reward': float('-inf'),
            'best_epoch': 0
        }
        
        # Performance tracking for adaptive scoring
        self.performance_history = {
            'rewards': deque(maxlen=50),  # Last 50 rewards
            'detection_scores': deque(maxlen=50),
            'relationship_scores': deque(maxlen=50),
            'diversity_scores': deque(maxlen=50),
            'consistency_scores': deque(maxlen=50),
            'improvement_trend': deque(maxlen=20),  # Last 20 improvement scores
            'weight_history': deque(maxlen=20),  # Track weight changes
        }
        
        # Adaptive scoring parameters
        self.scaling_factor = 1.0
        self.baseline_performance = {
            'detection': 0.3,
            'relationship': 0.7,
            'diversity': 0.3,
            'consistency': 0.5,
        }

        # Deep Q-Network agent configuration
        self.rl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = [1, 3, 5, 7]
        self.state_dim = 5
        self.q_network = self._build_q_network(self.state_dim, len(self.action_space)).to(self.rl_device)
        self.target_network = self._build_q_network(self.state_dim, len(self.action_space)).to(self.rl_device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.q_optimizer = AdamW(self.q_network.parameters(), lr=1e-3)
        self.gamma = 0.95
        self.batch_size = 32
        self.target_update_interval = 20
        self.learn_step_counter = 0
        self.last_metrics = {
            'detection_loss': 1.0,
            'relationship_loss': 1.0,
            'reward': 0.0,
            'dataset_size': 0,
        }
        self.last_state = self._build_state_vector(self.last_metrics)
        self.last_action_index: Optional[int] = None
        self.latest_reward_components: Dict[str, float] = {}
        self.latest_detection_metrics: Dict[str, float] = {}
        self.latest_relationship_metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_reltr_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _normalize_label(label: str) -> str:
        return label.strip().lower().replace("_", " ").replace("-", " ")

    def _build_q_network(self, input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    @staticmethod
    def _normalize_scalar(value: float, scale: float = 1.0) -> float:
        if scale <= 0:
            scale = 1.0
        return math.tanh(value / scale)

    def _build_state_vector(self, metrics: Optional[Dict[str, float]] = None) -> torch.Tensor:
        metrics = metrics or self.last_metrics
        detection_loss = self._normalize_scalar(float(metrics.get('detection_loss', 1.0)), scale=5.0)
        relationship_loss = self._normalize_scalar(float(metrics.get('relationship_loss', 1.0)), scale=5.0)
        reward_value = self._normalize_scalar(float(metrics.get('reward', 0.0)), scale=1.0)
        dataset_size = metrics.get('dataset_size', len(self.dataset_samples))
        dataset_norm = self._normalize_scalar(float(dataset_size), scale=50.0)
        epsilon_value = self._normalize_scalar(float(self.epsilon), scale=1.0)
        state = torch.tensor(
            [detection_loss, relationship_loss, reward_value, dataset_norm, epsilon_value],
            dtype=torch.float32,
            device=self.rl_device,
        )
        return state

    def _select_action(self, state: torch.Tensor) -> Tuple[int, int]:
        if random.random() < self.epsilon:
            action_index = random.randrange(len(self.action_space))
        else:
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                action_index = int(q_values.argmax(dim=1).item())
        action_value = self.action_space[action_index]
        return action_index, action_value

    def decide_action(self) -> Dict[str, Any]:
        """
        Choose an action for the next training episode using epsilon-greedy DQN policy.
        Returns a context dictionary that should be passed back after the episode completes.
        """
        state = self._build_state_vector()
        action_index, action_value = self._select_action(state)
        self.last_state = state
        self.last_action_index = action_index
        return {
            'state': state.clone().detach(),
            'action_index': action_index,
            'num_variations': action_value,
            'epsilon': self.epsilon,
        }

    def _ensure_detection_model(self):
        if self.detection_model is not None:
            return self.detection_model

        weights_path = self.data_paths.get('yolo_weights')
        if weights_path and os.path.exists(weights_path):
            resolved = weights_path
        else:
            fallback_path = Path.cwd() / "yolov5s.pt"
            resolved = weights_path if weights_path else str(fallback_path)
            if not os.path.exists(resolved):
                raise FileNotFoundError(
                    f"YOLO weights not found. Expected at '{weights_path}' or '{fallback_path}'."
                )

        print(f"[RL] Loading YOLO model from {resolved}")
        self.detection_model = YOLO(resolved)
        return self.detection_model

    def _build_reltr_args(self) -> SimpleNamespace:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SimpleNamespace(
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
            device=device,
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

    def _ensure_relationship_model(self):
        if self.relationship_model is not None and self.reltr_criterion is not None:
            return self.relationship_model, self.reltr_criterion

        args = self.reltr_args if self.reltr_args is not None else self._build_reltr_args()
        model, criterion, postprocessors = build_model(args)

        checkpoint_path = self.data_paths.get('reltr_checkpoint')
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[RL] Loading RelTR checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.reltr_device, weights_only=False)
            state_dict = checkpoint.get('model') if isinstance(checkpoint, dict) else None
            if state_dict:
                model.load_state_dict(state_dict, strict=False)

        model.to(self.reltr_device)
        criterion.to(self.reltr_device)

        self.relationship_model = model
        self.reltr_criterion = criterion
        self.reltr_postprocessors = postprocessors
        self.reltr_args = args
        return self.relationship_model, self.reltr_criterion

    def _load_original_detection_and_relationships(self) -> Optional[Dict[str, Any]]:
        image_path = self.data_paths.get('image')
        detections_path = self.data_paths.get('converted_bboxes')
        relationships_path = self.data_paths.get('relationships')

        if not image_path or not os.path.exists(image_path):
            print("[RL] Original image path missing, skipping dataset preparation.")
            return None
        if not detections_path or not os.path.exists(detections_path):
            print("[RL] Detection JSON missing, cannot prepare dataset.")
            return None

        with open(detections_path, 'r', encoding='utf-8') as f:
            detection_data = json.load(f)

        if not detection_data:
            print("[RL] Detection JSON empty, cannot prepare dataset.")
            return None

        objects = detection_data[0].get('objects', [])
        if not objects:
            print("[RL] No objects found in detection data.")
            return None

        with Image.open(image_path) as img:
            width, height = img.size

        normalized_objects: List[Dict[str, Any]] = []
        for obj in objects:
            new_obj = dict(obj)
            label_value = new_obj.get('class') or new_obj.get('label')
            if 'class' not in new_obj and label_value:
                new_obj['class'] = label_value
            if 'yolo_class' not in new_obj and label_value:
                new_obj['yolo_class'] = label_value
            normalized_objects.append(new_obj)

        relationships = []
        global_context = detection_data[0].get('global_context', [])
        if not global_context:
            try:
                _, _, _, _, global_context = detection_pipeline.detect_objects(image_path)
            except Exception:
                global_context = []
        if relationships_path and os.path.exists(relationships_path):
            with open(relationships_path, 'r', encoding='utf-8') as f:
                relationships = json.load(f)

        return {
            'image_path': image_path,
            'width': width,
            'height': height,
            'objects': normalized_objects,
            'relationships': relationships,
            'global_context': global_context,
        }

    def _match_detection_label(self, class_name: str, names_map: Dict[int, str]) -> Optional[int]:
        target = self._normalize_label(class_name)
        for idx, name in names_map.items():
            if self._normalize_label(name) == target:
                return int(idx)
        return None

    def _write_yolo_label_file(
        self,
        label_path: Path,
        objects: List[Dict[str, Any]],
        image_size: Tuple[int, int],
        names_map: Dict[int, str],
    ) -> int:
        width, height = image_size
        lines: List[str] = []
        for obj in objects:
            bbox = obj.get('bbox')
            class_name = obj.get('yolo_class') or obj.get('class')
            if not bbox or class_name is None:
                continue
            class_idx = self._match_detection_label(class_name, names_map)
            if class_idx is None:
                continue
            x1, y1, x2, y2 = bbox
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            x_c = (x1 + x2) / 2.0
            y_c = (y1 + y2) / 2.0
            line = f"{class_idx} {x_c / width:.6f} {y_c / height:.6f} {bw / width:.6f} {bh / height:.6f}"
            lines.append(line)

        if not lines:
            return 0

        with open(label_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        return len(lines)

    def _create_dataset_yaml(self, dataset_dir: Path, names_map: Dict[int, str]) -> Path:
        yaml_path = dataset_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {dataset_dir.resolve()}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("names:\n")
            for idx, name in sorted(names_map.items()):
                f.write(f"  {idx}: {name}\n")
        return yaml_path

    def _prepare_detection_dataset(self) -> Optional[Path]:
        cached = self.detection_dataset_dir
        if cached and cached.exists():
            return cached
        if not self.dataset_samples:
            fallback = self._load_original_detection_and_relationships()
            if fallback:
                self.dataset_samples = [fallback]

        if not self.dataset_samples:
            print("[RL] No dataset samples available for detection fine-tuning.")
            return None

        experiment_dir = Path(self.model_manager.current_experiment_dir or Path.cwd())
        dataset_dir = experiment_dir / "rl_detection_dataset"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

        detection_model = self._ensure_detection_model()
        names_map = detection_model.model.names if hasattr(detection_model.model, "names") else detection_model.names

        samples = list(self.dataset_samples)
        if not samples:
            print("[RL] No samples to export for detection dataset.")
            return None

        split_index = max(1, int(len(samples) * 0.8))
        train_samples = samples[:split_index]
        val_samples = samples[split_index:] if split_index < len(samples) else samples[-1:]

        def export_split(split_name: str, split_samples: List[Dict[str, Any]]) -> int:
            written = 0
            for idx, sample in enumerate(split_samples):
                image_path = Path(sample['image_path'])
                unique_name = f"{image_path.stem}_{idx:04d}{image_path.suffix}"
                target_image = dataset_dir / "images" / split_name / unique_name
                shutil.copy(image_path, target_image)

                label_target = dataset_dir / "labels" / split_name / f"{target_image.stem}.txt"
                written += self._write_yolo_label_file(
                    label_target,
                    sample['objects'],
                    (sample['width'], sample['height']),
                    names_map,
                )
            return written

        train_written = export_split("train", train_samples)
        val_written = export_split("val", val_samples)

        if train_written == 0 and val_written == 0:
            print("[RL] Warning: no labels written for detection dataset.")

        yaml_path = self._create_dataset_yaml(dataset_dir, names_map)
        print(f"[RL] Detection dataset prepared at {dataset_dir} (yaml: {yaml_path})")

        self.detection_dataset_dir = dataset_dir
        self._save_dataset_snapshot()
        return dataset_dir

    def _dataset_snapshot_path(self) -> Optional[Path]:
        experiment_dir = self.model_manager.current_experiment_dir
        if not experiment_dir:
            return None
        snapshot_dir = Path(experiment_dir) / "dataset"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        return snapshot_dir / "samples.json"

    def _save_dataset_snapshot(self) -> None:
        path = self._dataset_snapshot_path()
        if not path:
            return
        payload = {
            "samples": self.dataset_samples,
            "detection_dataset_dir": str(self.detection_dataset_dir) if self.detection_dataset_dir else None,
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[RL] Warning: failed to save dataset snapshot: {exc}")

    def load_dataset_snapshot(self) -> bool:
        path = self._dataset_snapshot_path()
        if not path or not path.exists():
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[RL] Warning: failed to load dataset snapshot: {exc}")
            return False

        samples = payload.get("samples") or []
        detection_dir = payload.get("detection_dataset_dir")

        self.dataset_samples = samples
        if detection_dir and Path(detection_dir).exists():
            self.detection_dataset_dir = Path(detection_dir)
        else:
            self.detection_dataset_dir = None

        return bool(self.dataset_samples)

    def _ensure_entity_label_index(self, class_name: str) -> int:
        normalized = self._normalize_label(class_name)
        if normalized not in self.entity_label_to_idx:
            self.entity_label_to_idx[normalized] = len(self.entity_label_to_idx)
        return self.entity_label_to_idx[normalized]

    def _relation_to_index(self, relation: str) -> Optional[int]:
        normalized = self._normalize_label(relation)
        return self.relation_label_to_idx.get(normalized)

    @staticmethod
    def _xyxy_to_cxcywh_norm(bbox: List[float], width: int, height: int) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0 / max(width, 1)
        cy = (y1 + y2) / 2.0 / max(height, 1)
        w = (x2 - x1) / max(width, 1)
        h = (y2 - y1) / max(height, 1)
        return cx, cy, w, h

    def _find_object_index(self, objects: List[Dict[str, Any]], class_name: str) -> Optional[int]:
        normalized = self._normalize_label(class_name)
        matches = [
            (idx, obj) for idx, obj in enumerate(objects)
            if self._normalize_label(obj.get('class', '')) == normalized
        ]
        if not matches:
            return None
        return matches[0][0]

    def _build_relationship_from_original(
        self,
        objects: List[Dict[str, Any]],
        original_relationship: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not original_relationship:
            return []
        subject_idx = self._find_object_index(objects, original_relationship.get('subject', ''))
        object_idx = self._find_object_index(objects, original_relationship.get('object', ''))
        relation = original_relationship.get('relation')
        if subject_idx is None or object_idx is None or not relation:
            return []
        subject_label = objects[subject_idx].get('class', original_relationship.get('subject', 'unknown'))
        object_label = objects[object_idx].get('class', original_relationship.get('object', 'unknown'))
        confidence = float(original_relationship.get('confidence', 1.0))
        return [{
            'subject': subject_label,
            'relation': relation,
            'object': object_label,
            'confidence': max(0.0, min(confidence, 1.0)),
            'source': 'original',
        }]

    def _build_reltr_target(
        self,
        sample: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        width, height = sample['width'], sample['height']
        objects = sample['objects']
        relationships = sample.get('relationships', [])

        if not objects:
            print("[RL] No objects found in sample")
            return None

        boxes = []
        labels = []
        for obj in objects:
            bbox = obj.get('bbox')
            class_name = obj.get('class')
            if not bbox or not class_name:
                continue
            boxes.append(self._xyxy_to_cxcywh_norm(bbox, width, height))
            labels.append(self._ensure_entity_label_index(class_name))

        if not boxes:
            print("[RL] No valid bounding boxes found in sample")
            return None

        rel_annotations = []
        for rel in relationships:
            subj_idx = self._find_object_index(objects, rel.get('subject', ''))
            obj_idx = self._find_object_index(objects, rel.get('object', ''))
            rel_idx = self._relation_to_index(rel.get('relation', ''))
            if subj_idx is None or obj_idx is None or rel_idx is None:
                print(f"[RL] Skipping invalid relationship: {rel} (subj_idx={subj_idx}, obj_idx={obj_idx}, rel_idx={rel_idx})")
                continue
            rel_annotations.append([subj_idx, obj_idx, rel_idx])

        if not rel_annotations:
            print(f"[RL] No valid relationship annotations found for RelTR training. Found {len(relationships)} relationships but none were valid.")
            print(f"[RL] Objects in sample: {[obj.get('class', 'unknown') for obj in objects]}")
            print(f"[RL] Relationships: {relationships}")
            return None

        print(f"[RL] Successfully created {len(rel_annotations)} valid relationship annotations")
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'rel_annotations': torch.tensor(rel_annotations, dtype=torch.long),
            'image_id': torch.tensor([0], dtype=torch.long),
            'orig_size': torch.tensor([height, width], dtype=torch.long),
            'size': torch.tensor([height, width], dtype=torch.long),
        }
        return target

    def _load_image_tensor(self, image_path: str) -> torch.Tensor:
        pil_image = self._ensure_pil_image(image_path)
        if pil_image is None:
            raise FileNotFoundError(f"Unable to load image from {image_path}")
        return self.reltr_transform(pil_image)

    def _ensure_pil_image(self, image: Any) -> Optional[Image.Image]:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str) and os.path.exists(image):
            with Image.open(image) as img:
                return img.convert("RGB").copy()
        if isinstance(image, np.ndarray):
            array = image.astype(np.uint8)
            return Image.fromarray(array)
        return None

    def _ensure_synthetic_image_path(self, data: Dict[str, Any], index: int) -> Optional[str]:
        path_candidate = data.get('image_path') or data.get('saved_path')
        if path_candidate and os.path.exists(path_candidate):
            return path_candidate

        image_obj = data.get('image')
        if not isinstance(image_obj, Image.Image):
            return None

        experiment_dir = Path(self.model_manager.current_experiment_dir or Path.cwd())
        target_dir = experiment_dir / "synthetic_cache"
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"synthetic_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{index:04d}.jpg"
        output_path = target_dir / filename
        try:
            image_obj.save(output_path, format="JPEG", quality=95)
        except Exception as exc:
            print(f"[RL] Failed to persist synthetic image: {exc}")
            return None

        try:
            image_obj.close()
        except Exception:
            pass

        resolved = str(output_path)
        data['image_path'] = resolved
        data['saved_path'] = resolved
        return resolved

    def _prepare_global_context_tensor(self, context_vector: Optional[List[float]]) -> Optional[torch.Tensor]:
        if not context_vector:
            return None
        tensor = torch.tensor(context_vector, dtype=torch.float32, device=self.reltr_device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _detect_objects_for_image(self, pil_image: Image.Image) -> Tuple[List[Dict[str, Any]], Optional[List[float]]]:
        detection_model = self._ensure_detection_model()
        try:
            _, _, _, _, global_context = detection_pipeline.detect_objects(pil_image)
        except Exception:
            global_context = []
        results = detection_model.predict(
            pil_image,
            imgsz=640,
            conf=0.25,
            verbose=False,
            device=self.detection_device,
        )
        if not results:
            return [], global_context
        result = results[0]
        names_map = detection_model.model.names if hasattr(detection_model.model, "names") else detection_model.names
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        objects: List[Dict[str, Any]] = []
        for bbox, cls_id in zip(boxes_xyxy, classes):
            objects.append({
                'bbox': bbox.tolist(),
                'class': names_map[int(cls_id)],
            })
        return objects, global_context

    def _run_reltr_inference(
        self,
        image_tensor: torch.Tensor,
        objects: List[Dict[str, Any]],
        global_context: Optional[List[float]] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        if not objects:
            return []

        model, _ = self._ensure_relationship_model()
        samples = nested_tensor_from_tensor_list([image_tensor.to(self.reltr_device)])
        context_tensor = self._prepare_global_context_tensor(global_context)
        model.eval()
        with torch.no_grad():
            if context_tensor is not None:
                outputs = model(samples, global_context=context_tensor)
            else:
                outputs = model(samples)
        return self._decode_relationships(outputs, objects, image_size=image_size)

    def _extract_objects_with_clip(self, image_path: str) -> Optional[Dict[str, Any]]:
        try:
            detected_objects, yolo_labels, original_image, feature_map, global_context = detection_pipeline.detect_objects(image_path)
        except Exception as exc:
            print(f"[RL] Detection failed for {image_path}: {exc}")
            return None

        if not detected_objects:
            print(f"[RL] No objects detected in {image_path}")
            return None

        classified_results = detection_pipeline.classify_with_clip(detected_objects, yolo_labels)
        boxes = [bbox for _, bbox in classified_results]
        roi_features = detection_pipeline.extract_roi_features(feature_map, boxes, original_image.shape)

        height, width = original_image.shape[:2]
        objects: List[Dict[str, Any]] = []
        for idx, (label, bbox) in enumerate(classified_results):
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            feature_vector = roi_features[idx] if idx < len(roi_features) else []
            if feature_vector:
                feature_vector = [float(v) for v in feature_vector]

            objects.append({
                'class': label.strip(),
                'yolo_class': yolo_labels[idx] if idx < len(yolo_labels) else label.strip(),
                'bbox': [x1, y1, x2, y2],
                'feature': feature_vector,
            })

        if not objects:
            print(f"[RL] No valid objects after classification in {image_path}")
            return None

        sample = {
            'image_path': image_path,
            'width': width,
            'height': height,
            'objects': objects,
            'global_context': global_context,
        }
        return sample

    def _ingest_synthetic_samples(self, synthetic_data: List[Dict[str, Any]]) -> int:
        if not synthetic_data:
            print("[RL] No synthetic data provided for ingestion")
            return 0

        existing_paths = {
            str(Path(sample.get('image_path')).resolve())
            for sample in self.dataset_samples
            if sample.get('image_path')
        }
        ingested = 0

        print(f"[RL] Processing {len(synthetic_data)} synthetic samples for ingestion")
        for index, data in enumerate(synthetic_data):
            image_path = self._ensure_synthetic_image_path(data, index)
            if not image_path or not os.path.exists(image_path):
                print(f"[RL] Skipping sample {index+1}: invalid image path")
                continue

            resolved_path = str(Path(image_path).resolve())
            if resolved_path in existing_paths:
                print(f"[RL] Skipping sample {index+1}: already exists")
                continue

            print(f"[RL] Processing synthetic sample {index+1}/{len(synthetic_data)}")
            sample = self._extract_objects_with_clip(image_path)
            if not sample:
                print(f"[RL] Skipping sample {index+1}: failed to extract objects")
                continue

            original_relationship = data.get('original_relationship')
            print(f"[RL] Sample {index+1} original relationship: {original_relationship}")
            
            # First try to build relationship from original
            relationships = self._build_relationship_from_original(sample['objects'], original_relationship)
            print(f"[RL] Sample {index+1} built {len(relationships)} relationships from original")
            
            # If no relationships from original, try RelTR inference
            if not relationships:
                try:
                    print(f"[RL] Sample {index+1}: attempting RelTR inference for relationship prediction")
                    image_tensor = self._load_image_tensor(sample['image_path'])
                    relationships = self._run_reltr_inference(
                        image_tensor,
                        sample['objects'],
                        sample.get('global_context'),
                        (sample['width'], sample['height']),
                    )
                    print(f"[RL] Sample {index+1} RelTR inference produced {len(relationships)} relationships")
                except Exception as exc:
                    print(f"[RL] RelTR inference failed for synthetic image {image_path}: {exc}")
                    relationships = []
            
            # If still no relationships, create a fallback relationship from original_relationship
            if not relationships and original_relationship:
                print(f"[RL] Sample {index+1}: creating fallback relationship from original")
                fallback_relationship = {
                    'subject': original_relationship.get('subject', 'unknown'),
                    'relation': original_relationship.get('relation', 'unknown'),
                    'object': original_relationship.get('object', 'unknown'),
                    'confidence': 0.5,  # Low confidence for fallback
                    'source': 'fallback'
                }
                relationships = [fallback_relationship]
                print(f"[RL] Sample {index+1} created fallback relationship: {fallback_relationship}")

            sample['relationships'] = relationships
            sample['source'] = 'synthetic'
            sample['prompt'] = data.get('prompt')
            sample['original_relationship'] = original_relationship
            sample['generation_timestamp'] = data.get('generation_timestamp')
            sample['image_path'] = image_path

            print(f"[RL] Sample {index+1} final relationships: {len(relationships)}")
            self.dataset_samples.append(sample)
            existing_paths.add(resolved_path)
            ingested += 1

        if ingested:
            self.detection_dataset_dir = None
            self._save_dataset_snapshot()
            print(f"[RL] Successfully ingested {ingested} synthetic sample(s) into the training dataset.")
        else:
            print("[RL] No synthetic samples were successfully ingested")
        return ingested

    def build_dataset_from_directory(self, image_dir: str, clear_previous: bool = True) -> int:
        directory = Path(image_dir)
        if not directory.exists() or not directory.is_dir():
            print(f"[RL] Image directory not found: {image_dir}")
            return len(self.dataset_samples)

        image_paths = sorted([
            path for path in directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])

        if not image_paths:
            print(f"[RL] No images found in directory: {image_dir}")
            return len(self.dataset_samples)

        dataset: List[Dict[str, Any]] = [] if clear_previous else list(self.dataset_samples)
        print(f"[RL] Building dataset from {len(image_paths)} images in {directory}")

        for idx, image_path in enumerate(image_paths, start=1):
            try:
                sample = self._extract_objects_with_clip(str(image_path))
                if not sample:
                    continue

                image_tensor = self._load_image_tensor(sample['image_path'])
                relationships = self._run_reltr_inference(
                    image_tensor,
                    sample['objects'],
                    sample.get('global_context'),
                    (sample['width'], sample['height']),
                )
                sample['relationships'] = relationships
                dataset.append(sample)
                print(f"[RL] Processed image {idx}/{len(image_paths)}: {image_path.name} "
                      f"({len(sample['objects'])} objects, {len(relationships)} relationships)")
            except Exception as exc:
                print(f"[RL] Error processing {image_path}: {exc}")

        if not dataset:
            print("[RL] Dataset build produced no usable samples.")
        self.dataset_samples = dataset
        if self.model_manager.current_experiment_dir:
            self._save_dataset_snapshot()
        return len(self.dataset_samples)

    def _prepare_reltr_training_samples(self) -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[List[float]]]]:
        if not self.dataset_samples:
            print("[RL] No dataset samples available, trying to load original data")
            fallback = self._load_original_detection_and_relationships()
            if fallback:
                self.dataset_samples = [fallback]
                print(f"[RL] Loaded {len(self.dataset_samples)} fallback samples")

        if not self.dataset_samples:
            print("[RL] No dataset samples available for RelTR training")
            return []

        print(f"[RL] Preparing RelTR training samples from {len(self.dataset_samples)} dataset samples")
        prepared: List[Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[List[float]]]] = []
        valid_samples = 0
        
        for i, sample in enumerate(self.dataset_samples):
            print(f"[RL] Processing sample {i+1}/{len(self.dataset_samples)}")
            target = self._build_reltr_target(sample)
            if target is None:
                print(f"[RL] Skipping sample {i+1}: no valid RelTR target")
                continue
            
            try:
                image_tensor = self._load_image_tensor(sample['image_path'])
                prepared.append((image_tensor, target, sample.get('global_context')))
                valid_samples += 1
                print(f"[RL] Sample {i+1} prepared successfully")
            except Exception as exc:
                print(f"[RL] Failed to load image tensor for sample {i+1}: {exc}")
                continue
        
        print(f"[RL] Successfully prepared {valid_samples} RelTR training samples")
        return prepared

    def _move_target_to_device(self, target: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}

    def _decode_relationships(
        self,
        outputs: Dict[str, torch.Tensor],
        objects: List[Dict[str, Any]],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        rel_logits = outputs.get("rel_logits")
        if rel_logits is None or rel_logits.numel() == 0:
            return []

        try:
            rel_scores = rel_logits.softmax(-1)[0, :, :-1].detach().cpu()
        except Exception:
            return []
        if rel_scores.numel() == 0:
            return []

        relationships: List[Dict[str, Any]] = []
        width: Optional[int] = None
        height: Optional[int] = None
        if image_size and len(image_size) == 2:
            width, height = image_size

        use_geometric = (
            width is not None
            and height is not None
            and outputs.get("sub_boxes") is not None
            and outputs.get("obj_boxes") is not None
            and len(objects) >= 2
        )

        if use_geometric:
            try:
                object_boxes = torch.tensor(
                    [obj.get('bbox', [0.0, 0.0, 0.0, 0.0]) for obj in objects],
                    dtype=torch.float32,
                )
                if object_boxes.numel() > 0:
                    scale = torch.tensor(
                        [width, height, width, height],
                        dtype=torch.float32,
                    )
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
                        subj_iou_vals = box_ops.box_iou(
                            sub_boxes_xyxy[idx].unsqueeze(0),
                            object_boxes,
                        )[0]
                        obj_iou_vals = box_ops.box_iou(
                            obj_boxes_xyxy[idx].unsqueeze(0),
                            object_boxes,
                        )[0]
                        subj_iou, subj_idx = subj_iou_vals.max(dim=0)
                        obj_iou, obj_idx = obj_iou_vals.max(dim=0)
                        if subj_iou.item() < min_iou or obj_iou.item() < min_iou:
                            continue
                        relation_name = RELATION_CLASSES[int(rel_idx) % len(RELATION_CLASSES)]
                        confidence = float(
                            rel_conf.item()
                            * max(subj_iou.item(), min_iou)
                            * max(obj_iou.item(), min_iou)
                        )
                        relationships.append({
                            'subject': objects[int(subj_idx)].get('class', 'unknown'),
                            'relation': relation_name,
                            'object': objects[int(obj_idx)].get('class', 'unknown'),
                            'confidence': min(confidence, 1.0),
                            'source': 'model',
                        })
                    if relationships:
                        return relationships
            except Exception as exc:
                print(f"[RL] Geometric relationship decoding failed: {exc}")

        keep = rel_scores.max(-1).values > 0.4
        filtered = rel_scores[keep] if keep.any() else rel_scores
        num_queries = filtered.shape[0]
        if num_queries == 0:
            filtered = rel_scores
            num_queries = filtered.shape[0]

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
                relationships.append({
                    'subject': objects[i].get('class', 'unknown'),
                    'relation': relation_name,
                    'object': objects[j].get('class', 'unknown'),
                    'confidence': confidence,
                    'source': 'model_fallback',
                })
                pair_cursor += 1
        return relationships
        
    # ------------------------------------------------------------------ #
    # RL agent helpers
    # ------------------------------------------------------------------ #
    def _remember(
        self,
        state: Optional[torch.Tensor],
        action_index: Optional[int],
        reward: float,
        next_state: Optional[torch.Tensor],
        done: bool,
    ) -> None:
        if state is None or next_state is None or action_index is None:
            return
        experience = (
            state.detach().cpu(),
            int(action_index),
            float(reward),
            next_state.detach().cpu(),
            float(done),
        )
        self.memory.append(experience)

    def _optimize_q_network(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([exp[0] for exp in batch]).to(self.rl_device)
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long, device=self.rl_device)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32, device=self.rl_device)
        next_states = torch.stack([exp[3] for exp in batch]).to(self.rl_device)
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.float32, device=self.rl_device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1).values
            target_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_values)
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.q_optimizer.step()
        return float(loss.item())

    def _finalize_rl_step(
        self,
        state: Optional[torch.Tensor],
        action_index: Optional[int],
        reward: float,
        detection_loss: float,
        relationship_loss: float,
        done: bool = False,
    ) -> Optional[float]:
        metrics = {
            'detection_loss': detection_loss,
            'relationship_loss': relationship_loss,
            'reward': reward,
            'dataset_size': len(self.dataset_samples),
        }
        next_state = self._build_state_vector(metrics)
        self._remember(state, action_index, reward, next_state, done)
        optimization_loss = self._optimize_q_network()
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.last_metrics = metrics
        self.last_state = next_state
        self.last_action_index = action_index
        return optimization_loss

    # ------------------------------------------------------------------ #
    # Reward evaluation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
        if len(box_a) != 4 or len(box_b) != 4:
            return 0.0
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        intersection = inter_w * inter_h
        if intersection <= 0.0:
            return 0.0
        area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
        area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
        union = area_a + area_b - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    def _match_detections(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        iou_threshold: float,
    ) -> Tuple[int, int, int]:
        if not predictions:
            return 0, 0, len(ground_truths)

        matched_gt = set()
        tp = 0
        fp = 0

        for pred in predictions:
            pred_bbox = pred.get('bbox')
            pred_class = self._normalize_label(pred.get('class', ''))
            if not pred_bbox or not pred_class:
                fp += 1
                continue

            best_iou = 0.0
            best_idx: Optional[int] = None

            for idx, gt in enumerate(ground_truths):
                if idx in matched_gt:
                    continue
                gt_bbox = gt.get('bbox')
                gt_class = self._normalize_label(gt.get('class', ''))
                if not gt_bbox or not gt_class or gt_class != pred_class:
                    continue
                iou = self._compute_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx is not None and best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn = len(ground_truths) - len(matched_gt)
        return tp, fp, fn

    def _evaluate_detection_metrics(
        self,
        samples: List[Dict[str, Any]],
        max_samples: int = 25,
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        if not samples:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0, 'num_samples': 0}

        total_tp = total_fp = total_fn = 0
        evaluated = 0

        subset = samples[:max_samples]
        for sample in subset:
            ground_truths = [
                obj for obj in sample.get('objects', [])
                if obj.get('bbox') and obj.get('class')
            ]
            if not ground_truths:
                continue

            pil_image = self._ensure_pil_image(sample.get('image_path'))
            if pil_image is None:
                continue

            try:
                predictions, _ = self._detect_objects_for_image(pil_image)
            except Exception as exc:
                print(f"[RL] Detection evaluation failed for {sample.get('image_path')}: {exc}")
                continue

            tp, fp, fn = self._match_detections(predictions, ground_truths, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            evaluated += 1

        if evaluated == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0, 'num_samples': 0}

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'num_samples': evaluated,
        }

    def _normalize_relationship_tuple(self, relationship: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
        if not relationship:
            return ("", "", "")
        return (
            self._normalize_label(relationship.get('subject', '')),
            self._normalize_label(relationship.get('relation', '')),
            self._normalize_label(relationship.get('object', '')),
        )

    def _compute_relationship_confusion(
        self,
        ground_truth: List[Tuple[str, str, str]],
        predictions: List[Tuple[str, str, str]],
    ) -> Tuple[int, int, int]:
        if not ground_truth and not predictions:
            return 0, 0, 0
        gt_counter = Counter(ground_truth)
        pred_counter = Counter(predictions)
        tp = sum(min(gt_counter[key], pred_counter.get(key, 0)) for key in gt_counter)
        fp = sum(max(pred_counter[key] - gt_counter.get(key, 0), 0) for key in pred_counter)
        fn = sum(max(gt_counter[key] - pred_counter.get(key, 0), 0) for key in gt_counter)
        return tp, fp, fn

    def _evaluate_relationship_metrics(
        self,
        synthetic_data: List[Dict[str, Any]],
        original_relationships: List[Dict[str, Any]],
        max_samples: int = 30,
    ) -> Dict[str, Any]:
        if not synthetic_data:
            print("[RL] No synthetic data provided for relationship evaluation")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'f1_std': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'num_samples': 0,
                'per_sample_f1': [],
            }

        print(f"[RL] Evaluating relationship metrics on {len(synthetic_data)} synthetic samples")
        total_tp = total_fp = total_fn = 0
        per_sample_f1: List[float] = []
        evaluated = 0

        subset = synthetic_data[:max_samples]
        for i, data in enumerate(subset):
            target_rel = data.get('original_relationship')
            image_input = data.get('image') or data.get('image_path')
            if not target_rel or image_input is None:
                print(f"[RL] Skipping sample {i+1}: missing target relationship or image input")
                continue

            try:
                # Ensure relationship model is loaded
                self._ensure_relationship_model()
                predicted_relationships = self.predict_relationships(image_input) or []
                print(f"[RL] Sample {i+1}: predicted {len(predicted_relationships)} relationships")
            except Exception as exc:
                print(f"[RL] Relationship evaluation failed for sample {i+1}: {exc}")
                predicted_relationships = []

            gt_tuples = [self._normalize_relationship_tuple(target_rel)]
            pred_tuples = [self._normalize_relationship_tuple(rel) for rel in predicted_relationships if rel]

            tp, fp, fn = self._compute_relationship_confusion(gt_tuples, pred_tuples)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            print(f"[RL] Sample {i+1} metrics: TP={tp}, FP={fp}, FN={fn}, F1={f1:.3f}")
            total_tp += tp
            total_fp += fp
            total_fn += fn
            per_sample_f1.append(f1)
            evaluated += 1

        if evaluated == 0:
            print("[RL] No samples were successfully evaluated")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'f1_std': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'num_samples': 0,
                'per_sample_f1': [],
            }

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_f1 = sum(per_sample_f1) / len(per_sample_f1) if per_sample_f1 else 0.0
        variance = sum((score - mean_f1) ** 2 for score in per_sample_f1) / len(per_sample_f1) if per_sample_f1 else 0.0
        std_f1 = math.sqrt(variance)

        print(f"[RL] Relationship evaluation completed: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (evaluated {evaluated} samples)")
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f1_std': std_f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'num_samples': evaluated,
            'per_sample_f1': per_sample_f1,
        }

    def train_episode(self, original_relationships, synthetic_data=None, action_context: Optional[Dict[str, Any]] = None, done: bool = False):
        print(f"Starting training episode with {len(original_relationships)} relationships")
        action_variations = action_context.get('num_variations', 3) if action_context else 3
        
        # 1. Use provided synthetic data or generate new if none provided
        if synthetic_data is None:
            print(f"Step 1: Generating synthetic data (variations per relation: {action_variations})...")
            synthetic_data = []
            for i, rel in enumerate(original_relationships):
                print(f"  Processing relationship {i+1}/{len(original_relationships)}: {rel.get('subject', 'Unknown')} {rel.get('relation', 'Unknown')} {rel.get('object', 'Unknown')}")
                try:
                    generated_images = self.generator.generate_from_relationship(rel, num_variations=action_variations)
                    synthetic_data.extend(generated_images)
                    print(f"    SUCCESS: Generated {len(generated_images)} images")
                except Exception as e:
                    print(f"    ERROR: Error generating images for relationship {i+1}: {e}")
                    continue
            print(f"Total synthetic data generated: {len(synthetic_data)} images")
        else:
            print(f"Step 1: Using provided synthetic data: {len(synthetic_data)} images (variations per relation: {action_variations})")

        ingested_count = self._ingest_synthetic_samples(synthetic_data)
        if ingested_count == 0:
            print("[RL] Warning: no synthetic samples ingested into the training dataset.")
        else:
            print(f"[RL] Dataset now contains {len(self.dataset_samples)} sample(s).")

        # 2. Train detection model
        print("Step 2: \U0001f9e0 Training detection model...")
        detection_loss = self.train_detection_model(synthetic_data)
        print(f"    ✅ Detection loss: {detection_loss:.4f}")
        
        # 3. Train relationship model
        print("Step 3: 🧠 Training relationship model...")
        relationship_loss = self.train_relationship_model(synthetic_data)
        print(f"    ✅ Relationship loss: {relationship_loss:.4f}")
        
        # 4. Calculate reward
        print("Step 4: 📊 Calculating reward...")
        reward = self.calculate_reward(synthetic_data, original_relationships)
        print(f"    ✅ Reward: {reward:.4f}")
        reward_components = dict(self.latest_reward_components or {})
        if reward_components:
            print(
                "     Reward breakdown -> "
                f"Detection: {reward_components.get('detection_score', 0.0):.3f}, "
                f"Relationship: {reward_components.get('relationship_score', 0.0):.3f}, "
                f"Diversity: {reward_components.get('diversity_score', 0.0):.3f}, "
                f"Consistency: {reward_components.get('consistency_score', 0.0):.3f}, "
                f"Improvement: {reward_components.get('improvement_score', 0.0):.3f}"
            )
            print(f"     Dynamic weights -> Detection: {reward_components.get('dynamic_weights', {}).get('detection', 0.0):.3f}, "
                  f"Relationship: {reward_components.get('dynamic_weights', {}).get('relationship', 0.0):.3f}, "
                  f"Diversity: {reward_components.get('dynamic_weights', {}).get('diversity', 0.0):.3f}, "
                  f"Consistency: {reward_components.get('dynamic_weights', {}).get('consistency', 0.0):.3f}")
        detection_metrics_snapshot = dict(self.latest_detection_metrics or {})
        relationship_metrics_snapshot = dict(self.latest_relationship_metrics or {})
        
        # 5. Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        print(f"    Exploration rate: {self.epsilon:.4f}")
        
        # 6. Save model state if this is a good result
        if reward > self.training_history['best_reward']:
            self.training_history['best_reward'] = reward
            self.training_history['best_epoch'] = len(self.training_history['epochs']) + 1
            self.save_model_state(reward, detection_loss, relationship_loss)
        
        # 7. Update training history
        epoch_data = {
            'epoch': len(self.training_history['epochs']) + 1,
            'detection_loss': detection_loss,
            'relationship_loss': relationship_loss,
            'reward': reward,
            'epsilon': self.epsilon,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.training_history['epochs'].append(epoch_data)
        
        print("SUCCESS: Training episode completed!")
        
        experience_batch = self._build_experience_batch(
            original_relationships=original_relationships,
            synthetic_data=synthetic_data,
            epoch_index=len(self.training_history['epochs']) + 1,
            reward=reward,
            detection_loss=detection_loss,
            relationship_loss=relationship_loss,
        )

        rl_state = action_context.get('state') if action_context else None
        rl_action_index = action_context.get('action_index') if action_context else None
        rl_loss = self._finalize_rl_step(
            rl_state,
            rl_action_index,
            reward,
            detection_loss,
            relationship_loss,
            done=done,
        )
        if rl_loss is not None:
            print(f"[RL] Q-network optimization loss: {rl_loss:.6f}")
        
        return {
            'detection_loss': detection_loss,
            'relationship_loss': relationship_loss,
            'reward': reward,
            'epsilon': self.epsilon,
            'experience_batch': experience_batch,
            'rl_action_index': rl_action_index,
            'rl_num_variations': action_variations,
            'rl_loss': rl_loss,
            'reward_components': reward_components,
            'detection_metrics': detection_metrics_snapshot,
            'relationship_metrics': relationship_metrics_snapshot,
        }
    
    def calculate_reward(self, synthetic_data, original_relationships):
        """
        Tính điểm dựa trên thuật toán thích ứng và các chỉ số khách quan.
        Thay thế công thức chủ quan bằng hệ thống đánh giá có thể đo lường được.
        """
        # 1. Thu thập các chỉ số cơ bản
        detection_metrics = self._evaluate_detection_metrics(self.dataset_samples)
        relationship_metrics = self._evaluate_relationship_metrics(synthetic_data, original_relationships)
        per_sample_f1 = relationship_metrics.pop('per_sample_f1', [])
        
        # 2. Tính toán các thành phần điểm với thuật toán cụ thể
        detection_score = self._calculate_detection_score(detection_metrics)
        relationship_score = self._calculate_relationship_score(relationship_metrics)
        diversity_score = self._calculate_diversity_score(synthetic_data)
        consistency_score = self._calculate_consistency_score(per_sample_f1, relationship_metrics.get('f1_std'))
        improvement_score = self._calculate_improvement_score()
        
        # 3. Tính trọng số động dựa trên hiệu suất hiện tại
        dynamic_weights = self._calculate_dynamic_weights(
            detection_score, relationship_score, diversity_score, consistency_score
        )
        
        # 4. Tính điểm tổng hợp với trọng số thích ứng
        total_reward = (
            dynamic_weights['detection'] * detection_score +
            dynamic_weights['relationship'] * relationship_score +
            dynamic_weights['diversity'] * diversity_score +
            dynamic_weights['consistency'] * consistency_score +
            dynamic_weights['improvement'] * improvement_score
        )
        
        # 5. Áp dụng hàm điều chỉnh để đảm bảo điểm trong khoảng hợp lý
        total_reward = self._apply_reward_scaling(total_reward)
        
        # 6. Lưu trữ thông tin để phân tích
        self.latest_detection_metrics = detection_metrics
        self.latest_relationship_metrics = relationship_metrics
        self.latest_reward_components = {
            'detection_score': detection_score,
            'relationship_score': relationship_score,
            'diversity_score': diversity_score,
            'consistency_score': consistency_score,
            'improvement_score': improvement_score,
            'dynamic_weights': dynamic_weights,
            'total_reward': total_reward,
            'scaling_factor': self._get_current_scaling_factor(),
        }
        
        # 7. Cập nhật lịch sử để học từ kinh nghiệm
        self._update_performance_history(total_reward, dynamic_weights)
        
        return total_reward
    
    def train_detection_model(self, synthetic_data):
        """Fine-tune the YOLO detection model on available labeled data."""
        try:
            detection_model = self._ensure_detection_model()
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}")
            return 0.0

        dataset_dir = self._prepare_detection_dataset()
        if dataset_dir is None:
            print("[RL] Detection dataset unavailable, skipping detection training.")
            return 0.0

        yaml_path = dataset_dir / "dataset.yaml"
        if not yaml_path.exists():
            print("[RL] Dataset YAML not found, skipping detection training.")
            return 0.0

        epochs = max(1, min(5, len(self.training_history['epochs']) + 1))
        project_dir = Path(self.model_manager.current_experiment_dir or Path.cwd()) / "rl_detection_runs"
        project_dir.mkdir(parents=True, exist_ok=True)

        print(f"[RL] Fine-tuning detection model for {epochs} epoch(s) using {yaml_path}.")
        metrics = detection_model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=640,
            batch=4,
            device=self.detection_device,
            project=str(project_dir),
            name=f"epoch_{len(self.training_history['epochs']) + 1:03d}",
            exist_ok=True,
            save=False,
            verbose=False,
            workers=0,
        )

        trainer = detection_model.trainer
        loss_value = 0.0
        if trainer is not None:
            if trainer.loss is not None:
                loss_value = float(trainer.loss)
            elif trainer.tloss is not None:
                loss_value = float(trainer.tloss)
        elif isinstance(metrics, dict):
            loss_value = float(metrics.get('train/box_loss', 0.0))

        print(f"[RL] Detection loss after fine-tuning: {loss_value:.4f}")
        detection_model.model.eval()
        return loss_value
    
    def train_relationship_model(self, synthetic_data):
        """Fine-tune the RelTR relationship model on available relationship annotations."""
        print("[RL] Starting relationship model training...")
        model, criterion = self._ensure_relationship_model()
        prepared_samples = self._prepare_reltr_training_samples()
        if not prepared_samples:
            print("[RL] RelTR training samples unavailable, skipping relationship training.")
            return 0.0

        print(f"[RL] Training RelTR model with {len(prepared_samples)} samples")
        if self.reltr_optimizer is None:
            self.reltr_optimizer = AdamW(
                (param for param in model.parameters() if param.requires_grad),
                lr=1e-5,
                weight_decay=1e-4,
            )
        optimizer = self.reltr_optimizer

        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for i, (image_tensor, target, global_context) in enumerate(prepared_samples):
            print(f"[RL] Training on sample {i+1}/{len(prepared_samples)}")
            try:
                samples = nested_tensor_from_tensor_list([image_tensor.to(self.reltr_device)])
                targets = [self._move_target_to_device(target, self.reltr_device)]

                context_tensor = self._prepare_global_context_tensor(global_context)
                if context_tensor is not None:
                    outputs = model(samples, global_context=context_tensor)
                else:
                    outputs = model(samples)
                
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                loss = sum(loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict.keys() if k in weight_dict)

                loss.backward()
                total_loss += float(loss.item())
                print(f"[RL] Sample {i+1} loss: {loss.item():.4f}")
            except Exception as exc:
                print(f"[RL] Error training on sample {i+1}: {exc}")
                continue

        optimizer.step()

        average_loss = total_loss / max(len(prepared_samples), 1)
        print(f"[RL] Relationship loss after fine-tuning: {average_loss:.4f}")
        model.eval()
        return average_loss
    
    def predict_relationships(self, image):
        """Predict relationships from an input image using the fine-tuned RelTR model."""
        self._ensure_relationship_model()
        pil_image = self._ensure_pil_image(image)
        if pil_image is None:
            print("[RL] Unsupported image input for relationship prediction.")
            return []

        detection_objects, global_context = self._detect_objects_for_image(pil_image)
        if not detection_objects:
            print("[RL] No objects detected; skipping relationship prediction.")
            return []

        image_tensor = self.reltr_transform(pil_image)
        return self._run_reltr_inference(
            image_tensor,
            detection_objects,
            global_context,
            image_size=pil_image.size,
        )
    
    def _calculate_detection_score(self, detection_metrics: Dict[str, float]) -> float:
        """
        Tính điểm detection dựa trên các chỉ số khách quan.
        Sử dụng F1-score làm chỉ số chính với điều chỉnh cho precision và recall.
        """
        f1 = detection_metrics.get('f1', 0.0)
        precision = detection_metrics.get('precision', 0.0)
        recall = detection_metrics.get('recall', 0.0)
        num_samples = detection_metrics.get('num_samples', 0)
        
        # Điểm cơ bản từ F1-score
        base_score = f1
        
        # Điều chỉnh dựa trên số lượng mẫu (confidence adjustment)
        sample_confidence = min(num_samples / 10.0, 1.0)  # Normalize to [0,1]
        
        # Điều chỉnh dựa trên sự cân bằng giữa precision và recall
        balance_factor = 1.0 - abs(precision - recall) / max(precision + recall, 1e-6)
        
        # Tính điểm cuối cùng với các điều chỉnh
        final_score = base_score * sample_confidence * balance_factor
        
        return max(0.0, min(final_score, 1.0))
    
    def _calculate_relationship_score(self, relationship_metrics: Dict[str, Any]) -> float:
        """
        Tính điểm relationship dựa trên các chỉ số khách quan.
        Bao gồm F1-score, độ lệch chuẩn và số lượng mẫu được đánh giá.
        """
        f1 = relationship_metrics.get('f1', 0.0)
        f1_std = relationship_metrics.get('f1_std', 0.0)
        precision = relationship_metrics.get('precision', 0.0)
        recall = relationship_metrics.get('recall', 0.0)
        num_samples = relationship_metrics.get('num_samples', 0)
        
        # Điểm cơ bản từ F1-score
        base_score = f1
        
        # Điều chỉnh dựa trên độ ổn định (stability adjustment)
        # Độ lệch chuẩn thấp = điểm cao hơn
        stability_factor = max(0.0, 1.0 - f1_std)
        
        # Điều chỉnh dựa trên số lượng mẫu
        sample_confidence = min(num_samples / 20.0, 1.0)
        
        # Điều chỉnh dựa trên sự cân bằng precision-recall
        balance_factor = 1.0 - abs(precision - recall) / max(precision + recall, 1e-6)
        
        # Tính điểm cuối cùng
        final_score = base_score * stability_factor * sample_confidence * balance_factor
        
        return max(0.0, min(final_score, 1.0))
    
    def _calculate_diversity_score(self, synthetic_data: List[Dict[str, Any]]) -> float:
        """
        Tính điểm đa dạng dựa trên các thuộc tính khác nhau của dữ liệu synthetic.
        Bao gồm đa dạng về relationship types, object classes và spatial distribution.
        """
        if not synthetic_data:
            return 0.0
        
        # 1. Đa dạng về relationship types
        unique_relations = set()
        for data in synthetic_data:
            if 'original_relationship' in data:
                rel = data['original_relationship']
                unique_relations.add(rel.get('relation', ''))
        
        relation_diversity = min(len(unique_relations) / 10.0, 1.0)
        
        # 2. Đa dạng về object classes
        unique_subjects = set()
        unique_objects = set()
        for data in synthetic_data:
            if 'original_relationship' in data:
                rel = data['original_relationship']
                unique_subjects.add(rel.get('subject', ''))
                unique_objects.add(rel.get('object', ''))
        
        class_diversity = min(len(unique_subjects | unique_objects) / 15.0, 1.0)
        
        # 3. Đa dạng về spatial distribution (nếu có thông tin bbox)
        spatial_diversity = self._calculate_spatial_diversity(synthetic_data)
        
        # Tính điểm tổng hợp với trọng số
        final_score = (
            0.4 * relation_diversity +
            0.4 * class_diversity +
            0.2 * spatial_diversity
        )
        
        return max(0.0, min(final_score, 1.0))
    
    def _calculate_spatial_diversity(self, synthetic_data: List[Dict[str, Any]]) -> float:
        """
        Tính đa dạng không gian dựa trên vị trí thực tế của các objects.
        Sử dụng thuật toán Spatial Distribution Analysis với các metrics:
        - Position Diversity: Đa dạng về vị trí trung tâm
        - Size Diversity: Đa dạng về kích thước bounding box
        - Coverage Diversity: Đa dạng về độ phủ của ảnh
        """
        if not synthetic_data:
            return 0.0
        
        # Thu thập tất cả bounding boxes thực tế
        all_bboxes = []
        image_sizes = []
        
        for data in synthetic_data:
            # Lấy thông tin từ objects trong data
            objects = data.get('objects', [])
            if not objects:
                continue
                
            # Lấy kích thước ảnh
            image_path = data.get('image_path') or data.get('saved_path')
            if image_path and os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        image_sizes.append((width, height))
                except Exception:
                    continue
            
            # Thu thập bounding boxes từ objects
            for obj in objects:
                bbox = obj.get('bbox')
                if bbox and len(bbox) == 4:
                    all_bboxes.append(bbox)
        
        if len(all_bboxes) < 2:
            return 0.0
        
        # Tính các thành phần đa dạng không gian
        position_diversity = self._calculate_position_diversity(all_bboxes, image_sizes)
        size_diversity = self._calculate_size_diversity(all_bboxes)
        coverage_diversity = self._calculate_coverage_diversity(all_bboxes, image_sizes)
        
        # Kết hợp các thành phần với trọng số
        spatial_score = (
            0.4 * position_diversity +
            0.3 * size_diversity +
            0.3 * coverage_diversity
        )
        
        return max(0.0, min(spatial_score, 1.0))
    
    def _calculate_position_diversity(self, bboxes: List[List[float]], image_sizes: List[Tuple[int, int]]) -> float:
        """
        Tính đa dạng vị trí dựa trên phân bố của center points.
        Sử dụng thuật toán Spatial Clustering Analysis.
        """
        if not bboxes or not image_sizes:
            return 0.0
        
        # Normalize bboxes về [0,1] dựa trên kích thước ảnh trung bình
        avg_width = sum(size[0] for size in image_sizes) / len(image_sizes)
        avg_height = sum(size[1] for size in image_sizes) / len(image_sizes)
        
        normalized_centers = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0 / avg_width
            center_y = (y1 + y2) / 2.0 / avg_height
            normalized_centers.append([center_x, center_y])
        
        # Tính độ phân tán của center points
        if len(normalized_centers) < 2:
            return 0.0
        
        # Tính variance của x và y coordinates
        x_coords = [center[0] for center in normalized_centers]
        y_coords = [center[1] for center in normalized_centers]
        
        x_mean = sum(x_coords) / len(x_coords)
        y_mean = sum(y_coords) / len(y_coords)
        
        x_variance = sum((x - x_mean) ** 2 for x in x_coords) / len(x_coords)
        y_variance = sum((y - y_mean) ** 2 for y in y_coords) / len(y_coords)
        
        # Tính độ phân tán tổng hợp
        total_variance = x_variance + y_variance
        
        # Normalize về [0,1] - variance cao = đa dạng cao
        # Sử dụng tanh để smooth và giới hạn trong [0,1]
        position_diversity = math.tanh(total_variance * 4)  # Scale factor 4
        
        return position_diversity
    
    def _calculate_size_diversity(self, bboxes: List[List[float]]) -> float:
        """
        Tính đa dạng kích thước dựa trên area và aspect ratio của bounding boxes.
        Sử dụng thuật toán Size Distribution Analysis.
        """
        if not bboxes:
            return 0.0
        
        areas = []
        aspect_ratios = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Tính area
            area = width * height
            areas.append(area)
            
            # Tính aspect ratio
            if height > 0:
                aspect_ratio = width / height
                aspect_ratios.append(aspect_ratio)
        
        if not areas or not aspect_ratios:
            return 0.0
        
        # Tính coefficient of variation cho area
        area_mean = sum(areas) / len(areas)
        area_std = math.sqrt(sum((area - area_mean) ** 2 for area in areas) / len(areas))
        area_cv = area_std / area_mean if area_mean > 0 else 0
        
        # Tính coefficient of variation cho aspect ratio
        ar_mean = sum(aspect_ratios) / len(aspect_ratios)
        ar_std = math.sqrt(sum((ar - ar_mean) ** 2 for ar in aspect_ratios) / len(aspect_ratios))
        ar_cv = ar_std / ar_mean if ar_mean > 0 else 0
        
        # Kết hợp area và aspect ratio diversity
        size_diversity = 0.6 * min(area_cv, 2.0) / 2.0 + 0.4 * min(ar_cv, 3.0) / 3.0
        
        return max(0.0, min(size_diversity, 1.0))
    
    def _calculate_coverage_diversity(self, bboxes: List[List[float]], image_sizes: List[Tuple[int, int]]) -> float:
        """
        Tính đa dạng độ phủ dựa trên việc phân chia ảnh thành grid và đếm coverage.
        Sử dụng thuật toán Grid-Based Coverage Analysis.
        """
        if not bboxes or not image_sizes:
            return 0.0
        
        # Sử dụng kích thước ảnh trung bình để tính grid
        avg_width = sum(size[0] for size in image_sizes) / len(image_sizes)
        avg_height = sum(size[1] for size in image_sizes) / len(image_sizes)
        
        # Chia ảnh thành grid 4x4 = 16 cells
        grid_size = 4
        cell_width = avg_width / grid_size
        cell_height = avg_height / grid_size
        
        # Đếm số cells được cover bởi ít nhất một bbox
        covered_cells = set()
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # Tìm các cells mà bbox này cover
            start_col = max(0, int(x1 // cell_width))
            end_col = min(grid_size - 1, int(x2 // cell_width))
            start_row = max(0, int(y1 // cell_height))
            end_row = min(grid_size - 1, int(y2 // cell_height))
            
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    covered_cells.add((row, col))
        
        # Tính coverage ratio
        total_cells = grid_size * grid_size
        coverage_ratio = len(covered_cells) / total_cells
        
        # Tính distribution balance - các cells được cover đều hay không
        if len(covered_cells) < 2:
            return coverage_ratio
        
        # Tính entropy của distribution
        cell_counts = {}
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            start_col = max(0, int(x1 // cell_width))
            end_col = min(grid_size - 1, int(x2 // cell_width))
            start_row = max(0, int(y1 // cell_height))
            end_row = min(grid_size - 1, int(y2 // cell_height))
            
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    cell = (row, col)
                    cell_counts[cell] = cell_counts.get(cell, 0) + 1
        
        # Tính entropy
        total_bboxes = len(bboxes)
        entropy = 0.0
        for count in cell_counts.values():
            if count > 0:
                p = count / total_bboxes
                entropy -= p * math.log2(p)
        
        # Normalize entropy (max entropy = log2(total_cells))
        max_entropy = math.log2(total_cells)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Kết hợp coverage ratio và distribution entropy
        coverage_diversity = 0.7 * coverage_ratio + 0.3 * normalized_entropy
        
        return max(0.0, min(coverage_diversity, 1.0))
    
    def _calculate_consistency_score(self, f1_scores: List[float], precomputed_std: Optional[float] = None) -> float:
        """
        Tính điểm consistency dựa trên độ ổn định của các predictions.
        Sử dụng độ lệch chuẩn và trend analysis.
        """
        if not f1_scores and (precomputed_std is None or precomputed_std == 0.0):
            return 0.0
        
        if precomputed_std is not None:
            std = float(precomputed_std)
        else:
            if not f1_scores:
                return 0.0
            mean_score = sum(f1_scores) / len(f1_scores)
            variance = sum((score - mean_score) ** 2 for score in f1_scores) / len(f1_scores)
            std = math.sqrt(variance)
        
        # Điểm consistency dựa trên độ lệch chuẩn (thấp = tốt)
        consistency_from_std = max(0.0, 1.0 - min(std, 1.0))
        
        # Điểm consistency dựa trên trend (xu hướng cải thiện)
        trend_score = self._calculate_trend_score(f1_scores)
        
        # Kết hợp hai chỉ số
        final_score = 0.7 * consistency_from_std + 0.3 * trend_score
        
        return max(0.0, min(final_score, 1.0))
    
    def _calculate_trend_score(self, scores: List[float]) -> float:
        """Tính điểm dựa trên xu hướng cải thiện của scores."""
        if len(scores) < 3:
            return 0.5  # Neutral score for insufficient data
        
        # Tính slope của linear regression đơn giản
        n = len(scores)
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        
        numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.5
        
        slope = numerator / denominator
        
        # Chuyển slope thành điểm (slope > 0 = cải thiện)
        trend_score = 0.5 + 0.5 * math.tanh(slope * 10)  # Scale và normalize
        
        return max(0.0, min(trend_score, 1.0))
    
    def _calculate_improvement_score(self) -> float:
        """
        Tính điểm cải thiện dựa trên lịch sử performance.
        So sánh với baseline và xu hướng gần đây.
        """
        if len(self.performance_history['rewards']) < 3:
            return 0.5  # Neutral score for insufficient history
        
        recent_rewards = list(self.performance_history['rewards'])[-10:]  # Last 10 rewards
        baseline_reward = self.baseline_performance.get('overall', 0.5)
        
        # So sánh với baseline
        current_avg = sum(recent_rewards) / len(recent_rewards)
        baseline_improvement = (current_avg - baseline_reward) / max(baseline_reward, 1e-6)
        
        # Xu hướng cải thiện gần đây
        trend_improvement = self._calculate_trend_score(recent_rewards)
        
        # Kết hợp hai chỉ số
        final_score = 0.6 * (0.5 + 0.5 * math.tanh(baseline_improvement)) + 0.4 * trend_improvement
        
        return max(0.0, min(final_score, 1.0))
    
    def _calculate_dynamic_weights(self, detection_score: float, relationship_score: float, 
                                  diversity_score: float, consistency_score: float) -> Dict[str, float]:
        """
        Tính trọng số động dựa trên hiệu suất hiện tại và lịch sử.
        Trọng số sẽ thích ứng để tập trung vào các thành phần cần cải thiện.
        """
        # Trọng số cơ bản
        base_weights = {
            'detection': 0.25,
            'relationship': 0.45,
            'diversity': 0.15,
            'consistency': 0.10,
            'improvement': 0.05,
        }
        
        # Tính độ lệch so với baseline
        detection_deviation = abs(detection_score - self.baseline_performance['detection'])
        relationship_deviation = abs(relationship_score - self.baseline_performance['relationship'])
        diversity_deviation = abs(diversity_score - self.baseline_performance['diversity'])
        consistency_deviation = abs(consistency_score - self.baseline_performance['consistency'])
        
        # Điều chỉnh trọng số dựa trên độ lệch (thành phần nào kém sẽ có trọng số cao hơn)
        adjustment_factor = 0.2  # Mức độ điều chỉnh
        
        adjusted_weights = {
            'detection': base_weights['detection'] + adjustment_factor * detection_deviation,
            'relationship': base_weights['relationship'] + adjustment_factor * relationship_deviation,
            'diversity': base_weights['diversity'] + adjustment_factor * diversity_deviation,
            'consistency': base_weights['consistency'] + adjustment_factor * consistency_deviation,
            'improvement': base_weights['improvement'],
        }
        
        # Normalize để tổng = 1.0
        total_weight = sum(adjusted_weights.values())
        normalized_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        return normalized_weights
    
    def _apply_reward_scaling(self, raw_reward: float) -> float:
        """
        Áp dụng hàm điều chỉnh để đảm bảo điểm trong khoảng hợp lý.
        Sử dụng sigmoid function để normalize.
        """
        # Sử dụng sigmoid để đưa điểm về khoảng [0, 1]
        scaled_reward = 1.0 / (1.0 + math.exp(-self.scaling_factor * (raw_reward - 0.5)))
        
        return scaled_reward
    
    def _get_current_scaling_factor(self) -> float:
        """Lấy scaling factor hiện tại dựa trên lịch sử performance."""
        if len(self.performance_history['rewards']) < 5:
            return 1.0
        
        recent_rewards = list(self.performance_history['rewards'])[-10:]
        reward_variance = np.var(recent_rewards) if recent_rewards else 0.0
        
        # Scaling factor cao hơn khi variance thấp (performance ổn định)
        scaling_factor = 1.0 + (1.0 - min(reward_variance, 1.0))
        
        return scaling_factor
    
    def _update_performance_history(self, reward: float, weights: Dict[str, float]) -> None:
        """Cập nhật lịch sử performance để học từ kinh nghiệm."""
        self.performance_history['rewards'].append(reward)
        
        # Cập nhật baseline performance nếu có cải thiện
        if len(self.performance_history['rewards']) >= 10:
            recent_avg = sum(list(self.performance_history['rewards'])[-10:]) / 10
            if recent_avg > self.baseline_performance.get('overall', 0.5):
                self.baseline_performance['overall'] = recent_avg
        
        # Cập nhật scaling factor
        self.scaling_factor = self._get_current_scaling_factor()
    
    def calculate_diversity_reward(self, synthetic_data):
        """Calculate diversity reward based on synthetic data variety"""
        if not synthetic_data:
            return 0.0
        
        # Count unique relationship types
        unique_relations = set()
        for data in synthetic_data:
            if 'original_relationship' in data:
                rel = data['original_relationship']
                unique_relations.add(rel.get('relation', ''))
        
        # Diversity reward based on number of unique relations
        diversity_score = min(len(unique_relations) / 10.0, 1.0)  # Normalize to [0,1]
        return diversity_score
    
    def calculate_consistency_reward(self, f1_scores: List[float], precomputed_std: Optional[float] = None) -> float:
        """Calculate consistency reward based on the stability of relationship predictions."""
        if not f1_scores and (precomputed_std is None or precomputed_std == 0.0):
            return 0.0
        if precomputed_std is not None:
            std = float(precomputed_std)
        else:
            if not f1_scores:
                return 0.0
            mean_score = sum(f1_scores) / len(f1_scores)
            variance = sum((score - mean_score) ** 2 for score in f1_scores) / len(f1_scores)
            std = math.sqrt(variance)
        return max(0.0, 1.0 - min(std, 1.0))
    
    @staticmethod
    def _optimizer_state_to_cpu(state_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not state_dict:
            return None
        cpu_state = copy.deepcopy(state_dict)
        for state in cpu_state.get('state', {}).values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.detach().cpu()
        return cpu_state

    def save_model_state(self, reward, detection_loss, relationship_loss):
        """Persist the current model and optimizer states when an improvement is achieved."""
        if not self.model_manager.current_experiment_dir:
            print("WARNING: No experiment directory set, cannot save model state")
            return
        
        try:
            detection_state = None
            if self.detection_model is not None and hasattr(self.detection_model, "model"):
                detection_state = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in self.detection_model.model.state_dict().items()
                }

            relationship_state = None
            if self.relationship_model is not None:
                relationship_state = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in self.relationship_model.state_dict().items()
                }

            q_network_state = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in self.q_network.state_dict().items()
            }
            target_network_state = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in self.target_network.state_dict().items()
            }

            q_optimizer_state = self._optimizer_state_to_cpu(self.q_optimizer.state_dict())
            reltr_optimizer_state = self._optimizer_state_to_cpu(
                self.reltr_optimizer.state_dict() if self.reltr_optimizer is not None else None
            )

            reltr_args_payload = vars(self.reltr_args) if isinstance(self.reltr_args, SimpleNamespace) else self.reltr_args

            model_state = {
                'epsilon': self.epsilon,
                'reward': reward,
                'detection_loss': detection_loss,
                'relationship_loss': relationship_loss,
                'training_step': len(self.training_history['epochs']),
                'timestamp': datetime.datetime.now().isoformat(),
                'detection_model_state': detection_state,
                'relationship_model_state': relationship_state,
                'q_network_state': q_network_state,
                'target_network_state': target_network_state,
                'q_optimizer_state': q_optimizer_state,
                'reltr_optimizer_state': reltr_optimizer_state,
                'reltr_args': reltr_args_payload,
                'dataset_size': len(self.dataset_samples),
                'training_history': self.training_history,
            }

            metadata = {
                'reward': reward,
                'detection_loss': detection_loss,
                'relationship_loss': relationship_loss,
                'epsilon': self.epsilon,
                'dataset_size': len(self.dataset_samples),
            }
            epoch_index = len(self.training_history['epochs'])

            self.model_manager.save_model_state('detection_model', model_state, epoch=epoch_index, metadata=metadata)
            self.model_manager.save_model_state('relationship_model', model_state, epoch=epoch_index, metadata=metadata)

            self.model_manager.save_training_history(self.training_history)
            self._save_dataset_snapshot()

            print(f"Saved model state (reward: {reward:.4f})")

        except Exception as e:
            print(f"ERROR saving model state: {e}")
    
    def load_model_state(self, model_name='detection_model', experiment_dir=None):
        """Load model and optimizer states from a saved checkpoint."""
        try:
            checkpoint = self.model_manager.load_model_state(model_name, experiment_dir)
            if not checkpoint:
                print(f"No checkpoint found for {model_name}")
                return False

            model_state = checkpoint.get('model_state', {})
            if not model_state:
                print(f"[RL] Checkpoint for {model_name} is missing model state information.")
                return False

            reltr_args_payload = model_state.get('reltr_args')
            if reltr_args_payload:
                if isinstance(reltr_args_payload, dict):
                    self.reltr_args = SimpleNamespace(**reltr_args_payload)
                elif isinstance(reltr_args_payload, SimpleNamespace):
                    self.reltr_args = reltr_args_payload

            detection_state = model_state.get('detection_model_state')
            if detection_state:
                detection_model = self._ensure_detection_model()
                if hasattr(detection_model, "model"):
                    detection_model.model.load_state_dict(detection_state, strict=False)

            relationship_state = model_state.get('relationship_model_state')
            if relationship_state:
                model, _ = self._ensure_relationship_model()
                model.load_state_dict(relationship_state, strict=False)

            q_state = model_state.get('q_network_state')
            if q_state:
                self.q_network.load_state_dict(q_state, strict=False)

            target_state = model_state.get('target_network_state')
            if target_state:
                self.target_network.load_state_dict(target_state, strict=False)

            q_optimizer_state = model_state.get('q_optimizer_state')
            if q_optimizer_state:
                self.q_optimizer.load_state_dict(q_optimizer_state)

            reltr_optimizer_state = model_state.get('reltr_optimizer_state')
            if reltr_optimizer_state:
                if self.reltr_optimizer is None:
                    self.reltr_optimizer = AdamW(
                        (param for param in self.relationship_model.parameters() if param.requires_grad),
                        lr=1e-5,
                        weight_decay=1e-4,
                    )
                self.reltr_optimizer.load_state_dict(reltr_optimizer_state)

            self.epsilon = model_state.get('epsilon', self.epsilon)

            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            elif 'training_history' in model_state:
                self.training_history = model_state['training_history']

            print(f"Loaded model state from {model_name}")
            return True

        except Exception as e:
            print(f"ERROR loading model state: {e}")
            return False
    
    def get_best_model(self, metric='reward'):
        """Lấy model tốt nhất"""
        return self.model_manager.get_best_model('detection_model', metric=metric)
    
    def continue_training(self, experiment_dir):
        """Tiếp tục training từ experiment trước đó"""
        self.model_manager.set_experiment_dir(experiment_dir)
        
        # Load best model từ experiment trước
        best_model = self.get_best_model()
        if best_model:
            print(f"Continuing training from best model (reward: {best_model['model_state'].get('reward', 0):.4f})")
            return True
        else:
            print("No previous model found, starting fresh training")
            return False

    # ------------------------------------------------------------------ #
    # Experience helpers
    # ------------------------------------------------------------------ #
    def _build_experience_batch(
        self,
        original_relationships: List[Dict[str, Any]],
        synthetic_data: List[Dict[str, Any]],
        epoch_index: int,
        reward: float,
        detection_loss: float,
        relationship_loss: float,
    ) -> List[Dict[str, Any]]:
        """Create a batch of serialized experiences for replay buffer storage."""
        if not synthetic_data:
            summary_state = {
                'epoch': epoch_index,
                'epsilon': self.epsilon,
                'relationships_observed': len(original_relationships),
            }
            return [{
                'state': summary_state,
                'action': 'train_models',
                'reward': reward,
                'next_state': {**summary_state},
                'done': True,
                'metadata': {
                    'detection_loss': detection_loss,
                    'relationship_loss': relationship_loss,
                    'timestamp': datetime.datetime.now().isoformat(),
                }
            }]

        per_step_reward = reward / max(len(synthetic_data), 1)
        batch: List[Dict[str, Any]] = []

        for idx, data in enumerate(synthetic_data):
            relationship_info = self._extract_relationship_info(data)
            state = {
                'epoch': epoch_index,
                'step': idx,
                'epsilon': self.epsilon,
                'relationship': relationship_info,
            }
            next_state = {
                'epoch': epoch_index,
                'step': idx + 1,
                'epsilon': max(self.epsilon * self.epsilon_decay, self.epsilon_min),
                'relationship': relationship_info,
            }

            batch.append({
                'state': state,
                'action': 'train_relationship_models',
                'reward': per_step_reward,
                'next_state': next_state,
                'done': False,
                'metadata': {
                    'prompt': data.get('prompt'),
                    'is_mock': data.get('is_mock', False),
                    'detection_loss': detection_loss,
                    'relationship_loss': relationship_loss,
                    'generated_at': data.get('generation_timestamp'),
                }
            })

        if batch:
            batch[-1]['done'] = True
            batch[-1]['metadata']['epoch_reward'] = reward

        return batch

    @staticmethod
    def _extract_relationship_info(data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a serializable snapshot of the relationship sample."""
        relationship = data.get('original_relationship') or {}
        cleaned_relationship = {
            key: value
            for key, value in relationship.items()
            if key not in {'image', 'image_data'}
        }

        return {
            'subject': cleaned_relationship.get('subject'),
            'relation': cleaned_relationship.get('relation'),
            'object': cleaned_relationship.get('object'),
            'similarity': cleaned_relationship.get('visual_similarity'),
            'epoch': data.get('epoch'),
            'variation_index': data.get('variation_index'),
            'relationship_index': data.get('relationship_index'),
        }
    
    def get_spatial_diversity_analysis(self, synthetic_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Phân tích chi tiết về spatial diversity để hiểu rõ cách tính toán.
        """
        if not synthetic_data:
            return {
                'status': 'No synthetic data available',
                'message': 'Provide synthetic data to analyze spatial diversity'
            }
        
        # Thu thập dữ liệu
        all_bboxes = []
        image_sizes = []
        
        for data in synthetic_data:
            objects = data.get('objects', [])
            if not objects:
                continue
                
            image_path = data.get('image_path') or data.get('saved_path')
            if image_path and os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        image_sizes.append((width, height))
                except Exception:
                    continue
            
            for obj in objects:
                bbox = obj.get('bbox')
                if bbox and len(bbox) == 4:
                    all_bboxes.append(bbox)
        
        if len(all_bboxes) < 2:
            return {
                'status': 'Insufficient data',
                'message': f'Need at least 2 bounding boxes, found {len(all_bboxes)}',
                'bbox_count': len(all_bboxes),
                'image_count': len(image_sizes)
            }
        
        # Tính các thành phần
        position_diversity = self._calculate_position_diversity(all_bboxes, image_sizes)
        size_diversity = self._calculate_size_diversity(all_bboxes)
        coverage_diversity = self._calculate_coverage_diversity(all_bboxes, image_sizes)
        
        # Tính tổng điểm
        total_spatial_diversity = (
            0.4 * position_diversity +
            0.3 * size_diversity +
            0.3 * coverage_diversity
        )
        
        # Phân tích chi tiết
        analysis = {
            'total_spatial_diversity': total_spatial_diversity,
            'components': {
                'position_diversity': {
                    'value': position_diversity,
                    'weight': 0.4,
                    'contribution': 0.4 * position_diversity,
                    'description': 'Diversity of object center positions using variance analysis'
                },
                'size_diversity': {
                    'value': size_diversity,
                    'weight': 0.3,
                    'contribution': 0.3 * size_diversity,
                    'description': 'Diversity of bounding box areas and aspect ratios using coefficient of variation'
                },
                'coverage_diversity': {
                    'value': coverage_diversity,
                    'weight': 0.3,
                    'contribution': 0.3 * coverage_diversity,
                    'description': 'Diversity of spatial coverage using grid-based analysis and entropy'
                }
            },
            'data_summary': {
                'total_bboxes': len(all_bboxes),
                'total_images': len(image_sizes),
                'avg_image_size': (
                    sum(size[0] for size in image_sizes) / len(image_sizes),
                    sum(size[1] for size in image_sizes) / len(image_sizes)
                ) if image_sizes else (0, 0),
                'bboxes_per_image': len(all_bboxes) / len(image_sizes) if image_sizes else 0
            },
            'algorithms_used': [
                'Spatial Clustering Analysis for position diversity',
                'Coefficient of Variation for size diversity',
                'Grid-Based Coverage Analysis for coverage diversity',
                'Shannon Entropy for distribution balance',
                'Tanh Normalization for smooth scaling'
            ]
        }
        
        return analysis
    
    def print_spatial_diversity_breakdown(self, synthetic_data: List[Dict[str, Any]]) -> None:
        """In ra phân tích chi tiết về spatial diversity."""
        analysis = self.get_spatial_diversity_analysis(synthetic_data)
        
        if analysis.get('status'):
            print(f"[SPATIAL DIVERSITY] {analysis['message']}")
            return
        
        print("\n" + "="*80)
        print("🌍 PHÂN TÍCH SPATIAL DIVERSITY")
        print("="*80)
        print(f"🎯 Tổng điểm Spatial Diversity: {analysis['total_spatial_diversity']:.4f}")
        
        print("\n📊 CHI TIẾT CÁC THÀNH PHẦN:")
        print("-" * 60)
        
        for component_name, component_data in analysis['components'].items():
            print(f"\n{component_name.upper().replace('_', ' ')}:")
            print(f"  • Giá trị: {component_data['value']:.4f}")
            print(f"  • Trọng số: {component_data['weight']:.3f}")
            print(f"  • Đóng góp: {component_data['contribution']:.4f}")
            print(f"  • Mô tả: {component_data['description']}")
        
        print("\n📈 THÔNG TIN DỮ LIỆU:")
        print("-" * 30)
        data_summary = analysis['data_summary']
        print(f"  • Tổng số bounding boxes: {data_summary['total_bboxes']}")
        print(f"  • Tổng số ảnh: {data_summary['total_images']}")
        print(f"  • Kích thước ảnh trung bình: {data_summary['avg_image_size'][0]:.0f}x{data_summary['avg_image_size'][1]:.0f}")
        print(f"  • Bboxes trung bình/ảnh: {data_summary['bboxes_per_image']:.2f}")
        
        print("\n⚙️  THUẬT TOÁN ĐƯỢC SỬ DỤNG:")
        print("-" * 30)
        for algorithm in analysis['algorithms_used']:
            print(f"  ✓ {algorithm}")
        
        print("\n" + "="*80)
    
    def get_scoring_analysis(self) -> Dict[str, Any]:
        """
        Trả về phân tích chi tiết về hệ thống tính điểm hiện tại.
        Giúp hiểu rõ cách điểm được tính và các thành phần đóng góp.
        """
        if not self.latest_reward_components:
            return {
                'status': 'No scoring data available',
                'message': 'Run a training episode first to get scoring analysis'
            }
        
        components = self.latest_reward_components
        weights = components.get('dynamic_weights', {})
        
        analysis = {
            'scoring_method': 'Adaptive Algorithm-Based Scoring',
            'total_reward': components.get('total_reward', 0.0),
            'scaling_factor': components.get('scaling_factor', 1.0),
            'components': {
                'detection_score': {
                    'value': components.get('detection_score', 0.0),
                    'weight': weights.get('detection', 0.0),
                    'contribution': components.get('detection_score', 0.0) * weights.get('detection', 0.0),
                    'description': 'Based on F1-score, precision-recall balance, and sample confidence'
                },
                'relationship_score': {
                    'value': components.get('relationship_score', 0.0),
                    'weight': weights.get('relationship', 0.0),
                    'contribution': components.get('relationship_score', 0.0) * weights.get('relationship', 0.0),
                    'description': 'Based on F1-score, stability (low std), and sample confidence'
                },
                'diversity_score': {
                    'value': components.get('diversity_score', 0.0),
                    'weight': weights.get('diversity', 0.0),
                    'contribution': components.get('diversity_score', 0.0) * weights.get('diversity', 0.0),
                    'description': 'Based on relationship types, object classes, and spatial distribution'
                },
                'consistency_score': {
                    'value': components.get('consistency_score', 0.0),
                    'weight': weights.get('consistency', 0.0),
                    'contribution': components.get('consistency_score', 0.0) * weights.get('consistency', 0.0),
                    'description': 'Based on prediction stability and improvement trend'
                },
                'improvement_score': {
                    'value': components.get('improvement_score', 0.0),
                    'weight': weights.get('improvement', 0.0),
                    'contribution': components.get('improvement_score', 0.0) * weights.get('improvement', 0.0),
                    'description': 'Based on performance history and baseline comparison'
                }
            },
            'baseline_performance': self.baseline_performance,
            'performance_history_size': len(self.performance_history['rewards']),
            'algorithm_features': [
                'Dynamic weight adjustment based on current performance',
                'Confidence adjustment based on sample size',
                'Stability measurement using standard deviation',
                'Trend analysis using linear regression',
                'Sigmoid scaling for reward normalization',
                'Adaptive baseline updating'
            ]
        }
        
        return analysis
    
    def print_scoring_breakdown(self) -> None:
        """In ra phân tích chi tiết về cách tính điểm."""
        analysis = self.get_scoring_analysis()
        
        if analysis.get('status') == 'No scoring data available':
            print(analysis['message'])
            return
        
        print("\n" + "="*80)
        print("📊 PHÂN TÍCH HỆ THỐNG TÍNH ĐIỂM")
        print("="*80)
        print(f"🎯 Tổng điểm: {analysis['total_reward']:.4f}")
        print(f"⚖️  Scaling Factor: {analysis['scaling_factor']:.3f}")
        print(f"📈 Lịch sử Performance: {analysis['performance_history_size']} epochs")
        
        print("\n🔍 CHI TIẾT CÁC THÀNH PHẦN:")
        print("-" * 60)
        
        for component_name, component_data in analysis['components'].items():
            print(f"\n{component_name.upper().replace('_', ' ')}:")
            print(f"  • Giá trị: {component_data['value']:.4f}")
            print(f"  • Trọng số: {component_data['weight']:.4f}")
            print(f"  • Đóng góp: {component_data['contribution']:.4f}")
            print(f"  • Mô tả: {component_data['description']}")
        
        print("\n📋 BASELINE PERFORMANCE:")
        print("-" * 30)
        for metric, value in analysis['baseline_performance'].items():
            print(f"  • {metric}: {value:.3f}")
        
        print("\n⚙️  TÍNH NĂNG THUẬT TOÁN:")
        print("-" * 30)
        for feature in analysis['algorithm_features']:
            print(f"  ✓ {feature}")
        
        print("\n" + "="*80)
