# File: reinforcement_learning.py
import torch
from torch.optim import AdamW
import numpy as np
from collections import deque
import random
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

        args = self._build_reltr_args()
        model, criterion, postprocessors = build_model(args)

        checkpoint_path = self.data_paths.get('reltr_checkpoint')
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[RL] Loading RelTR checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.reltr_device)
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

        relationships = []
        if relationships_path and os.path.exists(relationships_path):
            with open(relationships_path, 'r', encoding='utf-8') as f:
                relationships = json.load(f)

        return {
            'image_path': image_path,
            'width': width,
            'height': height,
            'objects': objects,
            'relationships': relationships,
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
            class_name = obj.get('class')
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

        sample = self._load_original_detection_and_relationships()
        if sample is None:
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

        image_path = Path(sample['image_path'])
        train_image_path = dataset_dir / "images" / "train" / image_path.name
        val_image_path = dataset_dir / "images" / "val" / image_path.name
        shutil.copy(image_path, train_image_path)
        shutil.copy(image_path, val_image_path)

        train_label_path = dataset_dir / "labels" / "train" / (image_path.stem + ".txt")
        val_label_path = dataset_dir / "labels" / "val" / (image_path.stem + ".txt")

        labels_written = self._write_yolo_label_file(
            train_label_path,
            sample['objects'],
            (sample['width'], sample['height']),
            names_map,
        )
        if labels_written:
            shutil.copy(train_label_path, val_label_path)
        else:
            print("[RL] No labels written for detection dataset.")

        yaml_path = self._create_dataset_yaml(dataset_dir, names_map)
        print(f"[RL] Detection dataset prepared at {dataset_dir} (yaml: {yaml_path})")

        self.detection_dataset_dir = dataset_dir
        return dataset_dir

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

    def _build_reltr_target(
        self,
        sample: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        width, height = sample['width'], sample['height']
        objects = sample['objects']
        relationships = sample.get('relationships', [])

        if not objects:
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
            return None

        rel_annotations = []
        for rel in relationships:
            subj_idx = self._find_object_index(objects, rel.get('subject', ''))
            obj_idx = self._find_object_index(objects, rel.get('object', ''))
            rel_idx = self._relation_to_index(rel.get('relation', ''))
            if subj_idx is None or obj_idx is None or rel_idx is None:
                continue
            rel_annotations.append([subj_idx, obj_idx, rel_idx])

        if not rel_annotations:
            print("[RL] No valid relationship annotations found for RelTR training.")
            return None

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

    def _detect_objects_for_image(self, pil_image: Image.Image) -> List[Dict[str, Any]]:
        detection_model = self._ensure_detection_model()
        results = detection_model.predict(
            pil_image,
            imgsz=640,
            conf=0.25,
            verbose=False,
            device=self.detection_device,
        )
        if not results:
            return []
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
        return objects

    def _prepare_reltr_training_sample(self) -> Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor], List[Dict[str, Any]]]]:
        sample = self._load_original_detection_and_relationships()
        if sample is None:
            return None

        target = self._build_reltr_target(sample)
        if target is None:
            return None

        image_tensor = self._load_image_tensor(sample['image_path'])
        return image_tensor, target, sample['objects']

    def _move_target_to_device(self, target: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}

    def _decode_relationships(self, outputs: Dict[str, torch.Tensor], objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rel_logits = outputs.get("rel_logits")
        if rel_logits is None:
            return []

        rel_scores = rel_logits.softmax(-1)[0, :, :-1].detach().cpu()
        if rel_scores.numel() == 0:
            return []

        keep = rel_scores.max(-1).values > 0.4
        filtered = rel_scores[keep] if keep.any() else rel_scores
        num_queries = filtered.shape[0]
        if num_queries == 0:
            filtered = rel_scores
            num_queries = filtered.shape[0]

        relationships: List[Dict[str, Any]] = []
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
                })
                pair_cursor += 1
        return relationships
        
    def train_episode(self, original_relationships, synthetic_data=None):
        print(f"Starting training episode with {len(original_relationships)} relationships")
        
        # 1. Use provided synthetic data or generate new if none provided
        if synthetic_data is None:
            print("Step 1: Generating synthetic data...")
            synthetic_data = []
            for i, rel in enumerate(original_relationships):
                print(f"  Processing relationship {i+1}/{len(original_relationships)}: {rel.get('subject', 'Unknown')} {rel.get('relation', 'Unknown')} {rel.get('object', 'Unknown')}")
                try:
                    generated_images = self.generator.generate_from_relationship(rel, num_variations=3)
                    synthetic_data.extend(generated_images)
                    print(f"    SUCCESS: Generated {len(generated_images)} images")
                except Exception as e:
                    print(f"    ERROR: Error generating images for relationship {i+1}: {e}")
                    continue
            print(f"Total synthetic data generated: {len(synthetic_data)} images")
        else:
            print(f"Step 1: Using provided synthetic data: {len(synthetic_data)} images")
        
        # 2. Train detection model
        print("Step 2: 🧠 Training detection model...")
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
        
        return {
            'detection_loss': detection_loss,
            'relationship_loss': relationship_loss,
            'reward': reward,
            'epsilon': self.epsilon,
            'experience_batch': experience_batch
        }
    
    def calculate_reward(self, synthetic_data, original_relationships):
        # Accuracy reward
        accuracy_reward = self.calculate_accuracy_reward(synthetic_data, original_relationships)
        
        # Diversity reward
        diversity_reward = self.calculate_diversity_reward(synthetic_data)
        
        # Consistency reward
        consistency_reward = self.calculate_consistency_reward(synthetic_data)
        
        total_reward = 0.4 * accuracy_reward + 0.3 * diversity_reward + 0.3 * consistency_reward
        return total_reward
    
    def calculate_accuracy_reward(self, synthetic_data, original_relationships):
        # Simulate detection and relationship prediction
        correct_predictions = 0
        total_predictions = 0
        
        for data in synthetic_data:
            # Mock prediction (replace with actual model inference)
            predicted_relationships = self.predict_relationships(data['image'])
            
            # Compare with original relationships
            for pred_rel in predicted_relationships:
                for orig_rel in original_relationships:
                    if self.relationship_similarity(pred_rel, orig_rel) > 0.7:
                        correct_predictions += 1
                    total_predictions += 1
        
        return correct_predictions / max(total_predictions, 1)
    
    def relationship_similarity(self, rel1, rel2):
        # Calculate similarity between two relationships
        subject_sim = 1.0 if rel1['subject'] == rel2['subject'] else 0.0
        relation_sim = 1.0 if rel1['relation'] == rel2['relation'] else 0.0
        object_sim = 1.0 if rel1['object'] == rel2['object'] else 0.0
        
        return (subject_sim + relation_sim + object_sim) / 3.0
    
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
        model, criterion = self._ensure_relationship_model()
        prepared = self._prepare_reltr_training_sample()
        if prepared is None:
            print("[RL] RelTR training sample unavailable, skipping relationship training.")
            return 0.0

        image_tensor, target, _ = prepared
        samples = nested_tensor_from_tensor_list([image_tensor.to(self.reltr_device)])
        targets = [self._move_target_to_device(target, self.reltr_device)]

        model.train()
        optimizer = AdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=1e-5,
            weight_decay=1e-4,
        )

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        print(f"[RL] Relationship loss after fine-tuning: {loss_value:.4f}")
        model.eval()
        return loss_value
    
    def predict_relationships(self, image):
        """Predict relationships from an input image using the fine-tuned RelTR model."""
        model, _ = self._ensure_relationship_model()
        pil_image = self._ensure_pil_image(image)
        if pil_image is None:
            print("[RL] Unsupported image input for relationship prediction.")
            return []

        detection_objects = self._detect_objects_for_image(pil_image)
        if not detection_objects:
            print("[RL] No objects detected; skipping relationship prediction.")
            return []

        image_tensor = self.reltr_transform(pil_image)
        samples = nested_tensor_from_tensor_list([image_tensor.to(self.reltr_device)])

        model.eval()
        with torch.no_grad():
            outputs = model(samples)
        return self._decode_relationships(outputs, detection_objects)
    
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
    
    def calculate_consistency_reward(self, synthetic_data):
        """Calculate consistency reward based on synthetic data consistency"""
        if not synthetic_data:
            return 0.0
        
        # Mock consistency calculation
        # In real implementation, this would check consistency between
        # synthetic data and original relationships
        consistency_score = random.uniform(0.6, 0.9)
        return consistency_score
    
    def save_model_state(self, reward, detection_loss, relationship_loss):
        """Lưu trạng thái model khi có kết quả tốt"""
        if not self.model_manager.current_experiment_dir:
            print("WARNING: No experiment directory set, cannot save model state")
            return
        
        try:
            # Tạo model state (trong thực tế sẽ là actual model weights)
            model_state = {
                'epsilon': self.epsilon,
                'reward': reward,
                'detection_loss': detection_loss,
                'relationship_loss': relationship_loss,
                'training_step': len(self.training_history['epochs']),
                'model_weights': self.create_mock_model_weights(),  # Mock weights
                'optimizer_state': self.create_mock_optimizer_state()  # Mock optimizer state
            }
            
            # Lưu detection model state
            self.model_manager.save_model_state(
                'detection_model',
                model_state,
                epoch=len(self.training_history['epochs']),
                metadata={
                    'reward': reward,
                    'detection_loss': detection_loss,
                    'relationship_loss': relationship_loss,
                    'epsilon': self.epsilon
                }
            )
            
            # Lưu relationship model state
            self.model_manager.save_model_state(
                'relationship_model',
                model_state,
                epoch=len(self.training_history['epochs']),
                metadata={
                    'reward': reward,
                    'detection_loss': detection_loss,
                    'relationship_loss': relationship_loss,
                    'epsilon': self.epsilon
                }
            )
            
            # Lưu training history
            self.model_manager.save_training_history(self.training_history)
            
            print(f"Saved model state (reward: {reward:.4f})")
            
        except Exception as e:
            print(f"ERROR saving model state: {e}")
    
    def load_model_state(self, model_name='detection_model', experiment_dir=None):
        """Load trạng thái model từ checkpoint"""
        try:
            checkpoint = self.model_manager.load_model_state(model_name, experiment_dir)
            if checkpoint:
                # Restore model state
                self.epsilon = checkpoint['model_state'].get('epsilon', self.epsilon)
                
                # Restore training history
                if 'training_history' in checkpoint:
                    self.training_history = checkpoint['training_history']
                
                print(f"Loaded model state from {model_name}")
                return True
            else:
                print(f"No checkpoint found for {model_name}")
                return False
                
        except Exception as e:
            print(f"ERROR loading model state: {e}")
            return False
    
    def create_mock_model_weights(self):
        """Tạo mock model weights (trong thực tế sẽ là actual weights)"""
        return {
            'layer1_weight': np.random.randn(10, 10).tolist(),
            'layer1_bias': np.random.randn(10).tolist(),
            'layer2_weight': np.random.randn(5, 10).tolist(),
            'layer2_bias': np.random.randn(5).tolist(),
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def create_mock_optimizer_state(self):
        """Tạo mock optimizer state"""
        return {
            'step': len(self.training_history['epochs']),
            'learning_rate': 0.001,
            'momentum': 0.9,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
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
