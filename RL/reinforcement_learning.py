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

from RL.auto_annotator import get_annotator, AutoAnnotator
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
from RL.model_manager import ModelManager
from RL.uncertainty_estimator import UncertaintyEstimator
from RL.active_learning import ActiveLearningSelector
from RL.approximation_algorithm import GreedySubsetSelector
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
            'rewards': deque(maxlen=50),  # Scaled rewards (sau sigmoid)
            'raw_rewards': deque(maxlen=50),  # R_raw trước sigmoid (để tính σ_recent cho k)
            'detection_scores': deque(maxlen=50),
            'relationship_scores': deque(maxlen=50),
            'diversity_scores': deque(maxlen=50),
            'consistency_scores': deque(maxlen=50),
            'uncertainty_reduction_scores': deque(maxlen=50),
            'improvement_trend': deque(maxlen=20),
            'weight_history': deque(maxlen=20),
        }
        # Long-tail handling: tail_weights được tính từ tần suất quan hệ hiếm
        self.tail_weights: Dict[str, float] = {}
        
        # Adaptive scoring parameters
        self.scaling_factor = 1.0
        # α (alpha) dùng trong công thức C_n và S_pos theo báo cáo 4.2.2
        self.reward_alpha = 0.5
        self.baseline_performance = {
            'detection': 0.3,
            'relationship': 0.7,
            'diversity': 0.3,
            'consistency': 0.5,
            'improvement': 0.5,
            'uncertainty_reduction': 0.5,
        }

        # Deep Q-Network agent configuration
        self.rl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Mở rộng action space để bao gồm nhiều số lượng variations
        self.action_space = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # State dimension: khớp README §11.2 (Detection 2, Relationship 2, Training 3, Uncertainty 2; bỏ histogram R để cố định)
        self.state_dim = 9
        self.total_training_epochs = 100  # T trong t/T (episode progress)
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
        
        # Per-relationship performance tracking (NEW: để agent có thể chọn relationships cần tập trung)
        self.relationship_performance: Dict[str, Dict[str, Any]] = {}  # Key: relationship_tuple, Value: performance metrics
        self.relationship_generation_history: Dict[str, List[float]] = {}  # Track F1 scores per relationship
        
        # Training configuration: number of epochs to train on full dataset each episode
        # Increase this value (e.g., 3-5) to train multiple epochs on accumulated dataset
        self.reltr_training_epochs = 1  # Default: 1 epoch per episode (can be increased for better learning)

        # === NEW: Active Learning, Uncertainty Learning, Approximation Algorithm ===
        self.uncertainty_estimator: Optional[UncertaintyEstimator] = None  # Lazy init
        self.active_learner: Optional[ActiveLearningSelector] = None       # Lazy init
        self.subset_selector = GreedySubsetSelector(
            diversity_weight=0.5,
            quality_weight=0.3,
            representativeness_weight=0.2,
        )
        self._previous_epoch_uncertainties: Dict[str, float] = {}  # Track uncertainty reduction
        self.latest_uncertainty_mean: float = 0.5
        self.latest_uncertainty_max: float = 0.5

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_uncertainty_estimator(self):
        """Lazy initialization cho UncertaintyEstimator và ActiveLearningSelector."""
        if self.uncertainty_estimator is None:
            model, _ = self._ensure_relationship_model()
            self.uncertainty_estimator = UncertaintyEstimator(
                model, self.reltr_device, n_forward_passes=10
            )
            self.active_learner = ActiveLearningSelector(
                self.uncertainty_estimator, strategy='combined'
            )
            print("[RL] Initialized UncertaintyEstimator (MC Dropout, T=10) "
                  "and ActiveLearningSelector (combined strategy)")
        return self.uncertainty_estimator, self.active_learner
    def _build_reltr_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _normalize_label(label: str) -> str:
        return label.strip().lower().replace("_", " ").replace("-", " ")

    def _recompute_tail_weights(self) -> None:
        """Tính trọng số cho các quan hệ hiếm (long-tail) dựa trên tần suất xuất hiện trong dataset_samples."""
        if not self.dataset_samples:
            self.tail_weights = {}
            return

        freq: Dict[str, int] = {}
        for sample in self.dataset_samples:
            for rel in sample.get('relationships', []) or []:
                rel_name = self._normalize_label(rel.get('relation', ''))
                if not rel_name:
                    continue
                freq[rel_name] = freq.get(rel_name, 0) + 1

        if not freq:
            self.tail_weights = {}
            return

        # 1/sqrt(freq) để ưu tiên lớp hiếm, sau đó normalize
        raw_weights = {k: 1.0 / math.sqrt(v + 1e-3) for k, v in freq.items()}
        total = sum(raw_weights.values()) or 1.0
        self.tail_weights = {k: v / total for k, v in raw_weights.items()}

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

    def _build_state_vector(self, metrics: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Build state vector for DQN agent — khớp README §11.2 và tài liệu đánh giá.
        
        s_t = [ F_det (2), F_rel (2), r_{t-1}, ε_t, t/T, U_mean, U_max ] → 9 chiều.
        (Bỏ histogram phân phối loại quan hệ R chiều để state cố định.)
        
        Thành phần:
        1. Avg detection confidence (proxy: F1_det)
        2. Detection count (normalized) N/100
        3. Avg relationship confidence (proxy: F1_rel)
        4. Relationship count (normalized) |R|/200
        5. Previous reward r_{t-1}
        6. Epsilon ε_t
        7. Episode progress t/T
        8. Mean uncertainty U_mean
        9. Max uncertainty U_max
        """
        metrics = metrics or self.last_metrics
        
        # Detection: F1 (proxy cho avg confidence), N/100 với N = tổng đối tượng phát hiện (README §11.2)
        detection_f1 = float(metrics.get('detection_f1', 0.0))
        if detection_f1 == 0.0 and 'detection_loss' in metrics:
            detection_f1 = max(0.0, 1.0 - self._normalize_scalar(float(metrics['detection_loss']), scale=5.0))
        detection_count = float(metrics.get('detection_count', metrics.get('num_samples', len(self.dataset_samples))))
        if detection_count == 0 and getattr(self, 'latest_detection_metrics', None):
            det = self.latest_detection_metrics
            detection_count = float(det.get('tp', 0) + det.get('fp', 0))
        detection_count_norm = min(1.0, detection_count / 100.0)
        
        # Relationship: F1 (proxy cho avg confidence), |R|/200 với |R| = tổng quan hệ GT (README §11.2)
        relationship_f1 = float(metrics.get('relationship_f1', 0.0))
        if relationship_f1 == 0.0 and 'relationship_loss' in metrics:
            relationship_f1 = max(0.0, 1.0 - self._normalize_scalar(float(metrics['relationship_loss']), scale=5.0))
        relationship_count = float(metrics.get('relationship_count', metrics.get('num_samples', 0)))
        if relationship_count == 0 and self.latest_relationship_metrics:
            rel_tp = self.latest_relationship_metrics.get('tp', 0)
            rel_fn = self.latest_relationship_metrics.get('fn', 0)
            relationship_count = float(rel_tp + rel_fn)
        relationship_count_norm = min(1.0, relationship_count / 200.0)
        
        # Training state: r_{t-1}, ε_t, t/T
        reward_value = self._normalize_scalar(float(metrics.get('reward', 0.0)), scale=1.0)
        epsilon_value = self._normalize_scalar(float(self.epsilon), scale=1.0)
        current_epoch = len(self.training_history['epochs'])
        total_epochs = max(1, getattr(self, 'total_training_epochs', 100))
        progress_t_T = min(1.0, current_epoch / total_epochs)
        
        # Uncertainty: U_mean, U_max (từ MC Dropout, mặc định 0.5 nếu chưa có)
        u_mean = float(getattr(self, 'latest_uncertainty_mean', 0.5))
        u_max = float(getattr(self, 'latest_uncertainty_max', 0.5))
        
        state = torch.tensor(
            [
                detection_f1,
                detection_count_norm,
                relationship_f1,
                relationship_count_norm,
                reward_value,
                epsilon_value,
                progress_t_T,
                u_mean,
                u_max,
            ],
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
                # Log tất cả Q_values để debug
                q_values_list = q_values.squeeze(0).cpu().tolist()
                print(f"[RL] Q_values for all actions: {dict(zip(self.action_space, q_values_list))}")
                action_index = int(q_values.argmax(dim=1).item())
        action_value = self.action_space[action_index]
        return action_index, action_value

    def _get_relationship_key(self, relationship: Dict[str, Any]) -> str:
        """Tạo key duy nhất cho relationship để tracking"""
        subject = relationship.get('subject', 'unknown')
        relation = relationship.get('relation', 'unknown')
        obj = relationship.get('object', 'unknown')
        return f"{subject}|{relation}|{obj}".lower()
    
    def _update_relationship_performance(self, relationship: Dict[str, Any], f1_score: float):
        """Cập nhật hiệu suất cho một relationship cụ thể"""
        rel_key = self._get_relationship_key(relationship)
        
        if rel_key not in self.relationship_performance:
            self.relationship_performance[rel_key] = {
                'relationship': relationship,
                'f1_scores': deque(maxlen=20),  # Lưu 20 F1 scores gần nhất
                'avg_f1': 0.0,
                'min_f1': 1.0,
                'max_f1': 0.0,
                'generation_count': 0,
                'last_improvement': 0.0,
            }
            self.relationship_generation_history[rel_key] = []
        
        perf = self.relationship_performance[rel_key]
        perf['f1_scores'].append(f1_score)
        perf['avg_f1'] = sum(perf['f1_scores']) / len(perf['f1_scores'])
        perf['min_f1'] = min(perf['f1_scores'])
        perf['max_f1'] = max(perf['f1_scores'])
        
        # Tính improvement
        if len(perf['f1_scores']) > 1:
            perf['last_improvement'] = f1_score - list(perf['f1_scores'])[-2]
    
    def _get_relationship_priorities(self, original_relationships: List[Dict[str, Any]], 
                                     base_variations: int) -> Dict[str, int]:
        """
        Tính toán số lượng variations cần sinh cho mỗi relationship dựa trên hiệu suất.
        Relationships có F1 thấp sẽ được sinh nhiều ảnh hơn.
        
        Returns: Dict mapping relationship key -> số variations cần sinh
        """
        priorities = {}
        
        for rel in original_relationships:
            rel_key = self._get_relationship_key(rel)
            
            if rel_key not in self.relationship_performance:
                # Relationship mới chưa có data -> sinh số lượng cơ bản
                priorities[rel_key] = base_variations
            else:
                perf = self.relationship_performance[rel_key]
                avg_f1 = perf['avg_f1']
                
                # Tính priority score: F1 càng thấp -> cần sinh càng nhiều
                # F1 = 0.0 -> sinh 3x base_variations
                # F1 = 0.5 -> sinh 2x base_variations  
                # F1 = 0.8+ -> sinh 0.5x base_variations (ít hơn)
                
                if avg_f1 < 0.3:
                    # Rất yếu -> cần nhiều data
                    variations = max(int(base_variations * 3), 5)
                elif avg_f1 < 0.5:
                    # Yếu -> cần nhiều data
                    variations = max(int(base_variations * 2), 3)
                elif avg_f1 < 0.7:
                    # Trung bình -> sinh bình thường
                    variations = base_variations
                else:
                    # Tốt -> sinh ít hơn để tiết kiệm
                    variations = max(int(base_variations * 0.5), 1)
                
                # Nếu đang cải thiện nhưng vẫn thấp, vẫn cần nhiều data
                if perf['last_improvement'] > 0.05 and avg_f1 < 0.6:
                    variations = int(variations * 1.5)
                
                priorities[rel_key] = variations
        
        return priorities
    
    def decide_action(self, original_relationships: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Choose an action for the next training episode using epsilon-greedy DQN policy.
        NEW: Sử dụng Active Learning (uncertainty-based) để tạo generation plan.
        
        Returns a context dictionary that should be passed back after the episode completes.
        """
        state = self._build_state_vector()
        action_index, action_value = self._select_action(state)
        self.last_state = state
        self.last_action_index = action_index
        
        # === Active Learning: Uncertainty-based generation plan ===
        relationship_plan = {}
        active_learning_stats = {}
        if original_relationships:
            try:
                # Lazy init uncertainty estimator & active learner
                self._ensure_uncertainty_estimator()
                
                # Score relationships using uncertainty + performance + tail_weight
                scored_relationships = self.active_learner.score_relationships(
                    relationships=original_relationships,
                    evaluation_samples=self.dataset_samples or [],
                    relationship_performance=self.relationship_performance,
                    tail_weights=self.tail_weights,
                    transform_fn=self.reltr_transform,
                )
                
                # Create generation plan with total budget
                total_budget = action_value * len(original_relationships)
                relationship_plan = self.active_learner.create_generation_plan(
                    scored_relationships,
                    total_budget=total_budget,
                    min_per_rel=1,
                    max_per_rel=max(action_value * 3, 10),
                )
                
                # Log detailed reasoning
                self.active_learner.log_selection_reasoning(relationship_plan, scored_relationships)
                
                active_learning_stats = {
                    'scored_relationships': [
                        {
                            'rel_key': sr['rel_key'],
                            'acquisition_score': sr['acquisition_score'],
                            'uncertainty_score': sr['uncertainty_score'],
                            'performance_score': sr['performance_score'],
                            'tail_score': sr['tail_score'],
                        }
                        for sr in scored_relationships
                    ],
                    'total_budget': total_budget,
                    'strategy': self.active_learner.strategy,
                }
                
            except Exception as e:
                print(f"[RL] Active Learning scoring failed, falling back to F1-heuristic: {e}")
                relationship_plan = self._get_relationship_priorities(original_relationships, action_value)
        
        return {
            'state': state.clone().detach(),
            'action_index': action_index,
            'num_variations': action_value,  # Base variations (backward compatibility)
            'relationship_plan': relationship_plan,
            'active_learning_stats': active_learning_stats,  # NEW
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

    def load_dataset_snapshot(self, from_experiment_dir: str = None) -> bool:
        """Load dataset snapshot từ experiment directory được chỉ định hoặc từ experiment hiện tại"""
        if from_experiment_dir:
            # Load từ experiment cũ
            old_snapshot_path = Path(from_experiment_dir) / "dataset" / "samples.json"
            if not old_snapshot_path.exists():
                print(f"[RL] No dataset snapshot found in {from_experiment_dir}")
                return False
            
            try:
                with open(old_snapshot_path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                
                samples = payload.get("samples") or []
                detection_dir = payload.get("detection_dataset_dir")
                
                self.dataset_samples = samples
                if detection_dir and Path(detection_dir).exists():
                    self.detection_dataset_dir = Path(detection_dir)
                else:
                    self.detection_dataset_dir = None
                
                print(f"[RL] Loaded {len(samples)} samples from previous experiment")
                return bool(self.dataset_samples)
                
            except Exception as exc:
                print(f"[RL] Warning: failed to load dataset snapshot from {from_experiment_dir}: {exc}")
                return False
        else:
            # Load từ experiment hiện tại
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

    # Synonym mapping for fuzzy matching in relationship building
    _LABEL_SYNONYMS = {
        'person': ['man', 'woman', 'people', 'human', 'boy', 'girl', 'child', 'adult'],
        'car': ['vehicle', 'automobile', 'auto'],
        'bike': ['bicycle', 'cycle'],
        'motorcycle': ['motorbike', 'scooter'],
        'phone': ['smartphone', 'cellphone', 'mobile', 'cell phone'],
        'laptop': ['computer', 'notebook'],
        'dog': ['puppy', 'canine'],
        'cat': ['kitten', 'feline'],
        'chair': ['seat'],
        'table': ['desk'],
        'tv': ['television', 'monitor', 'screen'],
    }
    
    def _get_label_synonyms(self, label: str) -> List[str]:
        """Get all synonyms for a label including the label itself."""
        normalized = self._normalize_label(label)
        synonyms = [normalized]
        
        # Check if label is a key
        if normalized in self._LABEL_SYNONYMS:
            synonyms.extend(self._LABEL_SYNONYMS[normalized])
        
        # Check if label is a value (reverse lookup)
        for key, values in self._LABEL_SYNONYMS.items():
            if normalized in values:
                synonyms.append(key)
                synonyms.extend(values)
        
        return list(set(synonyms))

    def _find_object_index(self, objects: List[Dict[str, Any]], class_name: str) -> Optional[int]:
        """Find object index with fuzzy matching support for synonyms."""
        normalized = self._normalize_label(class_name)
        
        # First try exact match
        for idx, obj in enumerate(objects):
            if self._normalize_label(obj.get('class', '')) == normalized:
                return idx
        
        # If no exact match, try synonym matching
        synonyms = self._get_label_synonyms(class_name)
        for idx, obj in enumerate(objects):
            obj_label = self._normalize_label(obj.get('class', ''))
            if obj_label in synonyms:
                return idx
        
        # If still no match, try partial matching (label contains or is contained)
        for idx, obj in enumerate(objects):
            obj_label = self._normalize_label(obj.get('class', ''))
            if normalized in obj_label or obj_label in normalized:
                return idx
        
        return None

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
            
            original_relationship = data.get('original_relationship')
            print(f"[RL] Sample {index+1} original relationship: {original_relationship}")
            
            # ========== BƯỚC 6.5: AUTO-ANNOTATION ==========
            # Sử dụng GroundingDINO/OWL-ViT để detect bbox trong ảnh synthetic
            # Giải quyết vấn đề: SD chỉ trả về pixels, không có bbox
            sample = None
            try:
                annotator = get_annotator()
                if original_relationship:
                    annotation_result = annotator.annotate_from_relationship(
                        image_path, original_relationship
                    )
                    if annotation_result and annotation_result.get('objects'):
                        sample = {
                            'image_path': annotation_result['image_path'],
                            'width': annotation_result['width'],
                            'height': annotation_result['height'],
                            'objects': annotation_result['objects'],
                            'global_context': [],  # Will be computed if needed
                            'annotation_backend': annotation_result.get('annotation_backend', 'unknown'),
                        }
                        print(f"[RL] Sample {index+1}: AutoAnnotator ({annotation_result.get('annotation_backend')}) "
                              f"detected {len(annotation_result['objects'])} objects")
            except Exception as e:
                print(f"[RL] Sample {index+1}: AutoAnnotator failed: {e}")
            
            # Fallback to YOLO+CLIP if AutoAnnotator fails
            if not sample:
                print(f"[RL] Sample {index+1}: Falling back to YOLO+CLIP extraction")
                sample = self._extract_objects_with_clip(image_path)
            
            if not sample:
                print(f"[RL] Skipping sample {index+1}: failed to extract objects")
                continue
            
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
            # Cập nhật trọng số long-tail sau khi ingest dataset
            self._recompute_tail_weights()
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
        # Cập nhật trọng số long-tail sau khi build dataset
        self._recompute_tail_weights()
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
        long_tail_loss: float = 0.0,
        detection_metrics: Optional[Dict[str, float]] = None,
        relationship_metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """
        Finalize RL step by storing experience and updating Q-network.
        
        IMPROVED: Now accepts evaluation metrics (F1 scores) for better state representation.
        """
        # Build metrics dict with both losses and evaluation F1 scores
        metrics = {
            'detection_loss': detection_loss,
            'relationship_loss': relationship_loss,
            'long_tail_loss': long_tail_loss,
            'reward': reward,
            'dataset_size': len(self.dataset_samples),
        }
        
        # Add evaluation F1 scores if available (preferred for state building)
        if detection_metrics:
            metrics['detection_f1'] = detection_metrics.get('f1', 0.0)
            # N = số lượng đối tượng phát hiện (README §11.2): tổng dự đoán = tp + fp
            metrics['detection_count'] = detection_metrics.get('tp', 0) + detection_metrics.get('fp', 0)
        if relationship_metrics:
            metrics['relationship_f1'] = relationship_metrics.get('f1', 0.0)
            # |R| = số lượng quan hệ (README §11.2): tổng GT = tp + fn
            metrics['relationship_count'] = relationship_metrics.get('tp', 0) + relationship_metrics.get('fn', 0)
        
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

    def _calculate_mr_at_k(
        self,
        all_ground_truths: List[List[Dict[str, Any]]],
        all_predictions: List[List[Dict[str, Any]]],
        k_values: List[int] = [10, 20, 50, 100]
    ) -> Dict[str, float]:
        """
        Calculate mean Recall@K (mR@K) for relationship prediction.
        
        mR@K is the mean of Recall@K across all relationship types, which provides
        a fairer evaluation for rare relationships (long-tail) compared to R@K.
        
        Algorithm:
        1. For each relation type, calculate R@K = (# of correct predictions in top K) / (# of ground truths)
        2. mR@K = mean of all relation type R@K values
        
        Args:
            all_ground_truths: List of ground truth relationship lists for each sample
            all_predictions: List of predicted relationships (sorted by confidence) for each sample
            k_values: List of K values to compute (default: [10, 20, 50, 100])
        
        Returns:
            Dictionary mapping 'mr@10', 'mr@20', etc. to their values
        """
        if not all_ground_truths or not all_predictions or len(all_ground_truths) != len(all_predictions):
            return {f'mr@{k}': 0.0 for k in k_values}
        
        # Collect all unique relation types from ground truths
        relation_types = set()
        for gt_list in all_ground_truths:
            for gt in gt_list:
                if not gt:
                    continue
                rel_type = self._normalize_label(gt.get('relation', ''))
                if rel_type:
                    relation_types.add(rel_type)
        
        if not relation_types:
            return {f'mr@{k}': 0.0 for k in k_values}
        
        mr_at_k_results = {}
        
        for k in k_values:
            relation_recalls = []
            
            # Calculate R@K for each relation type
            for rel_type in relation_types:
                total_gt_count = 0
                hits = 0
                
                # For each sample
                for sample_idx, (gt_list, pred_list) in enumerate(zip(all_ground_truths, all_predictions)):
                    if not pred_list:
                        continue
                    
                    # Get GT relationships of this type in this sample
                    gt_tuples_of_type = []
                    for gt in gt_list:
                        if not gt:
                            continue
                        if self._normalize_label(gt.get('relation', '')) == rel_type:
                            gt_tuple = self._normalize_relationship_tuple(gt)
                            gt_tuples_of_type.append(gt_tuple)
                    
                    if not gt_tuples_of_type:
                        continue
                    
                    total_gt_count += len(gt_tuples_of_type)
                    
                    # Get top K predictions (sorted by confidence if available)
                    top_k_preds = pred_list[:k]
                    pred_tuples = [
                        self._normalize_relationship_tuple(rel) 
                        for rel in top_k_preds 
                        if rel
                    ]
                    
                    # Count how many GT tuples of this type are in top K
                    for gt_tuple in gt_tuples_of_type:
                        if gt_tuple in pred_tuples:
                            hits += 1
                
                # Calculate recall for this relation type
                recall = hits / total_gt_count if total_gt_count > 0 else 0.0
                relation_recalls.append(recall)
            
            # Mean Recall@K = average of all relation type recalls
            mr_at_k = sum(relation_recalls) / len(relation_recalls) if relation_recalls else 0.0
            mr_at_k_results[f'mr@{k}'] = mr_at_k
        
        return mr_at_k_results

    def _evaluate_relationship_metrics(
        self,
        evaluation_data: List[Dict[str, Any]],
        original_relationships: List[Dict[str, Any]],
        max_samples: int = 30,
    ) -> Dict[str, Any]:
        if not evaluation_data:
            print("[RL] No evaluation data provided for relationship evaluation")
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
                'mr@10': 0.0,
                'mr@20': 0.0,
                'mr@50': 0.0,
                'mr@100': 0.0,
            }

        print(f"[RL] Evaluating relationship metrics on {len(evaluation_data)} samples")
        
        # Debug: Kiểm tra loại dữ liệu
        dataset_samples_count = sum(1 for data in evaluation_data if 'relationships' in data)
        synthetic_samples_count = sum(1 for data in evaluation_data if 'original_relationship' in data)
        print(f"[RL] Debug - Dataset samples: {dataset_samples_count}, Synthetic samples: {synthetic_samples_count}")
        
        total_tp = total_fp = total_fn = 0
        per_sample_f1: List[float] = []
        evaluated = 0
        
        # For mR@K calculation
        all_ground_truths: List[Dict[str, Any]] = []
        all_predictions: List[List[Dict[str, Any]]] = []

        subset = evaluation_data[:max_samples]
        for i, data in enumerate(subset):
            # Đối với dataset_samples, lấy relationship từ relationships field
            # Đối với synthetic_data, lấy từ original_relationship field
            if 'relationships' in data and data['relationships']:
                target_rel = data['relationships'][0]  # Lấy relationship đầu tiên
                all_gt_rels = data['relationships']  # Tất cả relationships trong sample
            else:
                target_rel = data.get('original_relationship')
                all_gt_rels = [target_rel] if target_rel else []
            
            image_input = data.get('image') or data.get('image_path')
            if not target_rel or image_input is None:
                print(f"[RL] Skipping sample {i+1}: missing target relationship or image input")
                continue

            try:
                # Ensure relationship model is loaded
                self._ensure_relationship_model()
                predicted_relationships = self.predict_relationships(image_input) or []
                
                # Sort predictions by confidence if available
                if predicted_relationships:
                    predicted_relationships = sorted(
                        predicted_relationships,
                        key=lambda x: x.get('confidence', 0.0),
                        reverse=True
                    )
                
                print(f"[RL] Sample {i+1}: predicted {len(predicted_relationships)} relationships")
                
                # Debug: In ra target relationship để kiểm tra
                print(f"[RL] Sample {i+1} target: {target_rel}")
                if predicted_relationships:
                    print(f"[RL] Sample {i+1} predicted: {predicted_relationships[0] if predicted_relationships else 'None'}")
            except Exception as exc:
                print(f"[RL] Relationship evaluation failed for sample {i+1}: {exc}")
                predicted_relationships = []

            # Store for mR@K calculation
            all_ground_truths.append(all_gt_rels)
            all_predictions.append(predicted_relationships)

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
            
            # NEW: Track per-relationship performance
            if target_rel:
                self._update_relationship_performance(target_rel, f1)

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
                'mr@10': 0.0,
                'mr@20': 0.0,
                'mr@50': 0.0,
                'mr@100': 0.0,
            }

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_f1 = sum(per_sample_f1) / len(per_sample_f1) if per_sample_f1 else 0.0
        variance = sum((score - mean_f1) ** 2 for score in per_sample_f1) / len(per_sample_f1) if per_sample_f1 else 0.0
        std_f1 = math.sqrt(variance)

        # Calculate mR@K metrics
        mr_at_k_results = self._calculate_mr_at_k(all_ground_truths, all_predictions)

        print(f"[RL] Relationship evaluation completed: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (evaluated {evaluated} samples)")
        print(f"[RL] mR@K metrics: mR@10={mr_at_k_results.get('mr@10', 0.0):.4f}, mR@20={mr_at_k_results.get('mr@20', 0.0):.4f}, "
              f"mR@50={mr_at_k_results.get('mr@50', 0.0):.4f}, mR@100={mr_at_k_results.get('mr@100', 0.0):.4f}")
        
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
            **mr_at_k_results,  # Add mR@K metrics
        }

    def train_episode(self, original_relationships, synthetic_data=None, action_context: Optional[Dict[str, Any]] = None, done: bool = False):
        print(f"Starting training episode with {len(original_relationships)} relationships")
        action_variations = action_context.get('num_variations', 3) if action_context else 3
        
        # NEW: Lấy relationship-specific generation plan nếu có
        relationship_plan = action_context.get('relationship_plan', {}) if action_context else {}
        
        # 1. Use provided synthetic data or generate new if none provided
        if synthetic_data is None:
            if relationship_plan:
                print(f"Step 1: Generating synthetic data with relationship-specific plan...")
            else:
                print(f"Step 1: Generating synthetic data (variations per relation: {action_variations})...")
            synthetic_data = []
            for i, rel in enumerate(original_relationships):
                rel_key = self._get_relationship_key(rel)
                
                # NEW: Sử dụng số variations cụ thể cho relationship này nếu có plan
                if relationship_plan and rel_key in relationship_plan:
                    num_variations = relationship_plan[rel_key]
                    perf_info = self.relationship_performance.get(rel_key, {})
                    avg_f1 = perf_info.get('avg_f1', 0.0)
                    print(f"  Processing relationship {i+1}/{len(original_relationships)}: "
                          f"{rel.get('subject', 'Unknown')} {rel.get('relation', 'Unknown')} {rel.get('object', 'Unknown')} "
                          f"-> {num_variations} variations (F1: {avg_f1:.3f})")
                else:
                    num_variations = action_variations
                    print(f"  Processing relationship {i+1}/{len(original_relationships)}: "
                          f"{rel.get('subject', 'Unknown')} {rel.get('relation', 'Unknown')} {rel.get('object', 'Unknown')} "
                          f"-> {num_variations} variations (default)")
                
                try:
                    generated_images = self.generator.generate_from_relationship(rel, num_variations=num_variations)
                    synthetic_data.extend(generated_images)
                    print(f"    SUCCESS: Generated {len(generated_images)} images")
                except Exception as e:
                    print(f"    ERROR: Error generating images for relationship {i+1}: {e}")
                    continue
            print(f"Total synthetic data generated: {len(synthetic_data)} images")
        else:
            if relationship_plan:
                print(f"Step 1: Using provided synthetic data: {len(synthetic_data)} images (with relationship-specific plan)")
            else:
                print(f"Step 1: Using provided synthetic data: {len(synthetic_data)} images (variations per relation: {action_variations})")

        # === Approximation Algorithm: Greedy Submodular Subset Selection ===
        subset_selection_stats = {}
        if len(synthetic_data) > 3:
            try:
                # Step 1.5: Loại bỏ samples trùng lặp
                filtered_data = self.subset_selector.filter_redundant_samples(
                    synthetic_data, min_distance=0.08
                )
                
                # Step 1.6: Chọn subset tối ưu nếu pool lớn hơn budget hợp lý
                # Budget = 70% pool size (giữ lại đa dạng nhất, bỏ 30% kém chất lượng/trùng lặp)
                subset_budget = max(3, int(len(filtered_data) * 0.7))
                if len(filtered_data) > subset_budget:
                    synthetic_data, subset_selection_stats = self.subset_selector.select_optimal_subset(
                        filtered_data, budget=subset_budget
                    )
                    print(f"[RL] Approximation Algorithm: {subset_selection_stats['pool_size']} → "
                          f"{subset_selection_stats['selected']} samples "
                          f"(coverage: {subset_selection_stats['relationship_coverage']:.1%})")
                else:
                    synthetic_data = filtered_data
                    subset_selection_stats = {
                        'pool_size': len(filtered_data), 'selected': len(filtered_data),
                        'skipped': True, 'reason': 'pool_too_small',
                    }
            except Exception as e:
                print(f"[RL] Approximation Algorithm failed, using full pool: {e}")
                subset_selection_stats = {'error': str(e)}

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
        # Use configurable number of epochs (default: 1, but can be increased for better learning)
        num_training_epochs = getattr(self, 'reltr_training_epochs', 1)
        relationship_loss, long_tail_loss = self.train_relationship_model(synthetic_data, num_epochs=num_training_epochs)
        print(f"    ✅ Relationship loss: {relationship_loss:.4f}")
        print(f"    ✅ Long-tail loss: {long_tail_loss:.4f}")
        
        # 4. Calculate reward
        print("Step 4: 📊 Calculating reward...")
        reward = self.calculate_reward(synthetic_data, original_relationships)
        print(f"    ✅ Reward: {reward:.4f}")
        
        # Debug: Kiểm tra reward components
        reward_components = dict(self.latest_reward_components or {})
        print(f"    🔍 Debug - Reward components keys: {list(reward_components.keys())}")
        
        if reward_components:
            print(
                "     Reward breakdown -> "
                f"Detection: {reward_components.get('detection_score', 0.0):.3f}, "
                f"Relationship: {reward_components.get('relationship_score', 0.0):.3f}, "
                f"Diversity: {reward_components.get('diversity_score', 0.0):.3f}, "
                f"Consistency: {reward_components.get('consistency_score', 0.0):.3f}, "
                f"Improvement: {reward_components.get('improvement_score', 0.0):.3f}, "
                f"Uncertainty: {reward_components.get('uncertainty_reduction_score', 0.0):.3f}"
            )
            dynamic_weights = reward_components.get('dynamic_weights', {})
            if dynamic_weights:
                print(f"     Dynamic weights -> Detection: {dynamic_weights.get('detection', 0.0):.3f}, "
                      f"Relationship: {dynamic_weights.get('relationship', 0.0):.3f}, "
                      f"Diversity: {dynamic_weights.get('diversity', 0.0):.3f}, "
                      f"Consistency: {dynamic_weights.get('consistency', 0.0):.3f}, "
                      f"Improvement: {dynamic_weights.get('improvement', 0.0):.3f}, "
                      f"Uncertainty: {dynamic_weights.get('uncertainty_reduction', 0.0):.3f}")
            else:
                print("     ⚠️  No dynamic weights found!")
        else:
            print("     ⚠️  No reward components found!")
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
            'long_tail_loss': long_tail_loss,
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
            long_tail_loss=long_tail_loss,
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
            long_tail_loss=long_tail_loss,
            detection_metrics=detection_metrics_snapshot,  # Pass evaluation F1 for better state
            relationship_metrics=relationship_metrics_snapshot,  # Pass evaluation F1 for better state
        )
        if rl_loss is not None:
            print(f"[RL] Q-network optimization loss: {rl_loss:.6f}")
        
        return {
            'detection_loss': detection_loss,
            'relationship_loss': relationship_loss,
            'long_tail_loss': long_tail_loss,
            'reward': reward,
            'epsilon': self.epsilon,
            'experience_batch': experience_batch,
            'rl_action_index': rl_action_index,
            'rl_num_variations': action_variations,
            'rl_loss': rl_loss,
            'reward_components': reward_components,
            'detection_metrics': detection_metrics_snapshot,
            'relationship_metrics': relationship_metrics_snapshot,
            'subset_selection_stats': subset_selection_stats,  # NEW: Approximation Algorithm stats
        }
    
    def calculate_reward(self, synthetic_data, original_relationships):
        """
        Tính điểm dựa trên thuật toán thích ứng và các chỉ số khách quan.
        Thay thế công thức chủ quan bằng hệ thống đánh giá có thể đo lường được.
        """
        # Debug information
        print(f"    🔍 Debug - Synthetic data count: {len(synthetic_data) if synthetic_data else 0}")
        print(f"    🔍 Debug - Dataset samples count: {len(self.dataset_samples)}")
        print(f"    🔍 Debug - Original relationships count: {len(original_relationships) if original_relationships else 0}")
        
        # 1. Thu thập các chỉ số cơ bản
        detection_metrics = self._evaluate_detection_metrics(self.dataset_samples)
        
        # Sử dụng dataset_samples thay vì chỉ synthetic_data để đánh giá relationship
        # Điều này đảm bảo đánh giá trên toàn bộ dữ liệu training, không chỉ ảnh mới
        evaluation_samples = self.dataset_samples if self.dataset_samples else synthetic_data
        relationship_metrics = self._evaluate_relationship_metrics(evaluation_samples, original_relationships)
        per_sample_f1 = relationship_metrics.pop('per_sample_f1', [])
        
        # Debug metrics
        print(f"    🔍 Debug - Detection metrics: {detection_metrics}")
        print(f"    🔍 Debug - Relationship metrics: {relationship_metrics}")
        
        # 2. Tính toán các thành phần điểm với thuật toán cụ thể
        detection_score = self._calculate_detection_score(detection_metrics)
        relationship_score = self._calculate_relationship_score(relationship_metrics)
        
        # FIXED: Use dataset_samples which has objects field, not raw synthetic_data
        # Raw synthetic_data only has image, prompt, original_relationship
        # After ingestion, dataset_samples has objects, relationships, bbox info etc.
        diversity_score = self._calculate_diversity_score(self.dataset_samples if self.dataset_samples else synthetic_data)
        consistency_score = self._calculate_consistency_score(per_sample_f1, relationship_metrics.get('f1_std'))
        improvement_score = self._calculate_improvement_score()
        
        # === NEW: Uncertainty Reduction Score ===
        uncertainty_reduction_score = 0.0
        try:
            if self.uncertainty_estimator is not None and evaluation_samples:
                batch_unc = self.uncertainty_estimator.estimate_batch(
                    evaluation_samples, transform_fn=self.reltr_transform, max_samples=10
                )
                if batch_unc:
                    current_uncertainties = {
                        str(k): v.get('uncertainty_score', 0.5) for k, v in batch_unc.items()
                    }
                    # Lưu U_mean, U_max cho state vector (README §11.2)
                    unc_vals = list(current_uncertainties.values())
                    if unc_vals:
                        self.latest_uncertainty_mean = sum(unc_vals) / len(unc_vals)
                        self.latest_uncertainty_max = max(unc_vals)
                    else:
                        self.latest_uncertainty_mean = 0.5
                        self.latest_uncertainty_max = 0.5
                    uncertainty_reduction_score = self.uncertainty_estimator.compute_uncertainty_reduction(
                        current_uncertainties
                    )
                    # ρ có thể rất âm khi uncertainty tăng mạnh → bọc tanh để S_unc ∈ (0,1) (README §12.7)
                    lam = getattr(self, 'unc_rho_lambda', 2.0)
                    uncertainty_reduction_score = 0.5 + 0.5 * math.tanh(lam * uncertainty_reduction_score)
                    print(f"    🔍 Debug - Uncertainty reduction score: {uncertainty_reduction_score:.4f}")
        except Exception as e:
            print(f"    ⚠️ Uncertainty estimation in reward failed: {e}")
            uncertainty_reduction_score = 0.5  # Neutral on failure
            self.latest_uncertainty_mean = 0.5
            self.latest_uncertainty_max = 0.5
        
        # Debug scores
        print(f"    🔍 Debug - Detection score: {detection_score:.4f}")
        print(f"    🔍 Debug - Relationship score: {relationship_score:.4f}")
        print(f"    🔍 Debug - Diversity score: {diversity_score:.4f}")
        print(f"    🔍 Debug - Consistency score: {consistency_score:.4f}")
        print(f"    🔍 Debug - Improvement score: {improvement_score:.4f}")
        print(f"    🔍 Debug - Uncertainty reduction score: {uncertainty_reduction_score:.4f}")
        
        # 3. Tính trọng số động (gồm cả S_unc) theo công thức thích nghi
        dynamic_weights = self._calculate_dynamic_weights(
            detection_score,
            relationship_score,
            diversity_score,
            consistency_score,
            improvement_score=improvement_score,
            uncertainty_reduction_score=uncertainty_reduction_score,
        )

        # 4. Tính điểm tổng hợp với trọng số thích ứng
        total_reward_raw = (
            dynamic_weights['detection'] * detection_score +
            dynamic_weights['relationship'] * relationship_score +
            dynamic_weights['diversity'] * diversity_score +
            dynamic_weights['consistency'] * consistency_score +
            dynamic_weights['improvement'] * improvement_score +
            dynamic_weights['uncertainty_reduction'] * uncertainty_reduction_score
        )
        # 5. Cập nhật k từ σ_recent (R_raw) rồi áp dụng sigmoid (README §12.1)
        self.scaling_factor = self._get_current_scaling_factor()
        total_reward = self._apply_reward_scaling(total_reward_raw)
        
        # 6. Lưu trữ thông tin để phân tích
        self.latest_detection_metrics = detection_metrics
        self.latest_relationship_metrics = relationship_metrics
        self.latest_reward_components = {
            'detection_score': detection_score,
            'relationship_score': relationship_score,
            'diversity_score': diversity_score,
            'consistency_score': consistency_score,
            'improvement_score': improvement_score,
            'uncertainty_reduction_score': uncertainty_reduction_score,  # NEW
            'dynamic_weights': dynamic_weights,
            'total_reward': total_reward,
            'scaling_factor': self._get_current_scaling_factor(),
        }
        
        # 7. Cập nhật lịch sử (gồm R_raw để lần sau tính k)
        self._update_performance_history(
            total_reward,
            dynamic_weights,
            detection_score,
            relationship_score,
            diversity_score,
            consistency_score,
            improvement_score=improvement_score,
            uncertainty_reduction_score=uncertainty_reduction_score,
            raw_reward=total_reward_raw,
        )
        
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
    
    def train_relationship_model(self, synthetic_data, num_epochs: int = 1):
        """Fine-tune the RelTR relationship model on available relationship annotations.
        
        Args:
            synthetic_data: Synthetic data generated (for compatibility, not used directly)
            num_epochs: Number of training epochs to run on the full dataset (default: 1)
        
        Returns:
            tuple: (relationship_loss, long_tail_loss) where:
                - relationship_loss: Average loss across all samples
                - long_tail_loss: Weighted loss for rare relationships (long-tail)
        """
        print(f"[RL] Starting relationship model training (epochs: {num_epochs})...")
        model, criterion = self._ensure_relationship_model()
        prepared_samples = self._prepare_reltr_training_samples()
        if not prepared_samples:
            print("[RL] RelTR training samples unavailable, skipping relationship training.")
            return 0.0, 0.0

        print(f"[RL] Training RelTR model with {len(prepared_samples)} samples over {num_epochs} epoch(s)")
        if self.reltr_optimizer is None:
            self.reltr_optimizer = AdamW(
                (param for param in model.parameters() if param.requires_grad),
                lr=1e-5,
                weight_decay=1e-4,
            )
        optimizer = self.reltr_optimizer

        # Track losses across all epochs
        all_epoch_losses = []
        all_epoch_tail_losses = []
        all_tail_weighted_counts = []

        # Train for multiple epochs
        for epoch in range(num_epochs):
            print(f"[RL] Epoch {epoch + 1}/{num_epochs}")
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            total_tail_loss = 0.0
            tail_weighted_count = 0

            # Shuffle samples for each epoch (except first epoch to maintain reproducibility)
            import random
            if epoch > 0:
                shuffled_samples = list(prepared_samples)
                random.shuffle(shuffled_samples)
            else:
                shuffled_samples = prepared_samples

            for i, (image_tensor, target, global_context) in enumerate(shuffled_samples):
                if num_epochs > 1 and len(prepared_samples) > 10:
                    # Only print every 10th sample for large datasets
                    if i % 10 == 0 or i == len(shuffled_samples) - 1:
                        print(f"[RL] Epoch {epoch + 1}: Training on sample {i+1}/{len(shuffled_samples)}")
                else:
                    print(f"[RL] Training on sample {i+1}/{len(shuffled_samples)}")
                
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
                    sample_loss = float(loss.item())
                    total_loss += sample_loss
                    
                    # Calculate long-tail loss: weight by tail_weights if this sample contains rare relationships
                    tail_weight = self._get_sample_tail_weight(target)
                    if tail_weight > 0:
                        total_tail_loss += sample_loss * tail_weight
                        tail_weighted_count += 1
                    
                    if num_epochs == 1 or i % 10 == 0 or i == len(shuffled_samples) - 1:
                        print(f"[RL] Sample {i+1} loss: {sample_loss:.4f}")
                except Exception as exc:
                    print(f"[RL] Error training on sample {i+1}: {exc}")
                    continue

            # Update optimizer after processing all samples in this epoch
            optimizer.step()

            epoch_avg_loss = total_loss / max(len(shuffled_samples), 1)
            epoch_tail_loss = total_tail_loss / max(tail_weighted_count, 1) if tail_weighted_count > 0 else 0.0
            
            all_epoch_losses.append(epoch_avg_loss)
            all_epoch_tail_losses.append(epoch_tail_loss)
            all_tail_weighted_counts.append(tail_weighted_count)
            
            print(f"[RL] Epoch {epoch + 1} completed - Loss: {epoch_avg_loss:.4f}, Long-tail loss: {epoch_tail_loss:.4f}")

        # Return average loss across all epochs (or final epoch loss)
        average_loss = sum(all_epoch_losses) / len(all_epoch_losses) if all_epoch_losses else 0.0
        long_tail_loss = sum(all_epoch_tail_losses) / len(all_epoch_tail_losses) if all_epoch_tail_losses else 0.0
        total_tail_weighted_count = sum(all_tail_weighted_counts)
        
        print(f"[RL] Relationship loss after {num_epochs} epoch(s): {average_loss:.4f} (avg across epochs)")
        print(f"[RL] Long-tail loss: {long_tail_loss:.4f} (weighted for {total_tail_weighted_count} rare relationship samples across all epochs)")
        model.eval()
        return average_loss, long_tail_loss
    
    def _get_sample_tail_weight(self, target: Dict[str, Any]) -> float:
        """Calculate tail weight for a training sample based on its relationships.
        
        Args:
            target: RelTR target dictionary containing relationship annotations
            
        Returns:
            float: Average tail weight of relationships in this sample, or 0 if no tail relationships
        """
        if not self.tail_weights:
            return 0.0
        
        # Calculate average tail weight based on relationships in current dataset
        # Since we can't directly map target to relationship names, we use the average
        # tail weight of all relationships in the dataset as a proxy
        if not self.dataset_samples:
            return 0.0
        
        # Collect all relationship names from dataset and calculate average tail weight
        tail_weights_in_dataset = []
        for sample in self.dataset_samples:
            for rel in sample.get('relationships', []):
                rel_name = self._normalize_label(rel.get('relation', ''))
                if rel_name in self.tail_weights:
                    tail_weights_in_dataset.append(self.tail_weights[rel_name])
        
        if not tail_weights_in_dataset:
            return 0.0
        
        # Return average tail weight - this gives us a measure of how "rare" 
        # the relationships in the current training batch are
        return sum(tail_weights_in_dataset) / len(tail_weights_in_dataset)
    
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
    
    def _calculate_b_pr(self, precision: float, recall: float) -> float:
        """
        B_PR = 2√(P·R)/(P+R) = GM/AM (README §12.2). Luôn ∈ (0, 1], bằng 1 khi P=R.
        Phạt nặng hơn khi một trong hai rất nhỏ (phản ánh chất lượng thực tế).
        """
        p, r = float(precision), float(recall)
        if p <= 0 and r <= 0:
            return 0.0
        denom = p + r
        if denom <= 0:
            return 0.0
        return (2.0 * math.sqrt(p * r)) / denom

    def _calculate_detection_score(self, detection_metrics: Dict[str, float]) -> float:
        """
        S_det = F1_det * C_n * B_PR
        C_n = tanh(α * ln(n+1)); B_PR = 2√(P·R)/(P+R) (geometric ratio, README §12.2).
        """
        f1 = detection_metrics.get('f1', 0.0)
        precision = detection_metrics.get('precision', 0.0)
        recall = detection_metrics.get('recall', 0.0)
        n = detection_metrics.get('num_samples', 0)

        alpha = getattr(self, 'reward_alpha', 0.5)
        c_n = math.tanh(alpha * math.log(n + 1)) if n >= 0 else 0.0
        b_pr = self._calculate_b_pr(precision, recall)
        b_pr = max(0.0, min(b_pr, 1.0))

        s_det = f1 * c_n * b_pr
        return max(0.0, min(s_det, 1.0))
    
    def _calculate_relationship_score(self, relationship_metrics: Dict[str, Any]) -> float:
        """
        S_rel = F1_rel · C_n · B_PR + β · W_tail · F1_rel, rồi clip về [0, 1].
        Additive bonus cho quan hệ hiếm (W_tail) tránh S_rel > 1 khi dùng (1+W_tail) nhân trực tiếp.
        β ∈ (0, 1), ví dụ 0.5 (README §12.3).
        """
        f1 = relationship_metrics.get('f1', 0.0)
        precision = relationship_metrics.get('precision', 0.0)
        recall = relationship_metrics.get('recall', 0.0)
        n = relationship_metrics.get('num_samples', 0)

        # C_n: hệ số tin cậy mẫu 
        alpha = getattr(self, 'reward_alpha', 0.5)
        c_n = math.tanh(alpha * math.log(n + 1)) if n >= 0 else 0.0

        # B_PR: geometric ratio 2√(P·R)/(P+R) (README §12.2)
        b_pr = self._calculate_b_pr(precision, recall)
        b_pr = max(0.0, min(b_pr, 1.0))

        # W_tail: long-tail weight (đã chuẩn hóa từ 1/sqrt(freq+ε))
        w_tail = 0.0
        rel_name = self._normalize_label(
            relationship_metrics.get('relation_name', '') or relationship_metrics.get('relation', '')
        )
        if rel_name and self.tail_weights:
            w_tail = self.tail_weights.get(rel_name, 0.0)
        if not rel_name and self.tail_weights:
            w_tail = sum(self.tail_weights.values()) / len(self.tail_weights) if self.tail_weights else 0.0

        # Additive bonus bounded: S_rel = F1·Cn·B_PR + β·W_tail·F1, clip [0,1]
        beta = getattr(self, 'rel_tail_bonus_beta', 0.5)
        base = f1 * c_n * b_pr
        bonus = beta * w_tail * f1
        s_rel = base + bonus
        return max(0.0, min(s_rel, 1.0))
    
    def _calculate_diversity_score(self, synthetic_data: List[Dict[str, Any]]) -> float:
        """
        S_div = 0.4*D_type + 0.4*D_class + 0.2*S_spatial
        D_type, D_class: tỷ lệ số loại quan hệ/lớp vật thể xuất hiện trên tổng số loại khả dụng.
        S_spatial = 0.4*S_pos + 0.3*S_size + 0.3*S_coverage
        S_pos = tanh(α*Var_pos), S_size = σ_size/μ_size (CV), S_coverage = entropy vị trí.
        """
        if not synthetic_data:
            return 0.0

        # D_type: tỷ lệ loại quan hệ xuất hiện / tổng loại khả dụng (dùng 10 làm mẫu nếu không có vocab)
        unique_relations = set()
        for data in synthetic_data:
            if 'original_relationship' in data:
                rel = data['original_relationship']
                unique_relations.add(rel.get('relation', ''))
        num_relation_types = 10  # có thể lấy từ vocab nếu có
        d_type = min(len(unique_relations) / max(num_relation_types, 1), 1.0)

        # D_class: tỷ lệ lớp vật thể xuất hiện / tổng lớp khả dụng
        unique_subjects = set()
        unique_objects = set()
        for data in synthetic_data:
            if 'original_relationship' in data:
                rel = data['original_relationship']
                unique_subjects.add(rel.get('subject', ''))
                unique_objects.add(rel.get('object', ''))
        num_class_types = 15
        d_class = min(len(unique_subjects | unique_objects) / max(num_class_types, 1), 1.0)

        s_spatial = self._calculate_spatial_diversity(synthetic_data)

        s_div = 0.4 * d_type + 0.4 * d_class + 0.2 * s_spatial
        return max(0.0, min(s_div, 1.0))
    
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
        S_pos theo báo cáo (52): S_pos = tanh(α * Var_pos).
        Khuyến khích vị trí vật thể thay đổi.
        """
        if not bboxes or not image_sizes:
            return 0.0

        avg_width = sum(size[0] for size in image_sizes) / len(image_sizes)
        avg_height = sum(size[1] for size in image_sizes) / len(image_sizes)

        normalized_centers = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0 / avg_width
            center_y = (y1 + y2) / 2.0 / avg_height
            normalized_centers.append([center_x, center_y])

        if len(normalized_centers) < 2:
            return 0.0

        x_coords = [c[0] for c in normalized_centers]
        y_coords = [c[1] for c in normalized_centers]
        x_mean = sum(x_coords) / len(x_coords)
        y_mean = sum(y_coords) / len(y_coords)
        var_pos = (
            sum((x - x_mean) ** 2 for x in x_coords) / len(x_coords) +
            sum((y - y_mean) ** 2 for y in y_coords) / len(y_coords)
        )
        alpha = getattr(self, 'reward_alpha', 0.5)
        s_pos = math.tanh(alpha * var_pos)
        return s_pos
    
    def _calculate_size_diversity(self, bboxes: List[List[float]]) -> float:
        """
        S_size theo báo cáo (53): S_size = σ_size/μ_size (Coefficient of Variation).
        """
        if not bboxes:
            return 0.0

        areas = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            areas.append((x2 - x1) * (y2 - y1))

        if not areas:
            return 0.0
        mu_size = sum(areas) / len(areas)
        var_size = sum((a - mu_size) ** 2 for a in areas) / len(areas)
        sigma_size = math.sqrt(var_size)
        s_size = (sigma_size / mu_size) if mu_size > 0 else 0.0
        return max(0.0, min(s_size, 2.0))  # clip CV hợp lý
    
    def _calculate_coverage_diversity(self, bboxes: List[List[float]], image_sizes: List[Tuple[int, int]]) -> float:
        """
        S_coverage theo báo cáo (54): Entropy vị trí -Σ p_i log(p_i), khuyến khích vật thể rải đều trên lưới ảnh.
        """
        if not bboxes or not image_sizes:
            return 0.0

        avg_width = sum(size[0] for size in image_sizes) / len(image_sizes)
        avg_height = sum(size[1] for size in image_sizes) / len(image_sizes)
        grid_size = 4
        cell_width = avg_width / grid_size
        cell_height = avg_height / grid_size

        cell_counts: Dict[Tuple[int, int], int] = {}
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

        total = sum(cell_counts.values()) or 1
        entropy = 0.0
        for count in cell_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log(p + 1e-10)
        max_entropy = math.log(grid_size * grid_size + 1e-10)
        normalized = entropy / max_entropy if max_entropy > 0 else 0
        return max(0.0, min(normalized, 1.0))
    
    def _calculate_consistency_score(self, f1_scores: List[float], precomputed_std: Optional[float] = None) -> float:
        """
        S_cons = 0.7*S_std + 0.3*S_trend^cons (README §12.5).
        S_trend^cons = LinearSlope(F1_{t-20:t}) — chuỗi dài hạn để tách với S_imp (giảm tương quan).
        """
        if not f1_scores and (precomputed_std is None or precomputed_std == 0.0):
            return 0.0

        if precomputed_std is not None:
            sigma_f1 = float(precomputed_std)
        else:
            if not f1_scores:
                return 0.0
            mean_score = sum(f1_scores) / len(f1_scores)
            variance = sum((s - mean_score) ** 2 for s in f1_scores) / len(f1_scores)
            sigma_f1 = math.sqrt(variance)

        s_std = 1.0 / (1.0 + sigma_f1)

        # S_trend^cons: dùng lịch sử dài hạn (20 epoch gần nhất) để tách tín hiệu với S_imp
        window = getattr(self, 'cons_trend_window', 20)
        long_series = list(self.performance_history['relationship_scores'])[-window:]
        s_trend = self._calculate_trend_score(long_series)

        s_cons = 0.7 * s_std + 0.3 * s_trend
        return max(0.0, min(s_cons, 1.0))
    
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
        S_imp = 0.6*(0.5+0.5*tanh(F1_curr-F1_base)) + 0.4*S_trend^imp (README §12.6).
        S_trend^imp = LinearSlope(F1_{t-5:t}) — chuỗi ngắn hạn (5 epoch) để tách với S_cons.
        """
        if len(self.performance_history['rewards']) < 3:
            return 0.5

        window = getattr(self, 'imp_trend_window', 5)
        rel_scores = list(self.performance_history['relationship_scores'])[-window:]
        if not rel_scores:
            return 0.5
        f1_current = sum(rel_scores) / len(rel_scores)
        f1_base = self.baseline_performance.get('relationship', self.baseline_performance.get('overall', 0.5))

        # S_trend^imp: chỉ chuỗi 5 epoch gần nhất (tách nguồn với S_cons)
        s_trend = self._calculate_trend_score(rel_scores)

        s_imp = 0.6 * (0.5 + 0.5 * math.tanh(f1_current - f1_base)) + 0.4 * s_trend
        return max(0.0, min(s_imp, 1.0))
    
    def _calculate_dynamic_weights(self, detection_score: float, relationship_score: float,
                                  diversity_score: float, consistency_score: float,
                                  improvement_score: float = 0.5,
                                  uncertainty_reduction_score: float = 0.5) -> Dict[str, float]:
        """
        Trọng số thích nghi dạng softmax (README §12.8): đảm bảo W_k > 0 và tổng = 1.
        W_k = (W_k^0 * exp(α(b_k - S_k))) / Σ_j (W_j^0 * exp(α(b_j - S_j)))
        Thành phần yếu (S_k < b_k) nhận trọng số cao hơn; prior W_k^0 được giữ.
        """
        base_weights = {
            'detection': 0.25,
            'relationship': 0.45,
            'diversity': 0.15,
            'consistency': 0.10,
            'improvement': 0.05,
            'uncertainty_reduction': 0.10,
        }

        baseline = self.baseline_performance
        scores = {
            'detection': detection_score,
            'relationship': relationship_score,
            'diversity': diversity_score,
            'consistency': consistency_score,
            'improvement': improvement_score,
            'uncertainty_reduction': uncertainty_reduction_score,
        }
        baselines = {
            'detection': baseline.get('detection', 0.3),
            'relationship': baseline.get('relationship', 0.7),
            'diversity': baseline.get('diversity', 0.3),
            'consistency': baseline.get('consistency', 0.5),
            'improvement': baseline.get('improvement', 0.5),
            'uncertainty_reduction': baseline.get('uncertainty_reduction', 0.5),
        }

        alpha = 0.2
        # logits_k = α(b_k - S_k); trừ max để ổn định số học
        logits = {k: alpha * (baselines[k] - scores[k]) for k in base_weights}
        logits_max = max(logits.values())
        unnorm = {k: base_weights[k] * math.exp(logits[k] - logits_max) for k in base_weights}
        total = sum(unnorm.values())
        normalized_weights = {k: unnorm[k] / total for k in base_weights}
        return normalized_weights
    
    def _apply_reward_scaling(self, raw_reward: float) -> float:
        """
        R = σ(k·(R_raw - 0.5)) với k tường minh (README §12.1).
        k phụ thuộc độ ổn định σ_recent của R_raw gần đây.
        """
        scaled_reward = 1.0 / (1.0 + math.exp(-self.scaling_factor * (raw_reward - 0.5)))
        return scaled_reward
    
    def _get_current_scaling_factor(self) -> float:
        """
        k = k_min + (k_max - k_min) · 1/(1 + σ_recent) (README §12.1).
        σ_recent = độ lệch chuẩn của R_raw trên M bước gần nhất.
        Reward ổn định (σ→0) → k→k_max (sigmoid dốc); bất ổn → k nhỏ (tín hiệu mềm).
        """
        k_min = getattr(self, 'reward_sigmoid_k_min', 3.0)
        k_max = getattr(self, 'reward_sigmoid_k_max', 10.0)
        M = getattr(self, 'reward_sigmoid_M', 10)
        raw_rewards = list(self.performance_history.get('raw_rewards', []))[-M:]
        if len(raw_rewards) < 2:
            return (k_min + k_max) / 2.0
        sigma_recent = float(np.std(raw_rewards))
        k = k_min + (k_max - k_min) * (1.0 / (1.0 + sigma_recent))
        return max(k_min, min(k_max, k))
    
    def _update_performance_history(
        self,
        reward: float,
        weights: Dict[str, float],
        detection_score: float,
        relationship_score: float,
        diversity_score: float,
        consistency_score: float,
        improvement_score: Optional[float] = None,
        uncertainty_reduction_score: Optional[float] = None,
        raw_reward: Optional[float] = None,
    ) -> None:
        """Cập nhật lịch sử performance; raw_reward (R_raw) dùng cho tính k lần sau."""
        self.performance_history['rewards'].append(reward)
        if raw_reward is not None:
            self.performance_history.setdefault('raw_rewards', deque(maxlen=50)).append(raw_reward)
        self.performance_history['weight_history'].append(weights)

        # Lưu history các thành phần điểm
        self.performance_history['detection_scores'].append(detection_score)
        self.performance_history['relationship_scores'].append(relationship_score)
        self.performance_history['diversity_scores'].append(diversity_score)
        self.performance_history['consistency_scores'].append(consistency_score)
        if uncertainty_reduction_score is not None:
            self.performance_history['uncertainty_reduction_scores'].append(uncertainty_reduction_score)

        # Cập nhật baseline tổng thể dựa trên trung bình gần đây
        if len(self.performance_history['rewards']) >= 10:
            recent_rewards = list(self.performance_history['rewards'])[-10:]
            recent_overall_avg = sum(recent_rewards) / len(recent_rewards)
            prev_overall = self.baseline_performance.get('overall', 0.5)
            # EWMA để mượt hơn, ưu tiên kinh nghiệm gần đây
            alpha_overall = 0.3
            self.baseline_performance['overall'] = (
                alpha_overall * recent_overall_avg + (1 - alpha_overall) * prev_overall
            )

        # Cập nhật baseline cho detection/relationship/diversity/consistency dựa trên lịch sử gần đây
        window = 10
        alpha = 0.3  # hệ số EWMA để phản ánh xu hướng gần đây nhưng vẫn ổn định
        # Helper lấy trung bình gần đây an toàn
        def recent_avg(values) -> Optional[float]:
            data = list(values)[-window:]
            if not data:
                return None
            return float(sum(data) / len(data))

        # Detection
        det_avg = recent_avg(self.performance_history['detection_scores'])
        if det_avg is not None:
            prev = self.baseline_performance.get('detection', det_avg)
            self.baseline_performance['detection'] = alpha * det_avg + (1 - alpha) * prev

        # Relationship
        rel_avg = recent_avg(self.performance_history['relationship_scores'])
        if rel_avg is not None:
            prev = self.baseline_performance.get('relationship', rel_avg)
            self.baseline_performance['relationship'] = alpha * rel_avg + (1 - alpha) * prev

        # Diversity
        div_avg = recent_avg(self.performance_history['diversity_scores'])
        if div_avg is not None:
            prev = self.baseline_performance.get('diversity', div_avg)
            self.baseline_performance['diversity'] = alpha * div_avg + (1 - alpha) * prev

        # Consistency
        cons_avg = recent_avg(self.performance_history['consistency_scores'])
        if cons_avg is not None:
            prev = self.baseline_performance.get('consistency', cons_avg)
            self.baseline_performance['consistency'] = alpha * cons_avg + (1 - alpha) * prev

        # Improvement (từ relationship_scores/reward trend, không có deque riêng — dùng improvement_trend nếu có)
        if improvement_score is not None:
            prev_imp = self.baseline_performance.get('improvement', 0.5)
            self.baseline_performance['improvement'] = alpha * improvement_score + (1 - alpha) * prev_imp

        # Uncertainty reduction
        if uncertainty_reduction_score is not None:
            unc_avg = recent_avg(self.performance_history['uncertainty_reduction_scores'])
            if unc_avg is not None:
                prev_unc = self.baseline_performance.get('uncertainty_reduction', unc_avg)
                self.baseline_performance['uncertainty_reduction'] = alpha * unc_avg + (1 - alpha) * prev_unc

        # Cập nhật scaling factor theo phân phối phần thưởng gần đây
        self.scaling_factor = self._get_current_scaling_factor()
    
    # Các phương thức cũ đã được thay thế bằng hệ thống mới
    # calculate_diversity_reward và calculate_consistency_reward đã được tích hợp vào hệ thống mới
    
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
        # Tạm thời set experiment directory để load model
        original_experiment_dir = self.model_manager.current_experiment_dir
        self.model_manager.set_experiment_dir(experiment_dir)
        
        # Load best model từ experiment trước
        best_model = self.get_best_model()
        if best_model:
            print(f"Continuing training from best model (reward: {best_model['model_state'].get('reward', 0):.4f})")
            
            # Kiểm tra và sửa dataset_samples nếu cần
            self._validate_and_fix_dataset_samples()
            
            # Khôi phục experiment directory gốc
            if original_experiment_dir:
                self.model_manager.set_experiment_dir(original_experiment_dir)
            
            return True
        else:
            print("No previous model found, starting fresh training")
            # Khôi phục experiment directory gốc
            if original_experiment_dir:
                self.model_manager.set_experiment_dir(original_experiment_dir)
            return False
    
    def _validate_and_fix_dataset_samples(self):
        """Kiểm tra và sửa dataset_samples để đảm bảo có đủ thông tin relationship"""
        if not self.dataset_samples:
            print("[RL] No dataset samples to validate")
            return
        
        print(f"[RL] Validating {len(self.dataset_samples)} dataset samples...")
        
        fixed_count = 0
        for i, sample in enumerate(self.dataset_samples):
            # Kiểm tra xem có relationships không
            if 'relationships' not in sample or not sample['relationships']:
                # Nếu không có relationships, thử tạo từ original_relationship
                if 'original_relationship' in sample and sample['original_relationship']:
                    sample['relationships'] = [sample['original_relationship']]
                    fixed_count += 1
                    print(f"[RL] Fixed sample {i+1}: added relationship from original_relationship")
                else:
                    print(f"[RL] Warning: Sample {i+1} has no relationships or original_relationship")
        
        if fixed_count > 0:
            print(f"[RL] Fixed {fixed_count} samples with missing relationships")
            # Lưu lại dataset snapshot
            self._save_dataset_snapshot()

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
        long_tail_loss: float = 0.0,
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
                    'long_tail_loss': long_tail_loss,
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
                    'long_tail_loss': long_tail_loss,
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
    
    def test_relationship_evaluation(self, evaluation_data: List[Dict[str, Any]], original_relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test relationship evaluation để debug vấn đề Relationship F1 = 0.
        """
        print("\n" + "="*80)
        print("🔍 TEST RELATIONSHIP EVALUATION")
        print("="*80)
        
        if not evaluation_data:
            print("❌ No evaluation data provided")
            return {}
        
        print(f"📊 Evaluation data count: {len(evaluation_data)}")
        
        # Phân tích loại dữ liệu
        dataset_samples_count = sum(1 for data in evaluation_data if 'relationships' in data)
        synthetic_samples_count = sum(1 for data in evaluation_data if 'original_relationship' in data)
        print(f"📊 Dataset samples: {dataset_samples_count}, Synthetic samples: {synthetic_samples_count}")
        
        # Kiểm tra từng sample
        valid_samples = 0
        for i, data in enumerate(evaluation_data[:5]):  # Chỉ kiểm tra 5 samples đầu
            print(f"\n🔍 Sample {i+1}:")
            print(f"  - Has relationships: {'relationships' in data}")
            print(f"  - Has original_relationship: {'original_relationship' in data}")
            print(f"  - Has image_path: {'image_path' in data}")
            print(f"  - Has image: {'image' in data}")
            
            if 'relationships' in data and data['relationships']:
                print(f"  - Relationships: {data['relationships']}")
                valid_samples += 1
            elif 'original_relationship' in data and data['original_relationship']:
                print(f"  - Original relationship: {data['original_relationship']}")
                valid_samples += 1
            else:
                print(f"  - ❌ No valid relationship data")
        
        print(f"\n✅ Valid samples: {valid_samples}/{min(5, len(evaluation_data))}")
        
        # Test relationship prediction
        if valid_samples > 0:
            print(f"\n🧠 Testing relationship prediction...")
            try:
                self._ensure_relationship_model()
                test_sample = evaluation_data[0]
                image_input = test_sample.get('image') or test_sample.get('image_path')
                
                if image_input:
                    predicted_relationships = self.predict_relationships(image_input) or []
                    print(f"  - Predicted relationships: {len(predicted_relationships)}")
                    if predicted_relationships:
                        print(f"  - First prediction: {predicted_relationships[0]}")
                    else:
                        print(f"  - ❌ No relationships predicted")
                else:
                    print(f"  - ❌ No image input available")
            except Exception as exc:
                print(f"  - ❌ Prediction failed: {exc}")
        
        print("\n" + "="*80)
        return {
            'total_samples': len(evaluation_data),
            'dataset_samples': dataset_samples_count,
            'synthetic_samples': synthetic_samples_count,
            'valid_samples': valid_samples
        }
    
    def test_new_scoring_system(self, synthetic_data: List[Dict[str, Any]], original_relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test hệ thống tính điểm mới để đảm bảo nó hoạt động đúng.
        """
        print("\n" + "="*80)
        print("🧪 TEST HỆ THỐNG TÍNH ĐIỂM MỚI")
        print("="*80)
        
        # Test từng thành phần
        print("\n1️⃣ TEST DETECTION SCORING:")
        detection_metrics = self._evaluate_detection_metrics(self.dataset_samples)
        detection_score = self._calculate_detection_score(detection_metrics)
        print(f"   Detection metrics: {detection_metrics}")
        print(f"   Detection score: {detection_score:.4f}")
        
        print("\n2️⃣ TEST RELATIONSHIP SCORING:")
        relationship_metrics = self._evaluate_relationship_metrics(synthetic_data, original_relationships)
        per_sample_f1 = relationship_metrics.pop('per_sample_f1', [])
        relationship_score = self._calculate_relationship_score(relationship_metrics)
        print(f"   Relationship metrics: {relationship_metrics}")
        print(f"   Relationship score: {relationship_score:.4f}")
        
        print("\n3️⃣ TEST DIVERSITY SCORING:")
        diversity_score = self._calculate_diversity_score(synthetic_data)
        print(f"   Diversity score: {diversity_score:.4f}")
        
        print("\n4️⃣ TEST CONSISTENCY SCORING:")
        consistency_score = self._calculate_consistency_score(per_sample_f1, relationship_metrics.get('f1_std'))
        print(f"   Consistency score: {consistency_score:.4f}")
        
        print("\n5️⃣ TEST IMPROVEMENT SCORING:")
        improvement_score = self._calculate_improvement_score()
        print(f"   Improvement score: {improvement_score:.4f}")

        uncertainty_reduction_score = 0.5  # Test không gọi uncertainty estimator
        print(f"   Uncertainty reduction score (test): {uncertainty_reduction_score:.4f}")
        
        print("\n6️⃣ TEST DYNAMIC WEIGHTS (gồm S_unc):")
        dynamic_weights = self._calculate_dynamic_weights(
            detection_score, relationship_score, diversity_score, consistency_score,
            improvement_score=improvement_score,
            uncertainty_reduction_score=uncertainty_reduction_score,
        )
        print(f"   Dynamic weights: {dynamic_weights}")
        
        print("\n7️⃣ TEST TOTAL REWARD:")
        total_reward = (
            dynamic_weights['detection'] * detection_score +
            dynamic_weights['relationship'] * relationship_score +
            dynamic_weights['diversity'] * diversity_score +
            dynamic_weights['consistency'] * consistency_score +
            dynamic_weights['improvement'] * improvement_score +
            dynamic_weights['uncertainty_reduction'] * uncertainty_reduction_score
        )
        scaled_reward = self._apply_reward_scaling(total_reward)
        print(f"   Raw reward: {total_reward:.4f}")
        print(f"   Scaled reward: {scaled_reward:.4f}")
        
        # Tạo test result
        test_result = {
            'detection_score': detection_score,
            'relationship_score': relationship_score,
            'diversity_score': diversity_score,
            'consistency_score': consistency_score,
            'improvement_score': improvement_score,
            'dynamic_weights': dynamic_weights,
            'total_reward': scaled_reward,
            'raw_reward': total_reward,
            'scaling_factor': self.scaling_factor,
            'baseline_performance': self.baseline_performance,
        }
        
        print("\n✅ TEST HOÀN THÀNH!")
        print("="*80)
        
        return test_result
    
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
    
    def validate_data_flow(self) -> Dict[str, Any]:
        """
        Validate entire RL data flow and identify potential issues.
        Call this to debug training problems.
        
        Returns a diagnostic report with identified issues and recommendations.
        """
        print("\n" + "="*80)
        print("🔍 RL DATA FLOW VALIDATION")
        print("="*80)
        
        issues = []
        warnings = []
        info = []
        
        # 1. Check dataset samples
        print("\n📊 1. DATASET SAMPLES CHECK:")
        if not self.dataset_samples:
            issues.append("No dataset samples available - model cannot be trained")
            print("  ❌ No dataset samples")
        else:
            print(f"  ✅ {len(self.dataset_samples)} samples available")
            
            # Check sample quality
            samples_with_objects = sum(1 for s in self.dataset_samples if s.get('objects'))
            samples_with_relationships = sum(1 for s in self.dataset_samples if s.get('relationships'))
            samples_with_images = sum(1 for s in self.dataset_samples if s.get('image_path') and os.path.exists(s.get('image_path', '')))
            
            print(f"  📦 Samples with objects: {samples_with_objects}/{len(self.dataset_samples)}")
            print(f"  🔗 Samples with relationships: {samples_with_relationships}/{len(self.dataset_samples)}")
            print(f"  🖼️  Samples with valid image paths: {samples_with_images}/{len(self.dataset_samples)}")
            
            if samples_with_relationships < len(self.dataset_samples) * 0.5:
                warnings.append(f"Only {samples_with_relationships}/{len(self.dataset_samples)} samples have relationships - relationship training may be ineffective")
            
            if samples_with_objects < len(self.dataset_samples) * 0.5:
                issues.append(f"Only {samples_with_objects}/{len(self.dataset_samples)} samples have objects - detection training will fail")
        
        # 2. Check models
        print("\n🧠 2. MODEL STATUS:")
        if self.detection_model is not None:
            print("  ✅ Detection model loaded")
        else:
            info.append("Detection model not loaded yet (will be loaded on first use)")
            print("  ⚠️ Detection model not loaded (lazy loading)")
        
        if self.relationship_model is not None:
            print("  ✅ Relationship model loaded")
        else:
            info.append("Relationship model not loaded yet (will be loaded on first use)")
            print("  ⚠️ Relationship model not loaded (lazy loading)")
        
        # 3. Check Q-network
        print("\n🎮 3. Q-NETWORK STATUS:")
        print(f"  📊 Memory buffer size: {len(self.memory)}/{self.memory.maxlen}")
        print(f"  🎯 Epsilon (exploration): {self.epsilon:.4f}")
        print(f"  📈 Learn step counter: {self.learn_step_counter}")
        print(f"  🎲 Action space: {self.action_space}")
        
        if len(self.memory) < self.batch_size:
            warnings.append(f"Memory buffer ({len(self.memory)}) < batch_size ({self.batch_size}) - Q-network cannot be trained yet")
        
        # 4. Check performance history
        print("\n📈 4. PERFORMANCE HISTORY:")
        print(f"  🎁 Rewards recorded: {len(self.performance_history['rewards'])}")
        print(f"  🔍 Detection scores: {len(self.performance_history['detection_scores'])}")
        print(f"  🔗 Relationship scores: {len(self.performance_history['relationship_scores'])}")
        
        if len(self.performance_history['rewards']) > 0:
            recent_rewards = list(self.performance_history['rewards'])[-5:]
            print(f"  📊 Recent rewards: {[f'{r:.3f}' for r in recent_rewards]}")
        
        # 5. Check relationship performance tracking
        print("\n📋 5. RELATIONSHIP TRACKING:")
        print(f"  🗂️  Tracked relationships: {len(self.relationship_performance)}")
        if self.relationship_performance:
            low_performers = [k for k, v in self.relationship_performance.items() if v.get('avg_f1', 0) < 0.3]
            if low_performers:
                print(f"  ⚠️ Low-performing relationships (F1<0.3): {len(low_performers)}")
                for rel_key in low_performers[:3]:
                    perf = self.relationship_performance[rel_key]
                    print(f"     - {rel_key}: avg F1 = {perf.get('avg_f1', 0):.3f}")
        
        # 6. Check training history
        print("\n📚 6. TRAINING HISTORY:")
        print(f"  📖 Epochs completed: {len(self.training_history['epochs'])}")
        print(f"  🏆 Best reward: {self.training_history['best_reward']:.4f}")
        print(f"  ⭐ Best epoch: {self.training_history['best_epoch']}")
        
        # Summary
        print("\n" + "="*80)
        print("📋 VALIDATION SUMMARY")
        print("="*80)
        
        if issues:
            print("\n❌ ISSUES (must fix):")
            for issue in issues:
                print(f"  • {issue}")
        
        if warnings:
            print("\n⚠️ WARNINGS (should address):")
            for warning in warnings:
                print(f"  • {warning}")
        
        if info:
            print("\n💡 INFO:")
            for i in info:
                print(f"  • {i}")
        
        if not issues and not warnings:
            print("\n✅ All checks passed! Data flow appears healthy.")
        
        print("\n" + "="*80)
        
        return {
            'issues': issues,
            'warnings': warnings,
            'info': info,
            'dataset_size': len(self.dataset_samples),
            'memory_size': len(self.memory),
            'epochs_completed': len(self.training_history['epochs']),
            'best_reward': self.training_history['best_reward'],
            'is_healthy': len(issues) == 0,
        }
