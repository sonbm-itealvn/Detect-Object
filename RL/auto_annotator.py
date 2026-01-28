# File: auto_annotator.py
"""
Bước 6.5: Auto-Annotation cho ảnh synthetic từ Stable Diffusion.

Vấn đề: SD chỉ trả về pixels, không có bounding box.
Giải pháp: Sử dụng Open-Vocabulary Detector (GroundingDINO hoặc OWL-ViT) 
để tự động phát hiện và gán bbox cho các đối tượng trong ảnh.

Flow:
1. Nhận ảnh synthetic + prompt/relationship (subject, relation, object)
2. Trích xuất text prompts: ["dog", "surfboard"]
3. Chạy GroundingDINO/OWL-ViT với text prompts
4. Trả về danh sách objects với bbox
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Try to import GroundingDINO
GROUNDINGDINO_AVAILABLE = False
try:
    from groundingdino.util.inference import load_model, load_image, predict
    GROUNDINGDINO_AVAILABLE = True
    print("[AutoAnnotator] GroundingDINO available - using for open-vocabulary detection")
except ImportError:
    pass

# Try to import OWL-ViT as fallback
OWLVIT_AVAILABLE = False
try:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    OWLVIT_AVAILABLE = True
    print("[AutoAnnotator] OWL-ViT available as fallback")
except ImportError:
    pass

# Fallback to YOLO+CLIP
YOLO_CLIP_AVAILABLE = False
try:
    import detect_objects as detection_pipeline
    YOLO_CLIP_AVAILABLE = True
    print("[AutoAnnotator] YOLO+CLIP available as fallback")
except ImportError:
    pass


class AutoAnnotator:
    """
    Tự động gán nhãn (bbox + class) cho ảnh synthetic.
    
    Ưu tiên sử dụng:
    1. GroundingDINO (SOTA, open-vocabulary)
    2. OWL-ViT (lighter, từ Hugging Face)
    3. YOLO+CLIP (fallback, vocabulary hạn chế)
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        groundingdino_config: Optional[str] = None,
        groundingdino_checkpoint: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Model instances
        self.groundingdino_model = None
        self.owlvit_model = None
        self.owlvit_processor = None
        
        # Determine which backend to use
        self.backend = self._initialize_backend(groundingdino_config, groundingdino_checkpoint)
        print(f"[AutoAnnotator] Initialized with backend: {self.backend}")
    
    def _initialize_backend(
        self,
        groundingdino_config: Optional[str],
        groundingdino_checkpoint: Optional[str],
    ) -> str:
        """Khởi tạo backend detector theo thứ tự ưu tiên."""
        
        # 1. Try GroundingDINO
        if GROUNDINGDINO_AVAILABLE:
            try:
                # Auto-detect paths if not provided
                if groundingdino_config is None:
                    # Common paths
                    candidates = [
                        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",  # Thư mục GroundingDINO trong project
                        "groundingdino/config/GroundingDINO_SwinT_OGC.py",  # Relative path
                        os.path.expanduser("~/.cache/groundingdino/GroundingDINO_SwinT_OGC.py"),  # Cache directory
                    ]
                    for c in candidates:
                        if os.path.exists(c):
                            groundingdino_config = c
                            print(f"[AutoAnnotator] Found GroundingDINO config at: {c}")
                            break
                
                if groundingdino_checkpoint is None:
                    candidates = [
                        "GroundingDINO/weights/groundingdino_swint_ogc.pth",  # Thư mục GroundingDINO trong project
                        "groundingdino_swint_ogc.pth",  # Root directory
                        "weights/groundingdino_swint_ogc.pth",  # Relative weights folder
                        os.path.expanduser("~/.cache/groundingdino/groundingdino_swint_ogc.pth"),  # Cache directory
                    ]
                    for c in candidates:
                        if os.path.exists(c):
                            groundingdino_checkpoint = c
                            print(f"[AutoAnnotator] Found GroundingDINO checkpoint at: {c}")
                            break
                
                if groundingdino_config and groundingdino_checkpoint:
                    print(f"[AutoAnnotator] Loading GroundingDINO model...")
                    print(f"  Config: {groundingdino_config}")
                    print(f"  Checkpoint: {groundingdino_checkpoint}")
                    self.groundingdino_model = load_model(
                        groundingdino_config,
                        groundingdino_checkpoint,
                        device=self.device,
                    )
                    print(f"[AutoAnnotator] ✅ GroundingDINO loaded successfully!")
                    return "groundingdino"
                else:
                    if not groundingdino_config:
                        print(f"[AutoAnnotator] ⚠️ GroundingDINO config not found in candidates")
                    if not groundingdino_checkpoint:
                        print(f"[AutoAnnotator] ⚠️ GroundingDINO checkpoint not found in candidates")
            except Exception as e:
                print(f"[AutoAnnotator] ❌ Failed to load GroundingDINO: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. Try OWL-ViT
        if OWLVIT_AVAILABLE:
            try:
                self.owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
                self.owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
                self.owlvit_model.to(self.device)
                self.owlvit_model.eval()
                return "owlvit"
            except Exception as e:
                print(f"[AutoAnnotator] Failed to load OWL-ViT: {e}")
        
        # 3. Fallback to YOLO+CLIP
        if YOLO_CLIP_AVAILABLE:
            return "yolo_clip"
        
        return "none"
    
    def annotate(
        self,
        image_path: str,
        text_prompts: List[str],
        original_relationship: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Annotate ảnh với bounding boxes dựa trên text prompts.
        
        Args:
            image_path: Đường dẫn đến ảnh
            text_prompts: Danh sách text để detect, ví dụ ["dog", "surfboard"]
            original_relationship: Quan hệ gốc {subject, relation, object} để tham khảo
        
        Returns:
            Dict với keys:
                - image_path: str
                - width, height: int
                - objects: List[Dict] với mỗi object có {class, bbox, confidence, source}
                - annotation_backend: str (groundingdino/owlvit/yolo_clip)
        """
        if not os.path.exists(image_path):
            print(f"[AutoAnnotator] Image not found: {image_path}")
            return None
        
        # Load image để lấy kích thước
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Annotate dựa trên backend
        if self.backend == "groundingdino":
            objects = self._annotate_groundingdino(image_path, text_prompts, width, height)
        elif self.backend == "owlvit":
            objects = self._annotate_owlvit(image_path, text_prompts, width, height)
        elif self.backend == "yolo_clip":
            objects = self._annotate_yolo_clip(image_path, text_prompts)
        else:
            print("[AutoAnnotator] No annotation backend available!")
            objects = []
        
        # Nếu không detect được, thử tạo pseudo-bbox từ relationship
        if not objects and original_relationship:
            objects = self._create_pseudo_annotations(
                original_relationship, width, height
            )
        
        return {
            'image_path': image_path,
            'width': width,
            'height': height,
            'objects': objects,
            'annotation_backend': self.backend,
            'num_detected': len(objects),
        }
    
    def _annotate_groundingdino(
        self,
        image_path: str,
        text_prompts: List[str],
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        """Annotate sử dụng GroundingDINO."""
        try:
            # Load image
            image_source, image_tensor = load_image(image_path)
            
            # Create text prompt (GroundingDINO expects "dog . surfboard" format)
            text_prompt = " . ".join(text_prompts)
            
            # Predict
            boxes, logits, phrases = predict(
                model=self.groundingdino_model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
            
            objects = []
            for box, score, phrase in zip(boxes, logits, phrases):
                # GroundingDINO returns normalized coords [cx, cy, w, h]
                cx, cy, bw, bh = box.tolist()
                x1 = int((cx - bw / 2) * width)
                y1 = int((cy - bh / 2) * height)
                x2 = int((cx + bw / 2) * width)
                y2 = int((cy + bh / 2) * height)
                
                # Clamp to image bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                
                objects.append({
                    'class': phrase.strip().lower(),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score),
                    'source': 'groundingdino',
                })
            
            print(f"[AutoAnnotator] GroundingDINO detected {len(objects)} objects")
            return objects
            
        except Exception as e:
            print(f"[AutoAnnotator] GroundingDINO error: {e}")
            return []
    
    def _annotate_owlvit(
        self,
        image_path: str,
        text_prompts: List[str],
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        """Annotate sử dụng OWL-ViT."""
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs
            texts = [[f"a photo of a {t}" for t in text_prompts]]
            inputs = self.owlvit_processor(text=texts, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.owlvit_model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([[height, width]], device=self.device)
            results = self.owlvit_processor.post_process_object_detection(
                outputs, threshold=self.box_threshold, target_sizes=target_sizes
            )[0]
            
            objects = []
            for box, score, label in zip(
                results["boxes"], results["scores"], results["labels"]
            ):
                x1, y1, x2, y2 = box.tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Clamp
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                
                objects.append({
                    'class': text_prompts[label.item()].strip().lower(),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(score),
                    'source': 'owlvit',
                })
            
            print(f"[AutoAnnotator] OWL-ViT detected {len(objects)} objects")
            return objects
            
        except Exception as e:
            print(f"[AutoAnnotator] OWL-ViT error: {e}")
            return []
    
    def _annotate_yolo_clip(
        self,
        image_path: str,
        text_prompts: List[str],
    ) -> List[Dict[str, Any]]:
        """Annotate sử dụng YOLO+CLIP (fallback)."""
        try:
            # Sử dụng pipeline hiện có
            detected_objects, yolo_labels, original_image, feature_map, global_context = \
                detection_pipeline.detect_objects(image_path)
            
            if not detected_objects:
                return []
            
            classified_results = detection_pipeline.classify_with_clip(
                detected_objects, yolo_labels
            )
            
            objects = []
            for idx, (label, bbox) in enumerate(classified_results):
                if not bbox or len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Check if label matches any of the text prompts (fuzzy)
                label_lower = label.strip().lower()
                matched = any(
                    p.lower() in label_lower or label_lower in p.lower()
                    for p in text_prompts
                )
                
                objects.append({
                    'class': label_lower,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.7 if matched else 0.5,
                    'source': 'yolo_clip',
                    'matched_prompt': matched,
                })
            
            print(f"[AutoAnnotator] YOLO+CLIP detected {len(objects)} objects")
            return objects
            
        except Exception as e:
            print(f"[AutoAnnotator] YOLO+CLIP error: {e}")
            return []
    
    def _create_pseudo_annotations(
        self,
        relationship: Dict[str, str],
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        """
        Tạo pseudo-bbox khi không detect được gì.
        Sử dụng heuristics dựa trên loại quan hệ.
        
        WARNING: Đây là phương án cuối cùng, chất lượng thấp!
        """
        subject = relationship.get('subject', 'unknown')
        relation = relationship.get('relation', 'unknown')
        obj = relationship.get('object', 'unknown')
        
        # Heuristics: đặt subject bên trái, object bên phải
        # Hoặc subject trên, object dưới tùy relation
        
        margin = 0.1
        if relation in ['on', 'above', 'over', 'riding']:
            # Subject trên object
            subject_bbox = [
                int(width * 0.3), int(height * 0.1),
                int(width * 0.7), int(height * 0.45)
            ]
            object_bbox = [
                int(width * 0.2), int(height * 0.5),
                int(width * 0.8), int(height * 0.9)
            ]
        elif relation in ['under', 'below']:
            # Subject dưới object
            subject_bbox = [
                int(width * 0.2), int(height * 0.5),
                int(width * 0.8), int(height * 0.9)
            ]
            object_bbox = [
                int(width * 0.3), int(height * 0.1),
                int(width * 0.7), int(height * 0.45)
            ]
        elif relation in ['holding', 'carrying', 'using']:
            # Subject lớn, object nhỏ gần subject
            subject_bbox = [
                int(width * 0.2), int(height * 0.1),
                int(width * 0.7), int(height * 0.9)
            ]
            object_bbox = [
                int(width * 0.5), int(height * 0.3),
                int(width * 0.8), int(height * 0.6)
            ]
        else:
            # Default: subject trái, object phải
            subject_bbox = [
                int(width * 0.05), int(height * 0.2),
                int(width * 0.45), int(height * 0.8)
            ]
            object_bbox = [
                int(width * 0.55), int(height * 0.2),
                int(width * 0.95), int(height * 0.8)
            ]
        
        print(f"[AutoAnnotator] Created PSEUDO annotations (low quality!) for: {subject} {relation} {obj}")
        
        return [
            {
                'class': subject.strip().lower(),
                'bbox': subject_bbox,
                'confidence': 0.3,  # Low confidence
                'source': 'pseudo',
                'is_pseudo': True,
            },
            {
                'class': obj.strip().lower(),
                'bbox': object_bbox,
                'confidence': 0.3,
                'source': 'pseudo',
                'is_pseudo': True,
            },
        ]
    
    def annotate_from_relationship(
        self,
        image_path: str,
        relationship: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Annotate ảnh dựa trên relationship triplet.
        Tự động trích xuất text prompts từ subject/object.
        """
        subject = relationship.get('subject', '').strip()
        obj = relationship.get('object', '').strip()
        
        # Build text prompts
        text_prompts = []
        if subject:
            text_prompts.append(subject)
        if obj and obj.lower() != subject.lower():
            text_prompts.append(obj)
        
        if not text_prompts:
            print(f"[AutoAnnotator] No valid prompts from relationship: {relationship}")
            return None
        
        return self.annotate(image_path, text_prompts, relationship)
    
    def get_status(self) -> Dict[str, Any]:
        """Trả về trạng thái của annotator."""
        return {
            'backend': self.backend,
            'groundingdino_available': GROUNDINGDINO_AVAILABLE,
            'owlvit_available': OWLVIT_AVAILABLE,
            'yolo_clip_available': YOLO_CLIP_AVAILABLE,
            'box_threshold': self.box_threshold,
            'text_threshold': self.text_threshold,
            'device': self.device,
        }


# Singleton instance for easy access
_annotator_instance: Optional[AutoAnnotator] = None


def get_annotator(force_reinit: bool = False, **kwargs) -> AutoAnnotator:
    """Get or create AutoAnnotator singleton."""
    global _annotator_instance
    if _annotator_instance is None or force_reinit:
        _annotator_instance = AutoAnnotator(**kwargs)
    return _annotator_instance

