import json
import time
from collections import Counter
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import detect_objects as detection_pipeline
from RL.reinforcement_learning import RELATION_CLASSES
from models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None


FrameCallback = Optional[Callable[[np.ndarray], None]]
RelationCallback = Optional[Callable[[Dict[str, List[Dict[str, object]]]], None]]


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
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
        if state:
            model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model
        return self.model

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
                        relationships.append({
                            'subject': objects[int(subj_idx)].get('class', 'unknown'),
                            'relation': relation_name,
                            'object': objects[int(obj_idx)].get('class', 'unknown'),
                            'confidence': min(confidence, 1.0),
                            'subject_track_id': objects[int(subj_idx)].get('track_id'),
                            'object_track_id': objects[int(obj_idx)].get('track_id'),
                            'source': 'model',
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
                relationships.append({
                    'subject': objects[i].get('class', 'unknown'),
                    'relation': relation_name,
                    'object': objects[j].get('class', 'unknown'),
                    'confidence': confidence,
                    'subject_track_id': objects[i].get('track_id'),
                    'object_track_id': objects[j].get('track_id'),
                    'source': 'model_fallback',
                })
                pair_cursor += 1
        return relationships


class VideoRelationPipeline:
    """Process an entire video: detect -> track -> infer relationships -> voice announce."""

    def __init__(
        self,
        reltr_checkpoint: str = "reltr_finetuned.pth",
        tracker_config: Optional[str] = "bytetrack.yaml",
        min_confidence: float = 0.55,
        announce_min_confidence: float = 0.6,
        voice_enabled: bool = True,
    ):
        self.rel_engine = RelTRInferenceEngine(reltr_checkpoint)
        self.yolo_model = detection_pipeline.yolo_model
        self.tracker_config = tracker_config
        self.min_confidence = min_confidence
        self.announce_threshold = announce_min_confidence
        self.voice_enabled = voice_enabled and pyttsx3 is not None
        self.spoken_relations: Dict[str, float] = {}
        self.voice_engine = None
        if self.voice_enabled:
            try:
                self.voice_engine = pyttsx3.init()
                self.voice_engine.setProperty("rate", 185)
            except Exception:
                self.voice_engine = None
                self.voice_enabled = False

    def _announce(self, relations: List[Dict[str, object]]):
        if not self.voice_enabled or not self.voice_engine or not relations:
            return
        now = time.time()
        cooldown = 3.0
        utterances: List[str] = []
        for rel in relations:
            if rel.get("confidence", 0.0) < self.announce_threshold:
                continue
            key = f"{rel.get('subject_track_id')}-{rel.get('relation')}-{rel.get('object_track_id')}"
            last = self.spoken_relations.get(key, 0.0)
            if now - last < cooldown:
                continue
            utterance = f"{rel.get('subject', 'vat the')} {rel.get('relation', 'lien quan')} {rel.get('object', 'vat the')}"
            self.spoken_relations[key] = now
            utterances.append(utterance)
        if utterances:
            for sentence in utterances:
                self.voice_engine.say(sentence)
            self.voice_engine.runAndWait()

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
        summary_path = output_root / f"{Path(video_path).stem}_relations.json"
        stats_path = output_root / f"{Path(video_path).stem}_summary.json"

        fps = self._read_video_fps(video_path)
        writer = None
        summaries: List[Dict[str, object]] = []
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
                objects = self._prepare_objects(result, frame, feature_map)
                for obj in objects:
                    label = obj.get("class")
                    if label:
                        object_counter[label] += 1
                relations = [
                    rel for rel in self.rel_engine.infer(frame, objects, global_context)
                    if rel.get("confidence", 0.0) >= self.min_confidence
                ]
                for rel in relations:
                    key = f"{rel.get('subject','unknown')}|{rel.get('relation','')}|{rel.get('object','unknown')}"
                    relation_counter[key] += 1
                annotated = self._draw_annotations(frame.copy(), objects, relations)
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(annotated_path),
                        cv2.VideoWriter_fourcc(*"XVID"),
                        fps,
                        (width, height),
                    )
                writer.write(annotated)
                summaries.append({
                    "frame": frame_idx,
                    "timestamp": time.time(),
                    "objects": objects,
                    "relations": relations,
                })
                if on_frame:
                    on_frame(annotated.copy())
                if on_relations:
                    on_relations({"frame": frame_idx, "relations": relations, "objects": objects})
                self._announce(relations)
        finally:
            if writer is not None:
                writer.release()
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summaries, f, ensure_ascii=False, indent=2)
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
            }
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats_payload, f, ensure_ascii=False, indent=2)
        return {"video": str(annotated_path), "json": str(summary_path), "summary": str(stats_path)}

    def _safe_consume_feature_map(self):
        try:
            return detection_pipeline._consume_feature_map()
        except Exception:
            return None

    def _prepare_objects(self, result, frame, feature_map):
        objects: List[Dict[str, object]] = []
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy().tolist() if result.boxes.conf is not None else [0.0] * len(boxes)
        track_ids = []
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(boxes)
        detected_objects = []
        yolo_labels = []
        metadata = []
        for idx, (bbox, cls_id) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = map(int, bbox)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
            cropped = detection_pipeline.add_padding(frame, (x1, y1, x2, y2))
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            detected_objects.append((cropped_pil, (x1, y1, x2, y2)))
            names_map = self.yolo_model.model.names if hasattr(self.yolo_model, "model") and hasattr(self.yolo_model.model, "names") else self.yolo_model.names
            class_name = names_map[int(cls_id)]
            yolo_labels.append(class_name)
            metadata.append({
                "track_id": track_ids[idx] if idx < len(track_ids) else None,
                "confidence": float(confs[idx]) if idx < len(confs) else 0.0,
            })
        if not detected_objects:
            return []
        classified = detection_pipeline.classify_with_clip(detected_objects, yolo_labels)
        box_list = [bbox for _, bbox in classified]
        roi_features = detection_pipeline.extract_roi_features(feature_map, box_list, frame.shape) if feature_map is not None else []
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

    def _draw_annotations(self, frame, objects, relations):
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            track_id = obj.get("track_id")
            label = obj.get("class", "obj")
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            caption = f"{label}"
            if track_id is not None:
                caption = f"ID {track_id}: {label}"
            cv2.putText(frame, caption, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        for idx, rel in enumerate(relations[:10]):
            text = f"{rel.get('subject', '?')} {rel.get('relation', '?')} {rel.get('object', '?')} ({rel.get('confidence', 0.0):.2f})"
            cv2.putText(frame, text, (15, 25 + 20 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame

    @staticmethod
    def _read_video_fps(video_path: str) -> float:
        capture = cv2.VideoCapture(video_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
        capture.release()
        return fps
