# gradcam_utils.py
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Optional
from PIL import Image
from util.misc import nested_tensor_from_tensor_list
from models import build_model
from detect_objects import yolo_model

REL_CLASSES = [
    "__background__", "above", "across", "against", "along", "and", "at", "attached to",
    "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating",
    "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in",
    "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near",
    "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing",
    "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in",
    "walking on", "watching", "wearing", "wears", "with"
]

_reltr_transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(dim=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _scale_boxes(boxes: torch.Tensor, img_size: tuple[int, int]) -> torch.Tensor:
    h, w = img_size
    scaled = boxes.clone()
    scaled[..., 0::2] *= w
    scaled[..., 1::2] *= h
    return scaled


def _box_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(x2 - x1, 0.0)
    inter_h = max(y2 - y1, 0.0)
    inter_area = inter_w * inter_h
    area1 = max(box1[2] - box1[0], 0.0) * max(box1[3] - box1[1], 0.0)
    area2 = max(box2[2] - box2[0], 0.0) * max(box2[3] - box2[1], 0.0)
    denom = area1 + area2 - inter_area + 1e-6
    return float(inter_area / denom) if denom > 0 else 0.0


def _match_relation_query(
    subject_box: np.ndarray,
    object_box: np.ndarray,
    rel_sub_boxes: np.ndarray,
    rel_obj_boxes: np.ndarray,
) -> int:
    best_idx = -1
    best_score = 0.0
    for idx, (r_sub, r_obj) in enumerate(zip(rel_sub_boxes, rel_obj_boxes)):
        iou_sub = _box_iou_xyxy(subject_box, r_sub)
        iou_obj = _box_iou_xyxy(object_box, r_obj)
        score = iou_sub * iou_obj
        if score > best_score:
            best_idx = idx
            best_score = score
    if best_idx < 0 or best_score < 1e-4:
        raise ValueError(
            "Không thể khớp cặp (subject, object) với bất kỳ quan hệ nào do IoU quá thấp."
        )
    return best_idx


def _sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label.strip().lower())
    return cleaned or "relation"


def _clamp_box(box: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    x1 = int(np.clip(box[0], 0, img_width - 1))
    y1 = int(np.clip(box[1], 0, img_height - 1))
    x2 = int(np.clip(box[2], x1 + 1, img_width))
    y2 = int(np.clip(box[3], y1 + 1, img_height))
    return np.array([x1, y1, x2, y2], dtype=int)


def _compute_cam_map(activations: torch.Tensor, gradients: torch.Tensor, image_size: tuple[int, int]) -> np.ndarray:
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image_size, mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def _blend_cam_with_image(cam: np.ndarray, base_image: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return np.uint8(np.clip(0.4 * heatmap + 0.6 * base_image, 0, 255))


def _assign_detection_tag(box: np.ndarray, det_boxes: np.ndarray, det_labels: List[str], min_iou: float = 0.25) -> Optional[str]:
    if det_boxes.size == 0 or not det_labels:
        return None
    best_idx = -1
    best_score = 0.0
    for idx, det_box in enumerate(det_boxes):
        score = _box_iou_xyxy(box, det_box)
        if score > best_score:
            best_idx = idx
            best_score = score
    if best_idx < 0 or best_score < min_iou:
        return None
    return det_labels[best_idx]


def _relation_filename(base_dir: Path, base_name: str, ext: str, rank: int, label: str) -> Path:
    safe_label = _sanitize_label(label)
    filename = f"{base_name}_{rank:02d}_{safe_label}{ext}"
    return base_dir / filename


def _collect_relation_candidates(
    rel_logits: torch.Tensor,
    score_threshold: float,
    max_relations: int,
) -> List[Dict[str, float]]:
    probs = rel_logits.softmax(-1)
    scores, labels_idx = probs.max(dim=-1)
    candidates: List[Dict[str, float]] = []
    for query_idx in range(rel_logits.shape[0]):
        label_idx = int(labels_idx[query_idx].item())
        score = float(scores[query_idx].item())
        if label_idx <= 0:  # skip background
            continue
        if score < score_threshold:
            continue
        candidates.append(
            {
                "query_index": query_idx,
                "label_idx": label_idx,
                "score": score,
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    if max_relations > 0:
        candidates = candidates[:max_relations]
    return candidates


def _draw_relation_overlay(
    base_image: np.ndarray,
    cam: np.ndarray,
    subj_box: np.ndarray,
    obj_box: np.ndarray,
    relation_label: str,
    score: float,
    subj_tag: Optional[str],
    obj_tag: Optional[str],
) -> np.ndarray:
    overlay = _blend_cam_with_image(cam, base_image)
    subj_text = subj_tag or "subject"
    obj_text = obj_tag or "object"
    cv2.rectangle(overlay, (subj_box[0], subj_box[1]), (subj_box[2], subj_box[3]), (0, 255, 0), 2)
    cv2.rectangle(overlay, (obj_box[0], obj_box[1]), (obj_box[2], obj_box[3]), (255, 0, 0), 2)
    cv2.putText(
        overlay,
        f"{subj_text}",
        (subj_box[0], max(subj_box[1] - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"{obj_text}",
        (obj_box[0], max(obj_box[1] - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"{relation_label} ({score:.2f})",
        (10, overlay.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay

def _load_reltr_model(checkpoint_path: str, device: torch.device):
    args = SimpleNamespace(
        dataset='vg', lr_backbone=1e-5, backbone='resnet50', dilation=False,
        position_embedding='sine', enc_layers=6, dec_layers=6, dim_feedforward=2048,
        hidden_dim=256, dropout=0.1, nheads=8, num_entities=100, num_triplets=200,
        pre_norm=False, aux_loss=False, device=str(device), set_cost_class=1,
        set_cost_bbox=5, set_cost_giou=2, set_iou_threshold=0.7, bbox_loss_coef=5,
        giou_loss_coef=2, rel_loss_coef=1, eos_coef=0.1, return_interm_layers=False,
    )
    model, _, _ = build_model(args)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model") if isinstance(ckpt, dict) else None
    if state_dict:
        model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

def generate_reltr_gradcam(
    image_path: str,
    reltr_checkpoint: str,
    subject_index: Optional[int] = None,
    object_index: Optional[int] = None,
    relation_label: Optional[str] = None,
    save_path: str = "gradcam_reltr.jpg",
    score_threshold: float = 0.35,
    max_relations: int = 20,
) -> List[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_image = cv2.imread(image_path)
    if base_image is None:
        raise FileNotFoundError(f"Không thể mở ảnh {image_path}")
    img_height, img_width = base_image.shape[:2]

    save_path_obj = Path(save_path)
    if save_path_obj.suffix:
        save_dir = save_path_obj.parent if save_path_obj.parent != Path("") else Path(".")
        base_name = save_path_obj.stem
        file_ext = save_path_obj.suffix
    else:
        save_dir = save_path_obj if save_path_obj != Path("") else Path(".")
        base_name = save_path_obj.name or "gradcam_reltr"
        file_ext = ".jpg"
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        results = yolo_model(image_path, verbose=False)
    if not results:
        raise RuntimeError("Không nhận được kết quả từ YOLO.")
    result = results[0]
    if result.boxes is None or result.boxes.xyxy is None:
        boxes_xyxy = torch.empty((0, 4), device=device)
        yolo_labels: List[str] = []
    else:
        boxes_xyxy = result.boxes.xyxy.to(device)
        cls_list = result.boxes.cls.tolist() if result.boxes.cls is not None else []
        yolo_labels = [result.names[int(cls_idx)] for cls_idx in cls_list]

    reltr_model = _load_reltr_model(reltr_checkpoint, device)
    pil_img = Image.open(image_path).convert("RGB")
    img_tensor = _reltr_transform(pil_img).to(device)
    samples = nested_tensor_from_tensor_list([img_tensor]).to(device)

    cam_state = {"activations": None, "grads": None}

    def _forward_cam_hook(module, _inp, output):
        cam_state["activations"] = output

    def _backward_cam_hook(module, _grad_in, grad_out):
        cam_state["grads"] = grad_out[0]

    hook_forward = reltr_model.input_proj.register_forward_hook(_forward_cam_hook)
    hook_backward = reltr_model.input_proj.register_full_backward_hook(_backward_cam_hook)

    saved_paths: List[str] = []
    try:
        outputs = reltr_model(samples)
        rel_logits_raw = outputs["rel_logits"][0]
        if rel_logits_raw.shape[-1] == len(REL_CLASSES) + 1:
            rel_logits_raw = rel_logits_raw[:, :-1]
        elif rel_logits_raw.shape[-1] != len(REL_CLASSES):
            raise RuntimeError(
                f"Số lượng lớp quan hệ ({rel_logits_raw.shape[-1]}) không khớp REL_CLASSES ({len(REL_CLASSES)})."
            )

        sub_boxes_rel = outputs["sub_boxes"][0]
        obj_boxes_rel = outputs["obj_boxes"][0]
        rel_boxes_sub_xyxy = _scale_boxes(_cxcywh_to_xyxy(sub_boxes_rel), (img_height, img_width)).detach().cpu().numpy()
        rel_boxes_obj_xyxy = _scale_boxes(_cxcywh_to_xyxy(obj_boxes_rel), (img_height, img_width)).detach().cpu().numpy()
        det_boxes_np = boxes_xyxy.detach().cpu().numpy() if boxes_xyxy.numel() else np.empty((0, 4))

        activations = cam_state["activations"]
        if activations is None:
            raise RuntimeError("Không thu được feature map cần thiết cho Grad-CAM.")
        activations = activations.detach()

        targeted_mode = (
            relation_label is not None and subject_index is not None and object_index is not None
        )

        if targeted_mode:
            subject_index = int(subject_index)
            object_index = int(object_index)
            if boxes_xyxy.numel() == 0:
                raise RuntimeError("YOLO không tìm thấy bbox nào để tham chiếu subject/object.")
            if not (0 <= subject_index < boxes_xyxy.shape[0] and 0 <= object_index < boxes_xyxy.shape[0]):
                raise ValueError("subject_index hoặc object_index nằm ngoài phạm vi bbox YOLO.")

            subject_box = boxes_xyxy[subject_index].detach().cpu().numpy()
            object_box = boxes_xyxy[object_index].detach().cpu().numpy()
            matched_query = _match_relation_query(subject_box, object_box, rel_boxes_sub_xyxy, rel_boxes_obj_xyxy)

            try:
                rel_idx = REL_CLASSES.index(relation_label)
            except ValueError:
                raise ValueError(f"Relation '{relation_label}' không nằm trong REL_CLASSES.")

            reltr_model.zero_grad(set_to_none=True)
            target_logit = rel_logits_raw[matched_query, rel_idx]
            target_logit.backward()
            gradients = cam_state["grads"]
            if gradients is None:
                raise RuntimeError("Không lấy được gradient cho Grad-CAM.")
            cam = _compute_cam_map(activations, gradients.detach(), (img_height, img_width))
            subj_box = _clamp_box(subject_box, img_width, img_height)
            obj_box = _clamp_box(object_box, img_width, img_height)
            subj_tag = yolo_labels[subject_index] if subject_index < len(yolo_labels) else "subject"
            obj_tag = yolo_labels[object_index] if object_index < len(yolo_labels) else "object"
            overlay = _draw_relation_overlay(base_image, cam, subj_box, obj_box, relation_label, 1.0, subj_tag, obj_tag)
            if save_path_obj.suffix:
                out_path = save_path_obj
            else:
                out_path = save_dir / f"{base_name}{file_ext}"
            cv2.imwrite(str(out_path), overlay)
            saved_paths.append(str(out_path))
        else:
            candidates = _collect_relation_candidates(rel_logits_raw, score_threshold, max_relations)
            if not candidates:
                raise RuntimeError("Không có quan hệ nào vượt ngưỡng score_threshold đã chọn.")

            total = len(candidates)
            for idx, candidate in enumerate(candidates):
                rel_idx = candidate["label_idx"]
                relation_name = REL_CLASSES[rel_idx] if rel_idx < len(REL_CLASSES) else f"class_{rel_idx}"
                reltr_model.zero_grad(set_to_none=True)
                target_logit = rel_logits_raw[candidate["query_index"], rel_idx]
                retain_graph = idx < total - 1
                target_logit.backward(retain_graph=retain_graph)
                gradients = cam_state["grads"]
                if gradients is None:
                    raise RuntimeError("Không lấy được gradient cho Grad-CAM.")
                cam = _compute_cam_map(activations, gradients.detach(), (img_height, img_width))
                cam_state["grads"] = None
                subj_box = _clamp_box(rel_boxes_sub_xyxy[candidate["query_index"]], img_width, img_height)
                obj_box = _clamp_box(rel_boxes_obj_xyxy[candidate["query_index"]], img_width, img_height)
                subj_tag = _assign_detection_tag(subj_box, det_boxes_np, yolo_labels)
                obj_tag = _assign_detection_tag(obj_box, det_boxes_np, yolo_labels)
                overlay = _draw_relation_overlay(
                    base_image,
                    cam,
                    subj_box,
                    obj_box,
                    relation_name,
                    candidate["score"],
                    subj_tag,
                    obj_tag,
                )
                out_path = _relation_filename(save_dir, base_name, file_ext, idx, relation_name)
                cv2.imwrite(str(out_path), overlay)
                saved_paths.append(str(out_path))
    finally:
        hook_forward.remove()
        hook_backward.remove()

    return saved_paths

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subject", type=int, default=None, help="Index bbox YOLO cho subject (chế độ mục tiêu).")
    parser.add_argument("--object", type=int, default=None, help="Index bbox YOLO cho object (chế độ mục tiêu).")
    parser.add_argument("--relation", default=None, help="Tên quan hệ cần Grad-CAM (chế độ mục tiêu).")
    parser.add_argument("--all", action="store_true", help="Bỏ qua subject/object và xuất toàn bộ quan hệ tìm được.")
    parser.add_argument("--score-threshold", type=float, default=0.35, help="Ngưỡng probability để giữ quan hệ.")
    parser.add_argument("--max-relations", type=int, default=20, help="Số lượng quan hệ tối đa sẽ xuất.")
    parser.add_argument("--out", default="gradcam_reltr.jpg")
    args = parser.parse_args()

    subject_arg = None if args.all else args.subject
    object_arg = None if args.all else args.object
    relation_arg = None if args.all else args.relation

    paths = generate_reltr_gradcam(
        image_path=args.image,
        reltr_checkpoint=args.checkpoint,
        subject_index=subject_arg,
        object_index=object_arg,
        relation_label=relation_arg,
        save_path=args.out,
        score_threshold=args.score_threshold,
        max_relations=args.max_relations,
    )
    print("Saved Grad-CAM overlays:")
    for p in paths:
        print("  ", p)
