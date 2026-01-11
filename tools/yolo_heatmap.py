"""
Generate a YOLO Grad-CAM style heatmap for a single image.

Usage:
  python tools/yolo_heatmap.py --image demo/vg1.jpg --weights yolov5s.pt --output outputs/yolo_heatmap.png
  # chỉ định class mục tiêu (theo id, dựa trên model.names)
  python tools/yolo_heatmap.py --image demo/vg1.jpg --weights yolov5s.pt --target-class 0
"""
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Grad-CAM heatmap for one image")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--weights", type=Path, required=True, help="Path to YOLO weights (e.g., yolov5s.pt)")
    parser.add_argument("--output", type=Path, default=Path("outputs/yolo_heatmap.png"), help="Output path")
    parser.add_argument("--target-class", type=int, default=None, help="Target class id; if None use all classes present")
    parser.add_argument("--fast", action="store_true", help="Hook only last Conv2d (faster, usually enough)")
    return parser.parse_args()


def run_cam_for_classes(args, device, target_classes=None):
    model = YOLO(str(args.weights))
    model.to(device)
    model.eval()

    names = model.names

    # Hooks: capture last 4D fmap and its grad via tensor hook (avoid module backward hook issues)
    feature_maps = {}
    grads = {}
    handles = []

    def f_hook(module, inp, out):
        outs = out if isinstance(out, (list, tuple)) else [out]
        for o in outs:
            if torch.is_tensor(o) and o.dim() == 4:
                feature_maps["feat"] = o
                if "b_handle" not in grads:
                    grads["b_handle"] = o.register_hook(lambda g: grads.setdefault("feat", g))

    if args.fast:
        last_conv = None
        for m in model.model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No Conv2d found for fast hook.")
        handles.append(last_conv.register_forward_hook(f_hook))
    else:
        for m in model.model.modules():
            handles.append(m.register_forward_hook(f_hook))

    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Manual letterbox preprocess (keeps gradients)
    h0, w0 = img_rgb.shape[:2]
    imgsz = 640
    r = min(imgsz / h0, imgsz / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = imgsz - new_unpad[0], imgsz - new_unpad[1]
    dw /= 2
    dh /= 2
    img_resized = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    x = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
    x.requires_grad_(True)

    # Forward with gradients through the core model (bypass postprocess detaching)
    with torch.enable_grad():
        raw_preds = model.model(x)  # may be list/tuple (pred, train_out) or tensor
        pred_raw = raw_preds[0] if isinstance(raw_preds, (list, tuple)) else raw_preds
        if pred_raw.dim() == 2:
            pred_raw = pred_raw.unsqueeze(0)
        # If the head returns multiple levels, stack anchors along dim=1
        if isinstance(pred_raw, (list, tuple)):
            pred_raw = torch.cat(pred_raw, dim=1)
        # If shape is [B, C, H, W], flatten anchors
        if pred_raw.dim() == 4:
            pred_raw = pred_raw.permute(0, 2, 3, 1).reshape(pred_raw.shape[0], -1, pred_raw.shape[1])

        obj = pred_raw[..., 4].sigmoid()                # (1, A)
        cls_probs = pred_raw[..., 5:].sigmoid()         # (1, A, num_cls)
        scores = obj.unsqueeze(-1) * cls_probs          # (1, A, num_cls)
        scores = scores.view(-1, scores.shape[-1])      # (A, num_cls)

        # Determine target classes
        if target_classes is None:
            present = (scores > 0.0).any(dim=0).nonzero(as_tuple=True)[0].tolist()
            target_classes = present

        cams = []
        for c in target_classes:
            if c >= scores.shape[1]:
                continue
            best_score, _ = scores[:, c].max(0)
            if best_score <= 0:
                continue
            model.model.zero_grad()
            best_score.backward(retain_graph=True)

            if "feat" not in feature_maps or "feat" not in grads:
                continue
            fmap = feature_maps["feat"][0]  # (C, H, W)
            grad = grads["feat"][0]         # (C, H, W)
            if fmap.shape[0] != grad.shape[0]:
                common_c = min(fmap.shape[0], grad.shape[0])
                fmap = fmap[:common_c]
                grad = grad[:common_c]

            weights_cam = grad.mean(dim=(1, 2))   # (C,)
            cam = (weights_cam[:, None, None] * fmap).sum(dim=0)
            cam = torch.relu(cam)

            cam_np = cam.detach().cpu().numpy()
            cam_np = cam_np - cam_np.min()
            cam_np = cam_np / (cam_np.max() + 1e-8)
            lo, hi = np.percentile(cam_np, 60), np.percentile(cam_np, 99)
            cam_np = np.clip((cam_np - lo) / (hi - lo + 1e-8), 0, 1)
            cams.append((c, cam_np))

    if "feat" not in feature_maps or "feat" not in grads:
        for h in handles:
            h.remove()
        raise RuntimeError("Failed to capture spatial feature map/grad; try a different image or weights.")

    for h in handles:
        h.remove()
    if "b_handle" in grads:
        try:
            grads["b_handle"].remove()
        except Exception:
            pass

    if not cams:
        raise RuntimeError("No heatmaps generated. Try another image/weights or specify classes.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for c, cam_np in cams:
        cam_resized = cv2.resize(cam_np, (img.shape[1], img.shape[0]))
        heatmap = (cam_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
        out_path = args.output.with_name(f"{args.output.stem}_class{c}{args.output.suffix}")
        cv2.imwrite(str(out_path), overlay)
        print(f"Saved heatmap for class={names.get(c, c)} to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    targets = [args.target_class] if args.target_class is not None else None
    run_cam_for_classes(args, device, targets)

