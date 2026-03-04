"""
Visualize YOLO feature-map heatmaps using multiple methods.

Supported methods:
  - eigen   : EigenCAM  – first principal component of feature maps (no gradient, fast)
  - gradcam : Grad-CAM  – gradient-weighted class activation map
  - activations : raw channel-mean activation from a chosen layer

Usage examples:
  # EigenCAM on default backbone layer
  python tools/visualize_heatmap.py --image demo/vg1.jpg --weights fine-tune.pt

  # Grad-CAM targeting class 0
  python tools/visualize_heatmap.py --image demo/vg1.jpg --weights fine-tune.pt --method gradcam --target-class 0

  # Visualise activations of multiple layers side-by-side
  python tools/visualize_heatmap.py --image demo/vg1.jpg --weights fine-tune.pt --method activations --layers 2 4 6 9

  # Process every image in a folder
  python tools/visualize_heatmap.py --image demo/ --weights fine-tune.pt --output outputs/heatmaps/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def letterbox(img: np.ndarray, new_shape: int = 640):
    """Resize + pad to square, return (padded_img, ratio, (dw, dh))."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, (dw, dh)


def preprocess(img_bgr: np.ndarray, imgsz: int = 640, device: torch.device = None):
    """BGR image -> (1,3,H,W) float32 tensor normalised to [0,1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    padded, ratio, (dw, dh) = letterbox(img_rgb, imgsz)
    x = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    if device is not None:
        x = x.to(device)
    return x, ratio, (dw, dh)


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Min-max normalise and apply contrast stretch."""
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    lo, hi = np.percentile(cam, 5), np.percentile(cam, 95)
    cam = np.clip((cam - lo) / (hi - lo + 1e-8), 0, 1)
    return cam


def overlay_heatmap(img_bgr: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET):
    """Overlay a [0,1] heatmap onto a BGR image."""
    cam_resized = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return blended, heatmap_color


# ---------------------------------------------------------------------------
# Hook helpers – capture intermediate feature maps & gradients
# ---------------------------------------------------------------------------

class FeatureHook:
    """Register forward hooks on selected layers and store their outputs."""

    def __init__(self):
        self.features: dict[str, torch.Tensor] = {}
        self._handles: list = []

    def register(self, model: nn.Module, layer_indices: list[int]):
        """Hook into model.model[i] for each i in *layer_indices*."""
        backbone = model.model.model  # ultralytics: YOLO -> model -> Sequential
        for idx in layer_indices:
            if idx >= len(backbone):
                print(f"[warn] layer index {idx} out of range (max {len(backbone)-1}), skipped")
                continue
            tag = f"layer_{idx}"
            handle = backbone[idx].register_forward_hook(self._make_hook(tag))
            self._handles.append(handle)

    def _make_hook(self, tag: str):
        def hook_fn(_module, _input, output):
            out = output
            if isinstance(out, (list, tuple)):
                out = out[0]
            if torch.is_tensor(out) and out.dim() == 4:
                self.features[tag] = out
        return hook_fn

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class GradHook(FeatureHook):
    """Like FeatureHook but also captures gradients (for Grad-CAM)."""

    def __init__(self):
        super().__init__()
        self.grads: dict[str, torch.Tensor] = {}
        self._grad_handles: list = []

    def _make_hook(self, tag: str):
        def hook_fn(_module, _input, output):
            out = output
            if isinstance(out, (list, tuple)):
                out = out[0]
            if torch.is_tensor(out) and out.dim() == 4:
                self.features[tag] = out
                gh = out.register_hook(lambda g, t=tag: self.grads.update({t: g}))
                self._grad_handles.append(gh)
        return hook_fn

    def clear(self):
        super().clear()
        self.grads.clear()

    def remove(self):
        super().remove()
        for h in self._grad_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._grad_handles.clear()


# ---------------------------------------------------------------------------
# Heatmap generation methods
# ---------------------------------------------------------------------------

def _resolve_layers(model: YOLO, user_layers: list[int] | None) -> list[int]:
    """Pick sensible default layers if the user didn't specify any."""
    if user_layers:
        return user_layers
    n = len(model.model.model)
    candidates = []
    for i, m in enumerate(model.model.model):
        name = type(m).__name__
        if name in ("SPPF", "C2f", "C3", "C2fAttn"):
            candidates.append(i)
    if candidates:
        return [candidates[-1]]
    return [n - 2]


def eigen_cam(model: YOLO, img_bgr: np.ndarray, layers: list[int] | None,
              imgsz: int, device: torch.device) -> list[tuple[str, np.ndarray]]:
    """EigenCAM: first principal component of each selected feature map."""
    layers = _resolve_layers(model, layers)
    hook = FeatureHook()
    hook.register(model, layers)

    x, *_ = preprocess(img_bgr, imgsz, device)
    with torch.no_grad():
        model.model(x)

    results = []
    for tag in sorted(hook.features):
        fmap = hook.features[tag][0]  # (C, H, W)
        C, H, W = fmap.shape
        reshaped = fmap.reshape(C, H * W).cpu().float().numpy()  # (C, HW)
        U, S, Vt = np.linalg.svd(reshaped, full_matrices=False)
        proj = Vt[0].reshape(H, W)  # first principal component
        cam = normalize_cam(np.abs(proj))
        results.append((tag, cam))

    hook.remove()
    return results


def grad_cam(model: YOLO, img_bgr: np.ndarray, layers: list[int] | None,
             imgsz: int, device: torch.device,
             target_class: int | None = None) -> list[tuple[str, np.ndarray]]:
    """Grad-CAM: gradient-weighted feature activation for a target class."""
    layers = _resolve_layers(model, layers)
    hook = GradHook()
    hook.register(model, layers)

    x, *_ = preprocess(img_bgr, imgsz, device)
    x.requires_grad_(True)

    with torch.enable_grad():
        raw = model.model.model(x)
        pred = raw[0] if isinstance(raw, (list, tuple)) else raw
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)

        # ultralytics v8+/v11: pred shape (B, num_classes+4, anchors) or (B, anchors, num_classes+4)
        # try to find objectness/class scores
        if pred.shape[-1] > pred.shape[1]:
            pred = pred.permute(0, 2, 1)  # -> (B, anchors, C)

        nc = pred.shape[-1] - 4  # assume first 4 are box coords
        cls_scores = pred[..., 4:].sigmoid()  # (B, A, nc)

        if target_class is not None and target_class < nc:
            score = cls_scores[0, :, target_class].max()
        else:
            score = cls_scores.max()

        model.model.zero_grad()
        score.backward()

    results = []
    for tag in sorted(hook.features):
        if tag not in hook.grads:
            continue
        fmap = hook.features[tag][0]  # (C, H, W)
        grad = hook.grads[tag][0]
        weights = grad.mean(dim=(1, 2))  # GAP over spatial dims
        cam = (weights[:, None, None] * fmap).sum(dim=0)
        cam = torch.relu(cam)
        cam_np = normalize_cam(cam.detach().cpu().numpy())
        results.append((tag, cam_np))

    hook.remove()
    return results


def activation_maps(model: YOLO, img_bgr: np.ndarray, layers: list[int] | None,
                    imgsz: int, device: torch.device) -> list[tuple[str, np.ndarray]]:
    """Channel-mean activation of specified layers."""
    layers = _resolve_layers(model, layers)
    hook = FeatureHook()
    hook.register(model, layers)

    x, *_ = preprocess(img_bgr, imgsz, device)
    with torch.no_grad():
        model.model(x)

    results = []
    for tag in sorted(hook.features):
        fmap = hook.features[tag][0]  # (C, H, W)
        cam = fmap.mean(dim=0).cpu().numpy()
        cam = normalize_cam(cam)
        results.append((tag, cam))

    hook.remove()
    return results


METHOD_MAP = {
    "eigen": eigen_cam,
    "gradcam": grad_cam,
    "activations": activation_maps,
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _layer_label(model: YOLO, tag: str) -> str:
    """Human-friendly layer label."""
    idx = int(tag.split("_")[1])
    mod = model.model.model[idx]
    return f"Layer {idx} ({type(mod).__name__})"


def save_figure(img_bgr: np.ndarray, cams: list[tuple[str, np.ndarray]],
                model: YOLO, out_path: Path, method_name: str,
                alpha: float = 0.5):
    """Save a matplotlib figure: original | heatmap | overlay per layer."""
    n = len(cams)
    fig = plt.figure(figsize=(5 * 3, 5 * max(n, 1)), dpi=120)
    gs = GridSpec(max(n, 1), 3, figure=fig, wspace=0.05, hspace=0.25)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    for row, (tag, cam) in enumerate(cams):
        overlay_bgr, heatmap_bgr = overlay_heatmap(img_bgr, cam, alpha)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        label = _layer_label(model, tag)

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(img_rgb)
        ax0.set_title("Original" if row == 0 else "", fontsize=11)
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(heatmap_rgb)
        ax1.set_title(f"{method_name} – {label}", fontsize=11)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(overlay_rgb)
        ax2.set_title("Overlay", fontsize=11)
        ax2.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  -> {out_path}")


def save_single_overlay(img_bgr: np.ndarray, cam: np.ndarray,
                        out_path: Path, alpha: float = 0.5):
    """Save just the overlay image (no matplotlib grid)."""
    overlay, _ = overlay_heatmap(img_bgr, cam, alpha)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO heatmap visualiser – EigenCAM / Grad-CAM / Activations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image", type=Path, required=True,
                   help="Path to a single image or a directory of images")
    p.add_argument("--weights", type=Path, required=True,
                   help="Path to YOLO weights (.pt)")
    p.add_argument("--method", choices=list(METHOD_MAP), default="eigen",
                   help="Heatmap method (default: eigen)")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Layer indices to hook (default: auto-select last SPPF/C2f)")
    p.add_argument("--target-class", type=int, default=None,
                   help="Target class id for Grad-CAM (ignored for other methods)")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--output", type=Path, default=Path("outputs/heatmaps"),
                   help="Output directory or file path")
    p.add_argument("--alpha", type=float, default=0.50,
                   help="Overlay blending alpha (0=original, 1=heatmap only)")
    p.add_argument("--simple", action="store_true",
                   help="Save only the overlay image (no side-by-side grid)")
    return p.parse_args()


def process_single(args, model: YOLO, img_path: Path, out_path: Path,
                   device: torch.device):
    """Run heatmap pipeline on one image."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[skip] Cannot read: {img_path}")
        return

    method_fn = METHOD_MAP[args.method]
    kwargs = dict(model=model, img_bgr=img_bgr, layers=args.layers,
                  imgsz=args.imgsz, device=device)
    if args.method == "gradcam":
        kwargs["target_class"] = args.target_class

    cams = method_fn(**kwargs)
    if not cams:
        print(f"[warn] No feature maps captured for {img_path}")
        return

    if args.simple:
        combined = np.mean([c for _, c in cams], axis=0)
        save_single_overlay(img_bgr, combined, out_path, args.alpha)
    else:
        save_figure(img_bgr, cams, model, out_path, args.method.upper(), args.alpha)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Weights: {args.weights}")
    print(f"Method : {args.method}")

    model = YOLO(str(args.weights))
    model.to(device)

    n_layers = len(model.model.model)
    print(f"Model has {n_layers} layers. Available indices: 0..{n_layers - 1}")
    for i, m in enumerate(model.model.model):
        print(f"  [{i:>2}] {type(m).__name__}")

    images: list[Path] = []
    if args.image.is_dir():
        images = sorted(p for p in args.image.iterdir()
                        if p.suffix.lower() in IMAGE_EXTENSIONS)
    elif args.image.is_file():
        images = [args.image]
    else:
        print(f"[error] Not found: {args.image}")
        sys.exit(1)

    print(f"\nProcessing {len(images)} image(s)...\n")
    for img_path in images:
        out_name = f"{img_path.stem}_{args.method}.png"
        if args.output.suffix in (".png", ".jpg", ".jpeg") and len(images) == 1:
            out_path = args.output
        else:
            out_path = args.output / out_name
        print(f"[{args.method}] {img_path.name}")
        process_single(args, model, img_path, out_path, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
