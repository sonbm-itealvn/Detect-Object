"""
RelTR & RL visualization examples.

Modes:
  - attention: run RelTR on an image, extract subject/object attention maps, save overlays.
  - rl_metrics: plot RL reward components and total reward from a metrics JSON file.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# Ensure repository root is on sys.path so that "models" and "util" are importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models import build_model
from util.misc import nested_tensor_from_tensor_list


# -----------------------------
# Helpers to load RelTR
# -----------------------------
def _default_reltr_args(device: str = "cuda") -> SimpleNamespace:
    return SimpleNamespace(
        dataset="vg",
        lr_backbone=1e-5,
        backbone="resnet50",
        dilation=False,
        position_embedding="sine",
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


def load_reltr(checkpoint: Optional[Path], device: torch.device):
    args = _default_reltr_args(str(device))
    model, _, _ = build_model(args)
    if checkpoint and checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model") if isinstance(ckpt, dict) else None
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"[RelTR] Loaded checkpoint: {checkpoint}")
    else:
        print("[RelTR] No checkpoint provided or file not found, using random init.")
    model.to(device)
    model.eval()
    transform = T.Compose(
        [
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, transform


def _prepare_image(image_path: Path, transform: T.Compose, device: torch.device):
    with Image.open(image_path) as img:
        pil = img.convert("RGB")
    w, h = pil.size
    tensor = transform(pil)
    samples = nested_tensor_from_tensor_list([tensor.to(device)])
    return pil, (w, h), samples


def _resize_map(attn: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    # attn: (H, W) -> resize to size (w, h)
    attn = attn.unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    attn = F.interpolate(attn, size=(size[1], size[0]), mode="bilinear", align_corners=False)
    attn = attn.squeeze(0).squeeze(0)
    attn = attn.clamp(min=0).cpu().numpy()
    if attn.max() > 0:
        attn = attn / attn.max()
    return attn


def _save_overlay(image: Image.Image, heatmap: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[RelTR] Saved: {out_path}")


def run_attention(image_path: Path, checkpoint: Optional[Path], output_prefix: Path, device: torch.device):
    model, transform = load_reltr(checkpoint, device)
    pil_img, orig_size, samples = _prepare_image(image_path, transform, device)

    with torch.no_grad():
        features, pos = model.backbone(samples)
        src, mask = features[-1].decompose()
        hs, hs_t, so_masks, _ = model.transformer(
            model.input_proj(src),
            mask,
            model.entity_embed.weight,
            model.triplet_embed.weight,
            pos[-1],
            model.so_embed.weight,
        )

    # so_masks: [layers, B, num_triplets, 2, H, W]
    attn_last = so_masks[-1, 0]  # take last decoder layer, batch 0
    sub_map = attn_last[:, 0].mean(0)  # mean over triplet queries
    obj_map = attn_last[:, 1].mean(0)

    sub_resized = _resize_map(sub_map, orig_size)
    obj_resized = _resize_map(obj_map, orig_size)

    _save_overlay(pil_img, sub_resized, output_prefix.with_name(output_prefix.name + "_sub_heatmap.png"), "Subject attention")
    _save_overlay(pil_img, obj_resized, output_prefix.with_name(output_prefix.name + "_obj_heatmap.png"), "Object attention")


# -----------------------------
# RL metrics plotting
# -----------------------------
def _load_metrics_series(path_pattern: str) -> list:
    """
    Load a series of metrics files.
    - If a file contains `training_progress`, return that list.
    - Otherwise, treat each matched JSON as one epoch record and sort by epoch.
    """
    paths = []
    p = Path(path_pattern)
    if p.is_dir():
        paths = sorted(p.glob("*.json"))
    else:
        paths = sorted(Path().glob(path_pattern))
    if not paths:
        raise FileNotFoundError(f"No metrics file matched pattern or directory: {path_pattern}")

    series = []
    for fp in paths:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Case 1: aggregated file with training_progress
        if isinstance(data, dict) and data.get("training_progress"):
            print(f"[RL] Loaded aggregated metrics: {fp}")
            return data["training_progress"]
        # Case 2: single-epoch file
        data["_source_file"] = str(fp)
        series.append(data)
    # Sort single-epoch records by epoch if present, else by file name
    series.sort(key=lambda d: d.get("epoch", 0))
    print(f"[RL] Loaded {len(series)} single-epoch metric files")
    return series


# -----------------------------
# Q-network utils (lightweight re-implementation)
# -----------------------------
def _build_q_network(input_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
    )


def _normalize_scalar(value: float, scale: float = 1.0) -> float:
    if scale <= 0:
        scale = 1.0
    return math.tanh(value / scale)


def _build_state_vector_from_record(rec: dict, device: torch.device) -> torch.Tensor:
    detection_loss = _normalize_scalar(float(rec.get("detection_loss", 1.0)), scale=5.0)
    relationship_loss = _normalize_scalar(float(rec.get("relationship_loss", 1.0)), scale=5.0)
    reward_value = _normalize_scalar(float(rec.get("reward", 0.0)), scale=1.0)
    dataset_size = float(rec.get("dataset_size", rec.get("ai_images_count", rec.get("experience_count", 0))))
    dataset_norm = _normalize_scalar(dataset_size, scale=50.0)
    epsilon = _normalize_scalar(float(rec.get("epsilon", 0.0)), scale=1.0)
    state = torch.tensor(
        [detection_loss, relationship_loss, reward_value, dataset_norm, epsilon],
        dtype=torch.float32,
        device=device,
    )
    return state


def plot_rl_metrics(metrics_path: str, output_prefix: Path):
    progress = _load_metrics_series(metrics_path)
    if not progress:
        print("[RL] No metrics records found.")
        return

    epochs = [p.get("epoch", i + 1) for i, p in enumerate(progress)]
    rewards = [p.get("reward", 0.0) for p in progress]
    components = [p.get("reward_components", {}) or {} for p in progress]

    # Plot total reward
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, rewards, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("RL reward per epoch")
    plt.grid(True, linestyle="--", alpha=0.5)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_reward = output_prefix.with_name(output_prefix.name + "_reward_vs_epoch.png")
    plt.savefig(out_reward, bbox_inches="tight")
    plt.close()
    print(f"[RL] Saved: {out_reward}")

    # Plot reward components
    keys = ["detection_score", "relationship_score", "diversity_score", "consistency_score", "improvement_score"]
    plt.figure(figsize=(9, 5))
    for k in keys:
        vals = [c.get(k, 0.0) for c in components]
        plt.plot(epochs, vals, marker=".", label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Component value")
    plt.title("Reward components per epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    out_comp = output_prefix.with_name(output_prefix.name + "_reward_components.png")
    plt.savefig(out_comp, bbox_inches="tight")
    plt.close()
    print(f"[RL] Saved: {out_comp}")


def plot_q_values(metrics_path: str, output_prefix: Path, q_checkpoint: Optional[Path]):
    progress = _load_metrics_series(metrics_path)
    if not progress:
        print("[RL] No metrics records found.")
        return

    epochs = [p.get("epoch", i + 1) for i, p in enumerate(progress)]
    actions = [p.get("rl_action_index", None) for p in progress]
    rewards = [p.get("reward", 0.0) for p in progress]

    # Plot action frequency and reward vs action
    plt.figure(figsize=(7, 4))
    plt.hist([a for a in actions if a is not None], bins=range(0, 12), align="left", rwidth=0.7)
    plt.xlabel("Action index")
    plt.ylabel("Count")
    plt.title("Action frequency")
    out_actions = output_prefix.with_name(output_prefix.name + "_action_hist.png")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_actions, bbox_inches="tight")
    plt.close()
    print(f"[RL] Saved: {out_actions}")

    plt.figure(figsize=(7, 4))
    plt.scatter(actions, rewards)
    plt.xlabel("Action index")
    plt.ylabel("Reward")
    plt.title("Reward vs action")
    plt.grid(True, linestyle="--", alpha=0.5)
    out_reward_action = output_prefix.with_name(output_prefix.name + "_reward_vs_action.png")
    plt.savefig(out_reward_action, bbox_inches="tight")
    plt.close()
    print(f"[RL] Saved: {out_reward_action}")

    # Optional: load q-network and compute Q-values per epoch/state
    if q_checkpoint is None:
        print("[RL] No Q-network checkpoint provided; skipping Q-value heatmap.")
        return

    q_checkpoint = Path(q_checkpoint)
    if not q_checkpoint.exists():
        print(f"[RL] Q-network checkpoint not found: {q_checkpoint} (skipping Q-value heatmap)")
        return

    device = torch.device("cpu")
    q_net = _build_q_network(5, 10).to(device)
    try:
        ckpt = torch.load(q_checkpoint, map_location=device, weights_only=False)
    except Exception as exc:
        print(f"[RL] Failed to load Q-network checkpoint: {exc} (skipping heatmap)")
        return
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        state_dict = ckpt
    missing, unexpected = q_net.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[RL] Warning: checkpoint keys mismatch. Missing: {missing}, Unexpected: {unexpected}")
    q_net.eval()

    states: List[torch.Tensor] = [_build_state_vector_from_record(rec, device) for rec in progress]
    with torch.no_grad():
        q_vals = torch.stack([q_net(s.unsqueeze(0)).squeeze(0) for s in states])  # [epochs, actions]
    q_np = q_vals.cpu().numpy()

    plt.figure(figsize=(9, 4))
    plt.imshow(q_np, aspect="auto", cmap="viridis")
    plt.colorbar(label="Q-value")
    plt.xlabel("Action index")
    plt.ylabel("Epoch")
    plt.title("Q-values heatmap (epoch x action)")
    plt.xticks(range(q_np.shape[1]))
    plt.yticks(range(len(epochs)), epochs)
    out_q = output_prefix.with_name(output_prefix.name + "_q_values_heatmap.png")
    plt.savefig(out_q, bbox_inches="tight")
    plt.close()
    print(f"[RL] Saved: {out_q}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="RelTR and RL visualization examples")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    attn = subparsers.add_parser("attention", help="Visualize RelTR subject/object attention")
    attn.add_argument("--image", type=Path, required=True, help="Path to input image")
    attn.add_argument("--checkpoint", type=Path, help="Path to RelTR checkpoint (.pth)")
    attn.add_argument("--output", type=Path, default=Path("outputs/reltr_attn"), help="Output prefix (file name without extension)")
    attn.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run model")

    rl = subparsers.add_parser("rl_metrics", help="Plot RL reward metrics")
    rl.add_argument("--metrics", type=str, required=True, help="Metrics JSON path or glob (e.g., experiments/exp_001/metrics/*.json)")
    rl.add_argument("--output", type=Path, default=Path("outputs/rl_metrics"), help="Output prefix (file name without extension)")

    qv = subparsers.add_parser("q_values", help="Visualize Q-values / action stats from RL metrics (optional checkpoint)")
    qv.add_argument("--metrics", type=str, required=True, help="Metrics JSON path, glob, or directory")
    qv.add_argument("--output", type=Path, default=Path("outputs/rl_qvalues"), help="Output prefix (file name without extension)")
    qv.add_argument("--q-checkpoint", type=Path, help="Path to saved Q-network state_dict (optional)")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "attention":
        device = torch.device(args.device)
        run_attention(args.image, args.checkpoint, args.output, device)
    elif args.mode == "rl_metrics":
        plot_rl_metrics(args.metrics, args.output)
    elif args.mode == "q_values":
        plot_q_values(args.metrics, args.output, args.q_checkpoint)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()

