# File: RL/visual_features.py
"""
Visual Feature Extraction Module for LLM-Enhanced Open-Vocab VRD.

Cung cấp các features để gửi cho LLM:
1. Union Box - Vùng bao chung của subject + object
2. Interaction Heatmap - Bản đồ nhiệt vùng giao thoa
3. Gaze Guided Attention Vector - Xử lý quan hệ không chạm

Author: Auto-generated for DATN project
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple


def compute_union_box(
    subject_bbox: List[int],
    object_bbox: List[int]
) -> Tuple[int, int, int, int]:
    """
    Tính union bounding box của subject và object.
    
    Args:
        subject_bbox: [x1, y1, x2, y2] - bounding box của subject
        object_bbox: [x1, y1, x2, y2] - bounding box của object
    
    Returns:
        Tuple[x1, y1, x2, y2] - vùng bao chung
    
    Example:
        >>> compute_union_box([10, 10, 50, 50], [40, 40, 100, 100])
        (10, 10, 100, 100)
    """
    x1 = min(int(subject_bbox[0]), int(object_bbox[0]))
    y1 = min(int(subject_bbox[1]), int(object_bbox[1]))
    x2 = max(int(subject_bbox[2]), int(object_bbox[2]))
    y2 = max(int(subject_bbox[3]), int(object_bbox[3]))
    return (x1, y1, x2, y2)


def crop_union_region(
    image: np.ndarray,
    subject_bbox: List[int],
    object_bbox: List[int],
    padding: int = 20
) -> Image.Image:
    """
    Crop vùng union từ ảnh gốc với padding.
    
    Args:
        image: Full image (BGR numpy array từ cv2)
        subject_bbox: [x1, y1, x2, y2]
        object_bbox: [x1, y1, x2, y2]
        padding: Pixels padding xung quanh union box
    
    Returns:
        PIL.Image - Cropped region in RGB format
    """
    union = compute_union_box(subject_bbox, object_bbox)
    x1, y1, x2, y2 = union
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Add padding with boundary checks
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Ensure valid crop region
    if x2 <= x1 or y2 <= y1:
        # Return a small default region if invalid
        x1, y1 = 0, 0
        x2, y2 = min(100, w), min(100, h)
    
    # Crop and convert to RGB PIL Image
    cropped = image[y1:y2, x1:x2]
    
    # Handle grayscale images
    if len(cropped.shape) == 2:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
    else:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(cropped)


def compute_interaction_heatmap(
    image_shape: Tuple[int, int],
    subject_bbox: List[int],
    object_bbox: List[int],
    method: str = "iou"
) -> np.ndarray:
    """
    Tạo heat map biểu thị vùng giao thoa giữa subject và object.
    
    Args:
        image_shape: (height, width) của ảnh gốc
        subject_bbox: [x1, y1, x2, y2]
        object_bbox: [x1, y1, x2, y2]
        method: "iou" hoặc "gaussian"
            - iou: Binary intersection map (nhanh hơn)
            - gaussian: Gaussian-weighted overlap (smooth hơn)
    
    Returns:
        np.ndarray: Heatmap [H, W] với giá trị [0, 1]
    """
    h, w = image_shape[:2]
    
    # Convert to int
    sx1, sy1, sx2, sy2 = [int(x) for x in subject_bbox]
    ox1, oy1, ox2, oy2 = [int(x) for x in object_bbox]
    
    # Clamp to image bounds
    sx1, sy1 = max(0, sx1), max(0, sy1)
    sx2, sy2 = min(w, sx2), min(h, sy2)
    ox1, oy1 = max(0, ox1), max(0, oy1)
    ox2, oy2 = min(w, ox2), min(h, oy2)
    
    if method == "iou":
        # Fast binary intersection method
        subject_mask = np.zeros((h, w), dtype=np.float32)
        object_mask = np.zeros((h, w), dtype=np.float32)
        
        subject_mask[sy1:sy2, sx1:sx2] = 1.0
        object_mask[oy1:oy2, ox1:ox2] = 1.0
        
        # Intersection region gets value 1.0
        # Subject-only or object-only regions get 0.5
        heatmap = np.clip(subject_mask + object_mask, 0, 2) / 2.0
        
    else:  # gaussian
        # Gaussian-weighted method (smoother but slower)
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Subject Gaussian
        scx, scy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
        sw, sh = max(sx2 - sx1, 1), max(sy2 - sy1, 1)
        sigma = 0.3
        
        y_indices, x_indices = np.ogrid[sy1:sy2, sx1:sx2]
        if sy2 > sy1 and sx2 > sx1:
            dist_x = (x_indices - scx) / (sw / 2 + 1e-6)
            dist_y = (y_indices - scy) / (sh / 2 + 1e-6)
            subject_gaussian = np.exp(-(dist_x**2 + dist_y**2) / (2 * sigma**2))
            heatmap[sy1:sy2, sx1:sx2] += subject_gaussian
        
        # Object Gaussian
        ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
        ow, oh = max(ox2 - ox1, 1), max(oy2 - oy1, 1)
        
        y_indices, x_indices = np.ogrid[oy1:oy2, ox1:ox2]
        if oy2 > oy1 and ox2 > ox1:
            dist_x = (x_indices - ocx) / (ow / 2 + 1e-6)
            dist_y = (y_indices - ocy) / (oh / 2 + 1e-6)
            object_gaussian = np.exp(-(dist_x**2 + dist_y**2) / (2 * sigma**2))
            heatmap[oy1:oy2, ox1:ox2] *= object_gaussian  # Multiply for intersection
        
        # Normalize
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
    
    return heatmap


def compute_gaze_attention_vector(
    subject_bbox: List[int],
    object_bbox: List[int],
    subject_class: str,
    image_shape: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Tính Gaze Guided Attention Vector để phát hiện quan hệ không chạm.
    
    Rất hữu ích cho các quan hệ như:
    - "looking at", "watching", "facing"
    - "approaching", "following"
    
    Args:
        subject_bbox: [x1, y1, x2, y2]
        object_bbox: [x1, y1, x2, y2]
        subject_class: Tên class của subject (e.g., "person", "dog")
        image_shape: (height, width) của ảnh
    
    Returns:
        Dict với:
            - direction: (dx, dy) normalized direction từ subject đến object
            - angle_degrees: Góc từ subject đến object
            - distance: Khoảng cách Euclidean giữa centers
            - proximity_score: 0-1, cao = gần nhau
            - non_contact_likelihood: 0-1, cao = có thể là quan hệ không chạm
            - gaze_vector: List 6 floats cho LLM context
    """
    h, w = image_shape[:2]
    
    # Centers of bboxes
    sx1, sy1, sx2, sy2 = [float(x) for x in subject_bbox]
    ox1, oy1, ox2, oy2 = [float(x) for x in object_bbox]
    
    scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
    ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
    
    # Direction vector từ subject đến object
    dx = ocx - scx
    dy = ocy - scy
    distance = np.sqrt(dx**2 + dy**2)
    
    if distance > 0:
        direction = (dx / distance, dy / distance)
    else:
        direction = (0.0, 0.0)
    
    # Angle in degrees (0 = right, 90 = down, -90 = up)
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Proximity score (closer = higher)
    max_dist = np.sqrt(h**2 + w**2)
    proximity = 1.0 - (distance / max_dist) if max_dist > 0 else 0.0
    
    # Check for physical overlap (IoU > 0)
    overlap_x = max(0, min(sx2, ox2) - max(sx1, ox1))
    overlap_y = max(0, min(sy2, oy2) - max(sy1, oy1))
    has_overlap = overlap_x > 0 and overlap_y > 0
    
    # Calculate overlap ratio
    if has_overlap:
        overlap_area = overlap_x * overlap_y
        subject_area = max((sx2 - sx1) * (sy2 - sy1), 1)
        object_area = max((ox2 - ox1) * (oy2 - oy1), 1)
        min_area = min(subject_area, object_area)
        overlap_ratio = overlap_area / min_area
    else:
        overlap_ratio = 0.0
    
    # Animate entities that can "gaze"
    gazers = {
        "person", "man", "woman", "child", "boy", "girl", "people",
        "dog", "cat", "horse", "cow", "animal", "bird",
        "player", "worker", "pedestrian"
    }
    subject_lower = subject_class.lower().strip()
    is_gazer = any(g in subject_lower for g in gazers)
    
    # Is horizontal gaze (subject looking sideways at object)?
    is_horizontal = abs(angle) < 60 or abs(angle) > 120
    
    # Vertical alignment check (are they at similar height?)
    vertical_alignment = 1.0 - min(abs(scy - ocy) / max(h, 1), 1.0)
    
    # Non-contact relationship likelihood
    # High if: animate subject, not overlapping, reasonable proximity
    if is_gazer:
        if not has_overlap:
            # No overlap = possible "looking at" relationship
            non_contact_score = proximity * 0.7 + vertical_alignment * 0.3
        else:
            # Has overlap = more likely contact relationship
            non_contact_score = max(0.3 - overlap_ratio, 0.1)
    else:
        # Inanimate subjects don't gaze
        non_contact_score = 0.05
    
    # Clamp to [0, 1]
    non_contact_score = max(0.0, min(1.0, non_contact_score))
    
    # Build gaze vector for LLM context
    gaze_vector = [
        direction[0],                    # dx normalized
        direction[1],                    # dy normalized
        proximity,                       # distance score (0-1)
        1.0 if is_horizontal else 0.0,   # horizontal gaze flag
        overlap_ratio,                   # how much overlap (0-1)
        non_contact_score,               # non-contact likelihood
    ]
    
    return {
        "direction": direction,
        "angle_degrees": float(angle),
        "distance": float(distance),
        "proximity_score": float(proximity),
        "overlap_ratio": float(overlap_ratio),
        "vertical_alignment": float(vertical_alignment),
        "is_horizontal_gaze": bool(is_horizontal),
        "has_physical_contact": bool(has_overlap),
        "non_contact_likelihood": float(non_contact_score),
        "is_animate_subject": bool(is_gazer),
        "gaze_vector": gaze_vector,
    }


def compute_spatial_description(
    subject_bbox: List[int],
    object_bbox: List[int],
    image_shape: Tuple[int, int]
) -> str:
    """
    Generate natural language description of spatial relationship.
    Useful for LLM prompt construction.
    
    Returns:
        String description like "subject is above and to the left of object"
    """
    sx1, sy1, sx2, sy2 = [float(x) for x in subject_bbox]
    ox1, oy1, ox2, oy2 = [float(x) for x in object_bbox]
    
    scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
    ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
    
    h, w = image_shape[:2]
    
    # Vertical position
    vertical_diff = scy - ocy
    vertical_threshold = h * 0.1  # 10% of image height
    
    if vertical_diff < -vertical_threshold:
        vertical_desc = "above"
    elif vertical_diff > vertical_threshold:
        vertical_desc = "below"
    else:
        vertical_desc = "at same height as"
    
    # Horizontal position
    horizontal_diff = scx - ocx
    horizontal_threshold = w * 0.1  # 10% of image width
    
    if horizontal_diff < -horizontal_threshold:
        horizontal_desc = "to the left of"
    elif horizontal_diff > horizontal_threshold:
        horizontal_desc = "to the right of"
    else:
        horizontal_desc = "aligned with"
    
    # Check overlap
    overlap_x = max(0, min(sx2, ox2) - max(sx1, ox1))
    overlap_y = max(0, min(sy2, oy2) - max(sy1, oy1))
    
    if overlap_x > 0 and overlap_y > 0:
        overlap_desc = "overlapping with"
    else:
        overlap_desc = None
    
    # Build description
    if overlap_desc:
        return f"Subject is {overlap_desc} object, positioned {vertical_desc} and {horizontal_desc} object"
    else:
        return f"Subject is {vertical_desc} and {horizontal_desc} object (no overlap)"


def heatmap_to_image(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> Image.Image:
    """
    Convert heatmap array to colorized PIL Image for visualization.
    
    Args:
        heatmap: [H, W] array with values [0, 1]
        colormap: OpenCV colormap (default: JET)
    
    Returns:
        PIL.Image in RGB format
    """
    # Scale to 0-255
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert BGR to RGB
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(colored_rgb)


def extract_all_visual_features(
    image: np.ndarray,
    subject_bbox: List[int],
    object_bbox: List[int],
    subject_class: str,
    object_class: str
) -> Dict[str, Any]:
    """
    Extract all visual features for a subject-object pair.
    
    This is the main entry point for the LLM relationship predictor.
    
    Args:
        image: Full image (BGR numpy array)
        subject_bbox: [x1, y1, x2, y2]
        object_bbox: [x1, y1, x2, y2]
        subject_class: Class name of subject
        object_class: Class name of object
    
    Returns:
        Dict containing all visual features:
            - union_bbox: Tuple[x1, y1, x2, y2]
            - union_crop: PIL.Image
            - interaction_heatmap: np.ndarray
            - gaze_info: Dict with gaze analysis
            - spatial_description: str
    """
    image_shape = image.shape[:2]  # (H, W)
    
    # Compute all features
    union_bbox = compute_union_box(subject_bbox, object_bbox)
    union_crop = crop_union_region(image, subject_bbox, object_bbox)
    interaction_heatmap = compute_interaction_heatmap(image_shape, subject_bbox, object_bbox)
    gaze_info = compute_gaze_attention_vector(subject_bbox, object_bbox, subject_class, image_shape)
    spatial_desc = compute_spatial_description(subject_bbox, object_bbox, image_shape)
    
    return {
        "subject_class": subject_class,
        "object_class": object_class,
        "subject_bbox": subject_bbox,
        "object_bbox": object_bbox,
        "union_bbox": union_bbox,
        "union_crop": union_crop,
        "interaction_heatmap": interaction_heatmap,
        "gaze_info": gaze_info,
        "spatial_description": spatial_desc,
    }


# ============ TESTING ============
if __name__ == "__main__":
    # Simple test
    print("Testing visual_features module...")
    
    # Create dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_image, (100, 100), (200, 200), (255, 0, 0), -1)  # Subject
    cv2.rectangle(dummy_image, (250, 150), (350, 250), (0, 255, 0), -1)  # Object
    
    subject_bbox = [100, 100, 200, 200]
    object_bbox = [250, 150, 350, 250]
    
    # Test all functions
    union = compute_union_box(subject_bbox, object_bbox)
    print(f"Union box: {union}")
    
    crop = crop_union_region(dummy_image, subject_bbox, object_bbox)
    print(f"Crop size: {crop.size}")
    
    heatmap = compute_interaction_heatmap((480, 640), subject_bbox, object_bbox)
    print(f"Heatmap shape: {heatmap.shape}, max: {heatmap.max():.2f}")
    
    gaze = compute_gaze_attention_vector(subject_bbox, object_bbox, "person", (480, 640))
    print(f"Gaze info: angle={gaze['angle_degrees']:.1f}°, proximity={gaze['proximity_score']:.2f}")
    
    spatial = compute_spatial_description(subject_bbox, object_bbox, (480, 640))
    print(f"Spatial: {spatial}")
    
    # Test full extraction
    features = extract_all_visual_features(
        dummy_image, subject_bbox, object_bbox, "person", "car"
    )
    print(f"\nFull features keys: {list(features.keys())}")
    print("✅ All tests passed!")
