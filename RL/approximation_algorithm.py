# File: RL/approximation_algorithm.py
"""
Approximation Algorithm Module – Greedy Submodular Maximization.

Chọn tập con tối ưu từ pool ảnh synthetic để đảm bảo **đa dạng** và **chất lượng**
mà không cần duyệt hết tất cả tổ hợp (NP-hard).

Thuật toán Greedy đạt approximation ratio (1 - 1/e) ≈ 63.2% so với optimal
(Nemhauser et al. 1978) khi objective function là submodular.

Submodular Objective:
    f(S) = λ₁·Diversity(S) + λ₂·Quality(S) + λ₃·Representativeness(S)

    Diversity: Khoảng cách giữa các samples trong tập S (càng xa nhau càng tốt)
    Quality: Confidence scores, annotation quality
    Representativeness: Coverage – S đại diện tốt cho toàn bộ pool

Author: Auto-generated for DATN project
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GreedySubsetSelector:
    """Greedy Submodular Maximization để chọn tập con tối ưu từ pool samples."""

    def __init__(
        self,
        max_subset_size: Optional[int] = None,
        diversity_weight: float = 0.5,
        quality_weight: float = 0.3,
        representativeness_weight: float = 0.2,
    ):
        """
        Args:
            max_subset_size: Giới hạn tối đa số samples chọn (None = dùng budget)
            diversity_weight: Trọng số cho diversity gain (λ₁)
            quality_weight: Trọng số cho quality gain (λ₂)
            representativeness_weight: Trọng số cho representativeness gain (λ₃)
        """
        self.max_subset_size = max_subset_size
        self.diversity_weight = diversity_weight
        self.quality_weight = quality_weight
        self.representativeness_weight = representativeness_weight

        # Cache relationship type encoding
        self._rel_type_cache: Dict[str, int] = {}
        self._obj_class_cache: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Core: Greedy Selection
    # ------------------------------------------------------------------ #

    def select_optimal_subset(
        self,
        candidate_samples: List[Dict[str, Any]],
        budget: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Chọn tập con tối ưu bằng Greedy Submodular Maximization.

        Algorithm:
            S = {}
            for i = 1 to budget:
                s* = argmax_{s ∈ Pool ∖ S} [ f(S ∪ {s}) − f(S) ]   (marginal gain)
                S = S ∪ {s*}
            return S

        Approximation guarantee: f(S) ≥ (1 − 1/e) · f(S*) ≈ 0.632 · OPT

        Args:
            candidate_samples: Pool ảnh synthetic
            budget: Số samples cần chọn

        Returns:
            Tuple of:
                - List[Dict]: Selected subset
                - Dict: Stats (gains, coverage, etc.)
        """
        n = len(candidate_samples)
        if n == 0:
            return [], {'total_gain': 0.0, 'selected': 0, 'pool_size': 0}

        effective_budget = min(budget, n)
        if self.max_subset_size is not None:
            effective_budget = min(effective_budget, self.max_subset_size)

        # Pre-compute features for all candidates
        all_features = [self._extract_sample_features(s) for s in candidate_samples]

        selected_indices: List[int] = []
        remaining_indices = set(range(n))
        total_gain = 0.0
        gains_per_step: List[float] = []

        print(f"[ApproxAlgo] Starting Greedy selection: pool={n}, budget={effective_budget}")

        for step in range(effective_budget):
            best_idx = -1
            best_gain = -float('inf')

            for idx in remaining_indices:
                gain = self._compute_marginal_gain(
                    selected_indices, idx, all_features, candidate_samples
                )
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx < 0:
                # Fallback: random selection khi tất cả gains = 0
                best_idx = next(iter(remaining_indices))
                best_gain = 0.0

            selected_indices.append(best_idx)
            remaining_indices.discard(best_idx)
            total_gain += best_gain
            gains_per_step.append(best_gain)

            # Diminishing returns check (marginal gain quá nhỏ → dừng sớm)
            if best_gain < 1e-6 and step > effective_budget * 0.5:
                print(f"[ApproxAlgo] Early stopping at step {step+1}: "
                      f"marginal gain {best_gain:.6f} < threshold")
                break

        # Build result
        selected_samples = [candidate_samples[i] for i in selected_indices]

        # Compute coverage stats
        rel_types_pool = set()
        rel_types_selected = set()
        for s in candidate_samples:
            rel_types_pool.add(self._get_relation_type(s))
        for s in selected_samples:
            rel_types_selected.add(self._get_relation_type(s))

        coverage = len(rel_types_selected) / max(len(rel_types_pool), 1)

        stats = {
            'pool_size': n,
            'selected': len(selected_samples),
            'budget': effective_budget,
            'total_gain': total_gain,
            'avg_marginal_gain': total_gain / max(len(selected_indices), 1),
            'gains_per_step': gains_per_step,
            'relationship_coverage': coverage,
            'unique_rel_types_pool': len(rel_types_pool),
            'unique_rel_types_selected': len(rel_types_selected),
        }

        print(f"[ApproxAlgo] Selected {stats['selected']}/{n} samples. "
              f"Total gain: {total_gain:.4f}, Coverage: {coverage:.2%}")

        return selected_samples, stats

    # ------------------------------------------------------------------ #
    # Marginal Gain Computation
    # ------------------------------------------------------------------ #

    def _compute_marginal_gain(
        self,
        current_indices: List[int],
        candidate_idx: int,
        all_features: List[np.ndarray],
        all_samples: List[Dict[str, Any]],
    ) -> float:
        """
        Tính marginal gain f(S ∪ {candidate}) − f(S).

        Submodular components:
            1. Diversity: min distance từ candidate đến S (max = best diversity gain)
            2. Quality: annotation confidence, image quality
            3. Representativeness: candidate gần với bao nhiêu samples chưa chọn
        """
        candidate_features = all_features[candidate_idx]
        candidate_sample = all_samples[candidate_idx]

        # 1. Diversity Gain
        diversity_gain = self._compute_diversity_gain(
            current_indices, candidate_features, all_features
        )

        # 2. Quality Gain
        quality_gain = self._compute_quality_gain(candidate_sample)

        # 3. Representativeness Gain
        representativeness_gain = self._compute_representativeness_gain(
            current_indices, candidate_idx, all_features
        )

        # Weighted combination
        total_gain = (
            self.diversity_weight * diversity_gain
            + self.quality_weight * quality_gain
            + self.representativeness_weight * representativeness_gain
        )

        return total_gain

    def _compute_diversity_gain(
        self,
        current_indices: List[int],
        candidate_features: np.ndarray,
        all_features: List[np.ndarray],
    ) -> float:
        """
        Diversity gain = khoảng cách nhỏ nhất từ candidate đến S.

        Nếu S rỗng → gain = 1.0 (mọi sample đều đa dạng khi S rỗng).
        Nếu candidate xa tất cả elements trong S → gain cao (tốt).
        Nếu candidate gần 1 element trong S → gain thấp (trùng lặp).
        """
        if not current_indices:
            return 1.0

        min_distance = float('inf')
        for idx in current_indices:
            dist = self._compute_distance(candidate_features, all_features[idx])
            min_distance = min(min_distance, dist)

        # Normalize distance to [0, 1]
        return min(min_distance, 1.0)

    def _compute_quality_gain(self, sample: Dict[str, Any]) -> float:
        """
        Quality gain dựa trên:
        - Annotation confidence (từ auto-annotator)
        - Image quality indicators
        - Relationship completeness
        """
        quality_score = 0.0
        n_factors = 0

        # 1. Annotation confidence
        objects = sample.get('objects', [])
        if objects:
            confidences = [o.get('confidence', 0.0) for o in objects]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            quality_score += avg_conf
            n_factors += 1

        # 2. Annotation backend quality
        ann_backend = sample.get('annotation_backend', '')
        backend_quality = {
            'groundingdino': 0.9,
            'owlvit': 0.7,
            'yolo_clip': 0.8,
            'pseudo': 0.3,
        }
        if ann_backend:
            quality_score += backend_quality.get(ann_backend, 0.5)
            n_factors += 1

        # 3. Relationship completeness (có đủ subject, relation, object)
        rel = (
            sample.get('original_relationship')
            or (sample.get('relationships', [{}]) or [{}])[0]
        )
        if rel:
            completeness = sum([
                1 for k in ['subject', 'relation', 'object']
                if rel.get(k)
            ]) / 3.0
            quality_score += completeness
            n_factors += 1

        # 4. Has valid bounding boxes
        if objects and any(o.get('bbox') for o in objects):
            quality_score += 0.8
            n_factors += 1

        return quality_score / max(n_factors, 1)

    def _compute_representativeness_gain(
        self,
        current_indices: List[int],
        candidate_idx: int,
        all_features: List[np.ndarray],
    ) -> float:
        """
        Representativeness gain = candidate "đại diện" cho bao nhiêu unselected samples.

        Tính bằng: Số samples chưa chọn mà candidate là nearest neighbor sau khi thêm vào S.
        Normalize bởi tổng số unselected samples.
        """
        n = len(all_features)
        selected_set = set(current_indices) | {candidate_idx}
        unselected = [i for i in range(n) if i not in selected_set]

        if not unselected:
            return 0.0

        # Đếm bao nhiêu unselected samples gần candidate nhất
        count_represented = 0
        candidate_features = all_features[candidate_idx]

        for u_idx in unselected:
            u_features = all_features[u_idx]
            dist_to_candidate = self._compute_distance(candidate_features, u_features)

            # So sánh với distance đến nearest element trong S hiện tại
            if current_indices:
                min_dist_to_s = min(
                    self._compute_distance(all_features[s_idx], u_features)
                    for s_idx in current_indices
                )
                if dist_to_candidate < min_dist_to_s:
                    count_represented += 1
            else:
                count_represented += 1  # Nếu S rỗng, candidate đại diện cho tất cả

        return count_represented / len(unselected)

    # ------------------------------------------------------------------ #
    # Feature Extraction
    # ------------------------------------------------------------------ #

    def _extract_sample_features(self, sample: Dict[str, Any]) -> np.ndarray:
        """
        Trích xuất feature vector từ sample.

        Features:
            [0-1]: Normalized bbox center (x, y) – nếu có
            [2]: Bbox area ratio
            [3]: Relationship type index (encoded)
            [4]: Subject class index
            [5]: Object class index
            [6]: Annotation confidence
        """
        features = np.zeros(7, dtype=np.float32)

        # BBox features
        objects = sample.get('objects', [])
        if objects:
            bboxes = [o.get('bbox', [0, 0, 0, 0]) for o in objects]
            if bboxes:
                # Center of all bboxes
                centers_x = []
                centers_y = []
                areas = []
                for bbox in bboxes:
                    if bbox and len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                        centers_x.append((x1 + x2) / 2.0)
                        centers_y.append((y1 + y2) / 2.0)
                        areas.append(abs((x2 - x1) * (y2 - y1)))

                if centers_x:
                    # Normalize by assuming max image dimension ~ 1000
                    features[0] = np.mean(centers_x) / 1000.0
                    features[1] = np.mean(centers_y) / 1000.0
                    features[2] = min(np.mean(areas) / (1000 * 1000), 1.0)

        # Relationship features
        rel = (
            sample.get('original_relationship')
            or (sample.get('relationships', [{}]) or [{}])[0]
        )
        if rel:
            rel_type = (rel.get('relation', '') or '').lower().strip()
            subject = (rel.get('subject', '') or '').lower().strip()
            obj = (rel.get('object', '') or '').lower().strip()

            features[3] = self._encode_string(rel_type, self._rel_type_cache) / 100.0
            features[4] = self._encode_string(subject, self._obj_class_cache) / 100.0
            features[5] = self._encode_string(obj, self._obj_class_cache) / 100.0

        # Confidence
        if objects:
            confs = [o.get('confidence', 0.0) for o in objects]
            features[6] = sum(confs) / max(len(confs), 1)

        return features

    def _encode_string(self, s: str, cache: Dict[str, int]) -> int:
        """Encode string to integer index."""
        if s not in cache:
            cache[s] = len(cache)
        return cache[s]

    # ------------------------------------------------------------------ #
    # Distance & Deduplication
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
        """Euclidean distance between two feature vectors, normalized."""
        diff = features_a - features_b
        dist = float(np.sqrt(np.sum(diff ** 2)))
        # Normalize: typical max distance for 7-dim unit features ≈ sqrt(7) ≈ 2.65
        return min(dist / 2.65, 1.0)

    def filter_redundant_samples(
        self,
        samples: List[Dict[str, Any]],
        min_distance: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Loại bỏ samples quá giống nhau (deduplication nhanh).

        Args:
            samples: Pool ảnh
            min_distance: Khoảng cách tối thiểu giữa 2 samples được giữ lại

        Returns:
            Filtered list (đã loại bỏ duplicates)
        """
        if len(samples) <= 1:
            return list(samples)

        features = [self._extract_sample_features(s) for s in samples]
        kept_indices: List[int] = [0]  # Luôn giữ sample đầu tiên

        for i in range(1, len(samples)):
            is_unique = True
            for j in kept_indices:
                dist = self._compute_distance(features[i], features[j])
                if dist < min_distance:
                    is_unique = False
                    break
            if is_unique:
                kept_indices.append(i)

        removed = len(samples) - len(kept_indices)
        if removed > 0:
            print(f"[ApproxAlgo] Deduplication: removed {removed}/{len(samples)} redundant samples")

        return [samples[i] for i in kept_indices]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_relation_type(sample: Dict[str, Any]) -> str:
        """Get relation type from sample."""
        rel = (
            sample.get('original_relationship')
            or (sample.get('relationships', [{}]) or [{}])[0]
        )
        if rel:
            return (rel.get('relation', '') or '').lower().strip()
        return 'unknown'


# ============ TESTING ============
if __name__ == "__main__":
    print("Testing GreedySubsetSelector module...")

    selector = GreedySubsetSelector(
        diversity_weight=0.5,
        quality_weight=0.3,
        representativeness_weight=0.2,
    )

    # Create dummy pool of samples
    pool = [
        {
            'original_relationship': {'subject': 'person', 'relation': 'riding', 'object': 'horse'},
            'objects': [
                {'bbox': [10, 10, 100, 200], 'confidence': 0.9, 'class': 'person'},
                {'bbox': [50, 100, 300, 350], 'confidence': 0.85, 'class': 'horse'},
            ],
        },
        {
            'original_relationship': {'subject': 'person', 'relation': 'riding', 'object': 'horse'},
            'objects': [
                {'bbox': [15, 15, 110, 210], 'confidence': 0.88, 'class': 'person'},
                {'bbox': [55, 105, 310, 360], 'confidence': 0.82, 'class': 'horse'},
            ],
        },
        {
            'original_relationship': {'subject': 'dog', 'relation': 'sitting on', 'object': 'chair'},
            'objects': [
                {'bbox': [200, 200, 350, 400], 'confidence': 0.75, 'class': 'dog'},
                {'bbox': [180, 300, 400, 500], 'confidence': 0.80, 'class': 'chair'},
            ],
        },
        {
            'original_relationship': {'subject': 'cat', 'relation': 'on', 'object': 'table'},
            'objects': [
                {'bbox': [100, 50, 200, 150], 'confidence': 0.70, 'class': 'cat'},
                {'bbox': [80, 100, 350, 250], 'confidence': 0.90, 'class': 'table'},
            ],
        },
        {
            'original_relationship': {'subject': 'person', 'relation': 'holding', 'object': 'phone'},
            'objects': [
                {'bbox': [300, 50, 450, 350], 'confidence': 0.92, 'class': 'person'},
                {'bbox': [350, 200, 400, 270], 'confidence': 0.65, 'class': 'phone'},
            ],
        },
    ]

    # Test selection
    selected, stats = selector.select_optimal_subset(pool, budget=3)
    print(f"\nSelection stats: {stats}")
    assert len(selected) == 3, f"Expected 3, got {len(selected)}"

    # Test that selected set is diverse (at least 2 different relation types)
    rel_types = set(selector._get_relation_type(s) for s in selected)
    print(f"Relation types in selected: {rel_types}")
    assert len(rel_types) >= 2, "Selected subset should be diverse"

    # Test deduplication
    filtered = selector.filter_redundant_samples(pool, min_distance=0.05)
    print(f"\nDeduplication: {len(pool)} -> {len(filtered)}")

    # Test edge cases
    empty_selected, empty_stats = selector.select_optimal_subset([], budget=5)
    assert len(empty_selected) == 0

    single_selected, single_stats = selector.select_optimal_subset([pool[0]], budget=5)
    assert len(single_selected) == 1

    print("✅ All GreedySubsetSelector tests passed!")
