# File: RL/active_learning.py
"""
Active Learning Module for VRD Pipeline.

Sử dụng kết quả uncertainty estimation để chọn relationships cần được tập trung
sinh thêm ảnh synthetic. Thay vì sinh đều cho tất cả relationships (uniform),
Active Learning ưu tiên các relationships mà model **không chắc chắn nhất**
→ tối đa hóa lượng thông tin thu được từ mỗi ảnh synthetic.

Strategies:
    - uncertainty: Chỉ dùng MC Dropout uncertainty scores
    - performance: Chỉ dùng F1 hiện tại
    - combined: Kết hợp uncertainty + F1 + tail_weight (mặc định)
    - random: Random sampling (baseline)

Author: Auto-generated for DATN project
"""

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from RL.uncertainty_estimator import UncertaintyEstimator
except ModuleNotFoundError:
    from uncertainty_estimator import UncertaintyEstimator


class ActiveLearningSelector:
    """Chọn relationships cần sinh thêm ảnh dựa trên uncertainty-informed scoring."""

    def __init__(
        self,
        uncertainty_estimator: UncertaintyEstimator,
        strategy: str = 'combined',
    ):
        """
        Args:
            uncertainty_estimator: Instance của UncertaintyEstimator
            strategy: Chiến lược scoring: 'uncertainty', 'performance', 'combined', 'random'
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.strategy = strategy

        # Weights for combined strategy
        self.uncertainty_weight = 0.40
        self.performance_weight = 0.35
        self.tail_weight = 0.25

    # ------------------------------------------------------------------ #
    # Core: Score Relationships
    # ------------------------------------------------------------------ #

    def score_relationships(
        self,
        relationships: List[Dict[str, Any]],
        evaluation_samples: List[Dict[str, Any]],
        relationship_performance: Dict[str, Dict[str, Any]],
        tail_weights: Dict[str, float],
        decode_fn=None,
        transform_fn=None,
    ) -> List[Dict[str, Any]]:
        """
        Tính điểm cho từng relationship để quyết định mức ưu tiên sinh ảnh.

        Quy trình:
        1. Đo uncertainty cho từng relationship qua MC Dropout
        2. Lấy performance (F1) hiện tại từ relationship_performance
        3. Kết hợp thành acquisition score

        Args:
            relationships: List of relationship dicts
            evaluation_samples: Samples để MC Dropout đánh giá
            relationship_performance: Dict {rel_key: {avg_f1, history, ...}}
            tail_weights: Dict {relation_name: weight} cho long-tail
            decode_fn: Function decode RelTR outputs
            transform_fn: Function transform image to tensor

        Returns:
            List[Dict] with added fields: 'uncertainty_score', 'performance_score',
                'tail_score', 'acquisition_score'
        """
        acquisition_fn = self.get_acquisition_function(self.strategy)

        # Step 1: Estimate uncertainty cho evaluation samples
        print(f"[ActiveLearning] Estimating uncertainty for {len(evaluation_samples)} samples...")
        uncertainty_results = self.uncertainty_estimator.estimate_batch(
            evaluation_samples,
            decode_fn=decode_fn,
            transform_fn=transform_fn,
            max_samples=min(len(evaluation_samples), 15),
        )

        # Step 2: Map uncertainty scores to relationships
        scored_relationships = []
        for i, rel in enumerate(relationships):
            rel_key = self._get_relationship_key(rel)

            # Uncertainty score: Tính trung bình từ các samples liên quan
            uncertainty_score = self._get_relationship_uncertainty(
                rel_key, rel, uncertainty_results, evaluation_samples
            )

            # Performance score: Từ F1 tracking
            perf_data = relationship_performance.get(rel_key, {})
            avg_f1 = perf_data.get('avg_f1', 0.0)
            performance_score = 1.0 - avg_f1  # F1 thấp → cần tập trung hơn

            # Tail weight: Relationship hiếm → ưu tiên
            relation_name = rel.get('relation', '').lower().strip()
            tail_score = tail_weights.get(relation_name, 0.0)

            # Acquisition score
            acquisition_score = acquisition_fn(
                uncertainty_score, performance_score, tail_score
            )

            scored_rel = dict(rel)
            scored_rel.update({
                'rel_key': rel_key,
                'uncertainty_score': uncertainty_score,
                'performance_score': performance_score,
                'tail_score': tail_score,
                'acquisition_score': acquisition_score,
            })
            scored_relationships.append(scored_rel)

        # Sort by acquisition score (high → low)
        scored_relationships.sort(key=lambda x: x['acquisition_score'], reverse=True)

        return scored_relationships

    def _get_relationship_uncertainty(
        self,
        rel_key: str,
        relationship: Dict[str, Any],
        uncertainty_results: Dict[int, Dict[str, float]],
        evaluation_samples: List[Dict[str, Any]],
    ) -> float:
        """
        Tính uncertainty score cho 1 relationship dựa trên batch uncertainty results.

        Logic: Lấy uncertainty từ các samples có liên quan đến relationship này.
        Nếu không tìm thấy sample liên quan → trả về uncertainty trung bình.
        """
        if not uncertainty_results:
            return 0.5  # Neutral khi không có data

        relevant_scores = []
        subject = (relationship.get('subject', '') or '').lower().strip()
        obj = (relationship.get('object', '') or '').lower().strip()

        for sample_idx, unc_metrics in uncertainty_results.items():
            if sample_idx >= len(evaluation_samples):
                continue
            sample = evaluation_samples[sample_idx]

            # Kiểm tra sample có liên quan đến relationship không
            is_relevant = False

            # Check qua original_relationship
            orig_rel = sample.get('original_relationship', {}) or {}
            if orig_rel:
                sample_subject = (orig_rel.get('subject', '') or '').lower().strip()
                sample_object = (orig_rel.get('object', '') or '').lower().strip()
                if sample_subject == subject or sample_object == obj:
                    is_relevant = True

            # Check qua relationships list
            for sample_rel in (sample.get('relationships') or []):
                s = (sample_rel.get('subject', '') or '').lower().strip()
                o = (sample_rel.get('object', '') or '').lower().strip()
                if s == subject or o == obj:
                    is_relevant = True
                    break

            if is_relevant:
                relevant_scores.append(unc_metrics.get('uncertainty_score', 0.5))

        if relevant_scores:
            return sum(relevant_scores) / len(relevant_scores)

        # Fallback: Trả về uncertainty trung bình của tất cả samples
        all_scores = [m.get('uncertainty_score', 0.5) for m in uncertainty_results.values()]
        return sum(all_scores) / len(all_scores) if all_scores else 0.5

    # ------------------------------------------------------------------ #
    # Generation Plan
    # ------------------------------------------------------------------ #

    def create_generation_plan(
        self,
        scored_relationships: List[Dict[str, Any]],
        total_budget: int,
        min_per_rel: int = 1,
        max_per_rel: int = 10,
    ) -> Dict[str, int]:
        """
        Phân bổ budget (tổng số ảnh cần sinh) cho từng relationship dựa trên score.

        Algorithm: Proportional Allocation
        1. Mỗi rel được ít nhất min_per_rel ảnh
        2. Budget còn lại phân bổ theo tỷ lệ acquisition_score
        3. Cap tại max_per_rel

        Args:
            scored_relationships: Đã có acquisition_score
            total_budget: Tổng số ảnh cần sinh
            min_per_rel: Tối thiểu ảnh mỗi relationship
            max_per_rel: Tối đa ảnh mỗi relationship

        Returns:
            Dict {rel_key: num_images_to_generate}
        """
        n = len(scored_relationships)
        if n == 0:
            return {}

        # Ensure budget đủ cho minimum
        total_budget = max(total_budget, n * min_per_rel)

        plan: Dict[str, int] = {}

        # Phase 1: Mỗi rel nhận min_per_rel
        for scored_rel in scored_relationships:
            plan[scored_rel['rel_key']] = min_per_rel

        remaining_budget = total_budget - n * min_per_rel

        if remaining_budget <= 0:
            return plan

        # Phase 2: Phân bổ remaining theo acquisition_score
        total_score = sum(r['acquisition_score'] for r in scored_relationships)

        if total_score <= 0:
            # Equal distribution nếu tất cả score = 0
            extra_each = remaining_budget // n
            for scored_rel in scored_relationships:
                plan[scored_rel['rel_key']] += extra_each
            remaining_budget -= extra_each * n
            # Distribute phần dư cho top relationships
            for scored_rel in scored_relationships[:remaining_budget]:
                plan[scored_rel['rel_key']] += 1
        else:
            # Proportional allocation
            allocated = 0
            allocations = []
            for scored_rel in scored_relationships:
                proportion = scored_rel['acquisition_score'] / total_score
                raw_alloc = proportion * remaining_budget
                floor_alloc = int(math.floor(raw_alloc))
                allocations.append((scored_rel['rel_key'], floor_alloc, raw_alloc - floor_alloc))
                plan[scored_rel['rel_key']] += floor_alloc
                allocated += floor_alloc

            # Distribute remainder using largest fractional part
            remaining = remaining_budget - allocated
            allocations.sort(key=lambda x: x[2], reverse=True)
            for j in range(min(remaining, len(allocations))):
                plan[allocations[j][0]] += 1

        # Phase 3: Cap at max_per_rel
        for key in plan:
            plan[key] = min(plan[key], max_per_rel)

        return plan

    # ------------------------------------------------------------------ #
    # Acquisition Functions
    # ------------------------------------------------------------------ #

    def get_acquisition_function(
        self, strategy: str
    ) -> Callable[[float, float, float], float]:
        """
        Trả về hàm acquisition phù hợp với strategy.

        Args:
            strategy: 'uncertainty', 'performance', 'combined', 'random'

        Returns:
            Function(uncertainty, performance, tail) -> float
        """
        strategies = {
            'uncertainty': self._acquisition_uncertainty,
            'performance': self._acquisition_performance,
            'combined': self._acquisition_combined,
            'random': self._acquisition_random,
        }
        return strategies.get(strategy, self._acquisition_combined)

    @staticmethod
    def _acquisition_uncertainty(
        uncertainty: float, performance: float, tail: float
    ) -> float:
        """Chỉ dùng uncertainty score."""
        return uncertainty

    @staticmethod
    def _acquisition_performance(
        uncertainty: float, performance: float, tail: float
    ) -> float:
        """Chỉ dùng inverse F1 (performance gap)."""
        return performance

    def _acquisition_combined(
        self, uncertainty: float, performance: float, tail: float
    ) -> float:
        """Kết hợp tất cả signals."""
        return (
            self.uncertainty_weight * uncertainty
            + self.performance_weight * performance
            + self.tail_weight * min(tail, 1.0)
        )

    @staticmethod
    def _acquisition_random(
        uncertainty: float, performance: float, tail: float
    ) -> float:
        """Random scoring (baseline)."""
        return random.random()

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def log_selection_reasoning(
        self,
        generation_plan: Dict[str, int],
        scored_relationships: List[Dict[str, Any]],
    ) -> None:
        """In thông tin chi tiết tại sao mỗi relationship được sinh N ảnh."""
        print("\n" + "=" * 70)
        print("[ActiveLearning] 📊 Generation Plan Reasoning")
        print("=" * 70)
        print(f"  Strategy: {self.strategy}")
        print(f"  Total budget: {sum(generation_plan.values())} images")
        print(f"  Relationships: {len(generation_plan)}")
        print("-" * 70)
        print(f"  {'Relationship':<40} {'Score':>6} {'Unc':>5} {'Perf':>5} {'Tail':>5} {'Imgs':>5}")
        print("-" * 70)

        for scored_rel in scored_relationships:
            rel_key = scored_rel['rel_key']
            subject = scored_rel.get('subject', '?')
            relation = scored_rel.get('relation', '?')
            obj = scored_rel.get('object', '?')
            label = f"{subject} {relation} {obj}"
            if len(label) > 38:
                label = label[:35] + "..."

            num_imgs = generation_plan.get(rel_key, 0)
            print(
                f"  {label:<40} "
                f"{scored_rel['acquisition_score']:>6.3f} "
                f"{scored_rel['uncertainty_score']:>5.2f} "
                f"{scored_rel['performance_score']:>5.2f} "
                f"{scored_rel['tail_score']:>5.2f} "
                f"{num_imgs:>5d}"
            )

        print("=" * 70 + "\n")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_relationship_key(relationship: Dict[str, Any]) -> str:
        """Tạo key duy nhất cho relationship (tương thích với reinforcement_learning.py)."""
        subject = (relationship.get('subject', '') or '').lower().strip()
        relation = (relationship.get('relation', '') or '').lower().strip()
        obj = (relationship.get('object', '') or '').lower().strip()
        return f"{subject}|{relation}|{obj}"


# ============ TESTING ============
if __name__ == "__main__":
    print("Testing ActiveLearningSelector module...")

    # Create dummy uncertainty estimator (no model needed for testing plan logic)
    class MockUncertaintyEstimator:
        def estimate_batch(self, *args, **kwargs):
            return {
                0: {'uncertainty_score': 0.8, 'entropy': 0.7, 'mutual_information': 0.5, 'variation_ratio': 0.3, 'mean_confidence': 0.4},
                1: {'uncertainty_score': 0.3, 'entropy': 0.2, 'mutual_information': 0.1, 'variation_ratio': 0.1, 'mean_confidence': 0.8},
                2: {'uncertainty_score': 0.6, 'entropy': 0.5, 'mutual_information': 0.3, 'variation_ratio': 0.2, 'mean_confidence': 0.6},
            }

    mock_estimator = MockUncertaintyEstimator()
    selector = ActiveLearningSelector(mock_estimator, strategy='combined')

    # Test relationships
    relationships = [
        {'subject': 'person', 'relation': 'riding', 'object': 'horse'},
        {'subject': 'dog', 'relation': 'sitting on', 'object': 'chair'},
        {'subject': 'person', 'relation': 'holding', 'object': 'phone'},
    ]

    evaluation_samples = [
        {'original_relationship': relationships[0]},
        {'original_relationship': relationships[1]},
        {'original_relationship': relationships[2]},
    ]

    performance = {
        'person|riding|horse': {'avg_f1': 0.8},
        'dog|sitting on|chair': {'avg_f1': 0.2},
        'person|holding|phone': {'avg_f1': 0.5},
    }

    tail_weights = {'riding': 0.1, 'sitting on': 0.8, 'holding': 0.3}

    # Score relationships
    scored = selector.score_relationships(
        relationships, evaluation_samples, performance, tail_weights
    )
    print("\nScored relationships:")
    for s in scored:
        print(f"  {s['rel_key']}: acq={s['acquisition_score']:.3f}, unc={s['uncertainty_score']:.2f}, perf={s['performance_score']:.2f}")

    # Create generation plan
    plan = selector.create_generation_plan(scored, total_budget=15, min_per_rel=2, max_per_rel=8)
    print(f"\nGeneration plan (budget=15): {plan}")
    assert sum(plan.values()) <= 15 + len(relationships), f"Budget exceeded: {sum(plan.values())}"

    # Log reasoning
    selector.log_selection_reasoning(plan, scored)

    # Test edge cases
    plan_small = selector.create_generation_plan(scored, total_budget=3, min_per_rel=1, max_per_rel=1)
    print(f"Small budget plan (budget=3): {plan_small}")

    plan_empty = selector.create_generation_plan([], total_budget=10)
    assert plan_empty == {}, "Empty plan should return empty dict"

    print("✅ All ActiveLearningSelector tests passed!")
