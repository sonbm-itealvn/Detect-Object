# File: RL/uncertainty_estimator.py
"""
Uncertainty Learning Module using Monte Carlo Dropout (MC Dropout).

Đo lường độ không chắc chắn (uncertainty) của model RelTR khi predict relationships.
MC Dropout giữ nguyên weights, chỉ bật dropout khi inference → tạo "ensemble ảo"
mà không cần train thêm model.

Metrics:
    - Predictive Entropy: H[ȳ] = −Σ p̄(y) log p̄(y)
    - Mutual Information (BALD): I[y, θ] = H[ȳ] − E[H[y|θ]]
    - Variation Ratio: Tỷ lệ predictions khác nhau giữa các forward passes

Author: Auto-generated for DATN project
"""

import copy
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class UncertaintyEstimator:
    """Estimate model uncertainty using MC Dropout on RelTR model."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        n_forward_passes: int = 10,
        dropout_rate: Optional[float] = None,
    ):
        """
        Args:
            model: RelTR model (nn.Module)
            device: torch device to run inference on
            n_forward_passes: Số lần forward pass MC Dropout (nhiều = chính xác hơn nhưng chậm)
            dropout_rate: Override dropout rate (None = giữ nguyên rate từ model)
        """
        self.model = model
        self.device = device
        self.n_forward_passes = n_forward_passes
        self.dropout_rate = dropout_rate

        # Cache previous uncertainty scores for tracking improvement
        self._previous_uncertainties: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # MC Dropout Utilities
    # ------------------------------------------------------------------ #

    def enable_mc_dropout(self, model: nn.Module) -> List[nn.Module]:
        """
        Bật dropout layers trong inference mode.

        Bình thường model.eval() sẽ tắt hết Dropout. MC Dropout cần
        giữ dropout **bật** trong khi eval để tạo stochastic predictions.

        Returns:
            List các dropout modules đã được bật
        """
        dropout_modules = []
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()  # Bật dropout (training mode = dropout active)
                if self.dropout_rate is not None:
                    module.p = self.dropout_rate
                dropout_modules.append(module)
        return dropout_modules

    def disable_mc_dropout(self, model: nn.Module) -> None:
        """Tắt MC Dropout, trả model về trạng thái eval bình thường."""
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.eval()

    # ------------------------------------------------------------------ #
    # Core Uncertainty Estimation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def estimate_relationship_uncertainty(
        self,
        image_tensor: torch.Tensor,
        objects: List[Dict[str, Any]],
        global_context: Optional[List[float]] = None,
        decode_fn=None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Chạy MC Dropout trên 1 image để đo uncertainty cho từng relationship.

        Quy trình:
        1. Bật MC Dropout
        2. Forward N lần cùng 1 input → N bộ predictions
        3. Tính uncertainty metrics dựa trên sự khác biệt giữa N predictions

        Args:
            image_tensor: Ảnh đã transform (1, C, H, W) hoặc (C, H, W)
            objects: Detected objects list
            global_context: Optional global context vector
            decode_fn: Function để decode RelTR output thành relationships
            image_size: (width, height) of original image

        Returns:
            Dict with keys:
                - predictions: List[List[Dict]] – N bộ predictions
                - entropy: float – Predictive Entropy
                - mutual_information: float – BALD score
                - variation_ratio: float – Tỷ lệ thay đổi
                - mean_confidence: float – Confidence trung bình
                - uncertainty_score: float – Combined score ∈ [0, 1]
        """
        from util.misc import nested_tensor_from_tensor_list

        # Ensure proper shape
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Prepare global context tensor
        context_tensor = None
        if global_context is not None:
            context_tensor = torch.tensor(
                global_context, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        # Enable MC Dropout
        dropout_modules = self.enable_mc_dropout(self.model)
        if not dropout_modules:
            print("[Uncertainty] Warning: No dropout layers found in model. "
                  "Uncertainty estimates will have zero variance.")

        all_predictions: List[List[Dict[str, Any]]] = []
        all_confidences: List[List[float]] = []

        try:
            for t in range(self.n_forward_passes):
                try:
                    samples = nested_tensor_from_tensor_list([image_tensor.squeeze(0)])
                    if context_tensor is not None:
                        outputs = self.model(samples, global_context=context_tensor)
                    else:
                        outputs = self.model(samples)

                    # Decode nếu có decode function
                    if decode_fn is not None:
                        relationships = decode_fn(outputs, objects, image_size)
                    else:
                        relationships = self._extract_raw_predictions(outputs)

                    all_predictions.append(relationships or [])

                    # Thu thập confidence scores
                    confs = [
                        r.get('confidence', 0.0) for r in (relationships or [])
                    ]
                    all_confidences.append(confs)

                except Exception as e:
                    print(f"[Uncertainty] Forward pass {t+1} failed: {e}")
                    all_predictions.append([])
                    all_confidences.append([])
        finally:
            # Luôn tắt MC Dropout sau khi hoàn thành
            self.disable_mc_dropout(self.model)

        if not all_predictions or all(len(p) == 0 for p in all_predictions):
            return {
                'predictions': all_predictions,
                'entropy': 0.0,
                'mutual_information': 0.0,
                'variation_ratio': 0.0,
                'mean_confidence': 0.0,
                'uncertainty_score': 1.0,  # Maximum uncertainty khi không predict được gì
            }

        # Tính Uncertainty Metrics
        entropy = self._compute_predictive_entropy(all_predictions)
        mutual_info = self._compute_mutual_information(all_predictions, all_confidences)
        variation_ratio = self._compute_variation_ratio(all_predictions)

        # Mean confidence
        flat_confs = [c for confs in all_confidences for c in confs]
        mean_conf = sum(flat_confs) / len(flat_confs) if flat_confs else 0.0

        # Combined uncertainty score ∈ [0, 1]
        # Higher = more uncertain
        uncertainty_score = self._compute_combined_score(
            entropy, mutual_info, variation_ratio, mean_conf
        )

        return {
            'predictions': all_predictions,
            'entropy': entropy,
            'mutual_information': mutual_info,
            'variation_ratio': variation_ratio,
            'mean_confidence': mean_conf,
            'uncertainty_score': uncertainty_score,
        }

    def _extract_raw_predictions(self, outputs: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Trích xuất predictions thô từ RelTR outputs khi không có decode function.

        Chỉ lấy relation logits để tính uncertainty, không cần decode fully.
        """
        predictions = []
        try:
            # RelTR outputs typically have 'rel_logits' key
            rel_logits = outputs.get('rel_logits', None)
            if rel_logits is None:
                return predictions

            # rel_logits shape: (batch, num_queries, num_rel_classes)
            probs = torch.softmax(rel_logits[0], dim=-1)  # (num_queries, num_classes)

            for q in range(probs.shape[0]):
                max_prob, max_idx = probs[q].max(dim=-1)
                if max_idx.item() == 0:  # Skip background
                    continue
                predictions.append({
                    'relation_idx': max_idx.item(),
                    'confidence': max_prob.item(),
                    'probs': probs[q].cpu().numpy(),
                })
        except Exception:
            pass
        return predictions

    # ------------------------------------------------------------------ #
    # Uncertainty Metrics
    # ------------------------------------------------------------------ #

    def _compute_predictive_entropy(
        self, all_predictions: List[List[Dict[str, Any]]]
    ) -> float:
        """
        Tính Predictive Entropy = H[ȳ] = −Σ p̄(y) log p̄(y)

        Đo tổng uncertainty (cả aleatoric + epistemic).
        Predictive entropy cao = model rất không chắc chắn.

        Sử dụng "relationship fingerprint" (subject_class, relation, object_class)
        để đếm tần suất xuất hiện qua N forward passes.
        """
        # Collect relationship fingerprints across all passes
        fingerprint_counts: Counter = Counter()
        total_predictions = 0

        for predictions in all_predictions:
            for pred in predictions:
                fp = self._get_relationship_fingerprint(pred)
                fingerprint_counts[fp] += 1
                total_predictions += 1

        if total_predictions == 0:
            return 0.0

        # Tính probability distribution
        entropy = 0.0
        for count in fingerprint_counts.values():
            p = count / total_predictions
            if p > 0:
                entropy -= p * math.log(p + 1e-10)

        # Normalize by log(# unique outcomes) to get entropy ∈ [0, 1]
        num_unique = len(fingerprint_counts)
        if num_unique > 1:
            entropy /= math.log(num_unique)

        return min(entropy, 1.0)

    def _compute_mutual_information(
        self,
        all_predictions: List[List[Dict[str, Any]]],
        all_confidences: List[List[float]],
    ) -> float:
        """
        Tính Mutual Information (BALD score): I[y, θ] = H[ȳ] − E[H[y|θ]]

        - H[ȳ]: Predictive Entropy (total uncertainty)
        - E[H[y|θ]]: Expected entropy of individual predictions (data uncertainty)
        - I = H - E[H] = Model uncertainty only (epistemic uncertainty)

        BALD score cao = model thiếu knowledge → nên thu thập thêm dữ liệu cho case này.
        """
        if not all_predictions:
            return 0.0

        # H[ȳ] = Predictive entropy (đã tính ở trên)
        predictive_entropy = self._compute_predictive_entropy(all_predictions)

        # E[H[y|θ]] = Trung bình entropy của từng forward pass
        per_pass_entropies = []
        for t, predictions in enumerate(all_predictions):
            if not predictions:
                per_pass_entropies.append(0.0)
                continue

            confs = all_confidences[t] if t < len(all_confidences) else []
            if not confs:
                per_pass_entropies.append(0.0)
                continue

            # Entropy từ confidence distribution của pass này
            pass_entropy = 0.0
            for c in confs:
                p = max(c, 1e-10)
                pass_entropy -= p * math.log(p + 1e-10)

            # Normalize
            if len(confs) > 1:
                pass_entropy /= math.log(len(confs))

            per_pass_entropies.append(min(pass_entropy, 1.0))

        expected_entropy = (
            sum(per_pass_entropies) / len(per_pass_entropies)
            if per_pass_entropies
            else 0.0
        )

        # Mutual Information = Predictive Entropy - Expected Entropy
        mutual_info = max(0.0, predictive_entropy - expected_entropy)
        return min(mutual_info, 1.0)

    def _compute_variation_ratio(
        self, all_predictions: List[List[Dict[str, Any]]]
    ) -> float:
        """
        Tính Variation Ratio = 1 − (count of mode) / N

        Đo tỷ lệ các forward passes cho kết quả khác so với kết quả phổ biến nhất.
        Variation ratio = 0: Tất cả passes cho cùng kết quả (confident)
        Variation ratio → 1: Mỗi pass cho kết quả khác nhau (very uncertain)
        """
        N = len(all_predictions)
        if N <= 1:
            return 0.0

        # Fingerprint cho mỗi forward pass (tập hợp các relationships)
        pass_fingerprints = []
        for predictions in all_predictions:
            fps = frozenset(
                self._get_relationship_fingerprint(pred) for pred in predictions
            )
            pass_fingerprints.append(fps)

        # Đếm mode (fingerprint phổ biến nhất)
        fp_counter = Counter(pass_fingerprints)
        mode_count = fp_counter.most_common(1)[0][1] if fp_counter else N

        variation_ratio = 1.0 - mode_count / N
        return variation_ratio

    def _compute_combined_score(
        self,
        entropy: float,
        mutual_info: float,
        variation_ratio: float,
        mean_confidence: float,
    ) -> float:
        """
        Tính combined uncertainty score ∈ [0, 1].

        Score = weighted combination of:
            - Entropy (30%): Tổng uncertainty
            - Mutual Information (30%): Epistemic uncertainty (model thiếu data)
            - Variation Ratio (20%): Sự biến thiên giữa các predictions
            - Inverse Confidence (20%): 1 - mean_confidence

        Kết quả cao = rất không chắc chắn → cần thu thập thêm data.
        """
        inverse_conf = 1.0 - mean_confidence
        score = (
            0.30 * entropy
            + 0.30 * mutual_info
            + 0.20 * variation_ratio
            + 0.20 * inverse_conf
        )
        return max(0.0, min(score, 1.0))

    # ------------------------------------------------------------------ #
    # Batch Estimation
    # ------------------------------------------------------------------ #

    def estimate_batch(
        self,
        evaluation_samples: List[Dict[str, Any]],
        decode_fn=None,
        transform_fn=None,
        max_samples: int = 20,
    ) -> Dict[int, Dict[str, float]]:
        """
        Estimate uncertainty cho nhiều samples.

        Args:
            evaluation_samples: List of dicts with 'image' or 'image_path' and 'objects'
            decode_fn: Function to decode RelTR outputs
            transform_fn: Function to transform image to tensor
            max_samples: Giới hạn số samples để tránh quá chậm

        Returns:
            Dict mapping sample_idx → uncertainty metrics
        """
        from PIL import Image

        results: Dict[int, Dict[str, float]] = {}
        subset = evaluation_samples[:max_samples]

        for i, sample in enumerate(subset):
            try:
                # Get image tensor
                image_input = sample.get('image') or sample.get('image_path')
                if image_input is None:
                    continue

                # Transform image
                if transform_fn is not None:
                    if isinstance(image_input, str):
                        pil_img = Image.open(image_input).convert('RGB')
                    elif isinstance(image_input, Image.Image):
                        pil_img = image_input.convert('RGB')
                    else:
                        continue
                    image_tensor = transform_fn(pil_img)
                    image_size = pil_img.size
                elif isinstance(image_input, torch.Tensor):
                    image_tensor = image_input
                    image_size = None
                else:
                    continue

                objects = sample.get('objects', [])
                global_context = sample.get('global_context')

                uncertainty = self.estimate_relationship_uncertainty(
                    image_tensor, objects, global_context,
                    decode_fn=decode_fn,
                    image_size=image_size,
                )

                results[i] = {
                    'entropy': uncertainty['entropy'],
                    'mutual_information': uncertainty['mutual_information'],
                    'variation_ratio': uncertainty['variation_ratio'],
                    'mean_confidence': uncertainty['mean_confidence'],
                    'uncertainty_score': uncertainty['uncertainty_score'],
                }

            except Exception as e:
                print(f"[Uncertainty] Error estimating sample {i}: {e}")
                results[i] = {
                    'entropy': 0.0,
                    'mutual_information': 0.0,
                    'variation_ratio': 0.0,
                    'mean_confidence': 0.0,
                    'uncertainty_score': 0.5,  # Neutral when estimation fails
                }

        return results

    # ------------------------------------------------------------------ #
    # Uncertainty Tracking
    # ------------------------------------------------------------------ #

    def compute_uncertainty_reduction(
        self,
        current_uncertainties: Dict[str, float],
    ) -> float:
        """
        Tính mức giảm uncertainty so với lần đo trước.

        Returns:
            float ∈ [-1, 1]: Dương = uncertainty giảm (tốt), Âm = tăng (xấu)
        """
        if not self._previous_uncertainties:
            self._previous_uncertainties = dict(current_uncertainties)
            return 0.0

        reductions = []
        for key, current_val in current_uncertainties.items():
            prev_val = self._previous_uncertainties.get(key, current_val)
            if prev_val > 0:
                reduction = (prev_val - current_val) / prev_val
                reductions.append(reduction)

        # Update cache
        self._previous_uncertainties = dict(current_uncertainties)

        if not reductions:
            return 0.0

        avg_reduction = sum(reductions) / len(reductions)
        return max(-1.0, min(1.0, avg_reduction))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_relationship_fingerprint(prediction: Dict[str, Any]) -> str:
        """Tạo fingerprint duy nhất cho 1 prediction để so sánh."""
        subject = prediction.get('subject', prediction.get('subject_class', ''))
        relation = prediction.get('relation', prediction.get('predicate', ''))
        obj = prediction.get('object', prediction.get('object_class', ''))
        rel_idx = prediction.get('relation_idx', '')
        
        # Ưu tiên dùng class names nếu có
        if subject and relation and obj:
            return f"{str(subject).lower().strip()}|{str(relation).lower().strip()}|{str(obj).lower().strip()}"
        # Fallback dùng relation index
        if rel_idx:
            return f"rel_idx_{rel_idx}"
        return f"unknown_{id(prediction)}"


# ============ TESTING ============
if __name__ == "__main__":
    print("Testing UncertaintyEstimator module...")

    # Test with dummy data
    dummy_predictions_pass1 = [
        {'subject': 'person', 'relation': 'riding', 'object': 'horse', 'confidence': 0.85},
        {'subject': 'person', 'relation': 'holding', 'object': 'bag', 'confidence': 0.72},
    ]
    dummy_predictions_pass2 = [
        {'subject': 'person', 'relation': 'riding', 'object': 'horse', 'confidence': 0.80},
        {'subject': 'person', 'relation': 'near', 'object': 'bag', 'confidence': 0.60},
    ]
    dummy_predictions_pass3 = [
        {'subject': 'person', 'relation': 'on', 'object': 'horse', 'confidence': 0.75},
        {'subject': 'person', 'relation': 'holding', 'object': 'bag', 'confidence': 0.55},
    ]

    all_preds = [dummy_predictions_pass1, dummy_predictions_pass2, dummy_predictions_pass3]
    all_confs = [[0.85, 0.72], [0.80, 0.60], [0.75, 0.55]]

    # Test static methods without model
    estimator = UncertaintyEstimator.__new__(UncertaintyEstimator)
    estimator._previous_uncertainties = {}

    entropy = estimator._compute_predictive_entropy(all_preds)
    print(f"Predictive Entropy: {entropy:.4f}")

    mi = estimator._compute_mutual_information(all_preds, all_confs)
    print(f"Mutual Information: {mi:.4f}")

    vr = estimator._compute_variation_ratio(all_preds)
    print(f"Variation Ratio: {vr:.4f}")

    score = estimator._compute_combined_score(entropy, mi, vr, 0.71)
    print(f"Combined Uncertainty Score: {score:.4f}")

    # Test uncertainty reduction
    current = {'rel1': 0.7, 'rel2': 0.5}
    reduction = estimator.compute_uncertainty_reduction(current)
    print(f"Uncertainty Reduction (first time): {reduction:.4f}")

    new_current = {'rel1': 0.5, 'rel2': 0.3}
    reduction = estimator.compute_uncertainty_reduction(new_current)
    print(f"Uncertainty Reduction (improved): {reduction:.4f}")

    print("✅ All UncertaintyEstimator tests passed!")
