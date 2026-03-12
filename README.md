# Hình thức hóa Toán học Hệ thống Phát hiện Quan hệ Thị giác (VRD)

> **Tài liệu khoa học** mô tả chi tiết kiến trúc, thuật toán, công thức toán học và chứng minh tính đúng đắn của toàn bộ hệ thống Visual Relationship Detection Pipeline.

**Tham chiếu chính:**
- Cong et al., "RelTR: Relation Transformer for Scene Graph Generation," *ACMMM 2022*
- Redmon et al., "You Only Look Once," *CVPR 2016*; Jocher et al., "Ultralytics YOLOv11," 2024
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision (CLIP)," *ICML 2021*
- Gal & Ghahramani, "Dropout as a Bayesian Approximation," *ICML 2016*
- Nemhauser, Wolsey & Fisher, "An analysis of approximations for maximizing submodular set functions," *Math. Programming 1978*
- Watkins & Dayan, "Q-Learning," *Machine Learning 1992*

---

## Mục lục

1. [Ký hiệu và Quy ước](#1-ký-hiệu-và-quy-ước)
2. [Tổng quan Kiến trúc](#2-tổng-quan-kiến-trúc)
3. [Object Detection – YOLOv11](#3-object-detection--yolov11)
4. [Zero-shot Classification – CLIP](#4-zero-shot-classification--clip)
5. [ROI Feature Extraction – RoIAlign](#5-roi-feature-extraction--roialign)
6. [Scene Graph Generation – RelTR Transformer](#6-scene-graph-generation--reltr-transformer)
7. [Spatial-Semantic Validation](#7-spatial-semantic-validation)
8. [LLM Open-Vocabulary Enhancement](#8-llm-open-vocabulary-enhancement)
9. [Video Processing Pipeline](#9-video-processing-pipeline)
10. [Safety Classification System](#10-safety-classification-system)
11. [Deep Q-Network (DQN) Agent](#11-deep-q-network-dqn-agent)
12. [Adaptive Reward Function](#12-adaptive-reward-function)
13. [MC Dropout Uncertainty Estimation](#13-mc-dropout-uncertainty-estimation)
14. [Active Learning Acquisition](#14-active-learning-acquisition)
15. [Greedy Submodular Maximization](#15-greedy-submodular-maximization)
16. [Synthetic Data Generation & Auto-Annotation](#16-synthetic-data-generation--auto-annotation)
17. [Tổng kết End-to-End Pipeline](#17-tổng-kết-end-to-end-pipeline)

---

## 1. Ký hiệu và Quy ước

| Ký hiệu | Ý nghĩa | Miền |
|---|---|---|
| $I$ | Ảnh đầu vào | $\mathbb{R}^{H \times W \times 3}$ |
| $\mathcal{O} = \{o_i\}_{i=1}^{N}$ | Tập đối tượng phát hiện | $o_i = (b_i, c_i, s_i)$ |
| $b_i = (x_1, y_1, x_2, y_2)$ | Bounding box (pixel coords) | $\mathbb{R}^4$ |
| $c_i$ | Class label | $\{1, \ldots, C\}$ |
| $s_i$ | Confidence score | $[0, 1]$ |
| $\mathcal{R} = \{r_k\}$ | Tập quan hệ (relationships) | $r_k = (s_k, p_k, o_k, \sigma_k)$ |
| $(s_k, p_k, o_k)$ | Triplet: (subject, predicate, object) | |
| $\sigma_k$ | Confidence của quan hệ | $[0, 1]$ |
| $F \in \mathbb{R}^{C' \times H' \times W'}$ | Feature map backbone | |
| $g \in \mathbb{R}^{C'}$ | Global context vector | |
| $d_{\text{model}}$ | Hidden dimension (= 256) | $\mathbb{N}$ |
| $Q(s, a; \theta)$ | Q-value function | $\mathbb{R}$ |
| $\varepsilon$ | Exploration rate | $[0, 1]$ |
| $\gamma$ | Discount factor (= 0.95) | $(0, 1)$ |
| $U(r)$ | Uncertainty score | $[0, 1]$ |
| $f(S)$ | Submodular objective | $\mathbb{R}_{\geq 0}$ |
| $\mathfrak{S}_N$ | Permutation group of N elements | |

---

## 2. Tổng quan Kiến trúc

Hệ thống gồm 2 luồng xử lý chính: **Inference Pipeline** (suy luận thời gian thực) và **Data Loop** (vòng lặp cải thiện mô hình liên tục).

```
╔══════════════════════════ INFERENCE PIPELINE ═══════════════════════════╗
║ Image I ∈ ℝ^{H×W×3}                                                   ║
║   │                                                                     ║
║   ├──→ YOLOv11 ──→ {(bᵢ, cᵢ, sᵢ)}  ──→ CLIP verify ──→ 𝒪 (objects) ║
║   │      └── Fire Model (parallel)                                      ║
║   │                                                                     ║
║   ├──→ Backbone Hook ──→ F ∈ ℝ^{C'×H'×W'} ──→ RoIAlign ──→ vᵢ        ║
║   │                        └── GAP ──→ g ∈ ℝ^{C'} (global context)     ║
║   │                                                                     ║
║   └──→ RelTR(I, g) ──→ {(sₖ, pₖ, oₖ, σₖ)}                           ║
║          │                                                              ║
║          ├──→ Spatial Validation ──→ Semantic Validation                ║
║          └──→ LLM Enhancement (low-conf / missing pairs)               ║
║                │                                                        ║
║                └──→ Safety Classifier (3-tier) ──→ Scene Graph G       ║
╚═════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════ DATA LOOP (RL) ═══════════════════════════════╗
║ DQN Agent ──→ decide_action(s) ──→ aₜ (num_variations)                ║
║   │                                                                     ║
║   ├──→ Active Learning ──→ Acquisition Score α(r)                      ║
║   ├──→ Stable Diffusion ──→ Synthetic Images                           ║
║   ├──→ Greedy Submodular ──→ Optimal Subset S*                         ║
║   ├──→ Auto-Annotation ──→ Labeled Dataset D                           ║
║   ├──→ Fine-tune YOLO + RelTR                                          ║
║   └──→ Evaluate ──→ Reward rₜ ──→ Q-Network Update                    ║
╚═════════════════════════════════════════════════════════════════════════╝
```

**Source files:**

| Module | File(s) | Dòng code |
|---|---|---|
| Object Detection | `detect_objects.py` | ~490 |
| Relationship Inference | `boundingbox_objects.py` | ~1005 |
| Video Pipeline | `video_relation_pipeline.py` | ~1200 |
| RelTR Model | `models/reltr.py`, `models/transformer.py`, `models/matcher.py`, `models/backbone.py` | ~1143 |
| RL Training | `RL/reinforcement_learning.py` | ~3665 |
| Uncertainty | `RL/uncertainty_estimator.py` | ~570 |
| Active Learning | `RL/active_learning.py` | ~439 |
| Approximation | `RL/approximation_algorithm.py` | ~520 |
| Image Generation | `RL/ai_images_generator.py` | ~500 |
| Auto-Annotation | `RL/auto_annotator.py` | ~495 |
| Visual Features | `RL/visual_features.py` | ~460 |
| LLM Predictor | `RL/llm_relationship_predictor.py` | ~584 |
| Safety System | `RL/safety_classifier.py`, `RL/llm_safety_analyzer.py`, `RL/local_rules_db.py` | ~563+ |
| Orchestration | `RL/rl_enhancement.py` | ~563 |

---

## 3. Object Detection – YOLOv11

**File:** `detect_objects.py`

### 3.1 Định nghĩa hình thức

**Định nghĩa 3.1 (Object Detection Function).** Hàm phát hiện đối tượng $\mathcal{D}: \mathbb{R}^{H \times W \times 3} \rightarrow \mathcal{P}(\mathcal{B} \times \mathcal{C} \times [0,1])$ ánh xạ ảnh đầu vào sang tập hợp các phát hiện, trong đó $\mathcal{B} = \{(x_1, y_1, x_2, y_2) \in \mathbb{R}^4 : x_1 < x_2, y_1 < y_2\}$ là không gian bounding box, $\mathcal{C} = \{1, \ldots, C\}$ là tập nhãn lớp.

### 3.2 Kiến trúc YOLOv11

YOLOv11 (You Only Look Once v11) thuộc họ single-stage detector, xử lý toàn bộ ảnh trong một lần forward pass.

**Backbone (CSPDarknet):** Trích xuất feature maps tại nhiều scale.

**Neck (FPN + PAN):** Kết hợp features đa tầng:

$$F_{\text{fpn}}^l = \text{Conv}(\text{Upsample}(F^{l+1}) \oplus F^l), \quad l \in \{3, 4, 5\}$$

**Head:** Tại mỗi vị trí grid $(i, j)$ ở scale $l$, dự đoán:

$$\hat{y}_{i,j,l} = (\hat{t}_x, \hat{t}_y, \hat{t}_w, \hat{t}_h, \hat{p}_{\text{obj}}, \hat{p}_{c_1}, \ldots, \hat{p}_{c_C})$$

**Giải mã bounding box (anchor-free):**

$$\begin{aligned}
b_x &= 2\sigma(\hat{t}_x) - 0.5 + c_x \\
b_y &= 2\sigma(\hat{t}_y) - 0.5 + c_y \\
b_w &= (2\sigma(\hat{t}_w))^2 \cdot a_w \\
b_h &= (2\sigma(\hat{t}_h))^2 \cdot a_h
\end{aligned}$$

trong đó $(c_x, c_y)$ là offset grid cell, $(a_w, a_h)$ là anchor dimensions, $\sigma(\cdot)$ là sigmoid function.

**Objectness score cuối cùng:**

$$s_{\text{final}}(i, j, l) = \sigma(\hat{p}_{\text{obj}}) \cdot \max_{c \in \mathcal{C}} \sigma(\hat{p}_c)$$

### 3.3 Non-Maximum Suppression (NMS)

**Định nghĩa 3.2 (IoU).** Cho hai boxes $A, B \in \mathcal{B}$:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}$$

**Thuộc tính:** (i) $0 \leq \text{IoU}(A,B) \leq 1$; (ii) $\text{IoU}(A,B) = 1 \Leftrightarrow A = B$; (iii) $\text{IoU}(A,B) = \text{IoU}(B,A)$ (đối xứng).

**Thuật toán 3.1 (Greedy NMS):**

```
Input:  Detections D = {(bᵢ, cᵢ, sᵢ)}, ngưỡng IoU θ_nms = 0.45
Output: Filtered detections D'

1. Sắp xếp D theo sᵢ giảm dần
2. D' ← ∅
3. while D ≠ ∅:
4.   d* ← D[0] (detection với score cao nhất)
5.   D' ← D' ∪ {d*}
6.   D ← D \ {d*}
7.   for each d ∈ D:
8.     if class(d) = class(d*) AND IoU(box(d), box(d*)) > θ_nms:
9.       D ← D \ {d}   // Suppress
10. return D'
```

**Độ phức tạp:** $O(N^2)$ với $N$ detections, có thể giảm xuống $O(N \log N)$ với R-tree spatial indexing.

### 3.4 Dual-Model Detection

Hệ thống sử dụng 2 YOLO models song song:

$$\mathcal{D}_{\text{merged}}(I) = \mathcal{D}_{\text{COCO}}(I) \cup \mathcal{D}_{\text{fire}}(I)$$

**Implementation** (`detect_objects.py`, line 289-350):
- `yolo_model`: Fine-tuned trên COCO 80 classes + custom classes
- `fire_model`: Chuyên biệt phát hiện fire/smoke (2 classes)
- Merge strategy: Khi fire box overlap với COCO box (IoU > threshold), ưu tiên giữ fire detection (safety-first)

### 3.5 YOLO Loss Function (Training)

$$\mathcal{L}_{\text{YOLO}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$

**CIoU Loss** (Complete IoU):

$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

$$v = \frac{4}{\pi^2}\left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2, \quad \alpha = \frac{v}{(1 - \text{IoU}) + v}$$

trong đó $\rho(\cdot)$ là Euclidean distance giữa centers, $c$ là diagonal của smallest enclosing box.

---

## 4. Zero-shot Classification – CLIP

**File:** `detect_objects.py` → `classify_with_clip()`

### 4.1 Kiến trúc CLIP

**Định nghĩa 4.1 (CLIP Dual Encoder).** CLIP gồm 2 encoder:
- Image encoder $f_I: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^d$ (ViT-B/32, $d = 512$)
- Text encoder $f_T: \Sigma^* \rightarrow \mathbb{R}^d$ (Transformer, $d = 512$)

được huấn luyện contrastive trên 400M image-text pairs.

### 4.2 Contrastive Pre-training Loss

Cho batch $\{(I_i, T_i)\}_{i=1}^{N}$ cặp image-text:

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_i, T_j)/\tau)} + \log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_j, T_i)/\tau)}\right]$$

trong đó cosine similarity:

$$\text{sim}(I, T) = \frac{f_I(I)^\top f_T(T)}{\|f_I(I)\|_2 \cdot \|f_T(T)\|_2}$$

và $\tau$ là learnable temperature parameter.

### 4.3 Zero-shot Inference trong Pipeline

**Thuật toán 4.1 (CLIP Classification):**

```
Input:  Ảnh ROI x_roi (crop từ bounding box), 
        Tập labels L = {l₁, ..., lₖ}, 
        Ngưỡng θ_clip = 0.65
Output: Predicted class ĉ, confidence p̂

1. Pre-compute (1 lần duy nhất):
   tₖ ← f_T("a photo of a {lₖ}") ∀k         // Text features
   T ← [t₁/‖t₁‖, ..., tₖ/‖tₖ‖]              // Normalized text matrix

2. Per detection:
   v ← f_I(preprocess(x_roi))                  // Image feature
   v̂ ← v / ‖v‖                                // Normalize

3. Similarity:
   s ← v̂ᵀ T ∈ ℝᴷ                             // Cosine similarities

4. Softmax:
   p(cₖ | x_roi) = exp(sₖ / τ) / Σⱼ exp(sⱼ / τ)

5. Classification:
   ĉ ← argmax_k p(cₖ | x_roi)
   p̂ ← max_k p(cₖ | x_roi)

6. Override YOLO nếu p̂ > θ_clip VÀ ĉ ≠ c_yolo
```

**Implementation:** `detect_objects.py` line 115-175. Text features được pre-compute 1 lần (`_precompute_clip_features`) và cache.

### 4.4 Chứng minh tính đúng đắn

**Mệnh đề 4.1.** Softmax temperature scaling đảm bảo output là phân phối xác suất hợp lệ.

**Chứng minh.** Cần chứng minh: (i) $p(c_k) > 0 \;\forall k$, và (ii) $\sum_k p(c_k) = 1$.

(i) Vì $\exp(\cdot) > 0$ cho mọi đối số hữu hạn, nên tử số $\exp(s_k/\tau) > 0$ và mẫu số $\sum_j \exp(s_j/\tau) > 0$. Do đó $p(c_k) > 0$.

(ii) $\sum_{k=1}^{K} p(c_k) = \sum_{k=1}^{K} \frac{\exp(s_k/\tau)}{\sum_j \exp(s_j/\tau)} = \frac{\sum_k \exp(s_k/\tau)}{\sum_j \exp(s_j/\tau)} = 1$. $\blacksquare$

**Mệnh đề 4.2 (Cosine Similarity Bounds).** $\forall x, y \in \mathbb{R}^d \setminus \{\mathbf{0}\}: -1 \leq \text{sim}(x, y) \leq 1$.

**Chứng minh.** Theo bất đẳng thức Cauchy-Schwarz: $|x^\top y| \leq \|x\|_2 \|y\|_2$. Chia cả 2 vế cho $\|x\|_2 \|y\|_2 > 0$:

$$\left|\frac{x^\top y}{\|x\|_2 \|y\|_2}\right| \leq 1 \implies -1 \leq \text{sim}(x, y) \leq 1 \quad \blacksquare$$

---

## 5. ROI Feature Extraction – RoIAlign

**File:** `detect_objects.py` → `_extract_roi_features()`

### 5.1 Backbone Feature Hooking

**Thuật toán 5.1 (Feature Capture):**

```
1. Hook vào SPPF layer (layer 9) của YOLO backbone:
   hook = yolo_model.model.model[9].register_forward_hook(capture_fn)

2. Forward pass → capture feature map F ∈ ℝ^{C'×H'×W'}
   trong đó C' = 512 (SPPF output channels)
         H' = H/32, W' = W/32 (stride 32)

3. Remove hook sau khi capture
```

### 5.2 RoIAlign (He et al., 2017)

**Định nghĩa 5.1 (RoIAlign).** Cho feature map $F \in \mathbb{R}^{C' \times H' \times W'}$, bounding box $b = (x_1, y_1, x_2, y_2)$ trong ảnh gốc, output size $(k, k)$:

**Bước 1.** Scale bbox sang feature map coordinates:

$$b'_x = b_x \cdot \frac{W'}{W}, \quad b'_y = b_y \cdot \frac{H'}{H}$$

**Bước 2.** Chia ROI thành $k \times k$ bins. Mỗi bin $B_{i,j}$ có kích thước:

$$\Delta_x = \frac{x_2' - x_1'}{k}, \quad \Delta_y = \frac{y_2' - y_1'}{k}$$

**Bước 3.** Trong mỗi bin, sample 4 điểm (2×2 regular grid) bằng **bilinear interpolation**:

$$F(x, y) = \sum_{(i,j) \in \mathcal{N}(x,y)} F[i, j] \cdot \max(0, 1 - |x - i|) \cdot \max(0, 1 - |y - j|)$$

trong đó $\mathcal{N}(x,y)$ là 4 pixel neighbors gần nhất.

**Bước 4.** Max/Average pooling trong mỗi bin:

$$v_{i,j}^c = \frac{1}{|\mathcal{S}_{i,j}|}\sum_{(x,y) \in \mathcal{S}_{i,j}} F^c(x, y)$$

**Output:** $v_{\text{roi}} \in \mathbb{R}^{C' \times k \times k}$, với $k = 7$ (default).

### 5.3 So sánh RoIAlign vs RoIPool

| Thuộc tính | RoIPool | RoIAlign |
|---|---|---|
| Quantization | Có (round về integer) | Không (bilinear interp.) |
| Gradient flow | Gián đoạn tại biên | Liên tục (differentiable) |
| Misalignment | ±1 pixel | Sub-pixel accuracy |
| Phù hợp cho | Classification | Segmentation, precise localization |

### 5.4 Global Context Vector

$$g = \text{GAP}(F) = \frac{1}{H' \times W'}\sum_{h=1}^{H'}\sum_{w=1}^{W'} F[:, h, w] \in \mathbb{R}^{C'}$$

Vector $g$ encode thông tin toàn cục của scene (chiếu sáng, layout tổng thể, context). Được truyền vào RelTR thông qua projection layer.

---

## 6. Scene Graph Generation – RelTR Transformer

**Files:** `models/reltr.py`, `models/transformer.py`, `models/matcher.py`, `models/backbone.py`

### 6.1 Định nghĩa hình thức

**Định nghĩa 6.1 (Scene Graph).** Scene graph $G = (\mathcal{V}, \mathcal{E})$ trong đó:
- $\mathcal{V} = \{v_i = (b_i, c_i)\}$ — nodes (entities/objects)
- $\mathcal{E} = \{e_k = (v_s, p_k, v_o)\}$ — edges (relationships/predicates)

RelTR sinh $G$ trực tiếp từ ảnh trong một forward pass (end-to-end).

### 6.2 Backbone (ResNet-50 + FrozenBatchNorm2d)

**Implementation:** `models/backbone.py`

```python
class Backbone(BackboneBase):
    """ResNet-50 with frozen BatchNorm."""
    # Layers 1-3: frozen (không train)
    # Layer 4: trainable
    # Output: feature map F ∈ ℝ^{2048×H/32×W/32}
```

**FrozenBatchNorm2d** (line 25-61): Batch statistics cố định, chỉ áp dụng affine transform:

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

trong đó $\mu, \sigma^2$ là running statistics (cố định), $\gamma, \beta$ là learnable (cố định).

**Joiner** (line 129-142): Kết hợp backbone với positional encoding:

$$\text{Joiner}(I) = (\text{features}, \text{pos\_encoding})$$

### 6.3 Input Projection

$$F_{\text{proj}} = \text{Conv}_{1 \times 1}(F_{\text{backbone}}) \in \mathbb{R}^{d_{\text{model}} \times H' \times W'}$$

`input_proj`: Conv2d(2048, 256, kernel_size=1) — giảm channels từ 2048 xuống 256.

### 6.4 Positional Encoding (Sinusoidal 2D)

Cho vị trí $(x, y)$ trên feature map:

$$\text{PE}_{(x, 2i)} = \sin\left(\frac{x}{10000^{2i/d}}\right), \quad \text{PE}_{(x, 2i+1)} = \cos\left(\frac{x}{10000^{2i/d}}\right)$$

$$\text{PE}_{(y, 2i)} = \sin\left(\frac{y}{10000^{2i/d}}\right), \quad \text{PE}_{(y, 2i+1)} = \cos\left(\frac{y}{10000^{2i/d}}\right)$$

Concat: $\text{PE}_{2D} = [\text{PE}_x ; \text{PE}_y] \in \mathbb{R}^{d_{\text{model}}}$.

**Mệnh đề 6.1.** Sinusoidal encoding giữ khoảng cách tương đối thông qua tích vô hướng: $\text{PE}_{pos}^\top \text{PE}_{pos+k}$ chỉ phụ thuộc vào $k$, không phụ thuộc $pos$.

### 6.5 Transformer Encoder

**Implementation:** `models/transformer.py`, line 67-148

$L = 6$ encoder layers, mỗi layer gồm:

**Multi-Head Self-Attention (MHSA):**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V, \quad d_k = \frac{d_{\text{model}}}{n_{\text{heads}}} = \frac{256}{8} = 32$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

với $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

**Feed-Forward Network (FFN):**

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

với $W_1 \in \mathbb{R}^{256 \times 2048}$, $W_2 \in \mathbb{R}^{2048 \times 256}$.

**Full Encoder Layer (Post-Norm):**

$$\begin{aligned}
q &= k = x + \text{PE} \\
x' &= \text{LayerNorm}(x + \text{Dropout}(\text{MHSA}(q, k, x))) \\
\text{output} &= \text{LayerNorm}(x' + \text{Dropout}(\text{FFN}(x')))
\end{aligned}$$

**Độ phức tạp:** $O(n^2 \cdot d)$ với $n = H'W'$ tokens.

### 6.6 Coupled Entity-Triplet Decoder (Đóng góp chính của RelTR)

**Implementation:** `models/transformer.py`, line 191-331

Đây là **kiến trúc lõi** phân biệt RelTR với các phương pháp SGG khác. Decoder xử lý đồng thời 2 loại queries:

**Learnable Embeddings:**
- `entity_embed` $\in \mathbb{R}^{N_e \times 2d}$ ($N_e = 100$): Split thành content ($\mathbb{R}^d$) + positional ($\mathbb{R}^d$)
- `triplet_embed` $\in \mathbb{R}^{N_t \times 3d}$ ($N_t = 200$): Split thành content ($\mathbb{R}^d$) + positional ($\mathbb{R}^{2d}$)
- `so_embed` $\in \mathbb{R}^{2 \times d}$: Subject/Object role encoding

**Thuật toán 6.1 (Coupled Decoder Layer)** — chi tiết từng bước trong `TransformerDecoderLayer.forward()`:

```
Input:  entity ∈ ℝ^{Nₑ×B×d}, triplet ∈ ℝ^{Nₜ×B×2d}, memory ∈ ℝ^{HW×B×d}

── ENTITY BRANCH ──
Step 1: Entity Self-Attention
  q = k = entity + entity_pos
  entity ← LN(entity + Dropout(MHSA(q, k, entity)))

Step 2: Entity-Memory Cross-Attention  
  entity ← LN(entity + Dropout(CrossAttn(entity+pos, memory+pos, memory)))

Step 3: Entity FFN
  entity ← LN(entity + Dropout(FFN(entity)))

── TRIPLET BRANCH ──
Step 4: Split triplet → (sub ∈ ℝ^{Nₜ×B×d}, obj ∈ ℝ^{Nₜ×B×d})

Step 5: Coupled Subject-Object Self-Attention
  q_sub = sub + triplet_pos + so_embed[0]
  q_obj = obj + triplet_pos + so_embed[1]
  [sub; obj] ← LN([sub; obj] + Dropout(MHSA([q_sub; q_obj], [k_sub; k_obj], [sub; obj])))
  // Subject và Object "giao tiếp" với nhau trong self-attention

Step 6a: Subject Visual Cross-Attention (→ tạo sub_attention_maps)
  sub, sub_maps ← CrossAttn(sub+triplet_pos, memory+pos, memory)
  sub ← LN(sub + Dropout(sub_attn))

Step 6b: Subject Entity Cross-Attention (→ "chọn" entity nào làm subject)
  sub ← LN(sub + Dropout(CrossAttn(sub+triplet_pos, entity, entity)))
  sub ← LN(sub + Dropout(FFN(sub)))

Step 7a: Object Visual Cross-Attention (→ tạo obj_attention_maps)  
  obj, obj_maps ← CrossAttn(obj+triplet_pos, memory+pos, memory)
  obj ← LN(obj + Dropout(obj_attn))

Step 7b: Object Entity Cross-Attention (→ "chọn" entity nào làm object)
  obj ← LN(obj + Dropout(CrossAttn(obj+triplet_pos, entity, entity)))
  obj ← LN(obj + Dropout(FFN(obj)))

Step 8: Recombine
  triplet ← [sub; obj] ∈ ℝ^{Nₜ×B×2d}

Output: entity, triplet, sub_maps, obj_maps
```

**Ý nghĩa:** Subject/Object cross-attention với entities cho phép mỗi triplet query "chọn" entity nào làm subject và object, tạo nên cấu trúc triplet $(s, p, o)$ một cách tự nhiên.

### 6.7 Subject-Object Mask và Predicate Classification

**Implementation:** `models/reltr.py`, line 96-119

Từ attention maps của decoder:

```
so_masks = [sub_maps; obj_maps] ∈ ℝ^{L×B×Nₜ×2×H'×W'}
```

**Mask Processing Pipeline:**

$$\text{so\_masks} \xrightarrow{\text{Upsample}(28 \times 28)} \xrightarrow{\text{Conv2d}(2 \to 64)} \xrightarrow{\text{ReLU+BN+MaxPool}} \xrightarrow{\text{Conv2d}(64 \to 32)} \xrightarrow{\text{ReLU+BN}} \xrightarrow{\text{Flatten}} \xrightarrow{\text{FC}(2048 \to 512 \to 128)} m \in \mathbb{R}^{128}$$

**Predicate Classification:**

$$\hat{p}_{\text{rel}} = \text{MLP}_{640 \to 256 \to (R+1)}([\text{sub}_{\text{ctx}} \;;\; \text{obj}_{\text{ctx}} \;;\; m])$$

trong đó:
- $\text{sub}_{\text{ctx}} = h_{\text{sub}} + g' \in \mathbb{R}^{256}$ (subject representation + context)
- $\text{obj}_{\text{ctx}} = h_{\text{obj}} + g' \in \mathbb{R}^{256}$ (object representation + context)
- $m \in \mathbb{R}^{128}$ (spatial mask features)
- $R = 51$ (Visual Genome) hoặc $R = 31$ (Open Images)

### 6.8 Prediction Heads

Từ decoder output ở layer cuối:

| Head | Input dim | Output | Architecture |
|---|---|---|---|
| `entity_class_embed` | $d$ | $\mathbb{R}^{C+1}$ | Linear(256, 152) |
| `entity_bbox_embed` | $d$ | $\mathbb{R}^4$ sigmoid | MLP(256→256→256→4) |
| `sub_class_embed` | $d$ | $\mathbb{R}^{C+1}$ | Linear(256, 152) |
| `sub_bbox_embed` | $d$ | $\mathbb{R}^4$ sigmoid | MLP(256→256→256→4) |
| `obj_class_embed` | $d$ | $\mathbb{R}^{C+1}$ | Linear(256, 152) |
| `obj_bbox_embed` | $d$ | $\mathbb{R}^4$ sigmoid | MLP(256→256→256→4) |
| `rel_class_embed` | $2d + 128$ | $\mathbb{R}^{R+1}$ | MLP(640→256→52) |

Bbox format: $(c_x, c_y, w, h)$ normalized ∈ $[0, 1]^4$.

### 6.9 Hungarian Matching

**Implementation:** `models/matcher.py`

**Định nghĩa 6.2 (Bipartite Matching Problem).** Tìm phép gán $\hat{\sigma} \in \mathfrak{S}_N$ tối thiểu hóa tổng chi phí:

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}(\hat{y}_{\sigma(i)}, y_i)$$

**Entity Matching Cost** (Focal Loss-based):

$$C_{\text{entity}}(i, j) = \lambda_{\text{cls}} C_{\text{focal}}(i, j) + \lambda_{\text{box}} \|b_i - b_j\|_1 + \lambda_{\text{giou}} (-\text{GIoU}(b_i, b_j))$$

trong đó Focal Loss cost (theo Deformable DETR):

$$C_{\text{focal}}(i, j) = \alpha(1-p_{i,c_j})^\gamma \cdot (-\log(p_{i,c_j} + \epsilon)) - (1-\alpha)(p_{i,c_j})^\gamma \cdot (-\log(1 - p_{i,c_j} + \epsilon))$$

với $\alpha = 0.25$, $\gamma = 2.0$ (focal loss hyperparameters).

**Triplet Matching Cost:**

$$C_{\text{triplet}} = \lambda_{\text{box}}(C_{\text{sub\_bbox}} + C_{\text{obj\_bbox}}) + \lambda_{\text{cls}}(C_{\text{sub\_cls}} + C_{\text{obj\_cls}}) + 0.5 \cdot C_{\text{rel\_cls}} + \lambda_{\text{giou}}(C_{\text{sub\_giou}} + C_{\text{obj\_giou}})$$

**Giải bằng** `scipy.optimize.linear_sum_assignment` — thuật toán Hungarian, $O(N^3)$.

**Subject/Object Weight Strategy** (line 135-148 matcher.py):

Tránh gán background cho predictions tốt: Nếu prediction match ground-truth entity (cùng class VÀ IoU ≥ 0.7), đặt weight = 0 (không back-propagate loss cho prediction đó, trừ khi nó được chọn bởi Hungarian matching).

### 6.10 Loss Functions (SetCriterion)

**Implementation:** `models/reltr.py`, line 191-382

**Tổng loss:**

$$\mathcal{L}_{\text{total}} = \sum_{l \in \{\text{labels}, \text{boxes}, \text{relations}\}} w_l \mathcal{L}_l$$

Với auxiliary losses (mỗi decoder layer):

$$\mathcal{L}_{\text{total}} = \sum_{d=0}^{L-1} \sum_{l} w_l \mathcal{L}_l^{(d)}$$

**Classification Loss** (`loss_labels`):

$$\mathcal{L}_{\text{CE}} = \frac{\sum_{i} w_i \cdot \text{CE}(p_i, y_i)}{\sum_{i} w_{\text{empty}}[y_i]}$$

trong đó $w_{\text{empty}}[-1] = \text{eos\_coef}$ (thường 0.1) — giảm weight cho class "no-object" vì phần lớn queries không match.

Đặc biệt: entity, subject, object losses được **gộp chung** và tính 1 lần:

$$\mathcal{L}_{\text{CE}} = \text{CE}([\text{entity\_logits}; \text{sub\_logits}; \text{obj\_logits}], [\text{entity\_targets}; \text{sub\_targets}; \text{obj\_targets}])$$

Subject/Object loss weight nhân 0.5.

**Box Loss** (`loss_boxes`):

$$\mathcal{L}_{\text{box}} = \frac{1}{N_{\text{boxes}}}\left(\|b_{\text{pred}} - b_{\text{gt}}\|_1 + \lambda_{\text{giou}}(1 - \text{GIoU}(b_{\text{pred}}, b_{\text{gt}}))\right)$$

**GIoU** (Generalized IoU):

$$\text{GIoU}(A, B) = \text{IoU}(A, B) - \frac{|C \setminus (A \cup B)|}{|C|}$$

trong đó $C$ là smallest enclosing box. Phạm vi: $\text{GIoU} \in [-1, 1]$.

**Mệnh đề 6.2.** $\text{GIoU}$ khắc phục vấn đề gradient = 0 khi $\text{IoU} = 0$.

**Chứng minh.** Khi $A \cap B = \emptyset$: $\text{IoU} = 0$ và $\frac{\partial \text{IoU}}{\partial b} = 0$. Tuy nhiên, $\text{GIoU} = -\frac{|C \setminus (A \cup B)|}{|C|} < 0$, và $\frac{\partial \text{GIoU}}{\partial b} \neq 0$ vì dịch $A$ gần $B$ làm giảm $|C|$ → GIoU tăng. $\blacksquare$

**Relation Loss** (`loss_relations`):

$$\mathcal{L}_{\text{rel}} = \text{CE}(\hat{p}_{\text{rel}}, y_{\text{rel}})$$

với weight `eos_coef` cho class "no-relation".

### 6.11 Global Context Integration

**Implementation:** `models/reltr.py`, line 130-157

$$g' = \text{ReLU}(\text{flatten}(g) \cdot W_{\text{ctx}} + b_{\text{ctx}}) \in \mathbb{R}^{d_{\text{model}}}$$

$g'$ được **cộng** vào subject/object representations trước predicate classification:

$$\text{sub}_{\text{ctx}} = h_{\text{sub}} + g', \quad \text{obj}_{\text{ctx}} = h_{\text{obj}} + g'$$

`context_proj` được khởi tạo zero → ban đầu không ảnh hưởng, dần học tầm quan trọng.

---

## 7. Spatial-Semantic Validation

**File:** `boundingbox_objects.py` → `validate_and_correct_relationships()`

### 7.1 Mục đích

RelTR có thể dự đoán quan hệ không hợp lý (ví dụ: "sky riding person"). Module validation lọc và sửa các dự đoán sai dựa trên quy tắc heuristic.

### 7.2 Spatial Validation Rules

**Quy tắc dựa trên centroid và bounding box overlap:**

Gọi centroid của subject/object: $c_s = (\frac{x_1^s+x_2^s}{2}, \frac{y_1^s+y_2^s}{2})$, tương tự cho $c_o$.

| Rule | Predicate | Điều kiện spatial | Hành động nếu vi phạm |
|---|---|---|---|
| R1 | `riding` | $c_s^y < c_o^y$ (subject ở **trên** object) | Swap subject ↔ object |
| R2 | `sitting on` | $c_s^y < c_o^y$ | Swap |
| R3 | `standing on` | $c_s^y < c_o^y$ | Swap |
| R4 | `above` | $c_s^y > c_o^y$ (subject ở **dưới** object) | Swap |
| R5 | `below` | $c_s^y < c_o^y$ | Swap |
| R6 | `under` | $c_s^y < c_o^y$ | Swap |
| R7 | `on` | $c_s^y > c_o^y + \delta$ | Swap |

### 7.3 Semantic Validation Rules

**Quy tắc dựa trên ngữ nghĩa class:**

| Rule | Predicate | Subject class constraint | Object class constraint |
|---|---|---|---|
| S1 | `wearing` | animate (person, man, ...) | wearable (hat, shirt, ...) |
| S2 | `riding` | animate | rideable (horse, bike, ...) |
| S3 | `eating` | animate | edible (pizza, food, ...) |
| S4 | `driving` | animate | vehicle (car, bus, ...) |
| S5 | `flying in` | flyable (airplane, bird) | sky, air |

**Implementation** — hệ thống quy tắc kiểm tra:

```python
def semantic_validate(subject_class, predicate, object_class):
    # 1. Check subject constraint
    if predicate in REQUIRES_ANIMATE_SUBJECT:
        if subject_class not in ANIMATE_CLASSES:
            return False  # Reject
    
    # 2. Check object constraint
    if predicate in PREDICATE_OBJECT_MAP:
        valid_objects = PREDICATE_OBJECT_MAP[predicate]
        if object_class not in valid_objects:
            return False  # Reject
    
    return True  # Accept
```

### 7.4 Heuristic Fallback

Khi RelTR confidence $\sigma < \theta_{\min}$ (thường 0.3) cho tất cả predicates, hệ thống tạo quan hệ heuristic dựa trên spatial analysis:

$$p_{\text{heuristic}} = \begin{cases}
\texttt{"near"} & \text{nếu } IoU(b_s, b_o) > 0.1 \\
\texttt{"above"} & \text{nếu } c_s^y < c_o^y - \delta_y \\
\texttt{"next to"} & \text{otherwise}
\end{cases}$$

---

## 8. LLM Open-Vocabulary Enhancement

**Files:** `RL/llm_relationship_predictor.py`, `RL/visual_features.py`

### 8.1 Tổng quan

Khi RelTR không thể dự đoán quan hệ (confidence thấp hoặc predicate không thuộc vocabulary), sử dụng GPT-4 Vision để suy luận open-vocabulary.

### 8.2 Visual Features cho LLM

**File:** `RL/visual_features.py`

#### 8.2.1 Union Box Crop

$$\text{UnionBox}(A, B) = (\min(x_1^A, x_1^B), \min(y_1^A, y_1^B), \max(x_2^A, x_2^B), \max(y_2^A, y_2^B))$$

Crop vùng union với padding 20px, chuyển đổi BGR→RGB.

#### 8.2.2 Interaction Heatmap

**IoU-based method** (nhanh):

$$H(x, y) = \frac{\min(M_s(x,y) + M_o(x,y), \; 2)}{2}$$

$H = 1.0$ tại vùng intersection, $H = 0.5$ tại vùng chỉ thuộc 1 object, $H = 0$ ngoài.

**Gaussian method** (smooth hơn):

$$G_s(x,y) = \exp\left(-\frac{(x-c_s^x)^2/(w_s/2)^2 + (y-c_s^y)^2/(h_s/2)^2}{2\sigma^2}\right), \quad \sigma = 0.3$$

$$H_{\text{gaussian}} = G_s \cdot G_o$$

#### 8.2.3 Gaze Guided Attention Vector

**Định nghĩa 8.1 (Gaze Vector).** Cho subject bbox $B_s$ và object bbox $B_o$:

$$\vec{g} = \frac{c_o - c_s}{\|c_o - c_s\|_2} \in \mathbb{R}^2$$

**Proximity score:**

$$\text{proximity} = 1 - \frac{\|c_o - c_s\|_2}{\sqrt{H^2 + W^2}}$$

**Non-contact likelihood** (cho animate subjects):

$$\text{NCL} = \begin{cases}
0.7 \cdot \text{proximity} + 0.3 \cdot \text{vertical\_align} & \text{nếu animate AND no overlap} \\
\max(0.3 - \text{overlap\_ratio}, 0.1) & \text{nếu animate AND overlap} \\
0.05 & \text{inanimate subject}
\end{cases}$$

**Gaze vector gửi cho LLM** (6-dimensional):

$$\mathbf{v}_{\text{gaze}} = [g_x, g_y, \text{proximity}, \mathbb{1}[\text{horizontal}], \text{overlap\_ratio}, \text{NCL}]$$

#### 8.2.4 Spatial Description

Sinh mô tả ngôn ngữ tự nhiên dựa trên centroid comparison:

```
vertical_desc ∈ {"above", "below", "at same height as"}   (threshold: 10% image height)
horizontal_desc ∈ {"to the left of", "to the right of", "aligned with"}   (threshold: 10% width)
overlap_desc = "overlapping with" nếu IoU > 0
```

### 8.3 LLM Prompt Construction

**File:** `RL/llm_relationship_predictor.py`

```
System: "You are an expert in visual relationship detection. Analyze 
the spatial relationship between objects in the image."

User: 
"Image: [union_crop_base64]
Subject: {subject_class} at bbox {subject_bbox}
Object: {object_class} at bbox {object_bbox}
Spatial: {spatial_description}
Gaze: direction=({gx:.2f}, {gy:.2f}), proximity={prox:.2f}
Current prediction: {reltr_predicate} (confidence: {conf:.2f})

What is the most likely relationship? Respond in JSON:
{\"predicate\": \"...\", \"confidence\": 0.x, \"reasoning\": \"...\"}"
```

### 8.4 Confidence Assignment

LLM predictions được gán confidence dựa trên nguồn:

| Source | Confidence range |
|---|---|
| RelTR high-conf ($\sigma > 0.7$) | $\sigma$ (giữ nguyên) |
| LLM prediction (new) | $0.5 \cdot p_{\text{LLM}}$ |
| LLM override (vs RelTR) | $0.3 + 0.4 \cdot p_{\text{LLM}}$ |
| Heuristic fallback | $0.2$ (fixed) |

---

## 9. Video Processing Pipeline

**File:** `video_relation_pipeline.py`

### 9.1 Frame Processing

$$G_t = \text{Pipeline}(I_t), \quad t = 1, \ldots, T$$

Mỗi frame $I_t$ được xử lý qua toàn bộ inference pipeline (Sections 3-8).

### 9.2 Object Tracking (ByteTrack)

YOLO's built-in tracker (ByteTrack variant) sử dụng **Kalman Filter** để track objects across frames.

**Kalman Filter State:**

$$\mathbf{x}_t = [c_x, c_y, s, r, \dot{c}_x, \dot{c}_y, \dot{s}]^\top$$

trong đó $s = \sqrt{wh}$ (scale), $r = w/h$ (aspect ratio).

**Prediction:** $\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}\mathbf{x}_{t-1}$

**Update:** $\mathbf{x}_t = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t(\mathbf{z}_t - \mathbf{H}\hat{\mathbf{x}}_{t|t-1})$

**ByteTrack two-stage association:**
1. **First stage:** Match high-score detections ($s > \theta_{\text{high}}$) với existing tracks bằng IoU
2. **Second stage:** Match remaining low-score detections ($s > \theta_{\text{low}}$) với unmatched tracks

### 9.3 Safety Zone Monitoring

**Định nghĩa 9.1 (Safety Zone).** Vùng an toàn $Z = (x_1, y_1, x_2, y_2)$ do người dùng định nghĩa.

**Kiểm tra vi phạm:**

$$\text{ViolationCheck}(b, Z) = \text{IoU}(b, Z) > 0 \wedge \text{class}(b) \in \mathcal{C}_{\text{restricted}}$$

**3-tier Safety Classification:**

| Tier | Level | Color | Mô tả |
|---|---|---|---|
| 0 | SAFE | 🟢 Green | Quan hệ bình thường |
| 1 | SUSPICIOUS | 🟡 Yellow | Cần theo dõi |
| 2 | DANGEROUS | 🔴 Red | Cảnh báo ngay |

---

## 10. Safety Classification System

**Files:** `RL/safety_classifier.py`, `RL/llm_safety_analyzer.py`, `RL/local_rules_db.py`

### 10.1 3-Tier Architecture

```
Input: relationship (subject, predicate, object)
           │
   Tier 1: White/Black List Lookup ── O(1)
           │ (nếu không match)
   Tier 2: LLM Safety Analyzer ── O(API call)
           │ (nếu LLM unavailable)
   Tier 3: Local Rules Database ── O(|rules|)
           │
   Output: (safety_level, confidence, explanation)
```

### 10.2 Tier 1: White/Black List

**Implementation:** `RL/safety_classifier.py`

```python
WHITE_LIST = {
    ("person", "walking on", "street"): SafetyLevel.SAFE,
    ("*", "near", "*"): SafetyLevel.SAFE,  # wildcard match
    ...
}
BLACK_LIST = {
    ("child", "playing with", "knife"): SafetyLevel.DANGEROUS,
    ("person", "near", "fire"): SafetyLevel.DANGEROUS,
    ...
}
```

**Matching order:** (1) Exact match → (2) Wildcard `*` match → (3) Next tier.

### 10.3 Tier 2: LLM Safety Analyzer

**File:** `RL/llm_safety_analyzer.py`

LLM prompt:

```
"Phân tích mối quan hệ sau:
Subject: {subject}, Relation: {relation}, Object: {object}
Đánh giá: safe | suspicious | dangerous
Confidence: 0.0-1.0
Giải thích ngắn gọn."
```

**Caching:** Key = `f"{subject}|{relation}|{object}"`, lưu vào file JSON.

**Provider support:** OpenAI (GPT-4/3.5/o1), Gemini, với auto-detection `max_completion_tokens` vs `max_tokens`.

### 10.4 Tier 3: Local Rules Database

Context-specific rules (giao thông, công nghiệp, giáo dục, ...).

---

## 11. Deep Q-Network (DQN) Agent

**File:** `RL/reinforcement_learning.py`

### 11.1 Formalization

**Định nghĩa 11.1 (MDP).** Bài toán được mô hình hóa như Markov Decision Process $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:
- $\mathcal{S}$: Tập trạng thái (state vector)
- $\mathcal{A} = \{0, 1, 2, 3, 4\}$: Tập hành động (mỗi action → số variations khác nhau)
- $P$: Transition probability (stochastic do data generation)
- $R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$: Reward function
- $\gamma = 0.95$: Discount factor

### 11.2 State Representation

$$\mathbf{s}_t = [\underbrace{F_1^{\text{det}}, F_2^{\text{det}}}_{\text{Detection}}, \underbrace{F_1^{\text{rel}}, F_2^{\text{rel}}}_{\text{Relationship}}, \underbrace{r_{t-1}, \varepsilon_t, t/T}_{\text{Training state}}, \underbrace{U_{\text{mean}}, U_{\text{max}}}_{\text{Uncertainty}}] \in \mathbb{R}^{9}$$

**Implementation** (`RL/reinforcement_learning.py` → `_build_state_vector`): vector 9 chiều cố định (bỏ histogram phân phối loại quan hệ $R$ chiều để tránh kích thước biến đổi).

| Thành phần | Công thức / Nguồn | Chiều | Biến trong code |
|---|---|---|---|
| **$F_1^{\text{det}}$** — Độ tin cậy phát hiện (proxy) | $\frac{1}{N}\sum s_i$ được proxy bằng $F1_{\text{det}}$ từ đánh giá; nếu không có: $1 - \tanh(\text{loss}_{\text{det}}/5)$ | 1 | `detection_f1` |
| **$F_2^{\text{det}}$** — Số lượng phát hiện (chuẩn hóa) | $N / 100$, clip $\leq 1$, với $N = \text{TP} + \text{FP}$ (tổng số đối tượng dự đoán) | 1 | `detection_count_norm` |
| **$F_1^{\text{rel}}$** — Độ tin cậy quan hệ (proxy) | $\frac{1}{|\mathcal{R}|}\sum \sigma_k$ được proxy bằng $F1_{\text{rel}}$; nếu không có: từ loss | 1 | `relationship_f1` |
| **$F_2^{\text{rel}}$** — Số lượng quan hệ (chuẩn hóa) | $|\mathcal{R}| / 200$, clip $\leq 1$, với $|\mathcal{R}| = \text{TP} + \text{FN}$ (tổng quan hệ ground truth) | 1 | `relationship_count_norm` |
| **$r_{t-1}$** — Phần thưởng bước trước | Chuẩn hóa: $\tanh(r/1)$ | 1 | `reward_value` |
| **$\varepsilon_t$** — Tỷ lệ khám phá | $\varepsilon_t \in [0,1]$, chuẩn hóa tanh khi đưa vào state | 1 | `epsilon_value` |
| **$t/T$** — Tiến độ epoch | $\text{current\_epoch} / \text{total\_training\_epochs}$, clip $\leq 1$ | 1 | `progress_t_T` |
| **$\bar{U}$** — Độ bất định trung bình | Trung bình combined uncertainty (MC Dropout) trên batch đánh giá; mặc định $0{,}5$ nếu chưa có | 1 | `u_mean` |
| **$U_{\max}$** — Độ bất định tối đa | Max combined uncertainty trên batch; mặc định $0{,}5$ nếu chưa có | 1 | `u_max` |

*Mở rộng (chưa implement):* có thể thêm histogram phân phối loại quan hệ ($R$ chiều) → state $D_s = 9 + R$.

### 11.3 Q-Network Architecture

```
Q-Network: ℝ^9 → ℝ^{|A|}

Input(9) → Linear(9, 128) → ReLU → Dropout(0.1)
          → Linear(128, 64) → ReLU → Dropout(0.1)
          → Linear(64, |A|) → Output
```

**Target network:** $\theta^- \leftarrow \theta$ (hard copy) mỗi $C = 10$ steps.

### 11.4 Action Selection (ε-greedy)

$$a_t = \begin{cases}
\text{random action} \sim \text{Uniform}(\mathcal{A}) & \text{với xác suất } \varepsilon_t \\
\arg\max_a Q(\mathbf{s}_t, a; \theta) & \text{với xác suất } 1 - \varepsilon_t
\end{cases}$$

**Epsilon decay:**

$$\varepsilon_{t+1} = \max(\varepsilon_{\min}, \varepsilon_t \cdot \varepsilon_{\text{decay}})$$

Với $\varepsilon_0 = 1.0$, $\varepsilon_{\min} = 0.01$, $\varepsilon_{\text{decay}} = 0.995$.

### 11.5 Experience Replay

**Buffer:** Circular buffer $\mathcal{D}$ với capacity $N_{\text{buf}} = 10000$.

$$\mathcal{D} = \{(\mathbf{s}_t, a_t, r_t, \mathbf{s}_{t+1}, \text{done}_t)\}$$

**Sampling:** Random mini-batch $\mathcal{B} \sim \text{Uniform}(\mathcal{D})$, $|\mathcal{B}| = 32$.

### 11.6 Q-Learning Update

**Bellman optimality equation:**

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a')\right]$$

**Loss function (MSE):**

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{B}}\left[\left(r + \gamma(1-d)\max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

**Thuật toán 11.1 (DQN Training Step):**

```
1. Sample mini-batch B = {(sᵢ, aᵢ, rᵢ, s'ᵢ, dᵢ)} from buffer D
2. Compute targets: yᵢ = rᵢ + γ(1-dᵢ) max_a' Q(s'ᵢ, a'; θ⁻)
3. Compute predictions: ŷᵢ = Q(sᵢ, aᵢ; θ)
4. Loss: L = (1/|B|) Σᵢ (yᵢ - ŷᵢ)²
5. Gradient step: θ ← θ - α∇_θ L
6. Periodically: θ⁻ ← θ  (target update)
```

### 11.7 Chứng minh hội tụ

**Định lý 11.1 (Watkins & Dayan, 1992).** Q-Learning hội tụ đến $Q^*$ nếu:

1. $\forall (s, a): \sum_t \alpha_t(s,a) = \infty$ (mọi cặp state-action được thăm vô hạn lần)
2. $\forall (s, a): \sum_t \alpha_t^2(s,a) < \infty$ (learning rate giảm đủ nhanh)
3. $0 \leq \gamma < 1$ (discount factor)

**Chứng minh (sketch).** Xem Q-Learning update rule như stochastic approximation algorithm. Gọi $\Delta_t = Q_t(s,a) - Q^*(s,a)$:

$$\Delta_{t+1} = (1 - \alpha_t)\Delta_t + \alpha_t[r + \gamma\max_{a'} Q_t(s',a') - Q^*(s,a)]$$

Do contraction mapping: $\|\gamma\max_{a'} Q(s', a') - \gamma\max_{a'} Q^*(s', a')\| \leq \gamma\|Q - Q^*\|_\infty$, và $\gamma < 1$, toán tử Bellman backup là $\gamma$-contraction. Kết hợp điều kiện Robbins-Monro, $\Delta_t \rightarrow 0$ w.p. 1. $\blacksquare$

**Lưu ý:** Trong thực tế với function approximation (neural network), hội tụ không được đảm bảo lý thuyết, nhưng experience replay + target network + ε-greedy giúp ổn định (Mnih et al., 2015).

---

## 12. Adaptive Reward Function

**File:** `RL/reinforcement_learning.py` → `calculate_reward()`

### 12.1 Tổng quan

Hàm phần thưởng được chuẩn hóa qua **sigmoid** để tránh bùng nổ giá trị và đưa reward vào khoảng $(0, 1)$:

$$R = \sigma\big(k \cdot (R_{\text{raw}} - 0.5)\big) = \frac{1}{1 + e^{-k(R_{\text{raw}} - 0.5)}}$$

trong đó $k$ là **scaling factor** (phụ thuộc độ ổn định gần đây của reward). Điểm thô:

$$R_{\text{raw}} = W_{\text{det}} S_{\text{det}} + W_{\text{rel}} S_{\text{rel}} + W_{\text{div}} S_{\text{div}} + W_{\text{cons}} S_{\text{cons}} + W_{\text{imp}} S_{\text{imp}} + W_{\text{unc}} S_{\text{unc}}$$

Các thành phần $S_{\text{det}}, S_{\text{rel}}, S_{\text{div}}, S_{\text{cons}}, S_{\text{imp}}$ được định nghĩa dưới đây theo báo cáo; $S_{\text{unc}}$ là điểm giảm uncertainty (MC Dropout). Trọng số $W_*$ là **trọng số thích nghi** (adaptive weights).

**Ký hiệu chung:**
- $n$: số mẫu đánh giá (num_samples)
- $\alpha$: hằng số điều chỉnh (mặc định 0.5), dùng trong $C_n$ và $S_{\text{pos}}$
- $P, R$: Precision và Recall tương ứng (detection hoặc relationship)
- $\text{F1}$: F1-score ($2PR/(P+R)$ hoặc từ metrics)

---

### 12.2 Điểm phát hiện vật thể — $S_{\text{det}}$


$$S_{\text{det}} = F1_{\text{det}} \cdot C_n \cdot B_{PR}$$

- **Hệ số tin cậy mẫu** $C_n$ (giảm tác động khi $n$ ít — cold-start):

$$C_n = \tanh\big(\alpha \cdot \ln(n + 1)\big)$$

- **Hệ số cân bằng Precision–Recall** $B_{PR}$ (trừng phạt lệch P/R):

$$B_{PR} = 1 - |P - R|, \quad B_{PR} \in [0, 1]$$

**Implementation:** `_calculate_detection_score(detection_metrics)` — lấy `f1`, `precision`, `recall`, `num_samples` từ `detection_metrics`.

---

### 12.3 Điểm quan hệ — $S_{\text{rel}}$


$$S_{\text{rel}} = \big(F1_{\text{rel}} \cdot C_n \cdot B_{PR}\big) \times (1 + W_{\text{tail}})$$

$C_n$ và $B_{PR}$ giống mục 12.2 (dùng metrics của relationship). **Trọng số đuôi dài** $W_{\text{tail}}$:

- Trọng số thô theo nghịch đảo tần suất (49):

$$W_{\text{raw}}(r) = \frac{1}{\sqrt{\text{freq}(r) + \epsilon}}, \quad \epsilon = 10^{-3}$$

- Chuẩn hóa trên toàn bộ từ vựng quan hệ $R$ (48):

$$W_{\text{tail}}(r) = \frac{W_{\text{raw}}(r)}{\sum_{i=1}^{R} W_{\text{raw}}(i)}$$

Hệ thống duy trì từ điển `tail_weights` (đã chuẩn hóa). Khi đánh giá theo từng relation thì dùng $W_{\text{tail}}(r)$ tương ứng; khi đánh giá gộp thì dùng trung bình $W_{\text{tail}}$ trên các relation có trong batch.

**Implementation:** `_calculate_relationship_score(relationship_metrics)` — F1, P, R, n từ `relationship_metrics`; $W_{\text{tail}}$ từ `self.tail_weights` (tính bằng `_recompute_tail_weights()`).

---

### 12.4 Điểm đa dạng — $S_{\text{div}}$


$$S_{\text{div}} = 0.4\, D_{\text{type}} + 0.4\, D_{\text{class}} + 0.2\, S_{\text{spatial}}$$

- **$D_{\text{type}}$:** Tỷ lệ số **loại quan hệ** xuất hiện trên tổng số loại khả dụng (ví dụ chuẩn hóa với mẫu 10).
- **$D_{\text{class}}$:** Tỷ lệ số **lớp vật thể** (subject/object) xuất hiện trên tổng lớp khả dụng (ví dụ 15).

**Điểm đa dạng không gian** $S_{\text{spatial}}$:

$$S_{\text{spatial}} = 0.4\, S_{\text{pos}} + 0.3\, S_{\text{size}} + 0.3\, S_{\text{coverage}}$$

- **$S_{\text{pos}}$ (52):** Đa dạng vị trí — khuyến khích centroid bbox thay đổi:

$$S_{\text{pos}} = \tanh(\alpha \cdot \text{Var}_{\text{pos}})$$

với $\text{Var}_{\text{pos}}$ là tổng phương sai tọa độ tâm (sau khi chuẩn hóa theo kích thước ảnh).

- **$S_{\text{size}}$ (53):** Hệ số biến thiên (Coefficient of Variation) diện tích bbox:

$$S_{\text{size}} = \frac{\sigma_{\text{size}}}{\mu_{\text{size}}}$$

(clip về đoạn hợp lý, ví dụ $[0, 2]$).

- **$S_{\text{coverage}}$ (54):** Entropy vị trí — khuyến khích vật thể rải đều trên lưới ảnh (grid 4×4):

$$S_{\text{coverage}} = -\sum_i p_i \log(p_i + \epsilon)$$

$p_i$ là tỷ lệ bbox rơi vào ô $i$; entropy được chuẩn hóa theo entropy cực đại.

**Implementation:** `_calculate_diversity_score(synthetic_data)` và các helper `_calculate_spatial_diversity`, `_calculate_position_diversity`, `_calculate_size_diversity`, `_calculate_coverage_diversity`.

---

### 12.5 Điểm nhất quán — $S_{\text{cons}}$


$$S_{\text{cons}} = 0.7\, S_{\text{std}} + 0.3\, S_{\text{trend}}$$

- **$S_{\text{std}}$ (56):** Nghịch đảo độ lệch chuẩn F1 (ổn định qua các mẫu):

$$S_{\text{std}} = \frac{1}{1 + \sigma_{F1}}$$

- **$S_{\text{trend}}$ (57):** Hệ số góc hồi quy tuyến tính của chuỗi F1 (theo thời gian / theo mẫu). Slope được map vào $[0, 1]$ bằng hàm tanh để slope dương (cải thiện) cho điểm cao hơn.

**Implementation:** `_calculate_consistency_score(f1_scores, precomputed_std)` và `_calculate_trend_score(scores)`.

---

### 12.6 Điểm cải thiện — $S_{\text{imp}}$


$$S_{\text{imp}} = 0.6 \cdot \tanh(F1_{\text{current}} - F1_{\text{base}}) + 0.4 \cdot S_{\text{trend}}$$

- **$F1_{\text{current}}$:** F1 (hoặc proxy: trung bình relationship score) gần đây.
- **$F1_{\text{base}}$:** Baseline (ngưỡng mong muốn), lấy từ `baseline_performance['relationship']` hoặc `overall`.
- **$S_{\text{trend}}$:** Cùng định nghĩa xu hướng như trong $S_{\text{cons}}$ (chuỗi relationship scores gần đây).

Trong code, $\tanh(F1_{\text{current}} - F1_{\text{base}})$ được map về $[0,1]$ dạng $0.5 + 0.5\tanh(\cdot)$ rồi nhân 0.6.

**Implementation:** `_calculate_improvement_score()` — dùng `performance_history['relationship_scores']` và `baseline_performance`.

---

### 12.7 Điểm giảm uncertainty — $S_{\text{unc}}$

Thành phần bổ sung (không nằm trong báo cáo 4.2.2): đo **mức giảm uncertainty** của mô hình RelTR sau khi bổ sung dữ liệu, dựa trên **MC Dropout** (Section 13). Cách tính $S_{\text{unc}}$ gồm ba bước sau.

#### Bước 1: Ước lượng uncertainty theo từng mẫu

Với tập mẫu đánh giá (ví dụ `evaluation_samples`, tối đa 10 mẫu), gọi:

$$\text{batch\_unc} = \text{estimate\_batch}(\text{samples}; \text{transform\_fn}, \text{max\_samples}=10)$$

Với mỗi mẫu, **estimate_relationship_uncertainty** chạy $N$ lần forward (MC Dropout, $N=10$ mặc định), thu được nhiều bộ dự đoán quan hệ. Từ đó tính:

- **Predictive Entropy** $\mathcal{H}[\bar{p}]$ (tổng uncertainty)
- **Mutual Information (BALD)** $\mathcal{I}[y; \omega]$ (epistemic uncertainty)
- **Variation Ratio** VR (tỷ lệ dự đoán thay đổi giữa các lần forward)
- **Mean confidence** $\bar{c}$ của các prediction

**Combined uncertainty score** (mỗi mẫu) ∈ $[0, 1]$ — càng cao càng không chắc chắn:

$$U_{\text{sample}} = 0.30\,\mathcal{H} + 0.30\,\mathcal{I} + 0.20\,\text{VR} + 0.20\,(1 - \bar{c})$$

Kết quả batch: `current_uncertainties = { sample_id: U_sample }` (trong code dùng `uncertainty_score` từ mỗi phần tử của `batch_unc`).

#### Bước 2: Mức giảm uncertainty so với epoch trước

Module **UncertaintyEstimator** lưu cache `_previous_uncertainties` (uncertainty của lần đo trước). Hàm **compute_uncertainty_reduction** so sánh với bộ uncertainty hiện tại:

$$\text{reduction}_k = \frac{U_{\text{prev},k} - U_{\text{current},k}}{U_{\text{prev},k}} \quad \text{(chỉ với } U_{\text{prev},k} > 0\text{)}$$

$$\rho = \frac{1}{|\mathcal{K}|}\sum_{k \in \mathcal{K}} \text{reduction}_k, \quad \rho \in [-1, 1]$$

- $\rho > 0$: uncertainty giảm trung bình (tốt)
- $\rho < 0$: uncertainty tăng (xấu)
- $\rho = 0$: không đổi hoặc lần đầu (chưa có cache)

Sau khi tính xong, cache được cập nhật: `_previous_uncertainties ← current_uncertainties` cho epoch tiếp theo.

#### Bước 3: Map sang $S_{\text{unc}} \in [0, 1]$

Trong **calculate_reward**, giá trị $\rho \in [-1, 1]$ được đưa về khoảng $[0, 1]$ để dùng làm thành phần reward:

$$S_{\text{unc}} = 0.5 + 0.5\,\rho, \quad \text{clip về } [0, 1]$$

- $S_{\text{unc}} = 0.5$: không thay đổi uncertainty (trung tính) hoặc lỗi/không có estimator
- $S_{\text{unc}} = 1$: giảm mạnh uncertainty
- $S_{\text{unc}} = 0$: uncertainty tăng mạnh (bị clip)

**Implementation:** `RL/uncertainty_estimator.py` — `estimate_batch`, `compute_uncertainty_reduction`, `_compute_combined_score`; `reinforcement_learning.py` — trong `calculate_reward` gọi `estimate_batch` → `compute_uncertainty_reduction` → map bằng `0.5 + 0.5 * reduction`. Khi không có `uncertainty_estimator` hoặc ngoại lệ: $S_{\text{unc}} = 0.5$.

**Trọng số $W_{\text{unc}}$** tham gia công thức trọng số thích nghi (mục 12.8): trọng số cơ bản $W_{\text{unc}}^0 = 0.10$, sau đó điều chỉnh theo độ lệch so với baseline và chuẩn hóa cùng 5 thành phần còn lại (tổng 6 trọng số bằng 1).

---

### 12.8 Trọng số thích nghi (Adaptive Weights)

Trọng số $W_k$ cho thành phần $k$ được cập nhật theo độ lệch so với baseline:

$$W_k = \frac{W_k^0 + \alpha\, (b_k - S_k)}{\sum_j \big(W_j^0 + \alpha\, (b_j - S_j)\big)}$$

- $W_k^0$: trọng số khởi tạo (detection 0.25, relationship 0.45, diversity 0.15, consistency 0.10, improvement 0.05, **uncertainty_reduction 0.10**).
- $S_k$: điểm hiện tại của thành phần $k$ (gồm cả $S_{\text{unc}}$).
- $b_k$: baseline của thành phần $k$ (baseline cho uncertainty_reduction cũng được cập nhật theo lịch sử).
- $\alpha$: hệ số điều chỉnh độ nhạy (ví dụ 0.2).

Thành phần nào **điểm thấp hơn baseline** ($S_k < b_k$) thì nhận trọng số cao hơn, giúp tập trung cải thiện các mục tiêu còn yếu. **Implementation:** `_calculate_dynamic_weights(..., improvement_score=..., uncertainty_reduction_score=...)` tính cả 6 trọng số trong một lần và chuẩn hóa tổng bằng 1; trọng số $S_{\text{unc}}$ cũng tham gia thích nghi, không còn cố định 10%.

---

### 12.9 Tóm tắt công thức và luồng tính toán

| Thành phần | Công thức chính | Input chính |
|---|---|---|
| $S_{\text{det}}$ | $F1_{\text{det}} \cdot C_n \cdot B_{PR}$ | detection_metrics (f1, P, R, n) |
| $S_{\text{rel}}$ | $(F1_{\text{rel}} \cdot C_n \cdot B_{PR})(1 + W_{\text{tail}})$ | relationship_metrics + tail_weights |
| $S_{\text{div}}$ | $0.4 D_{\text{type}} + 0.4 D_{\text{class}} + 0.2 S_{\text{spatial}}$ | synthetic_data (relations, classes, bbox) |
| $S_{\text{cons}}$ | $0.7/(1+\sigma_{F1}) + 0.3\, S_{\text{trend}}$ | per-sample F1, $\sigma_{F1}$ |
| $S_{\text{imp}}$ | $0.6\tanh(F1_{\text{curr}}-F1_{\text{base}}) + 0.4 S_{\text{trend}}$ | relationship_scores, baseline |
| $S_{\text{unc}}$ | $0.5 + 0.5\,\rho$, $\rho = \text{mean}_k\big((U_{\text{prev},k}-U_{\text{curr},k})/U_{\text{prev},k}\big)$ | estimate_batch → compute_uncertainty_reduction, cache $U_{\text{prev}}$ |


## 13. MC Dropout Uncertainty Estimation

**File:** `RL/uncertainty_estimator.py`

### 13.1 Nền tảng lý thuyết

**Định lý 13.1 (Gal & Ghahramani, 2016).** Một mạng neural với dropout trước mỗi weight layer tương đương xấp xỉ variational inference trong deep Gaussian process.

**Chứng minh (sketch).** Xem weight matrices $\{W_l\}$ là biến ngẫu nhiên. Dropout tạo phân phối biến phân:

$$q(\mathbf{W}_l) = \prod_{i} q(w_{l,i}), \quad q(w_{l,i}) = p \cdot \delta(w_{l,i}) + (1-p) \cdot \delta(w_{l,i} - m_{l,i})$$

trong đó $m_{l,i}$ là learned weights, $p$ là dropout rate. Khi tối ưu:

$$\min_{q} \text{KL}(q(\mathbf{W}) \| p(\mathbf{W}|\mathcal{D}))$$

Tương đương cross-entropy loss + L2 regularization. $\blacksquare$

### 13.2 MC Dropout Inference

**Thuật toán 13.1 (MC Dropout Uncertainty):**

```
Input:  Model f_θ, input x, T = 30 forward passes, dropout rate p
Output: Uncertainty scores

1. Enable dropout ở test time
2. for t = 1, ..., T:
3.   ŷₜ = f_θ(x)  // với random dropout mask khác nhau
4.   pₜ = softmax(ŷₜ) ∈ ℝ^K
5. Compute:
   p̄ = (1/T) Σₜ pₜ                      // mean prediction
   H_pred = -Σₖ p̄ₖ log(p̄ₖ)              // predictive entropy
   H_exp = -(1/T) Σₜ Σₖ pₜₖ log(pₜₖ)    // expected entropy
   MI = H_pred - H_exp                    // mutual information (BALD)
   VR = 1 - max_k (1/T)|{t: argmax pₜ = k}|  // variation ratio
```

### 13.3 Uncertainty Metrics

**Định nghĩa 13.2 (Predictive Entropy).**

$$\mathcal{H}[\mathbf{y}|\mathbf{x}, \mathcal{D}] = -\sum_{k=1}^{K} \bar{p}_k \log \bar{p}_k, \quad \bar{p}_k = \frac{1}{T}\sum_{t=1}^{T} p_{t,k}$$

Đo tổng uncertainty (aleatoric + epistemic).

**Định nghĩa 13.3 (BALD — Bayesian Active Learning by Disagreement).**

$$\mathcal{I}[\mathbf{y}; \boldsymbol{\omega}|\mathbf{x}, \mathcal{D}] = \mathcal{H}[\mathbf{y}|\mathbf{x}, \mathcal{D}] - \mathbb{E}_{q(\boldsymbol{\omega})}[\mathcal{H}[\mathbf{y}|\mathbf{x}, \boldsymbol{\omega}]]$$

$$= \underbrace{-\sum_k \bar{p}_k \log \bar{p}_k}_{\text{predictive entropy}} + \underbrace{\frac{1}{T}\sum_{t=1}^{T}\sum_k p_{t,k}\log p_{t,k}}_{\text{expected entropy}}$$

Đo **epistemic uncertainty** only — uncertainty giảm được bằng thêm data.

**Mệnh đề 13.1.** BALD ≥ 0, và BALD = 0 khi và chỉ khi tất cả MC samples cho cùng prediction.

**Chứng minh.** BALD = $\mathcal{H}[\bar{p}] - \frac{1}{T}\sum_t \mathcal{H}[p_t]$. Theo Jensen's inequality: $\mathcal{H}[\frac{1}{T}\sum p_t] \geq \frac{1}{T}\sum \mathcal{H}[p_t]$ (entropy là concave). Dấu bằng khi $p_t = \bar{p} \;\forall t$. $\blacksquare$

**Định nghĩa 13.4 (Variation Ratio).**

$$\text{VR} = 1 - \frac{|\{t : \arg\max_k p_{t,k} = \hat{c}\}|}{T}, \quad \hat{c} = \text{mode}\{\arg\max_k p_{t,k}\}_{t=1}^{T}$$

VR = 0: tất cả MC passes đồng ý; VR → 1: MC passes không nhất quán.

### 13.4 Combined Uncertainty Score

$$U_{\text{combined}} = \alpha_1 \cdot \hat{\mathcal{H}} + \alpha_2 \cdot \hat{\mathcal{I}} + \alpha_3 \cdot \text{VR}$$

với $\alpha_1 = 0.4, \alpha_2 = 0.4, \alpha_3 = 0.2$. Hats ($\hat{\cdot}$) denote normalized values ∈ [0, 1].

---

## 14. Active Learning Acquisition

**File:** `RL/active_learning.py`

### 14.1 Acquisition Function

**Định nghĩa 14.1.** Acquisition score cho relationship $r_k$:

$$\alpha(r_k) = \underbrace{U(r_k)}_{\text{Uncertainty}} + \lambda_1 \underbrace{\Delta_{\text{perf}}(r_k)}_{\text{Performance Gap}} + \lambda_2 \underbrace{\tau(r_k)}_{\text{Tail Weight}}$$

**Uncertainty $U(r_k)$:** MC Dropout combined score (Section 13.4).

**Performance Gap:**

$$\Delta_{\text{perf}}(r_k) = \max(0, \text{target\_F1} - \text{current\_F1}(p_k))$$

**Tail Weight** (khuyến khích rare predicates):

$$\tau(r_k) = 1 - \frac{\text{count}(p_k)}{\max_{p'} \text{count}(p') + \epsilon}$$

Predicates hiếm ($\tau \to 1$) được ưu tiên sinh thêm data.

### 14.2 Budget Allocation

$$n_k = \text{round}\left(N_{\text{budget}} \cdot \frac{\alpha(r_k)}{\sum_{k'} \alpha(r_{k'})}\right)$$

$n_k$ = số ảnh synthetic cần sinh cho relationship $r_k$. Total budget $N_{\text{budget}}$ do DQN action quyết định.

### 14.3 Generation Plan

**Thuật toán 14.1 (Active Learning Plan):**

```
Input:  Relationships R, uncertainty U, performance metrics M, budget N
Output: Generation plan P = {(rₖ, nₖ)}

1. for each r ∈ R:
2.   α(r) = U(r) + λ₁·Δ_perf(r) + λ₂·τ(r)
3. Sort R by α(r) descending
4. Top-K ← R[:K]  (focus on most informative)
5. for each r ∈ Top-K:
6.   nₖ = round(N · α(r) / Σα)
7.   nₖ = clip(nₖ, 1, max_per_rel)
8. return P
```

---

## 15. Greedy Submodular Maximization

**File:** `RL/approximation_algorithm.py`

### 15.1 Submodular Function

**Định nghĩa 15.1 (Submodularity).** Hàm $f: 2^\Omega \rightarrow \mathbb{R}$ là submodular nếu $\forall A \subseteq B \subseteq \Omega$ và $\forall x \in \Omega \setminus B$:

$$f(A \cup \{x\}) - f(A) \geq f(B \cup \{x\}) - f(B)$$

(Diminishing returns property.)

### 15.2 Objective Function

$$f(S) = \underbrace{\lambda_{\text{div}} \sum_{c \in \mathcal{C}} \log(1 + |S \cap S_c|)}_{\text{Diversity (submodular)}} + \underbrace{\lambda_{\text{qual}} \sum_{x \in S} q(x)}_{\text{Quality (modular)}} + \underbrace{\lambda_{\text{rep}} \cdot \text{Coverage}(S, \Omega)}_{\text{Representativeness}}$$

**Diversity term:** Log-count per class — concave → submodular.

**Quality term:** $q(x) = $ quality score of sample $x$ (image quality, annotation quality).

**Representativeness:** Coverage of original data distribution.

**Mệnh đề 15.1.** $f(S)$ là submodular.

**Chứng minh.** (i) Log-count: $g(S) = \log(1 + |S \cap S_c|)$ là concave function of $|S \cap S_c|$, hence submodular. Sum of submodular functions is submodular. (ii) Quality: linear → modular → submodular. (iii) Sum preserves submodularity. $\blacksquare$

### 15.3 Greedy Algorithm

**Thuật toán 15.1 (Greedy Submodular Maximization):**

```
Input:  Ground set Ω = {x₁, ..., xₘ}, budget k, submodular function f
Output: Selected subset S* ⊆ Ω, |S*| ≤ k

1. S* ← ∅
2. for i = 1, ..., k:
3.   x* ← argmax_{x ∈ Ω\S*} [f(S* ∪ {x}) - f(S*)]   // marginal gain
4.   if f(S* ∪ {x*}) - f(S*) ≤ 0:
5.     break  // no improvement possible
6.   S* ← S* ∪ {x*}
7. return S*
```

**Độ phức tạp:** $O(k \cdot |\Omega| \cdot T_f)$ với $T_f$ = cost tính $f$.

### 15.4 Approximation Guarantee

**Định lý 15.1 (Nemhauser, Wolsey & Fisher, 1978).** Cho $f$ monotone submodular, normalized ($f(\emptyset) = 0$), thuật toán greedy đạt:

$$f(S_{\text{greedy}}) \geq \left(1 - \frac{1}{e}\right) \cdot f(S^*) \approx 0.632 \cdot f(S^*)$$

trong đó $S^* = \arg\max_{|S| \leq k} f(S)$ là nghiệm tối ưu.

**Chứng minh.** Gọi $S^* = \{o_1^*, \ldots, o_k^*\}$ và $S_i$ là tập sau $i$ bước greedy.

**Bước 1.** Do submodularity, tại bước $i$:

$$f(S^*) - f(S_i) \leq \sum_{j=1}^{k} [f(S_i \cup \{o_j^*\}) - f(S_i)] \leq k \cdot [f(S_{i+1}) - f(S_i)]$$

bất đẳng thức cuối do greedy chọn element có marginal gain lớn nhất.

**Bước 2.** Gọi $\delta_i = f(S^*) - f(S_i)$. Từ bước 1:

$$\delta_{i+1} \leq \delta_i - \frac{\delta_i}{k} = \delta_i\left(1 - \frac{1}{k}\right)$$

**Bước 3.** Bằng induction:

$$\delta_k \leq \delta_0 \left(1 - \frac{1}{k}\right)^k \leq f(S^*) \cdot \frac{1}{e}$$

Do đó: $f(S_k) = f(S^*) - \delta_k \geq f(S^*)(1 - 1/e)$. $\blacksquare$

---

## 16. Synthetic Data Generation & Auto-Annotation

### 16.1 Stable Diffusion Image Generation

**File:** `RL/ai_images_generator.py`

**Prompt Templates** cho mỗi relationship type:

$$\text{prompt}(s, p, o) = \text{Template}(p) + \text{", "} + s + \text{" "} + p + \text{" "} + o$$

Ví dụ: `Template("riding") = "A realistic photo of"` → `"A realistic photo of person riding horse"`.

**Biến thể prompt** (tăng diversity):
- Style variations: "photorealistic", "natural lighting", "outdoor scene"
- Negative prompt: "blurry, low quality, deformed, cartoon"
- Guidance scale: $\omega \in [7.5, 12.0]$

**Quality Filtering:**

$$\text{Accept}(I) = \text{sharpness}(I) > \theta_s \wedge \text{brightness} \in [\theta_l, \theta_h] \wedge \neg\text{IsDuplicate}(I)$$

**Duplicate Detection** sử dụng perceptual hashing (imagehash):

$$\text{IsDuplicate}(I) = \min_{J \in \mathcal{D}_{\text{existing}}} \text{HammingDist}(\text{pHash}(I), \text{pHash}(J)) < \theta_{\text{dup}}$$

### 16.2 Auto-Annotation Pipeline

**File:** `RL/auto_annotator.py`

**Định nghĩa 16.1.** Auto-Annotation function $\mathcal{A}: \text{Image} \times \text{TextPrompts} \rightarrow \{(\text{bbox}, \text{class}, \text{confidence})\}^*$

**Backend Priority Chain:**

```
GroundingDINO (box_threshold=0.25, text_threshold=0.20)
    ↓ fallback
OWL-ViT (score_threshold=0.15)
    ↓ fallback
YOLO + CLIP (detect → classify)
    ↓ fallback
Pseudo-Annotation (heuristic bboxes)
```

**Pseudo-Annotation** (last resort, low quality):

| Relationship Type | Subject Region | Object Region |
|---|---|---|
| `riding`, `sitting on` | Top 30-70% | Bottom 50-90% |
| `holding`, `using` | Left 20-55% | Right 45-80% |
| `near`, `next to` | Left 10-45% | Right 55-90% |

### 16.3 Data Augmentation

**File:** `RL/data_augmentation.py`

Albumentations pipeline tạo biến thể:

$$I' = T_{\text{aug}}(I), \quad T_{\text{aug}} = T_1 \circ T_2 \circ \ldots \circ T_7$$

| $T_i$ | Tham số | $P(T_i)$ |
|---|---|---|
| RandomBrightnessContrast | $\Delta_b = \pm 0.2$ | 0.5 |
| HueSaturationValue | $\Delta_h = \pm 20°$ | 0.5 |
| RandomRotate90 | $\{90°, 180°, 270°\}$ | 0.3 |
| HorizontalFlip | — | 0.5 |
| RandomScale | $\pm 20\%$ | 0.5 |
| GaussNoise | $\sigma^2 \in [10, 50]$ | 0.3 |
| Blur | kernel $\leq 3$ | 0.3 |

---

## 17. Tổng kết End-to-End Pipeline

### 17.1 Inference Pipeline (1 frame)

```
I ∈ ℝ^{H×W×3}
  │
  ├─ YOLOv11(I) → 𝒪_coco = {(bᵢ, cᵢ, sᵢ)}     [Section 3]
  ├─ FireModel(I) → 𝒪_fire                         [Section 3.4]
  ├─ Merge(𝒪_coco, 𝒪_fire) → 𝒪_merged
  │
  ├─ CLIP(crop(I, bᵢ)) → override cᵢ if needed     [Section 4]
  │
  ├─ Hook(YOLO.layer9) → F ∈ ℝ^{512×H/32×W/32}    [Section 5.1]
  ├─ GAP(F) → g ∈ ℝ^{512}                           [Section 5.4]
  ├─ RoIAlign(F, bᵢ) → vᵢ ∈ ℝ^{512×7×7}           [Section 5.2]
  │
  ├─ RelTR(I, g) → ℛ_raw = {(sₖ, pₖ, oₖ, σₖ)}    [Section 6]
  │    └─ Backbone → Encoder(×6) → Decoder(×6) → Heads
  │
  ├─ SpatialValidation(ℛ_raw) → ℛ_valid             [Section 7.2]
  ├─ SemanticValidation(ℛ_valid) → ℛ_filtered        [Section 7.3]
  │
  ├─ LLM_Enhance(ℛ_filtered, low-conf pairs)         [Section 8]
  │    └─ VisualFeatures → Prompt → GPT-4V → Parse
  │
  └─ SafetyClassify(ℛ_enhanced) → Scene Graph G      [Section 10]
       └─ Tier1(WB list) → Tier2(LLM) → Tier3(Rules)

Output: G = (𝒪, ℛ, Safety Labels)
```

### 17.2 Training Loop (Data Loop)

```
for epoch = 1, ..., E:
  │
  ├─ DQN.decide_action(sₜ) → aₜ, plan               [Section 11]
  │    └─ Active Learning → Acquisition Scores          [Section 14]
  │
  ├─ Generate AI Images (Stable Diffusion)             [Section 16.1]
  │    └─ Per-relationship variations (from plan)
  │
  ├─ Auto-Annotate(images) → labeled dataset           [Section 16.2]
  │    └─ GroundingDINO/OWL-ViT/YOLO+CLIP
  │
  ├─ Greedy Submodular Select(dataset, k)              [Section 15]
  │    └─ Maximize f(S) = diversity + quality + repr.
  │
  ├─ Fine-tune YOLO + RelTR on selected data
  │
  ├─ Evaluate → Detection F1, Relationship F1, mR@K
  │
  ├─ Calculate Reward R(sₜ, aₜ) (6 components)        [Section 12]
  │    └─ MC Dropout Uncertainty                        [Section 13]
  │
  └─ DQN Update: Q(s,a;θ) ← Bellman backup            [Section 11.6]
```

### 17.3 Evaluation Metrics

| Metric | Công thức | Mô tả |
|---|---|---|
| Precision@K | $\frac{\text{TP}@K}{\text{TP}@K + \text{FP}@K}$ | Tỷ lệ dự đoán đúng trong top-K |
| Recall@K | $\frac{\text{TP}@K}{|\text{GT}|}$ | Tỷ lệ GT được phát hiện trong top-K |
| mR@K | $\frac{1}{|\mathcal{P}|}\sum_{p}\text{R}@K(p)$ | Mean recall qua tất cả predicates |
| Detection F1 | $2pr/(p+r)$ | Harmonic mean P, R cho detection |
| Relationship F1 | $2pr/(p+r)$ | Harmonic mean P, R cho relationships |

### 17.4 Hyperparameters tổng hợp

| Parameter | Giá trị | Nguồn |
|---|---|---|
| $d_{\text{model}}$ (RelTR) | 256 | `transformer.py` |
| $n_{\text{heads}}$ | 8 | `transformer.py` |
| $d_{\text{ff}}$ | 2048 | `transformer.py` |
| $L_{\text{enc}}, L_{\text{dec}}$ | 6, 6 | `transformer.py` |
| $N_{\text{entity}}$ | 100 | `reltr.py` |
| $N_{\text{triplet}}$ | 200 | `reltr.py` |
| $\gamma$ (DQN discount) | 0.95 | `reinforcement_learning.py` |
| $\varepsilon_0, \varepsilon_{\min}, \varepsilon_{\text{decay}}$ | 1.0, 0.01, 0.995 | `reinforcement_learning.py` |
| Batch size (DQN) | 32 | `reinforcement_learning.py` |
| Buffer capacity | 10,000 | `reinforcement_learning.py` |
| Target update freq | 10 steps | `reinforcement_learning.py` |
| MC Dropout passes $T$ | 30 | `uncertainty_estimator.py` |
| CLIP threshold | 0.65 | `detect_objects.py` |
| NMS IoU threshold | 0.45 | `detect_objects.py` |
| GroundingDINO box threshold | 0.25 | `auto_annotator.py` |

---

## Tài liệu Tham khảo

1. **RelTR:** Y. Cong et al., "RelTR: Relation Transformer for Scene Graph Generation," *ACM Multimedia*, 2022.
2. **DETR:** N. Carion et al., "End-to-End Object Detection with Transformers," *ECCV*, 2020.
3. **YOLOv11:** G. Jocher et al., "Ultralytics YOLO," 2024. https://github.com/ultralytics/ultralytics
4. **CLIP:** A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," *ICML*, 2021.
5. **RoIAlign:** K. He et al., "Mask R-CNN," *ICCV*, 2017.
6. **GIoU:** H. Rezatofighi et al., "Generalized Intersection over Union," *CVPR*, 2019.
7. **Hungarian Algorithm:** H. W. Kuhn, "The Hungarian Method for the Assignment Problem," *Naval Research Logistics*, 1955.
8. **DQN:** V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015.
9. **Q-Learning Convergence:** C. J. C. H. Watkins and P. Dayan, "Q-Learning," *Machine Learning*, 1992.
10. **MC Dropout:** Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," *ICML*, 2016.
11. **BALD:** N. Houlsby et al., "Bayesian Active Learning for Classification and Preference Learning," *arXiv*, 2011.
12. **Submodular Maximization:** G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher, "An analysis of approximations for maximizing submodular set functions," *Mathematical Programming*, 1978.
13. **Stable Diffusion:** R. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," *CVPR*, 2022.
14. **ByteTrack:** Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," *ECCV*, 2022.
15. **GroundingDINO:** S. Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training," *arXiv*, 2023.
16. **Focal Loss:** T.-Y. Lin et al., "Focal Loss for Dense Object Detection," *ICCV*, 2017.

