# Phân Tích Chi Tiết Hệ Thống VRD Toolkit

**Visual Relationship Detection + Reinforcement Learning Self-Improvement**

---

## Mục Lục

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Tầng Phát Hiện Vật Thể](#2-tầng-phát-hiện-vật-thể)
3. [Tầng Dự Đoán Quan Hệ](#3-tầng-dự-đoán-quan-hệ)
4. [Video Pipeline](#4-video-pipeline)
5. [Module Học Tăng Cường (RL)](#5-module-học-tăng-cường-rl)
   - [5.1 Kiến Trúc Tổng Quan](#51-kiến-trúc-tổng-quan-rl)
   - [5.2 Deep Q-Network (DQN)](#52-deep-q-network-dqn-agent)
   - [5.3 Uncertainty Estimator (MC Dropout)](#53-uncertainty-estimator-mc-dropout)
   - [5.4 Active Learning Selector](#54-active-learning-selector)
   - [5.5 Approximation Algorithm](#55-approximation-algorithm-greedy-submodular)
   - [5.6 Reward Function](#56-reward-function-adaptive)
   - [5.7 Long-Tail Handling](#57-long-tail-handling)
   - [5.8 Sinh Ảnh AI](#58-sinh-ảnh-ai-stable-diffusion)
   - [5.9 Auto-Annotation](#59-auto-annotation)
   - [5.10 Evaluation Metrics](#510-evaluation-metrics)
6. [Training Episode Flow](#6-training-episode-flow)
7. [Application Layer](#7-application-layer)
8. [Các File Hỗ Trợ](#8-các-file-hỗ-trợ)

---

## 1. Tổng Quan Kiến Trúc

Hệ thống là một pipeline phát hiện quan hệ giữa vật thể trong ảnh/video (Visual Relationship Detection), được tăng cường bởi **Học Tăng Cường (RL)** để tự động cải thiện hiệu suất mô hình qua thời gian.

### Bốn tầng chính

| Tầng | Chức năng | File chính |
|------|-----------|------------|
| **Detection** | Phát hiện vật thể (bbox + class) | `detect_objects.py` |
| **Relationship** | Dự đoán quan hệ giữa các vật thể | `boundingbox_objects.py`, `models/reltr.py` |
| **RL Training** | Học tăng cường tự cải thiện mô hình | `RL/reinforcement_learning.py` và cả thư mục `RL/` |
| **Application** | Giao diện người dùng (GUI + Video) | `app.py`, `video_relation_pipeline.py` |

### Sơ đồ luồng dữ liệu tổng quan

```
Ảnh đầu vào
    │
    ▼
┌──────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│  YOLO + CLIP │────▶│  RelTR Model    │────▶│  Relationships JSON │
│  (Detection) │     │  (Relationship) │     │  (subject,rel,object)│
└──────────────┘     └─────────────────┘     └──────────┬──────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────────┐
                                              │  RL Training Loop   │
                                              │  (Self-Improvement) │
                                              └─────────────────────┘
```

---

## 2. Tầng Phát Hiện Vật Thể

**File**: `detect_objects.py`

Tầng này kết hợp **3 mô hình** để phân loại vật thể với độ chính xác cao:

### 2.1 YOLO (You Only Look Once)

- Load model từ `fine-tune.pt` (đã fine-tune trên dữ liệu custom)
- Chạy inference → bounding boxes, class IDs, confidence scores
- Hỗ trợ thêm **Fire Detection model** riêng biệt (tùy chọn)
- Kết quả 2 model được merge qua **NMS** (Non-Maximum Suppression) để loại bỏ trùng lặp

### 2.2 CLIP (Contrastive Language-Image Pre-training)

- Dùng **CLIP ViT-B/32** để phân loại vật thể bằng open-vocabulary (~200+ class)
- Text features được **pre-compute một lần** lúc load module → tối ưu tốc độ đáng kể
- Logic quyết định nhãn cuối cùng:
  - Nếu CLIP confidence < 0.3 → giữ nhãn YOLO
  - Nếu nhãn YOLO nằm trong `important_labels` (tất cả class YOLO + Fire model) → giữ nhãn YOLO
  - Còn lại → dùng nhãn CLIP

### 2.3 ROI Feature Extraction

- Hook vào backbone YOLO (layer SPPF, index 9) để lấy feature map
- Dùng **RoIAlign** (7×7 pooling) trích xuất feature vector cho mỗi bbox
- Tính **Global Context Vector**: Global Average Pooling → L2 normalization

### Output

```json
{
  "image_path": "path/to/image.jpg",
  "objects": [
    {
      "label": "person",
      "bbox": [x1, y1, x2, y2],
      "feature": [0.123, 0.456, ...]
    }
  ],
  "global_context": [0.01, 0.02, ...]
}
```

---

## 3. Tầng Dự Đoán Quan Hệ

**File**: `boundingbox_objects.py`

### 3.1 RelTR (Relationship Transformer)

- Kiến trúc dựa trên **DETR** (DEtection TRansformer):
  - **Backbone**: ResNet-50
  - **Encoder**: 6 Transformer encoder layers
  - **Decoder**: 6 Transformer decoder layers
  - **Hidden dim**: 256, **Heads**: 8, **FFN dim**: 2048
- Predict **51 loại quan hệ** (`RELATION_CLASSES`) giữa các cặp vật thể
- Sử dụng **subject/object boxes** từ output + **IoU matching** để gán quan hệ cho đúng cặp

### 3.2 Spatial Validation

Kiểm tra tính hợp lý không gian của quan hệ:

| Sai | Sửa thành |
|-----|-----------|
| person **wearing** car | person **riding** car |
| dog **wearing** skateboard | dog **on** skateboard |
| person **has** horse | person **riding** horse |

Các rule được hard-code trong `SEMANTIC_CORRECTIONS` list.

### 3.3 LLM Enhancement (tùy chọn)

- Tích hợp LLM predictor cho open-vocabulary relationship
- Kích hoạt khi confidence của RelTR < 0.6 (`LLM_CONFIDENCE_THRESHOLD`)
- Hoạt động như **fallback** khi mô hình chính không chắc chắn

---

## 4. Video Pipeline

**File**: `video_relation_pipeline.py`

Mở rộng pipeline ảnh sang xử lý video real-time:

| Tính năng | Mô tả |
|-----------|-------|
| **Frame Processing** | Xử lý từng frame với configurable stride (mặc định stride=2) |
| **Object Tracking** | Gán `track_id` cho vật thể qua các frame |
| **Safety Monitoring** | Phát hiện xâm nhập vùng an toàn (mặc định < 2m) |
| **Safety Classifier** | Phân loại mức nguy hiểm: Safe / Warning / Danger |
| **Callback Pattern** | `on_frame` (hiển thị) + `on_relations` (cập nhật UI) |

---

## 5. Module Học Tăng Cường (RL)

> **Đây là phần nặng nhất về toán và là trái tim của hệ thống cải thiện tự động.**

### 5.1 Kiến Trúc Tổng Quan RL

```
┌──────────────────────────────────────────────────────────────────────┐
│                      TRAINING EPISODE LOOP                            │
│                                                                       │
│  ┌───────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐ │
│  │ DQN Agent │───▶│  Active    │───▶│   Image    │───▶│  Approx.   │ │
│  │ (decide)  │    │  Learning  │    │  Generator │    │  Algorithm │ │
│  └───────────┘    └────────────┘    └────────────┘    └────────────┘ │
│       │                │                  │                 │         │
│       ▼                ▼                  ▼                 ▼         │
│  ┌───────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐ │
│  │ State →   │    │Uncertainty │    │  Stable    │    │  Greedy    │ │
│  │ Q-values  │    │ Estimator  │    │ Diffusion  │    │ Submodular │ │
│  │ → Action  │    │ MC Dropout │    │  Pipeline  │    │ Selection  │ │
│  └───────────┘    └────────────┘    └────────────┘    └────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                    REWARD CALCULATION                           │   │
│  │  R = Σ wᵢ(dynamic) × sᵢ  + uncertainty_reduction bonus       │   │
│  └────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

**Bốn kỹ thuật toán học cốt lõi:**

1. **DQN** (Deep Q-Learning) → Quyết định action
2. **MC Dropout Uncertainty** (BALD) → Đo độ không chắc chắn
3. **Active Learning** (Acquisition Functions) → Ưu tiên relationship yếu
4. **Greedy Submodular Maximization** → Chọn subset tối ưu (guarantee 63.2% OPT)

---

### 5.2 Deep Q-Network (DQN) Agent

**File**: `RL/reinforcement_learning.py`, class `RelationshipReinforcementLearning`

#### Q-Network Architecture

```
Q(s, a) : ℝ⁵ → ℝ¹⁰

Linear(5, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 10)
```

Hai mạng riêng biệt:
- **Q-network**: Dùng để chọn action, được update mỗi step
- **Target network**: Dùng để tính target value, sync mỗi 20 steps

#### State Vector (s ∈ ℝ⁵)

| Index | Thành phần | Ý nghĩa | Normalize |
|-------|-----------|---------|-----------|
| 0 | `detection_f1` | F1 score phát hiện vật thể | Trực tiếp [0, 1] |
| 1 | `relationship_f1` | F1 score dự đoán quan hệ | Trực tiếp [0, 1] |
| 2 | `reward` | Reward gần nhất | tanh(r / 1.0) |
| 3 | `dataset_size` | Kích thước dataset hiện tại | tanh(\|D\| / 50.0) |
| 4 | `epsilon` | Tỷ lệ exploration hiện tại | tanh(ε / 1.0) |

> **Fallback**: Nếu F1 không có, dùng `1 - tanh(loss / 5.0)` thay thế.

#### Action Space

```
A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
```

Mỗi action `a` quyết định **số lượng biến thể ảnh synthetic** (base_variations) cho mỗi relationship. Action 1 = sinh ít ảnh (tiết kiệm), action 10 = sinh nhiều ảnh (toàn diện).

#### Epsilon-Greedy Policy

```
         ┌ random action          với xác suất ε
a(s) =   │
         └ argmax_a Q(s, a)       với xác suất (1 - ε)
```

Epsilon decay:

```
ε ← max(ε_min, ε × ε_decay)

ε₀ = 0.9       (ban đầu explore 90%)
ε_decay = 0.995
ε_min = 0.01   (explore tối thiểu 1%)
```

#### Q-Network Update

**Bellman equation:**

```
Q_target(s, a) = r + γ × max_{a'} Q_target_network(s', a') × (1 - done)
```

**Loss function:**

```
L = MSE(Q(s, a), Q_target(s, a))
```

**Hyperparameters:**

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| γ (gamma) | 0.95 | Discount factor |
| Batch size | 32 | Số experience mỗi lần update |
| Replay buffer | 10,000 | Kích thước bộ nhớ kinh nghiệm |
| Target sync | Mỗi 20 steps | Tần suất sync target network |
| Learning rate | 1e-3 | AdamW optimizer |
| Gradient clip | max_norm = 1.0 | Tránh exploding gradients |

---

### 5.3 Uncertainty Estimator (MC Dropout)

**File**: `RL/uncertainty_estimator.py`

Module đo **mức độ không chắc chắn** của model RelTR bằng Monte Carlo Dropout.

#### Ý Tưởng Cốt Lõi

Bình thường `model.eval()` tắt hết Dropout → output deterministic.
MC Dropout **giữ dropout bật** khi inference → mỗi forward pass cho kết quả khác nhau (stochastic) → tạo "ensemble ảo" mà **không cần train thêm model**.

#### Thuật Toán

```
1. Giữ model weights cố định
2. Bật tất cả Dropout layers → module.train()
3. Forward T = 10 lần cùng 1 input → T bộ predictions khác nhau
4. Tính uncertainty metrics từ sự biến thiên giữa T predictions
5. Tắt dropout → module.eval()
```

#### Ba Uncertainty Metrics

##### 1) Predictive Entropy — Tổng Uncertainty (Aleatoric + Epistemic)

```
H[ȳ] = −Σ p̄(y) × log p̄(y)
```

Trong đó `p̄(y)` là tần suất xuất hiện của relationship fingerprint `(subject, relation, object)` qua T forward passes.

- Normalize bởi `log(|unique outcomes|)` để nằm trong `[0, 1]`
- **H cao** = model rất không chắc chắn tổng thể

##### 2) Mutual Information (BALD Score) — Chỉ Epistemic Uncertainty

```
I[y, θ] = H[ȳ] − E_θ[H[y|θ]]
```

| Thành phần | Ý nghĩa |
|-----------|---------|
| `H[ȳ]` | Predictive Entropy (total uncertainty) |
| `E_θ[H[y\|θ]]` | Trung bình entropy từng forward pass (data uncertainty) |
| `I = H − E[H]` | **Model uncertainty only** (epistemic) |

- **I cao** = model thiếu knowledge → **nên thu thập thêm dữ liệu cho case này**
- Đây là metric quan trọng nhất cho Active Learning

##### 3) Variation Ratio

```
VR = 1 − (count of mode) / T
```

| Giá trị | Ý nghĩa |
|---------|---------|
| VR = 0 | Tất cả T passes cho cùng kết quả (confident) |
| VR → 1 | Mỗi pass cho kết quả khác nhau (very uncertain) |

#### Combined Uncertainty Score

```
U = 0.30 × Entropy
  + 0.30 × Mutual Information
  + 0.20 × Variation Ratio
  + 0.20 × (1 − mean_confidence)
```

- **U ∈ [0, 1]**
- **Cao** = rất không chắc chắn → cần thu thập thêm data
- **Thấp** = model tự tin → không cần ưu tiên

---

### 5.4 Active Learning Selector

**File**: `RL/active_learning.py`

Quyết định **sinh bao nhiêu ảnh cho mỗi relationship** dựa trên uncertainty + performance + rarity.

#### Bốn Chiến Lược (Strategies)

| Strategy | Công thức | Khi nào dùng |
|----------|-----------|-------------|
| `uncertainty` | α(r) = U(r) | Chỉ dùng uncertainty |
| `performance` | α(r) = 1 − F₁(r) | Chỉ dùng inverse F1 |
| `combined` | α(r) = w_u·U + w_p·(1−F₁) + w_t·τ | **Mặc định** — kết hợp tất cả |
| `random` | α(r) = random() | Baseline để so sánh |

#### Acquisition Function (Combined Strategy)

```
α(r) = 0.40 × U(r) + 0.35 × (1 − F₁(r)) + 0.25 × min(τ(r), 1.0)
```

| Thành phần | Trọng số | Ý nghĩa |
|-----------|---------|---------|
| U(r) | 0.40 | Uncertainty score (từ MC Dropout) |
| 1 − F₁(r) | 0.35 | Performance gap (F1 thấp → cần tập trung hơn) |
| τ(r) | 0.25 | Tail weight (relationship hiếm → ưu tiên) |

#### Budget Allocation Algorithm (Proportional)

```
Phase 1: Mỗi relationship nhận min_per_rel = 1 ảnh
Phase 2: Budget còn lại phân bổ theo tỷ lệ acquisition score:

    n_r = floor( α(r) / Σα(r') × B_remaining )

Phase 3: Phần dư phân theo largest fractional part
Phase 4: Cap tại max_per_rel
```

#### Ví Dụ Cụ Thể

Giả sử 3 relationships, budget = 15:

| Relationship | U(r) | F₁ | τ(r) | α(r) | Ảnh sinh |
|-------------|------|-----|------|------|----------|
| person riding horse | 0.8 | 0.8 | 0.1 | 0.42 | **7** |
| dog sitting on chair | 0.3 | 0.2 | 0.8 | 0.60 | **8** |
| person holding phone | 0.6 | 0.5 | 0.3 | 0.42 | **7** |

→ "dog sitting on chair" được sinh **nhiều nhất** vì F1 thấp (0.2) và là relationship hiếm (tail=0.8).

---

### 5.5 Approximation Algorithm (Greedy Submodular)

**File**: `RL/approximation_algorithm.py`

Chọn **tập con tối ưu** từ pool ảnh synthetic đã sinh. Bài toán tìm tập con S tối ưu từ pool P là **NP-hard**, nhưng khi objective function f là **submodular** thì thuật toán Greedy đạt được bound lý thuyết.

#### Thuật Toán Greedy

```
S = ∅
for i = 1 to budget:
    s* = argmax_{s ∈ Pool \ S} [ f(S ∪ {s}) − f(S) ]   // marginal gain
    S = S ∪ {s*}
    if marginal_gain < 1e-6 and i > budget/2:
        break   // early stopping
return S
```

#### Approximation Guarantee

> **Định lý** (Nemhauser et al. 1978): Nếu f là submodular và monotone, thuật toán Greedy đạt:
>
> ```
> f(S_greedy) ≥ (1 − 1/e) × f(S_optimal) ≈ 0.632 × OPT
> ```
>
> Nghĩa là: kết quả Greedy **đảm bảo ít nhất 63.2%** giá trị tối ưu.

#### Submodular Objective Function

```
f(S) = λ₁ × Diversity(S) + λ₂ × Quality(S) + λ₃ × Representativeness(S)
```

| Thành phần | λ | Cách tính |
|-----------|---|----------|
| **Diversity** | 0.5 | Min Euclidean distance từ candidate đến S trong feature space 7D |
| **Quality** | 0.3 | Trung bình: annotation confidence + backend quality + relationship completeness + bbox validity |
| **Representativeness** | 0.2 | Tỷ lệ unselected samples mà candidate trở thành nearest neighbor |

#### Feature Space (7 chiều)

```
features[0-1]: Normalized bbox center (x, y)
features[2]:   Bbox area ratio
features[3]:   Relationship type index (encoded)
features[4]:   Subject class index
features[5]:   Object class index
features[6]:   Annotation confidence
```

#### Backend Quality Mapping

| Backend | Quality Score |
|---------|-------------|
| GroundingDINO | 0.9 |
| YOLO+CLIP | 0.8 |
| OWL-ViT | 0.7 |
| Pseudo | 0.3 |

#### Deduplication

Trước khi chọn subset, loại bỏ samples quá giống nhau (Euclidean distance < `min_distance = 0.08`).

---

### 5.6 Reward Function (Adaptive)

**File**: `RL/reinforcement_learning.py`, method `calculate_reward()`

Hệ thống tính reward đa thành phần với **trọng số động** (dynamic weights).

#### Công Thức Tổng Hợp

```
R = Σᵢ wᵢ(dynamic) × sᵢ
```

Với 6 thành phần:

```
R = w_det × s_det
  + w_rel × s_rel
  + w_div × s_div
  + w_con × s_con
  + w_imp × s_imp
  + w_unc × s_unc
```

#### Chi Tiết Từng Thành Phần

##### 1) Detection Score (s_det)

```
s_det = F₁(det) × min(n/10, 1) × (1 − |P−R| / (P+R))
```

| Yếu tố | Ý nghĩa |
|--------|---------|
| F₁(det) | F1 score phát hiện vật thể |
| min(n/10, 1) | Sample confidence (n = số sample đánh giá) |
| 1 − \|P−R\| / (P+R) | Balance factor (P và R cân bằng → tốt) |

##### 2) Relationship Score (s_rel)

```
s_rel = F₁(rel) × (1 − σ_F₁) × min(n/20, 1) × balance × (1 + τ)
```

| Yếu tố | Ý nghĩa |
|--------|---------|
| F₁(rel) | F1 score dự đoán quan hệ |
| (1 − σ_F₁) | Stability factor (std thấp = ổn định = tốt) |
| min(n/20, 1) | Sample confidence |
| balance | Precision-Recall balance |
| (1 + τ) | Long-tail bonus |

##### 3) Diversity Score (s_div) — 3 thành phần con

```
s_div = 0.4 × relation_diversity + 0.4 × class_diversity + 0.2 × spatial_diversity
```

| Thành phần | Cách tính |
|-----------|----------|
| Relation diversity | min(\|unique_relations\| / 10, 1) |
| Class diversity | min(\|unique_subjects ∪ objects\| / 15, 1) |
| Spatial diversity | 0.4×position + 0.3×size + 0.3×coverage |

**Spatial Diversity** chi tiết:
- **Position diversity**: Variance của normalized center points, smooth bằng tanh
- **Size diversity**: Coefficient of variation của area (60%) + aspect ratio (40%)
- **Coverage diversity**: Grid 4×4 analysis — coverage ratio (70%) + entropy (30%)

##### 4) Consistency Score (s_con)

```
s_con = 0.7 × max(0, 1 − σ) + 0.3 × trend_score
```

- σ = standard deviation của per-sample F1 scores
- trend_score = linear regression slope trên lịch sử F1 (cải thiện theo thời gian → điểm cao)

##### 5) Improvement Score (s_imp)

Tính từ xu hướng cải thiện qua các epoch trước đó trong `performance_history`.

##### 6) Uncertainty Reduction Score (s_unc) — **MỚI**

```
raw = (Ū_prev − Ū_current) / Ū_prev     // ∈ [-1, 1]
s_unc = 0.5 + 0.5 × raw                   // map → [0, 1]
```

| Giá trị | Ý nghĩa |
|---------|---------|
| s_unc > 0.5 | Uncertainty giảm (training đang hiệu quả) |
| s_unc = 0.5 | Không thay đổi |
| s_unc < 0.5 | Uncertainty tăng (training chưa tốt) |

#### Dynamic Weights

Trọng số được tính **adaptive** dựa trên performance hiện tại:
- Nếu detection score thấp → tăng w_det
- Nếu relationship score thấp → tăng w_rel
- Uncertainty reduction nhận **cố định 10%** (w_unc = 0.10)
- Các weight còn lại được **re-normalize** để tổng = 90%

#### Reward Scaling

Sau khi tính R, áp dụng hàm scaling để đảm bảo reward nằm trong khoảng hợp lý và tránh extreme values.

---

### 5.7 Long-Tail Handling

Quan hệ hiếm (ít xuất hiện trong dataset) cần được ưu tiên đặc biệt.

#### Công Thức Tail Weight

```
τ_raw(r) = 1 / √(freq(r) + ε)       // ε = 10⁻³

τ(r) = τ_raw(r) / Σᵣ' τ_raw(r')     // normalize
```

| Quan hệ | freq | τ(r) | Ý nghĩa |
|---------|------|------|---------|
| "on" | 100 | nhỏ | Rất phổ biến → ít ưu tiên |
| "riding" | 5 | trung bình | Tương đối hiếm |
| "flying in" | 1 | lớn | Rất hiếm → **ưu tiên cao** |

#### Tail weight được dùng ở 3 nơi:

1. **Acquisition function** (Active Learning): relationship hiếm nhận score cao hơn → sinh nhiều ảnh hơn
2. **Relationship score** (Reward): nhân với `(1 + τ)` để tăng reward khi cải thiện quan hệ hiếm
3. **RelTR training loss**: Weighted loss để model tập trung hơn vào quan hệ hiếm

---

### 5.8 Sinh Ảnh AI (Stable Diffusion)

**File**: `RL/ai_images_generator.py`

#### Pipeline Sinh Ảnh

```
Step 1: Relationship Triplet → Prompt Template
        ("person", "holding", "phone") → "person holding phone in hands"

Step 2: Prompt Engineering — Thêm variations
        + quality modifier: "high quality", "photorealistic", "4k"
        + lighting: "golden hour", "studio lighting", "sunset"
        + background: "on the street", "in a park", "indoors"
        + (70% chance) time/weather
        + (50% chance) style descriptor
        + (40% chance) composition hint

Step 3: Stable Diffusion v1.5 → Generate images
        - 25 inference steps, guidance scale 7.0
        - Batch generation (batch_size=2)
        - Optimizations: xFormers, VAE slicing, torch.compile

Step 4: Quality Filter → Reject bad images
```

#### Quality Filter (5 checks)

| Check | Tiêu chí | Ngưỡng |
|-------|---------|--------|
| Size/Aspect | Width, Height ≥ 512px, aspect ≤ 2.2 | Hard reject |
| Blur | Laplacian variance | ≥ 60.0 |
| Exposure | Mean brightness, clip ratio | [20, 235], ≤ 20% |
| Duplicate | Perceptual hash (pHash) | Exact match |
| CLIP Similarity | Cosine similarity image↔prompt | ≥ 0.23 |

---

### 5.9 Auto-Annotation

**File**: `RL/auto_annotator.py`

Stable Diffusion chỉ trả về **pixels**, không có bounding boxes. Cần auto-annotate:

```
Ảnh synthetic
    │
    ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  GroundingDINO   │────▶│  Có bbox?    │──Y──▶│  Dùng annotation │
│  (preferred)     │     │              │      │                   │
└─────────────────┘     └──────┬───────┘      └─────────────────┘
                               │ N
                               ▼
                        ┌──────────────┐     ┌─────────────────┐
                        │  YOLO + CLIP  │────▶│  Fallback bbox   │
                        │  (fallback)   │     │                   │
                        └──────────────┘     └─────────────────┘
```

- **GroundingDINO**: Open-set object detection, quality score = 0.9
- **OWL-ViT**: Alternative open-set detector, quality score = 0.7
- **YOLO+CLIP**: Fallback dùng detection pipeline chính, quality score = 0.8

---

### 5.10 Evaluation Metrics

#### Detection Evaluation

| Metric | Cách tính |
|--------|----------|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | 2 × P × R / (P + R) |
| **IoU threshold** | 0.5 (với class matching) |

#### Relationship Evaluation

| Metric | Cách tính |
|--------|----------|
| **Precision** | Relationship tuple matching |
| **Recall** | Relationship tuple matching |
| **F1** | 2 × P × R / (P + R) |
| **F1-std** | Standard deviation qua samples |
| **mR@K** | Mean Recall@K (K = 10, 20, 50, 100) |

#### mR@K (Mean Recall at K)

```
Cho mỗi relation type r:
    R@K(r) = (# correct predictions of type r in top K) / (# ground truths of type r)

mR@K = mean of R@K(r) across all relation types
```

> **mR@K quan trọng hơn R@K** cho long-tail evaluation vì nó tính mean theo từng relation type, không bị bias bởi relation phổ biến.

---

## 6. Training Episode Flow

Mỗi epoch trong RL training diễn ra theo **9 bước** tuần tự:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: DQN Agent → decide_action()                         │
│         Chọn action (số variations)                         │
│         + Active Learning tạo generation plan                │
├─────────────────────────────────────────────────────────────┤
│ Step 2: Sinh ảnh synthetic                                   │
│         Stable Diffusion + prompt engineering                │
│         Số lượng theo relationship-specific plan             │
├─────────────────────────────────────────────────────────────┤
│ Step 3: Greedy Subset Selection                              │
│         Dedup (distance < 0.08) → Greedy select 70% pool    │
│         Approximation guarantee: ≥ 63.2% OPT                │
├─────────────────────────────────────────────────────────────┤
│ Step 4: Ingest synthetic samples                             │
│         Auto-annotation (GroundingDINO) → thêm vào dataset  │
│         + Recompute tail weights                             │
├─────────────────────────────────────────────────────────────┤
│ Step 5: Train Detection Model                                │
│         Fine-tune YOLO trên accumulated dataset              │
├─────────────────────────────────────────────────────────────┤
│ Step 6: Train Relationship Model                             │
│         Fine-tune RelTR (configurable epochs)                │
│         + Long-tail weighted loss                            │
├─────────────────────────────────────────────────────────────┤
│ Step 7: Calculate Reward                                     │
│         6-component adaptive reward + dynamic weights        │
│         + Uncertainty reduction bonus                        │
├─────────────────────────────────────────────────────────────┤
│ Step 8: DQN Update                                           │
│         Store experience → Optimize Q-network                │
│         → Sync target network → Decay epsilon                │
├─────────────────────────────────────────────────────────────┤
│ Step 9: Save best model                                      │
│         Checkpoint nếu reward > best_reward                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Application Layer

**File**: `app.py`

Giao diện desktop Tkinter với thiết kế modern light theme:

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│  SIDEBAR (280px)  │           MAIN CONTENT                    │
│                   │                                           │
│  ┌─────────────┐  │  ┌─────────────────────────────────────┐  │
│  │ VRD Toolkit │  │  │         Status Bar                   │  │
│  │ ✓ Ready     │  │  └─────────────────────────────────────┘  │
│  └─────────────┘  │                                           │
│                   │  ┌────────────────┐ ┌─────────────────┐  │
│  IMAGE OPS        │  │                │ │ 📦 Detected     │  │
│  📁 Select Image  │  │  🖼️ Image      │ │    Objects       │  │
│  ▶️ Detect Objects │  │    Preview     │ │                 │  │
│  🔄 Reload Data   │  │                │ ├─────────────────┤  │
│                   │  │   (Canvas)     │ │ 🔗 Relationships│  │
│  VIDEO OPS        │  │                │ │                 │  │
│  📹 Select Video  │  │                │ │                 │  │
│  ▶️ Run Video Demo│  │                │ │                 │  │
│  ⏹️ Stop Video    │  └────────────────┘ └─────────────────┘  │
│                   │                                           │
│  TRAINING         │                                           │
│  🧠 RL Training   │                                           │
│  🎨 Gen Synthetic │                                           │
│  📊 Evaluate      │                                           │
└──────────────────────────────────────────────────────────────┘
```

### Chức năng chính

| Button | Chức năng |
|--------|----------|
| Select Image | Chọn ảnh → hiển thị + load JSON data |
| Detect Objects | Chạy full pipeline: YOLO → Convert → RelTR → Vẽ bbox + arrows |
| Reload Data | Tải lại JSON mà không chạy pipeline |
| Select Video | Chọn video → hiển thị thumbnail |
| Run Video Demo | Xử lý video real-time + safety monitoring |
| RL Training | Chạy RL training loop (chọn dataset directory) |
| Generate Synthetic | Sinh ảnh từ relationships hiện tại |
| Evaluate Training | Tạo báo cáo đánh giá toàn diện |

### Threading

Tất cả operations nặng chạy trong thread riêng để **không block UI** (daemon threads).

---

## 8. Các File Hỗ Trợ

### Thư mục RL/

| File | Vai trò |
|------|---------|
| `reinforcement_learning.py` | **Core**: DQN agent, training loop, reward, model training |
| `rl_enhancement.py` | Tầng trung gian giữa GUI (app.py) và RL core |
| `uncertainty_estimator.py` | MC Dropout uncertainty estimation |
| `active_learning.py` | Acquisition functions + budget allocation |
| `approximation_algorithm.py` | Greedy submodular subset selection |
| `ai_images_generator.py` | Stable Diffusion image generation + quality filter |
| `auto_annotator.py` | GroundingDINO/OWL-ViT auto-annotation |
| `model_manager.py` | Model checkpoint save/load |
| `experiment_manager.py` | Experiment directories, plots, metrics |
| `experience_manager.py` | Replay buffer persistence |
| `training_evaluator.py` | Comprehensive evaluation reports |
| `safety_classifier.py` | Scene danger classification |
| `llm_relationship_predictor.py` | LLM-based open-vocab relationship |
| `visual_features.py` | Visual feature extraction cho LLM |
| `data_augmentation.py` | Data augmentation utilities |
| `local_rules_db.py` | Local rules database |

### Thư mục models/

| File | Vai trò |
|------|---------|
| `reltr.py` | RelTR model definition (Transformer-based VRD) |
| `transformer.py` | Transformer encoder-decoder implementation |
| `backbone.py` | ResNet-50 backbone |
| `position_encoding.py` | Positional encoding (sine/cosine) |
| `matcher.py` | Hungarian matcher cho set prediction |
| `yolo.py` | YOLOv5 model definition |
| `common.py` | Shared model components |

### Thư mục utils/

Shared utilities: augmentation, dataloaders, metrics, plots, loss functions, loggers (WandB, ClearML, Comet).

### Thư mục convert/

Scripts chuyển đổi dataset formats: COCO, GQA, Open Images, Visual Genome → YOLO format.

---

## Tổng Kết

Hệ thống VRD Toolkit kết hợp **4 kỹ thuật toán học nặng** trong module RL:

| Kỹ thuật | Vai trò | Guarantee/Lợi ích |
|---------|---------|-------------------|
| **DQN** | Quyết định chiến lược sinh ảnh | Convergence tới optimal Q-function |
| **MC Dropout** | Đo uncertainty không cần train thêm model | Approximate Bayesian inference |
| **Active Learning** | Ưu tiên relationship yếu nhất | Tối đa information gain per image |
| **Greedy Submodular** | Chọn subset đa dạng + chất lượng | ≥ 63.2% optimal (provable) |

Tất cả phối hợp để trả lời câu hỏi: **"Sinh ảnh nào, bao nhiêu, cho relationship nào, và chọn subset nào để training hiệu quả nhất?"** — rồi đánh giá kết quả qua hệ thống reward 6 thành phần với trọng số tự thích nghi.
