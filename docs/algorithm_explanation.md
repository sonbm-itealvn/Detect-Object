# Giải Thích Chi Tiết 3 Thuật Toán: Active Learning, Uncertainty Learning & Approximation Algorithm

## Mục Lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Uncertainty Learning (MC Dropout)](#2-uncertainty-learning-mc-dropout)
3. [Active Learning](#3-active-learning)
4. [Approximation Algorithm (Greedy Submodular Maximization)](#4-approximation-algorithm-greedy-submodular-maximization)
5. [Tích Hợp Vào Training Loop](#5-tích-hợp-vào-training-loop)

---

## 1. Tổng Quan Hệ Thống

### 1.1 Hệ thống trước khi tích hợp

Hệ thống VRD (Visual Relationship Detection) đã có sẵn vòng lặp RL Training:

```
DQN Agent → quyết định số ảnh cần sinh
    → Stable Diffusion sinh ảnh synthetic
        → Auto-Annotation (GroundingDINO/OWL-ViT)
            → Fine-tune YOLO + RelTR
                → Tính Reward → Cập nhật DQN
```

**Vấn đề của hệ thống cũ:**

| Vấn đề | Mô tả |
|---|---|
| Sinh ảnh không thông minh | Dùng F1 score đơn giản để quyết định sinh bao nhiêu ảnh. F1 < 0.3 → sinh 3x, F1 > 0.8 → sinh 0.5x. Quá thô sơ. |
| Không đo được uncertainty | Model RelTR predict "person riding horse" với confidence 0.7, nhưng không biết model **thực sự tự tin** hay chỉ vì không có lựa chọn khác. |
| Train trên toàn bộ ảnh | Toàn bộ ảnh synthetic đều được dùng để train, kể cả ảnh trùng lặp hoặc chất lượng kém → lãng phí tài nguyên, có thể gây overfitting. |

### 1.2 Hệ thống sau khi tích hợp

```
DQN Agent → quyết định tổng budget
    → [MỚI] Uncertainty Estimator → đo uncertainty cho từng relationship
        → [MỚI] Active Learning → ưu tiên relationship cần tập trung
            → Stable Diffusion sinh ảnh (số lượng theo priority)
                → [MỚI] Approximation Algorithm → chọn subset tối ưu
                    → Auto-Annotation → Fine-tune
                        → Reward (+ uncertainty reduction)
                            → Cập nhật DQN
```

---

## 2. Uncertainty Learning (MC Dropout)

**File:** `RL/uncertainty_estimator.py`

### 2.1 Vấn đề: Tại sao cần đo Uncertainty?

Khi model RelTR predict một relationship, nó trả về:
```
{"subject": "person", "relation": "riding", "object": "horse", "confidence": 0.75}
```

**Confidence ≠ Uncertainty.** Confidence chỉ là output của softmax layer, nó cho biết "trong các lựa chọn mà model có, nó chọn cái nào nhiều nhất". Nhưng:
- Model có thể confident sai (overconfident)
- Model có thể có confidence thấp nhưng thực ra kết quả đúng
- Confidence không cho biết "nếu tôi cho model thêm data, nó có cải thiện không?"

**Uncertainty** trả lời câu hỏi quan trọng hơn: **"Model có thực sự biết, hay nó đang đoán?"**

### 2.2 Giải pháp: Monte Carlo Dropout (MC Dropout)

#### Ý tưởng cốt lõi

Dropout là kỹ thuật regularization: trong training, ngẫu nhiên tắt một số neurons. Khi inference, bình thường Dropout bị tắt (`model.eval()`).

**MC Dropout** giữ Dropout **bật** khi inference:

```
Bình thường:                    MC Dropout:
model.eval()                    model.eval()
→ Dropout OFF                   → Dropout ON (chỉ riêng Dropout layers)
→ 1 lần predict                 → Chạy N=10 lần predict
→ 1 kết quả                     → 10 kết quả khác nhau
→ Không biết uncertainty         → Đo sự biến thiên → Uncertainty!
```

#### Tại sao MC Dropout hoạt động?

Mỗi lần chạy với Dropout bật, một tập hợp neurons khác nhau bị tắt → tạo ra "phiên bản model khác nhau". Tương đương với việc có **10 model ngẫu nhiên** (approximate Bayesian inference):

```
Forward pass 1: Tắt neurons {3, 7, 12}  → Predict: "person riding horse"
Forward pass 2: Tắt neurons {5, 8, 11}  → Predict: "person riding horse"
Forward pass 3: Tắt neurons {2, 9, 14}  → Predict: "person on horse"      ← khác!
Forward pass 4: Tắt neurons {4, 6, 10}  → Predict: "person near horse"    ← khác!
...
```

Nếu tất cả 10 lần cho cùng kết quả → **model tự tin, uncertainty thấp**.
Nếu mỗi lần cho kết quả khác → **model đang đoán, uncertainty cao**.

#### Code triển khai

```python
def enable_mc_dropout(self, model):
    """Bật dropout trong inference mode."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()  # training mode = dropout active
    # Các layer khác (BatchNorm, Conv2d...) vẫn ở eval mode
```

**Quan trọng:** Chỉ bật `train()` cho Dropout layers, không phải toàn bộ model. BatchNorm, Conv2d... vẫn ở `eval()` mode.

### 2.3 Ba Metrics Đo Uncertainty

#### Metric 1: Predictive Entropy

**Công thức:** `H[ȳ] = −Σ p̄(y) · log p̄(y)`

**Ý nghĩa:** Đo **tổng uncertainty** bao gồm cả:
- **Aleatoric uncertainty**: Do noise trong data (không thể giảm bằng thêm data)
- **Epistemic uncertainty**: Do model thiếu knowledge (có thể giảm bằng thêm data)

**Ví dụ:**
```
10 lần predict:
- "person riding horse":    6 lần → p̄ = 0.6 → -0.6 × log(0.6) = 0.306
- "person on horse":        3 lần → p̄ = 0.3 → -0.3 × log(0.3) = 0.361
- "person near horse":      1 lần → p̄ = 0.1 → -0.1 × log(0.1) = 0.230

Entropy = 0.306 + 0.361 + 0.230 = 0.897  (cao → uncertain)
```

So sánh:
```
10 lần predict đều cho "person riding horse":
- p̄ = 1.0 → -1.0 × log(1.0) = 0.0

Entropy = 0.0  (thấp → confident)
```

Trong code, entropy được **normalize** về [0, 1] bằng cách chia cho log(số outcomes duy nhất).

#### Metric 2: Mutual Information (BALD Score)

**Công thức:** `I[y, θ] = H[ȳ] − E[H[y|θ]]`

**Ý nghĩa:**
- `H[ȳ]` = Predictive Entropy = Tổng uncertainty
- `E[H[y|θ]]` = Trung bình entropy của **từng forward pass riêng lẻ** = Data uncertainty
- `I` = Tổng − Data = **Model uncertainty (Epistemic)**

**Tại sao quan trọng?**
- BALD cao → Model thiếu knowledge → **Thu thập thêm data sẽ có ích**
- BALD thấp → Uncertainty chủ yếu do data noise → Thu thập thêm data **không giúp gì**

**Ví dụ:**
```
Predictive Entropy H[ȳ] = 0.897

Entropy mỗi forward pass:
  Pass 1: entropy = 0.200 (2 predictions, 1 dominant)
  Pass 2: entropy = 0.350
  Pass 3: entropy = 0.180
  ...
  Trung bình E[H] = 0.250

BALD = 0.897 - 0.250 = 0.647  (cao → cần thêm data cho case này!)
```

#### Metric 3: Variation Ratio

**Công thức:** `VR = 1 − (count of mode) / N`

**Ý nghĩa:** Đơn giản nhất — bao nhiêu % forward passes cho kết quả khác so với kết quả phổ biến nhất.

**Ví dụ:**
```
10 lần predict:
- 7 lần cho cùng kết quả (mode)
- 3 lần cho kết quả khác

VR = 1 - 7/10 = 0.3 (30% khác mode → uncertainty vừa phải)
```

#### Combined Uncertainty Score

Ba metrics trên được trộn lại thành 1 score duy nhất ∈ [0, 1]:

```python
score = 0.30 * entropy + 0.30 * mutual_info + 0.20 * variation_ratio + 0.20 * (1-confidence)
```

| Weight | Metric | Lý do |
|---|---|---|
| 30% | Entropy | Tổng uncertainty — bao quát nhất |
| 30% | Mutual Information | Epistemic uncertainty — actionable nhất (thêm data có ích không?) |
| 20% | Variation Ratio | Stability — predictions có ổn định không? |
| 20% | 1 − Confidence | Inverse confidence — bổ sung khi không đủ data cho metrics khác |

---

## 3. Active Learning

**File:** `RL/active_learning.py`

### 3.1 Vấn đề: Sinh ảnh cho relationship nào?

Giả sử có 5 relationships cần train:

| Relationship | F1 hiện tại |
|---|---|
| person riding horse | 0.80 |
| dog sitting on chair | 0.20 |
| cat on table | 0.50 |
| person holding phone | 0.35 |
| car parked on street | 0.90 |

**Cách cũ (F1 heuristic):**
```
F1 < 0.3 → sinh 3x ảnh (dog sitting on chair → 9 ảnh)
F1 < 0.5 → sinh 2x ảnh (person holding phone → 6 ảnh)
F1 < 0.7 → sinh 1x ảnh (cat on table → 3 ảnh)
F1 > 0.8 → sinh 0.5x ảnh (person riding horse → 1 ảnh, car parked → 1 ảnh)
```

**Vấn đề:** F1 thấp có thể do:
1. Model thiếu data → **Thêm data sẽ giúp** ✅
2. Data quality kém → Thêm data **không giúp** ❌
3. Task quá khó (ảnh mờ, occlusion) → Thêm data **ít giúp** ❌

F1 heuristic **không phân biệt** được 3 trường hợp trên.

### 3.2 Giải pháp: Acquisition Function

Active Learning sử dụng **Acquisition Function** — hàm tính "informativeness" cho mỗi relationship:

```python
acquisition_score = (
    0.40 × uncertainty_score      # MC Dropout: model thực sự không biết?
  + 0.35 × (1 − avg_f1)           # Performance gap: còn nhiều room để cải thiện?
  + 0.25 × tail_weight             # Long-tail: relationship hiếm cần được bảo vệ?
)
```

**Ví dụ so sánh:**

| Relationship | Uncertainty | 1−F1 | Tail | Score | Ảnh |
|---|---|---|---|---|---|
| dog sitting on chair | 0.85 | 0.80 | 0.70 | **0.79** | **8** |
| person holding phone | 0.70 | 0.65 | 0.30 | **0.58** | **5** |
| cat on table | 0.30 | 0.50 | 0.20 | **0.35** | **3** |
| person riding horse | 0.15 | 0.20 | 0.10 | **0.16** | **2** |
| car parked on street | 0.10 | 0.10 | 0.05 | **0.09** | **1** |

→ "dog sitting on chair" nhận nhiều ảnh nhất vì **cả 3 tín hiệu đều cao**: model không chắc chắn + F1 thấp + relationship hiếm.

### 3.3 Proportional Budget Allocation

Sau khi có score, phân bổ budget theo tỷ lệ:

```
Tổng budget = DQN_action × số_relationships = 4 × 5 = 20 ảnh

Phase 1: Mỗi rel tối thiểu 1 ảnh → 5 ảnh đã dùng, còn 15
Phase 2: 15 ảnh còn lại phân theo tỷ lệ score:
    dog:    0.79 / (0.79+0.58+0.35+0.16+0.09) = 40% × 15 = 6 → tổng 7
    phone:  0.58 / ... = 29% × 15 = 4 → tổng 5
    cat:    0.35 / ... = 18% × 15 = 3 → tổng 4
    horse:  0.16 / ... = 8% × 15 = 1 → tổng 2
    car:    0.09 / ... = 5% × 15 = 1 → tổng 2
                                         Tổng: 20 ảnh ✓
```

### 3.4 Tại sao dùng trọng số 0.40 / 0.35 / 0.25?

| Thành phần | Trọng số | Giải thích |
|---|---|---|
| Uncertainty | 40% | **Tín hiệu trực tiếp nhất.** MC Dropout cho biết chính xác model đang "đoán" ở đâu. Thêm data cho chỗ model đoán → hiệu quả nhất. |
| Performance | 35% | **Bổ sung cho uncertainty.** Có trường hợp uncertainty thấp nhưng F1 vẫn thấp (model tự tin nhưng sai). Performance gap bắt được case này. |
| Tail weight | 25% | **Bảo vệ fairness.** Relationship hiếm (ít xuất hiện trong training data) cần được ưu tiên để model không chỉ giỏi ở các relationship phổ biến. |

---

## 4. Approximation Algorithm (Greedy Submodular Maximization)

**File:** `RL/approximation_algorithm.py`

### 4.1 Vấn đề: Ảnh synthetic không phải ảnh nào cũng tốt

Sau khi Stable Diffusion sinh 20 ảnh, pool có thể chứa:
- 5 ảnh gần giống nhau (cùng người cưỡi ngựa, khác background nhẹ)
- 3 ảnh có annotation kém (auto-annotator không detect được object)
- 2 ảnh có chất lượng thấp (bị nhòe, bị cắt)

Nếu train trên cả 20 ảnh:
- **Tốn GPU time** cho ảnh dư thừa
- **Bias** về phía relationship có nhiều ảnh giống nhau
- **Overfitting** trên các ảnh tương tự

### 4.2 Mục tiêu: Chọn subset đa dạng nhất

**Bài toán tối ưu:**
```
Tìm S ⊆ Pool, |S| ≤ budget
sao cho f(S) = λ₁·Diversity(S) + λ₂·Quality(S) + λ₃·Representativeness(S)
là lớn nhất
```

**Bài toán này là NP-hard** — thử tất cả tổ hợp C(20, 14) = 38,760 → quá chậm cho pool lớn.

### 4.3 Giải pháp: Greedy Algorithm

#### Thuật toán

```
Input: Pool gồm n samples, budget k
Output: Subset S gồm k samples

S = {}                                              # Bắt đầu rỗng
for i = 1, 2, ..., k:
    for mỗi sample s ∈ Pool \ S:
        Tính marginal_gain(s) = f(S ∪ {s}) − f(S)   # Thêm s vào thì gain bao nhiêu?
    s* = sample có marginal_gain lớn nhất
    S = S ∪ {s*}                                     # Thêm sample tốt nhất
return S
```

#### Ví dụ minh họa

```
Pool: [A, B, C, D, E]

Bước 1: S = {}
  gain(A) = f({A}) - f({}) = 0.8     ← B, D tương tự A
  gain(B) = f({B}) - f({}) = 0.7     
  gain(C) = f({C}) - f({}) = 0.9     ← C khác biệt nhất
  gain(D) = f({D}) - f({}) = 0.6
  gain(E) = f({E}) - f({}) = 0.5
  → Chọn C (gain cao nhất). S = {C}

Bước 2: S = {C}
  gain(A) = f({C,A}) - f({C}) = 0.7  ← A khác C → gain vẫn cao
  gain(B) = f({C,B}) - f({C}) = 0.3  ← B tương tự C → gain giảm!
  gain(D) = f({C,D}) - f({C}) = 0.6
  gain(E) = f({C,E}) - f({C}) = 0.4
  → Chọn A. S = {C, A}

Bước 3: S = {C, A}
  gain(B) = f({C,A,B}) - f({C,A}) = 0.1  ← B giống A → gain rất thấp!
  gain(D) = f({C,A,D}) - f({C,A}) = 0.5  ← D khác cả C và A
  gain(E) = f({C,A,E}) - f({C,A}) = 0.3
  → Chọn D. S = {C, A, D}

Kết quả: Chọn {C, A, D} — đa dạng nhất, bỏ B (giống A) và E (ít hữu ích).
```

### 4.4 Tại sao Greedy hoạt động? (Submodularity)

Hàm mục tiêu f(S) có tính chất **submodular** (diminishing returns):
```
Nếu A ⊆ B, thì: f(A ∪ {s}) − f(A) ≥ f(B ∪ {s}) − f(B)
```

**Nghĩa là:** Thêm 1 sample vào tập nhỏ → gain lớn hơn so với thêm vào tập lớn.

Ví dụ: Thêm ảnh "person riding horse" vào tập có 0 ảnh ngựa → gain lớn. Thêm cùng ảnh đó vào tập đã có 5 ảnh ngựa → gain nhỏ.

**Định lý Nemhauser (1978):** Với hàm submodular, thuật toán Greedy đảm bảo:
```
f(S_greedy) ≥ (1 − 1/e) × f(S_optimal) ≈ 0.632 × f(S_optimal)
```

→ Greedy luôn đạt **ít nhất 63.2%** so với giải pháp tối ưu. Đây là lower bound tốt nhất có thể cho polynomial-time algorithm.

### 4.5 Ba thành phần của Marginal Gain

#### Diversity Gain (50%)

```python
diversity = min_distance(candidate, S)  # Khoảng cách nhỏ nhất đến bất kỳ element nào trong S
```

- Candidate xa tất cả elements trong S → gain cao (đa dạng)
- Candidate gần 1 element trong S → gain thấp (trùng lặp)

Feature vector cho mỗi sample: `[bbox_center_x, bbox_center_y, bbox_area, rel_type, subject_class, object_class, confidence]`

#### Quality Gain (30%)

```python
quality = mean(
    annotation_confidence,           # Confidence trung bình của detected objects
    backend_quality,                 # GroundingDINO (0.9) > OWL-ViT (0.7) > YOLO+CLIP (0.8)
    relationship_completeness,       # Có đủ subject, relation, object không?
    bbox_validity,                   # Có bounding box hợp lệ không?
)
```

#### Representativeness Gain (20%)

```python
representativeness = count(unselected samples mà candidate là nearest neighbor) / total_unselected
```

- Candidate gần nhiều samples chưa chọn → đại diện tốt cho pool
- Đảm bảo subset cover đều toàn bộ distribution của pool

### 4.6 Deduplication trước Greedy

Trước khi chạy Greedy, loại bỏ samples quá giống nhau:

```python
for mỗi sample mới:
    nếu distance(sample, bất kỳ sample đã giữ) < 0.08:
        bỏ sample  # Quá giống
    else:
        giữ sample
```

→ Giảm search space cho Greedy → nhanh hơn. Ví dụ: 20 ảnh → dedup → 15 ảnh → Greedy chọn 10 ảnh.

---

## 5. Tích Hợp Vào Training Loop

### 5.1 Luồng xử lý mới (theo thứ tự)

```
┌─────────────────────────────────────────────────────────┐
│ Epoch bắt đầu                                          │
│                                                         │
│ 1. DQN Agent chọn action (num_variations = 4)           │
│    └── decide_action()                                  │
│                                                         │
│ 2. [MỚI] MC Dropout đo uncertainty cho từng relationship│
│    └── uncertainty_estimator.estimate_batch()            │
│    └── Kết quả: {rel1: 0.85, rel2: 0.30, rel3: 0.60}   │
│                                                         │
│ 3. [MỚI] Active Learning tạo generation plan            │
│    └── active_learner.score_relationships()              │
│    └── active_learner.create_generation_plan()           │
│    └── Plan: {rel1: 8 ảnh, rel2: 2 ảnh, rel3: 5 ảnh}   │
│                                                         │
│ 4. Stable Diffusion sinh ảnh theo plan                  │
│    └── generator.generate_from_relationship()            │
│    └── Pool: 15 ảnh synthetic                            │
│                                                         │
│ 5. [MỚI] Dedup + Greedy chọn subset tối ưu             │
│    └── subset_selector.filter_redundant_samples()        │
│    └── subset_selector.select_optimal_subset()           │
│    └── Selected: 10/15 ảnh (coverage 100%)               │
│                                                         │
│ 6. Auto-Annotation + Ingest vào dataset                 │
│                                                         │
│ 7. Train YOLO + RelTR trên dataset                      │
│                                                         │
│ 8. Calculate Reward                                     │
│    └── detection_score (F1 detection)                    │
│    └── relationship_score (F1 relationship)              │
│    └── diversity_score                                   │
│    └── consistency_score                                 │
│    └── improvement_score                                 │
│    └── [MỚI] uncertainty_reduction_score                 │
│                                                         │
│ 9. Update DQN Q-network                                 │
│                                                         │
│ Epoch kết thúc                                          │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Uncertainty Reduction trong Reward

Thêm thành phần thứ 6 vào reward function:

```
Reward = w₁·Detection + w₂·Relationship + w₃·Diversity + w₄·Consistency 
       + w₅·Improvement + w₆·Uncertainty_Reduction

Trong đó w₆ = 0.10 (10% tổng weight)
Các weight khác được scale lại: wᵢ_mới = wᵢ_cũ × (1 - 0.10) / Σwᵢ_cũ
```

`Uncertainty_Reduction` đo: **Uncertainty ở epoch hiện tại có giảm so với epoch trước không?**

```python
reduction = (uncertainty_trước − uncertainty_sau) / uncertainty_trước

# Map từ [-1, 1] sang [0, 1]:
score = 0.5 + reduction × 0.5

# Uncertainty giảm 40%: score = 0.5 + 0.4×0.5 = 0.70 (reward cao)
# Uncertainty không đổi: score = 0.5 + 0.0×0.5 = 0.50 (neutral)
# Uncertainty tăng 20%: score = 0.5 + (-0.2)×0.5 = 0.40 (reward thấp)
```

→ DQN agent học được: **"Hành động nào giảm uncertainty nhiều nhất → reward cao nhất → lặp lại hành động đó"**

### 5.3 Fallback Mechanism

Tất cả 3 modules mới đều có **try/except fallback**:

```python
try:
    # Dùng Active Learning scoring
    scored_relationships = self.active_learner.score_relationships(...)
    relationship_plan = self.active_learner.create_generation_plan(...)
except Exception as e:
    # Nếu lỗi → quay về cách cũ (F1 heuristic)
    relationship_plan = self._get_relationship_priorities(...)
```

→ Hệ thống **không bao giờ crash** vì 3 modules mới. Nếu có lỗi, tự động quay về hành vi cũ.

---

## Tóm Tắt

| Thuật toán | Câu hỏi nó trả lời | Kết quả |
|---|---|---|
| **MC Dropout** | "Model có thực sự biết hay đang đoán?" | Uncertainty score ∈ [0,1] cho mỗi relationship |
| **Active Learning** | "Nên tập trung sinh ảnh cho relationship nào?" | Generation plan: {rel → số ảnh} |
| **Greedy Submodular** | "Trong pool ảnh, nên chọn ảnh nào để train?" | Subset tối ưu ≈ 70% pool, đa dạng nhất |
