# Hình Thức Hoá Toán Học Hệ Thống VRD — Thống Nhất Từ Đầu Đến Cuối

## Bảng Ký Hiệu Chung

Tất cả ký hiệu dưới đây được dùng **nhất quán** xuyên suốt tài liệu.

| Ký hiệu | Kiểu | Ý nghĩa |
|---|---|---|
| x | Tensor (C×H×W) | Ảnh đầu vào (RGB) |
| O = {o₁, ..., o_M} | Tập hợp | M objects detected trong ảnh |
| o_i = (c_i, b_i, p_i) | Tuple | Object: class label, bounding box [x₁,y₁,x₂,y₂], confidence |
| R = {r₁, ..., r_K} | Tập hợp | K relationship types (51 classes: on, near, riding, under, ...) |
| y_j = (s_j, p_j, o_j, γ_j) | Tuple | Relationship: subject, predicate, object, confidence |
| θ | Vector | Parameters của model (YOLO hoặc RelTR) |
| D_train | Tập hợp | Training dataset |
| B | Số nguyên | Tổng budget (số ảnh cần sinh) |
| T | Số nguyên | Số MC Dropout forward passes (mặc định T=10) |
| ε | Số thực [0,1] | Exploration rate của DQN |
| K | Số nguyên | Số relationship types |
| N_r | Số nguyên | Số samples hiện có cho relationship r |

---

## Bước 1: Object Detection (YOLO11)

### Đầu vào / Đầu ra

```
Đầu vào: x ∈ R^(3×H×W)           -- ảnh RGB
Đầu ra:  O = {o₁, o₂, ..., o_M}   -- M detected objects
         o_i = (c_i, b_i, p_i)
```

### Biến số

| Biến | Kiểu | Ý nghĩa |
|---|---|---|
| c_i ∈ {person, car, bike, ...} | String | Class label |
| b_i = [x₁, y₁, x₂, y₂] | Vector 4D | Bounding box (pixel coordinates) |
| p_i ∈ [0, 1] | Số thực | Detection confidence |

### Công thức

YOLO11 dự đoán qua feature pyramid network:

```
F = Backbone(x)                   -- Feature extraction (CSPDarknet)
P = Neck(F)                       -- Multi-scale features (PANet)
O = Head(P)                       -- Prediction heads

Với mỗi anchor a tại grid cell (i,j) ở scale s:
  p(c_i | a) = σ(z_cls)           -- Class probability (sigmoid)
  b_i = decode(z_box, a)          -- Box regression
  p_i = σ(z_obj) × max_c p(c|a)  -- Final confidence = objectness × class prob
```

**NMS (Non-Maximum Suppression):** Loại bỏ boxes trùng lặp:
```
Giữ o_i nếu: IoU(b_i, b_j) < 0.45   với mọi o_j đã chọn có p_j > p_i
```

---

## Bước 2: Relationship Detection (RelTR)

### Đầu vào / Đầu ra

```
Đầu vào: x ∈ R^(3×H×W)                    -- ảnh RGB
         O = {o₁, ..., o_M}                -- detected objects
Đầu ra:  Y = {y₁, y₂, ..., y_J}            -- J detected relationships
         y_j = (s_j, p_j, o_j, γ_j)
```

### Biến số

| Biến | Kiểu | Ý nghĩa |
|---|---|---|
| s_j ∈ O | Object | Subject (chủ thể) |
| p_j ∈ R | String | Predicate (quan hệ), ví dụ "riding", "under" |
| o_j ∈ O | Object | Object (đối tượng) |
| γ_j ∈ [0, 1] | Số thực | Relationship confidence |

### Công thức

RelTR là Visual Transformer:

```
1. Feature extraction:
   F = ResNet50_Backbone(x)          -- F ∈ R^(d×h×w), d=2048

2. Transformer Encoder:
   E = TransformerEncoder(F + PE)    -- PE = positional encoding
                                     -- E ∈ R^(N×d_model), d_model=256

3. Relationship queries:
   Q = {q₁, ..., q_J}               -- J learnable query vectors, q_j ∈ R^d_model

4. Transformer Decoder:
   D = TransformerDecoder(Q, E)      -- Cross-attention: Q attend to E
                                     -- D_j ∈ R^d_model mỗi query

5. Prediction heads:
   s_j = FFN_subject(D_j)           -- Subject class + box
   p_j = FFN_predicate(D_j)         -- Predicate class (51 classes)
   o_j = FFN_object(D_j)            -- Object class + box
   γ_j = σ(FFN_conf(D_j))           -- Confidence score
```

---

## Bước 3: LLM Fallback (GPT-4V)

### Đầu vào / Đầu ra

```
Đầu vào: x ∈ R^(3×H×W)            -- ảnh gốc
         b_s, b_o ∈ R^4            -- bounding boxes của subject và object
         c_s, c_o ∈ String         -- class labels
         γ_reltr ∈ [0, 1]          -- RelTR confidence (nếu có)

Đầu ra:  p_llm ∈ R                -- predicted predicate
         γ_llm ∈ [0, 1]            -- LLM confidence
```

### Điều kiện kích hoạt

```
LLM được gọi khi:
  γ_reltr < τ_llm    với τ_llm = 0.6 (confidence threshold)
  HOẶC
  RelTR không detect được relationship cho cặp (s, o) nào đó
```

### Biến số đặc trưng

```
Spatial features:
  Δx = center(b_s).x − center(b_o).x       -- Khoảng cách ngang
  Δy = center(b_s).y − center(b_o).y       -- Khoảng cách dọc
  IoU(b_s, b_o)                             -- Intersection over Union
  A_s / A_o                                  -- Tỷ lệ diện tích

Gaze features (nếu subject là person):
  θ_gaze = atan2(Δy, Δx)                   -- Hướng nhìn ước tính
  d_gaze = √(Δx² + Δy²)                    -- Khoảng cách gaze target
```

---

## Bước 4: DQN Agent — Chọn Action

### Đầu vào / Đầu ra

```
Đầu vào: metrics từ epoch trước    -- Evaluation results
Đầu ra:  a ∈ A = {1, 2, 3, 5, 8}  -- Số ảnh sinh mỗi relationship
         B_total = a × |R_active|   -- Tổng budget
```

### 4.1 State Vector

```
s = [F1_det, F1_rel, R_prev, D_norm, ε_norm] ∈ R^5
```

| Thành phần | Công thức | Miền giá trị | Ý nghĩa |
|---|---|---|---|
| F1_det | F1 score detection model | [0, 1] | Hiệu suất detection |
| F1_rel | F1 score relationship model | [0, 1] | Hiệu suất relationship |
| R_prev | σ(reward_prev) | [0, 1] | Reward epoch trước (normalized) |
| D_norm | σ(\|D_train\| / 50) | [0, 1] | Kích thước dataset (normalized) |
| ε_norm | ε | [0, 1] | Exploration rate hiện tại |

Hàm normalize:
```
σ(v, scale) = tanh(v / scale) ∈ [0, 1]
```

### 4.2 Q-Network Architecture

```
Q_θ(s, a) : R^5 → R^5    -- 5 Q-values cho 5 actions

Layer 1: z₁ = ReLU(W₁ · s + b₁)     W₁ ∈ R^(64×5),  b₁ ∈ R^64
Layer 2: z₂ = ReLU(W₂ · z₁ + b₂)    W₂ ∈ R^(32×64), b₂ ∈ R^32
Layer 3: q  = W₃ · z₂ + b₃           W₃ ∈ R^(5×32),  b₃ ∈ R^5

q = [Q(s,a=1), Q(s,a=2), Q(s,a=3), Q(s,a=5), Q(s,a=8)]
```

### 4.3 Epsilon-Greedy Policy

```
            ┌ random(A)           với xác suất ε        (explore)
a = π(s) = ┤
            └ argmax_a Q_θ(s, a)  với xác suất 1 − ε    (exploit)

ε cập nhật mỗi epoch:
  ε ← max(ε_min, ε × ε_decay)
  ε_min = 0.1,  ε_decay = 0.95
```

---

## Bước 5: Uncertainty Estimation (MC Dropout)

### Đầu vào / Đầu ra

```
Đầu vào: x ∈ R^(3×H×W)           -- ảnh
         f_θ                       -- RelTR model
         T = 10                    -- số forward passes

Đầu ra:  U(r) ∈ [0, 1]            -- Combined uncertainty score cho relationship r
         Gồm: H (entropy), I (mutual info), VR (variation ratio)
```

### 5.1 MC Dropout Forward Passes

```
Bật Dropout layers (các layer khác vẫn eval):
  ∀ module ∈ f_θ:
    nếu module ∈ {Dropout, Dropout2d, Dropout3d}:
      module.train()     -- dropout ACTIVE
    ngược lại:
      module.eval()      -- bình thường

Chạy T lần:
  ŷ^(t) = f_θ(x),    t = 1, 2, ..., T
  
  Mỗi ŷ^(t) là 1 tập relationships:
  ŷ^(t) = {(s_j, p_j, o_j, γ_j)}   t = 1..T
```

### 5.2 Predictive Entropy H[ȳ]

**Ý nghĩa:** Tổng uncertainty (aleatoric + epistemic)

```
Bước 1: Tính fingerprint cho mỗi prediction
  fp(y_j) = (class(s_j), p_j, class(o_j))
  
  Ví dụ: ("person", "riding", "horse")

Bước 2: Đếm tần suất qua T passes
  count(fp) = số lần fingerprint fp xuất hiện trong T passes
  N_total   = tổng số predictions qua tất cả T passes

Bước 3: Tính probability distribution
  p̄(fp) = count(fp) / N_total

Bước 4: Tính entropy
  H[ȳ] = − Σ_{fp} p̄(fp) × log(p̄(fp))

Bước 5: Normalize về [0, 1]
  H_norm = H[ȳ] / log(|unique fingerprints|)

Ví dụ:
  T=10 passes, tổng 30 predictions:
  fp₁ = ("person","riding","horse"): 18 lần → p̄ = 0.60
  fp₂ = ("person","on","horse"):      9 lần → p̄ = 0.30
  fp₃ = ("person","near","horse"):    3 lần → p̄ = 0.10

  H = −(0.60×log0.60 + 0.30×log0.30 + 0.10×log0.10)
    = −(−0.306 − 0.361 − 0.230) = 0.897

  H_norm = 0.897 / log(3) = 0.897 / 1.099 = 0.816
```

### 5.3 Mutual Information I[y,θ] (BALD Score)

**Ý nghĩa:** Chỉ epistemic uncertainty (model thiếu data, có thể giảm bằng thêm data)

```
I[y,θ] = H[ȳ] − E_θ[H[y|θ]]

Trong đó:
  H[ȳ]        = Predictive Entropy (đã tính ở 5.2)
  E_θ[H[y|θ]] = (1/T) × Σ_{t=1}^{T} H_t

  H_t = entropy của forward pass t:
    H_t = − Σ_{j} γ_j^(t) × log(γ_j^(t))    -- dựa trên confidence scores
    
    Normalize: H_t = H_t / log(|predictions_t|)  nếu |predictions_t| > 1

Ví dụ:
  H[ȳ] = 0.816

  Pass 1: confidences = [0.85, 0.72, 0.60] → H₁ = 0.200
  Pass 2: confidences = [0.90, 0.55, 0.45] → H₂ = 0.350
  ...
  E[H] = (0.200 + 0.350 + ... ) / 10 = 0.250

  I = 0.816 − 0.250 = 0.566

  I cao → model thiếu knowledge → thêm data SẼ CÓ ÍCH
  I thấp → uncertainty do data noise → thêm data KHÔNG GIÚP
```

### 5.4 Variation Ratio VR

**Ý nghĩa:** Tỷ lệ forward passes cho kết quả khác mode

```
Bước 1: Fingerprint toàn bộ output mỗi pass
  FP_t = frozenset({fp(y_j) : y_j ∈ ŷ^(t)})   t = 1..T

Bước 2: Đếm mode (FP phổ biến nhất)
  mode_count = max_FP count(FP)

Bước 3: Variation Ratio
  VR = 1 − mode_count / T

Ví dụ:
  T=10, mode xuất hiện 7 lần:
  VR = 1 − 7/10 = 0.30    (30% passes khác mode)
```

### 5.5 Combined Uncertainty Score

```
U(r) = 0.30 × H_norm + 0.30 × I_norm + 0.20 × VR + 0.20 × (1 − γ̄)

γ̄ = mean confidence = (1/|all_confs|) × Σ γ_j^(t)   ∀t, ∀j
```

| Trọng số | Thành phần | Lý do |
|---|---|---|
| 0.30 | H (entropy) | Tổng uncertainty — bao quát nhất |
| 0.30 | I (BALD) | Epistemic uncertainty — actionable nhất |
| 0.20 | VR | Stability measure — predictions ổn định không |
| 0.20 | 1 − γ̄ | Inverse confidence — bổ sung |

---

## Bước 6: Active Learning — Acquisition Function

### Đầu vào / Đầu ra

```
Đầu vào: R_active = {r₁, ..., r_K}    -- K relationships cần train
         U(r_i) ∈ [0, 1]               -- Uncertainty score (từ Bước 5)
         F1(r_i) ∈ [0, 1]              -- F1 score hiện tại
         T(r_i) ∈ [0, +∞)              -- Tail weight
         B_total ∈ Z⁺                   -- Tổng budget (từ Bước 4)

Đầu ra:  Plan = {r_i → n_i}            -- Số ảnh cần sinh cho mỗi relationship
         Σ n_i = B_total
```

### 6.1 Acquisition Score

```
α(r_i) = w_u × U(r_i) + w_p × P(r_i) + w_t × T(r_i)
```

| Biến | Công thức | Miền | Ý nghĩa |
|---|---|---|---|
| U(r_i) | Combined uncertainty (Bước 5.5) | [0, 1] | Model không chắc chắn? |
| P(r_i) | 1 − F1(r_i) | [0, 1] | Performance gap |
| T(r_i) | N_total / (K × N_{r_i}) | [0, +∞) | Inverse frequency (long-tail) |
| w_u | 0.40 | hằng số | Trọng số uncertainty |
| w_p | 0.35 | hằng số | Trọng số performance |
| w_t | 0.25 | hằng số | Trọng số tail |

```
Ví dụ:
  r₁ = "person under car":  U=0.85, P=1−0.15=0.85, T=0.70
  α(r₁) = 0.40×0.85 + 0.35×0.85 + 0.25×0.70
        = 0.340 + 0.298 + 0.175 = 0.813

  r₅ = "person near car":   U=0.20, P=1−0.80=0.20, T=0.10
  α(r₅) = 0.40×0.20 + 0.35×0.20 + 0.25×0.10
        = 0.080 + 0.070 + 0.025 = 0.175
```

### 6.2 Proportional Budget Allocation

```
Phase 1: Mỗi r_i nhận tối thiểu n_min ảnh
  n_i ← n_min    ∀i = 1..K
  B_remain = B_total − K × n_min

Phase 2: Phân bổ B_remain theo tỷ lệ α
  proportion(r_i) = α(r_i) / Σ_{j=1}^{K} α(r_j)
  n_i ← n_i + ⌊proportion(r_i) × B_remain⌋

Phase 3: Phân bổ phần dư (do floor)
  Remainder = B_remain − Σ ⌊proportion(r_i) × B_remain⌋
  Phân bổ 1 ảnh/rel cho Remainder rel có fractional part lớn nhất

Phase 4: Cap tối đa
  n_i ← min(n_i, n_max)

Tham số mặc định: n_min = 1, n_max = 10
```

---

## Bước 7: Sinh Ảnh (Stable Diffusion) + Quality Filter

### 7.1 Stable Diffusion

### Đầu vào / Đầu ra

```
Đầu vào: prompt_i ∈ String           -- Text prompt từ relationship template
         n_i ∈ Z⁺                     -- Số ảnh cần sinh cho rel i
         seed ∈ Z                      -- Random seed (optional)

Đầu ra:  Pool_raw = {x₁, x₂, ..., x_N}  -- N ảnh synthetic (N = Σ n_i)
```

### 7.2 Quality Filter

```
Đầu vào: x_k ∈ Pool_raw              -- 1 ảnh synthetic
         prompt_k ∈ String             -- Prompt dùng để sinh x_k

Đầu ra:  (pass, reason) ∈ {True,False} × String
```

**5 kiểm tra tuần tự (fail bất kỳ → reject):**

```
Check 1: Size & Aspect Ratio
  pass ⟺ width(x) ≥ 256 ∧ height(x) ≥ 256

Check 2: Blur Detection (Laplacian Variance)
  G = grayscale(x)                                -- Convert RGB → Gray
  L = Laplacian(G)                                -- Laplacian filter (edge detection)
  V_blur = Var(L)                                  -- Variance of Laplacian
  
  pass ⟺ V_blur ≥ τ_blur                          -- τ_blur = 100 (threshold)
  
  Ý nghĩa: V_blur thấp → ảnh mờ (ít edges) → REJECT

Check 3: Exposure
  G = grayscale(x)
  μ = mean(G)                                     -- Mean brightness [0, 255]
  clip_ratio = (count(G=0) + count(G=255)) / |G|  -- Tỷ lệ pixels bị clip
  
  pass ⟺ μ_min ≤ μ ≤ μ_max ∧ clip_ratio ≤ τ_clip
  
  Mặc định: μ_min=40, μ_max=220, τ_clip=0.15

Check 4: Duplicate Detection (pHash)
  h = pHash(x)                                    -- Perceptual hash (64-bit)
  
  pass ⟺ ∀ h' ∈ Hash_cache: hamming(h, h') > τ_dupe
  
  τ_dupe = 8 (bits khác nhau)
  
  Nếu pass: Hash_cache ← Hash_cache ∪ {h}

Check 5: CLIP Similarity
  e_img = CLIP_image_encoder(x)      ∈ R^512      -- Image embedding
  e_txt = CLIP_text_encoder(prompt)   ∈ R^512      -- Text embedding
  
  sim = cos(e_img, e_txt) = (e_img · e_txt) / (‖e_img‖ × ‖e_txt‖)
  
  pass ⟺ sim ≥ τ_clip_sim                         -- τ_clip_sim = 0.20

Kết quả:
  Pool_filtered = {x_k ∈ Pool_raw : tất cả 5 checks pass}
```

---

## Bước 8: Auto-Annotation

### Đầu vào / Đầu ra

```
Đầu vào: x_k ∈ Pool_filtered          -- 1 ảnh synthetic đã pass quality
         rel_k = (s, p, o)             -- Relationship gốc dùng để sinh x_k

Đầu ra:  A_k = {(c_i, b_i, γ_i)}      -- Detected objects + bounding boxes
         backend ∈ {groundingdino, owlvit, yolo_clip, pseudo}
```

### Cascade Fallback

```
Thử lần lượt (dừng khi thành công):

Backend 1: GroundingDINO (SOTA, chất lượng 0.9)
  text_prompts = [s, o]              -- Ví dụ: ["person", "car"]
  detections = GroundingDINO(x_k, text_prompts)
  pass ⟺ |detections| ≥ 2 ∧ min(γ_i) ≥ τ_box
  τ_box = 0.25 (box threshold), τ_text = 0.20 (text threshold)

Backend 2: OWL-ViT (chất lượng 0.7)
  detections = OWLViT(x_k, text_prompts)
  (tương tự)

Backend 3: YOLO+CLIP (chất lượng 0.8)
  objects_yolo = YOLO(x_k)           -- Detect objects
  Với mỗi object: class_clip = CLIP_classify(crop(x_k, b_i), [s, o])
  detections = match(objects_yolo, class_clip)

Backend 4: Pseudo-bbox (chất lượng 0.3)
  Heuristic rules dựa trên predicate:
    "riding" → subject ở trên (y thấp), object ở dưới (y cao)
    "under"  → subject ở dưới, object ở trên
    default  → subject trái, object phải
```

---

## Bước 9: Approximation Algorithm (Greedy Submodular)

### Đầu vào / Đầu ra

```
Đầu vào: Pool = {x₁, ..., x_N}       -- N ảnh đã annotated (từ Bước 8)
         budget k = ⌊0.70 × N⌋         -- Chọn 70% pool

Đầu ra:  S ⊆ Pool, |S| = k            -- Subset tối ưu
         stats                          -- Thống kê (coverage, gains, ...)
```

### 9.1 Feature Vector

```
φ(x_i) ∈ R^d    -- Feature vector cho mỗi sample

φ(x_i) = [
  center_x(b_s) / W,           -- x tâm subject (normalized)
  center_y(b_s) / H,           -- y tâm subject
  area(b_s) / (W×H),           -- diện tích subject (normalized)
  center_x(b_o) / W,           -- x tâm object
  center_y(b_o) / H,           -- y tâm object
  area(b_o) / (W×H),           -- diện tích object
  IoU(b_s, b_o),               -- Overlap giữa subject và object
  hash(predicate) / |R|,        -- Predicate encoded
  hash(class_s) / |C|,          -- Subject class encoded
  hash(class_o) / |C|,          -- Object class encoded
  γ_annotation,                 -- Annotation confidence
]
```

### 9.2 Deduplication (Trước Greedy)

```
Pool_dedup = {}
for x_i ∈ Pool:
  if ∀ x_j ∈ Pool_dedup: ‖φ(x_i) − φ(x_j)‖₂ ≥ τ_dedup:
    Pool_dedup ← Pool_dedup ∪ {x_i}

τ_dedup = 0.08
```

### 9.3 Greedy Selection

```
S ← ∅                                                     -- Tập chọn rỗng
for step = 1 to k:
  ∀ x_i ∈ Pool_dedup \ S:
    Δf(x_i | S) = f(S ∪ {x_i}) − f(S)                     -- Marginal gain
  x* = argmax_{x_i ∈ Pool \ S}  Δf(x_i | S)               -- Tốt nhất
  S ← S ∪ {x*}                                             -- Thêm vào
  
  Dừng sớm nếu: Δf(x*) < 10⁻⁶ ∧ step > k/2               -- Diminishing returns
```

### 9.4 Marginal Gain — 3 Thành Phần

```
Δf(x_i | S) = λ_d × Div(x_i, S) + λ_q × Qual(x_i) + λ_r × Rep(x_i, S)
```

| Biến | Giá trị | Ý nghĩa |
|---|---|---|
| λ_d | 0.50 | Trọng số diversity |
| λ_q | 0.30 | Trọng số quality |
| λ_r | 0.20 | Trọng số representativeness |

**Diversity gain:**
```
Div(x_i, S) = min_{x_j ∈ S}  ‖φ(x_i) − φ(x_j)‖₂

Nếu S = ∅: Div(x_i, S) = ‖φ(x_i)‖₂

Ý nghĩa: x_i càng xa mọi x_j ∈ S → gain càng cao → đa dạng
```

**Quality gain:**
```
Qual(x_i) = mean(
  γ_annotation,                              -- Confidence annotation
  backend_quality,                            -- {groundingdino:0.9, owlvit:0.7, yolo_clip:0.8, pseudo:0.3}
  1_{has_subject ∧ has_object ∧ has_relation}, -- Relationship completeness (0 hoặc 1)
  1_{valid_bbox},                              -- BBox hợp lệ (0 hoặc 1)
)
```

**Representativeness gain:**
```
Rep(x_i, S) = |{x_j ∈ Pool \ S : nn(x_j) = x_i}| / |Pool \ S|

nn(x_j) = argmin_{x_k ∈ S ∪ {x_i}}  ‖φ(x_j) − φ(x_k)‖₂

Ý nghĩa: x_i là nearest neighbor của bao nhiêu unselected samples
          → x_i "đại diện" cho nhiều samples → gain cao
```

### 9.5 Approximation Guarantee (Nemhauser 1978)

```
f(S_greedy) ≥ (1 − 1/e) × f(S*)   ≈ 0.632 × OPT

Điều kiện: f là submodular + monotone
  Submodular: ∀ A ⊆ B, ∀ x ∉ B:  Δf(x|A) ≥ Δf(x|B)   (diminishing returns)
  Monotone:   ∀ A ⊆ B:  f(A) ≤ f(B)                     (thêm = không giảm)
```

---

## Bước 10: Fine-tune Models

```
Đầu vào: D_train ∪ S               -- Dataset cũ + subset mới
Đầu ra:  θ_new                      -- Updated model parameters

YOLO Fine-tune:
  L_det(θ) = L_box + L_cls + L_obj
  θ ← θ − η × ∇L_det(θ)            -- SGD/Adam optimizer

RelTR Fine-tune:
  L_rel(θ) = L_subject + L_predicate + L_object + L_matching
  θ ← θ − η × ∇L_rel(θ)
```

---

## Bước 11: Calculate Reward

### Đầu vào / Đầu ra

```
Đầu vào: synthetic_data             -- Dữ liệu vừa được train
         original_relationships      -- Ground truth relationships
         model updated (θ_new)       -- Model sau fine-tune

Đầu ra:  R ∈ R                       -- Reward signal (scalar)
         components ∈ R^6             -- 6 thành phần chi tiết
```

### 11.1 Sáu Thành Phần Reward

```
R = Σ_{i=1}^{6}  w_i × c_i

w = dynamic weights (tự điều chỉnh), Σ w_i = 1
c = [c_det, c_rel, c_div, c_con, c_imp, c_unc]
```

| i | c_i | Công thức | Ý nghĩa |
|---|---|---|---|
| 1 | c_det | F1_det(θ_new, D_eval) | Detection F1 trên evaluation set |
| 2 | c_rel | F1_rel(θ_new, D_eval) | Relationship F1 trên evaluation set |
| 3 | c_div | entropy(class_distribution(D_train)) | Đa dạng class trong dataset |
| 4 | c_con | 1 − std(per_sample_f1) | Ổn định: F1 đồng đều giữa samples |
| 5 | c_imp | (F1_current − F1_previous) / max(F1_previous, 0.01) | Cải thiện so với epoch trước |
| 6 | c_unc | 0.5 + 0.5 × (Ū_prev − Ū_curr) / max(Ū_prev, 0.01) | Uncertainty giảm bao nhiêu |

### 11.2 Dynamic Weights

```
Bước 1: Tính base weights dựa trên tình trạng hiện tại
  w_det_base, w_rel_base, w_div_base, w_con_base, w_imp_base
  = _calculate_dynamic_weights(c_det, c_rel, c_div, c_con)
  
  Logic: Thành phần nào YẾU → weight CAO hơn (tập trung cải thiện)

Bước 2: Thêm uncertainty weight
  w_unc = 0.10 (cố định 10%)

Bước 3: Re-normalize để Σ w_i = 1
  scale = (1 − w_unc) / Σ w_base_i
  w_i = w_base_i × scale    ∀ i = 1..5
  w_6 = w_unc = 0.10

Bước 4: Tính tổng reward
  R = w₁×c_det + w₂×c_rel + w₃×c_div + w₄×c_con + w₅×c_imp + w₆×c_unc
```

### 11.3 Uncertainty Reduction Score Chi Tiết

```
Bước 1: Đo uncertainty hiện tại (sau khi train)
  Ū_curr = (1/|samples|) × Σ U(x_i)    ∀ x_i ∈ evaluation_samples
  U(x_i) = Combined Uncertainty Score (Bước 5.5)

Bước 2: So sánh với epoch trước
  reduction = (Ū_prev − Ū_curr) / max(Ū_prev, 0.01)
  
  reduction > 0  → uncertainty GIẢM (tốt)
  reduction = 0  → không đổi
  reduction < 0  → uncertainty TĂNG (xấu)

Bước 3: Map về [0, 1]
  c_unc = clamp(0.5 + 0.5 × reduction, 0, 1)
  
  c_unc = 0.70 khi uncertainty giảm 40%:  0.5 + 0.5×0.40 = 0.70
  c_unc = 0.50 khi không đổi:             0.5 + 0.5×0.00 = 0.50
  c_unc = 0.40 khi uncertainty tăng 20%:   0.5 + 0.5×(−0.20) = 0.40

Bước 4: Lưu cho epoch sau
  Ū_prev ← Ū_curr
```

---

## Bước 12: Update DQN

```
Bước 1: Store experience
  e_t = (s_t, a_t, R_t, s_{t+1})
  Replay Buffer D ← D ∪ {e_t}

Bước 2: Sample mini-batch
  B_train = random_sample(D, size=32)

Bước 3: Compute target Q-values
  ∀ (s, a, R, s') ∈ B_train:
    Q_target = R + γ × max_{a'} Q_target_network(s', a')
    
    γ = 0.99 (discount factor)

Bước 4: Compute loss
  L = (1/|B_train|) × Σ (Q_θ(s, a) − Q_target)²     -- MSE Loss

Bước 5: Update Q-network
  θ ← θ − η × ∇_θ L

Bước 6: Soft update target network (mỗi N steps)
  θ_target ← τ × θ + (1−τ) × θ_target
  τ = 0.005 (soft update rate)

Bước 7: Decay epsilon
  ε ← max(ε_min, ε × ε_decay)
  ε_min = 0.1, ε_decay = 0.95
```

---

## Tóm Tắt Toàn Bộ Pipeline

```
Bước    Đầu vào                 Công thức chính                    Đầu ra
─────── ─────────────────────── ─────────────────────────────────── ──────────────────────
1       x (ảnh)                 YOLO(x)                            O = {(c,b,p)} objects
2       x, O                    RelTR(x, O)                        Y = {(s,p,o,γ)} rels
3       x, b_s, b_o             GPT4V(x, spatial, gaze)            p_llm (nếu γ < 0.6)
4       metrics                 π(s) = ε-greedy(Q_θ(s))           a (budget per rel)
5       x, f_θ, T=10           MC Dropout → H, I, VR              U(r) ∈ [0,1]
6       U, F1, T, B             α(r) = weighted sum → allocate     Plan {r→n}
7       prompts, n              SD sinh → QualityFilter             Pool_filtered
8       Pool, rel               GroundingDINO/OWL-ViT/YOLO+CLIP    Pool_annotated
9       Pool_annotated, k       Greedy argmax Δf                   S (subset tối ưu)
10      D_train ∪ S             SGD: θ ← θ − η∇L                  θ_new
11      θ_new, D_eval           R = Σ w_i × c_i                    R (reward)
12      (s,a,R,s')              Q-learning: minimize MSE            θ_DQN updated
```
