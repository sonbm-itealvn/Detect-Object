# Tổng Quan Hệ Thống VRD — Từ Đầu Đến Cuối

## Mục Lục

1. [Kiến Trúc Tổng Thể](#1-kiến-trúc-tổng-thể)
2. [Pipeline Inference (Xử lý video/ảnh)](#2-pipeline-inference)
3. [RL Training Loop (Huấn luyện)](#3-rl-training-loop)
4. [Chi Tiết Từng Module](#4-chi-tiết-từng-module)
5. [Data Flow Cụ Thể](#5-data-flow-cụ-thể)

---

## 1. Kiến Trúc Tổng Thể

Hệ thống gồm **2 pipeline chính**:

```
┌─────────────────────────────────────────────────────┐
│                    PIPELINE 1                        │
│              INFERENCE (xử lý video)                 │
│                                                      │
│  Video → YOLO11 detect objects                       │
│       → RelTR detect relationships                   │
│       → LLM fallback (GPT-4V cho low-confidence)     │
│       → Heuristic rules (safety-critical cases)      │
│       → Safety Classification → Output               │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                    PIPELINE 2                        │
│              RL TRAINING (data loop)                 │
│                                                      │
│  DQN Agent quyết định budget                         │
│       → Active Learning chọn relationship ưu tiên    │
│       → Stable Diffusion sinh ảnh                    │
│       → Quality Filter lọc ảnh kém                   │
│       → Auto-Annotation gán bounding box             │
│       → Approximation Algorithm chọn subset          │
│       → Fine-tune YOLO + RelTR                       │
│       → Evaluate + Calculate Reward                  │
│       → Update DQN                                   │
└─────────────────────────────────────────────────────┘
```

---

## 2. Pipeline Inference

Pipeline này chạy khi bạn đưa video/ảnh vào hệ thống để phát hiện quan hệ.

### Bước 1: Object Detection (YOLO11)

```
Input:  1 frame ảnh từ video
Tool:   YOLO11 (detect_objects.py)
Output: Danh sách objects + bounding boxes

Ví dụ:
  [
    {class: "person", bbox: [100, 200, 300, 500], confidence: 0.92},
    {class: "car",    bbox: [400, 300, 700, 600], confidence: 0.88}
  ]
```

### Bước 2: Relationship Detection (RelTR)

```
Input:  Ảnh gốc + danh sách bounding boxes
Tool:   RelTR model (visual relationship transformer)
Output: Danh sách relationships (subject, predicate, object)

Ví dụ:
  [
    {subject: "person", relation: "near", object: "car", confidence: 0.85},
    {subject: "person", relation: "riding", object: "horse", confidence: 0.40}
  ]
```

RelTR có **51 predefined relationship classes** (on, near, riding, under, wearing...).

### Bước 3: LLM Fallback (GPT-4V)

**File:** `RL/llm_relationship_predictor.py`

**Khi nào kích hoạt:**
- RelTR confidence < 0.6 (threshold)
- RelTR không detect được relationship cho 1 cặp object nào đó

```
Input:  Ảnh gốc + subject_bbox + object_bbox + class names
Tool:   GPT-4o-mini (Vision API)
Output: Predicted relationship + confidence

Quy trình:
1. Crop ảnh vùng chứa 2 objects
2. Phân tích spatial features (vị trí, khoảng cách, overlap)
3. Phân tích gaze (hướng nhìn) nếu subject là person
4. Gửi prompt + ảnh lên GPT-4V → nhận relationship
5. Validate kết quả (có nằm trong danh sách hợp lệ không?)
```

**Ví dụ:**
```
RelTR predict: person → ??? → car (không detect được relation)
LLM fallback: person → "working under" → car (confidence 0.75)
```

### Bước 4: Heuristic Rules

**File:** `RL/boundingbox_objects.py` (phần heuristic logic)

Cho các trường hợp safety-critical mà cả RelTR và LLM đều bỏ sót:

```
Rule 1: Nếu person bbox NẰM TRONG car bbox + overlap > 30% → "under"
Rule 2: Nếu person bbox TRÊN car bbox + khoảng cách nhỏ → "on"
Rule 3: Nếu person bbox BÊN CẠNH car bbox → "near"
```

### Bước 5: Safety Classification

**File:** `RL/safety_classifier.py`

```
Input:  Relationships detected
Output: Safety level (safe / warning / danger)

Ví dụ:
  "person under car"  → DANGER
  "person near car"   → WARNING
  "person riding bike" → SAFE
```

---

## 3. RL Training Loop

Pipeline này chạy để **tự động cải thiện model** bằng synthetic data.

### Tổng quan luồng

```
Epoch 1 ─────────────────────────────────────────────────────────────────
│
│ BƯỚC 1: DQN Agent chọn action
│ │
│ │  State vector: [F1_detection, F1_relationship, dataset_size, epsilon, ...]
│ │  Action: num_variations = 4 (sinh 4 biến thể mỗi relationship)
│ │  Exploration: epsilon-greedy (ban đầu explore nhiều, dần exploit)
│ │
│ BƯỚC 2: Active Learning tạo generation plan
│ │
│ │  Đo uncertainty cho từng relationship (MC Dropout)
│ │  Kết hợp uncertainty + F1 + tail_weight → acquisition score
│ │  Phân bổ budget: {rel yếu nhất: 8 ảnh, rel mạnh nhất: 1 ảnh}
│ │
│ BƯỚC 3: Stable Diffusion sinh ảnh
│ │
│ │  Với mỗi relationship trong plan:
│ │    1. Map relationship → prompt template
│ │       "person riding horse" → "A person riding a horse, photorealistic..."
│ │    2. Tạo variations (đổi background, góc nhìn, lighting)
│ │    3. Stable Diffusion v1.5 generate ảnh
│ │    4. ⭐ Quality Filter kiểm tra ảnh
│ │       - Size/aspect ratio hợp lệ?
│ │       - Bị blur không?
│ │       - Exposure (quá sáng/tối)?
│ │       - Trùng lặp với ảnh trước? (pHash)
│ │       - CLIP similarity với prompt?
│ │    5. Ảnh PASS → giữ lại, ảnh FAIL → bỏ
│ │
│ │  Output: Pool ~15-20 ảnh synthetic
│ │
│ BƯỚC 4: Auto-Annotation
│ │
│ │  Ảnh synthetic chỉ có pixels, KHÔNG có bounding box.
│ │  Cần tự động gán bbox + class label.
│ │
│ │  Thứ tự ưu tiên backend:
│ │    1. GroundingDINO (SOTA, open-vocabulary) ← tốt nhất
│ │    2. OWL-ViT (lighter, từ HuggingFace)    ← fallback 1
│ │    3. YOLO+CLIP (detect + classify)         ← fallback 2
│ │    4. Pseudo-bbox (heuristic)               ← fallback cuối cùng
│ │
│ │  Output: Ảnh + bounding boxes + class labels + confidence scores
│ │
│ BƯỚC 5: Approximation Algorithm (Greedy Submodular)
│ │
│ │  Pool 15 ảnh → loại ảnh trùng lặp → Greedy chọn ~10 ảnh tối ưu
│ │
│ │  Tiêu chí chọn:
│ │    - Diversity (50%): ảnh khác nhau nhất có thể
│ │    - Quality (30%): annotation confidence cao
│ │    - Representativeness (20%): cover đều các relationship types
│ │
│ │  Output: Subset tối ưu ~70% pool
│ │
│ BƯỚC 6: Ingest vào dataset + Fine-tune
│ │
│ │  Thêm ảnh đã chọn vào training dataset
│ │  Fine-tune YOLO (detection) + RelTR (relationship)
│ │  Output: Updated models
│ │
│ BƯỚC 7: Evaluate
│ │
│ │  Chạy model mới trên evaluation set
│ │  Tính: detection F1, relationship F1, per-class metrics
│ │  Output: Evaluation metrics
│ │
│ BƯỚC 8: Calculate Reward
│ │
│ │  Reward = w1 × detection_score          (F1 detection)
│ │         + w2 × relationship_score       (F1 relationship)
│ │         + w3 × diversity_score          (dataset đa dạng)
│ │         + w4 × consistency_score        (kết quả ổn định)
│ │         + w5 × improvement_score        (cải thiện vs epoch trước)
│ │         + w6 × uncertainty_reduction    (uncertainty giảm)
│ │
│ BƯỚC 9: Update DQN
│ │
│ │  Lưu experience: (state, action, reward, next_state)
│ │  Train Q-network bằng experience replay
│ │  Giảm epsilon (ít explore hơn theo thời gian)
│ │
│ → Epoch 2 (lặp lại từ bước 1)
```

---

## 4. Chi Tiết Từng Module

### 4.1 File Mapping

| File | Vai trò | Thuộc Pipeline |
|---|---|---|
| `detect_objects.py` | YOLO11 object detection | Inference |
| `RL/llm_relationship_predictor.py` | LLM fallback (GPT-4V) | Inference |
| `RL/safety_classifier.py` | Safety level classification | Inference |
| `RL/visual_features.py` | Trích xuất spatial + gaze features | Inference |
| `RL/ai_images_generator.py` | Stable Diffusion + Quality Filter | Training |
| `RL/auto_annotator.py` | Auto-annotation (GroundingDINO/OWL-ViT) | Training |
| `RL/reinforcement_learning.py` | DQN agent + train_episode + reward | Training |
| `RL/rl_enhancement.py` | Orchestrator (chạy training loop) | Training |
| `RL/uncertainty_estimator.py` | MC Dropout uncertainty | Training |
| `RL/active_learning.py` | Acquisition function + budget allocation | Training |
| `RL/approximation_algorithm.py` | Greedy Submodular subset selection | Training |
| `RL/experiment_manager.py` | Lưu experiment configs + metrics | Training |
| `RL/experience_manager.py` | Replay buffer cho DQN | Training |
| `RL/training_evaluator.py` | Đánh giá training sessions | Training |
| `RL/model_manager.py` | Load/save model checkpoints | Training |
| `RL/data_augmentation.py` | Data augmentation cơ bản | Training |
| `RL/llm_safety_analyzer.py` | LLM phân tích safety | Both |

### 4.2 Quality Filter (Image Quality Filter)

**File:** `RL/ai_images_generator.py` → class `ImageQualityFilter`

**Vị trí trong pipeline:** Ngay SAU khi Stable Diffusion sinh ảnh, TRƯỚC khi Auto-Annotation.

**5 kiểm tra:**

```
1. Size/Aspect Ratio
   - Ảnh phải ≥ 256×256 pixels
   - Aspect ratio không quá lệch
   → Loại ảnh quá nhỏ hoặc bị méo

2. Blur Detection
   - Tính Laplacian variance của ảnh
   - Variance < threshold → ảnh bị mờ → REJECT
   → Loại ảnh nhòe, out of focus

3. Exposure Check
   - Tính histogram brightness
   - Quá sáng (>90% pixels sáng) hoặc quá tối (>90% pixels tối) → REJECT
   → Loại ảnh overexposed/underexposed

4. Duplicate Detection (pHash)
   - Tính perceptual hash (pHash) của ảnh
   - So sánh với tất cả ảnh đã sinh trước đó
   - Hash distance < threshold → DUPLICATE → REJECT
   → Loại ảnh gần giống ảnh đã có

5. CLIP Similarity
   - Tính cosine similarity giữa ảnh và prompt text
   - Similarity < threshold → ảnh không khớp prompt → REJECT
   → Loại ảnh mà SD sinh ra không đúng yêu cầu
```

**Trả lời câu hỏi của bạn:** ✅ **Quality Filter VẪN CÒN và VẪN HOẠT ĐỘNG.** Nó nằm ở bước 3 (sinh ảnh), lọc ảnh TRƯỚC KHI vào pool. Approximation Algorithm ở bước 5 lọc THÊM MỘT LẦN NỮA từ pool đã qua filter.

```
Stable Diffusion sinh 25 ảnh
    → Quality Filter loại 5 ảnh kém (blur, trùng lặp, ...)
    → Pool: 20 ảnh PASS quality
        → Dedup loại 3 ảnh quá giống nhau
        → Greedy chọn 12/17 ảnh tối ưu nhất
            → 12 ảnh vào training
```

**2 lớp lọc khác nhau:**

| | Quality Filter | Approximation Algorithm |
|---|---|---|
| Vị trí | Bước 3 (sau sinh ảnh) | Bước 5 (sau annotate) |
| Tiêu chí | Technical quality (blur, exposure...) | Diversity + Relevance |
| Câu hỏi | "Ảnh này có chất lượng tốt không?" | "Ảnh này có đa dạng/hữu ích không?" |
| Ví dụ loại | Ảnh mờ, quá tối, trùng pixel-level | Ảnh tốt nhưng quá giống ảnh khác |

### 4.3 Auto-Annotator

**File:** `RL/auto_annotator.py`

**Vấn đề:** Stable Diffusion chỉ trả về pixels (ảnh). Để train YOLO, cần bounding boxes. Để train RelTR, cần relationship annotations.

**Giải pháp:** Dùng open-vocabulary detector tự động gán nhãn.

```
Thứ tự ưu tiên:

1. GroundingDINO (tốt nhất)
   - SOTA zero-shot object detector
   - Input: ảnh + text "person. car."
   - Output: bounding boxes cho "person" và "car"
   - Quality: 0.9/1.0

2. OWL-ViT (fallback 1)
   - Lightweight, từ HuggingFace
   - Quality: 0.7/1.0

3. YOLO+CLIP (fallback 2)
   - YOLO detect → CLIP classify
   - Quality: 0.8/1.0

4. Pseudo-bbox (fallback cuối)
   - Heuristic: đặt bbox dựa trên loại relationship
   - "riding" → subject ở trên, object ở dưới
   - Quality: 0.3/1.0 (thấp, chỉ dùng khi hết cách)
```

### 4.4 LLM Relationship Predictor

**File:** `RL/llm_relationship_predictor.py`

```
Khi nào dùng:
  1. RelTR confidence < 0.6 → "Tôi không chắc lắm"
  2. RelTR không detect relationship cho 1 cặp object

Quy trình:
  1. Trích xuất visual features:
     - Spatial: "person is to the LEFT of car, 50px apart"
     - Gaze: "person is looking TOWARD the car"
     - Overlap: "25% overlap between bounding boxes"

  2. Build prompt cho GPT-4V:
     "Given the image showing a person and a car:
      - The person is below and slightly overlapping the car
      - The person appears to be looking at the car
      What is the most likely spatial relationship?"

  3. GPT-4V trả về: "working under"

  4. Validate:
     - "working under" → map to closest RelTR class → "under"
     - Confidence: 0.75 (từ LLM)
```

### 4.5 DQN Agent

**File:** `RL/reinforcement_learning.py` → class `RelationshipReinforcementLearning`

```
Architecture:
  Input:  State vector (8 dimensions)
          [f1_det, f1_rel, loss_det, loss_rel, reward, dataset_size, epsilon, epoch]

  Network: Linear(8, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 5)

  Output: Q-values cho 5 actions
          Action 0: sinh 1 ảnh/relationship (ít, tiết kiệm)
          Action 1: sinh 2 ảnh/relationship
          Action 2: sinh 3 ảnh/relationship
          Action 3: sinh 5 ảnh/relationship
          Action 4: sinh 8 ảnh/relationship (nhiều, tốn kém)

  Policy: Epsilon-greedy
          P(explore) = epsilon (chọn ngẫu nhiên)
          P(exploit) = 1 − epsilon (chọn action có Q-value cao nhất)
          epsilon giảm dần: 1.0 → 0.1 qua các epoch
```

---

## 5. Data Flow Cụ Thể

### 5.1 Một Epoch Đầy Đủ — Ví dụ

```
EPOCH 3:
  State: [f1_det=0.65, f1_rel=0.42, loss=0.8, epsilon=0.5, ...]

  ┌─ BƯỚC 1: DQN chọn action ─────────────────────────────────────┐
  │  Epsilon = 0.5 → 50% explore, 50% exploit                     │
  │  Random > 0.5 → exploit → Q-network chọn action 2 (3 ảnh/rel)│
  │  Total budget = 3 × 10 relationships = 30 ảnh                 │
  └────────────────────────────────────────────────────────────────┘
  
  ┌─ BƯỚC 2: Active Learning ──────────────────────────────────────┐
  │                                                                │
  │  MC Dropout đo uncertainty:                                    │
  │    "person under car":  uncertainty=0.85, F1=0.15, tail=0.70   │
  │    "person near car":   uncertainty=0.20, F1=0.80, tail=0.10   │
  │    "person riding bike": uncertainty=0.60, F1=0.35, tail=0.40  │
  │    ...                                                         │
  │                                                                │
  │  Acquisition score:                                            │
  │    "person under car":  0.40×0.85 + 0.35×0.85 + 0.25×0.70     │
  │                       = 0.34 + 0.30 + 0.175 = 0.81 ← cao nhất │
  │    "person near car":   0.40×0.20 + 0.35×0.20 + 0.25×0.10     │
  │                       = 0.08 + 0.07 + 0.025 = 0.18 ← thấp nhất│
  │                                                                │
  │  Plan: "person under car": 8 ảnh                               │
  │        "person riding bike": 5 ảnh                             │
  │        ... (phân bổ theo tỷ lệ score)                          │
  │        "person near car": 1 ảnh (minimum)                      │
  │        Tổng: 30 ảnh                                            │
  └────────────────────────────────────────────────────────────────┘
  
  ┌─ BƯỚC 3: Sinh ảnh + Quality Filter ────────────────────────────┐
  │                                                                │
  │  "person under car" × 8 variations:                            │
  │    Prompt: "A person lying under a car, working on engine..."  │
  │    SD sinh 8 ảnh                                               │
  │    Quality Filter:                                             │
  │      ✅ Ảnh 1: PASS (clear, good exposure)                     │
  │      ✅ Ảnh 2: PASS                                            │
  │      ❌ Ảnh 3: REJECT (blurry, Laplacian variance too low)     │
  │      ✅ Ảnh 4: PASS                                            │
  │      ❌ Ảnh 5: REJECT (duplicate of ảnh 2, pHash match)        │
  │      ✅ Ảnh 6: PASS                                            │
  │      ✅ Ảnh 7: PASS                                            │
  │      ❌ Ảnh 8: REJECT (CLIP similarity too low, wrong content) │
  │    → 5/8 ảnh PASS                                              │
  │                                                                │
  │  Tương tự cho các relationships khác...                        │
  │  Tổng pool sau Quality Filter: 22/30 ảnh PASS                 │
  └────────────────────────────────────────────────────────────────┘
  
  ┌─ BƯỚC 4: Auto-Annotation ──────────────────────────────────────┐
  │                                                                │
  │  22 ảnh → GroundingDINO annotate:                              │
  │    Ảnh 1: detect "person" [120,300,280,500] conf=0.91          │
  │           detect "car" [50,100,600,400] conf=0.88              │
  │           relationship: "person under car" (from original)     │
  │    ...                                                         │
  │                                                                │
  │  Output: 22 ảnh với bbox + class + relationship annotations    │
  └────────────────────────────────────────────────────────────────┘
  
  ┌─ BƯỚC 5: Approximation Algorithm ──────────────────────────────┐
  │                                                                │
  │  Pool: 22 ảnh annotated                                        │
  │                                                                │
  │  Step 5a: Dedup (min_distance=0.08)                            │
  │    Loại 3 ảnh quá giống → 19 ảnh                               │
  │                                                                │
  │  Step 5b: Budget = 70% × 19 = 13 ảnh                          │
  │                                                                │
  │  Step 5c: Greedy Submodular:                                   │
  │    Iteration 1: Chọn ảnh có diversity cao nhất                 │
  │    Iteration 2: Chọn ảnh XA ảnh đã chọn + quality cao         │
  │    ...                                                         │
  │    Iteration 13: Chọn ảnh cuối                                 │
  │                                                                │
  │  Selected: 13/19 ảnh (coverage: 100% relationship types)       │
  │  Bỏ: 6 ảnh (giống ảnh đã chọn hoặc quality kém nhất)          │
  └────────────────────────────────────────────────────────────────┘
  
  ┌─ BƯỚC 6: Fine-tune ───────────────────────────────────────────┐
  │                                                                │
  │  Thêm 13 ảnh vào training dataset                             │
  │  Fine-tune YOLO: detection → detection loss giảm               │
  │  Fine-tune RelTR: relationship → relationship loss giảm        │
  └────────────────────────────────────────────────────────────────┘
  
  ┌─ BƯỚC 7-8: Evaluate + Reward ──────────────────────────────────┐
  │                                                                │
  │  Evaluation:                                                    │
  │    F1_detection:    0.65 → 0.70 (+0.05)                        │
  │    F1_relationship: 0.42 → 0.48 (+0.06)                        │
  │                                                                │
  │  Reward components:                                             │
  │    detection_score:          0.70                               │
  │    relationship_score:       0.48                               │
  │    diversity_score:          0.82                               │
  │    consistency_score:        0.75                               │
  │    improvement_score:        0.60 (có cải thiện)                │
  │    uncertainty_reduction:    0.65 (uncertainty giảm)            │
  │                                                                │
  │  Total reward: 0.67 (positive → episode tốt!)                  │
  └────────────────────────────────────────────────────────────────┘
  
  ┌─ BƯỚC 9: Update DQN ──────────────────────────────────────────┐
  │                                                                │
  │  Store: (state, action=2, reward=0.67, next_state)             │
  │  Train Q-network: action 2 (3 ảnh/rel) → reward cao            │
  │  → Q(state, action=2) tăng → lần sau ưu tiên action 2         │
  │  Epsilon: 0.5 → 0.45 (explore ít hơn)                         │
  └────────────────────────────────────────────────────────────────┘
  
→ EPOCH 4 (lặp lại, nhưng bây giờ model đã tốt hơn)
```

### 5.2 Tóm Tắt Vai Trò Từng Module

```
Module                    Câu hỏi nó trả lời
───────────────────────── ──────────────────────────────────────────────
DQN Agent                 "Sinh bao nhiêu ảnh tổng cộng?"
Active Learning           "Ưu tiên sinh ảnh cho relationship nào?"
Stable Diffusion          "Sinh ảnh ra sao?"
Quality Filter            "Ảnh có đủ chất lượng kỹ thuật không?"
Auto-Annotator            "Bounding box và class label ở đâu?"
Approximation Algorithm   "Trong pool, chọn ảnh nào đa dạng/hữu ích nhất?"
Reward Function           "Episode vừa rồi tốt hay xấu?"
Uncertainty Estimator     "Model có thực sự tự tin hay đang đoán?"
LLM Predictor             "RelTR không chắc, GPT-4V nói gì?"
Safety Classifier         "Tình huống này nguy hiểm không?"
```

### 5.3 Diagram: 2 Lớp Lọc Ảnh

```
Stable Diffusion sinh 30 ảnh
        │
        ▼
┌─────────────────────────────┐
│   LỚP 1: QUALITY FILTER    │  ← "Ảnh có tốt không?"
│                             │
│  ❌ 3 ảnh blur              │
│  ❌ 2 ảnh quá tối           │
│  ❌ 2 ảnh duplicate         │
│  ❌ 1 ảnh CLIP mismatch     │
│  ✅ 22 ảnh PASS             │
└─────────────┬───────────────┘
              │
              ▼
     Auto-Annotation (22 ảnh)
     Gán bbox + class + relationship
              │
              ▼
┌─────────────────────────────────┐
│   LỚP 2: APPROXIMATION ALG.    │  ← "Ảnh nào đa dạng/hữu ích nhất?"
│                                 │
│  Dedup: loại 3 ảnh giống nhau   │
│  Greedy: chọn 13/19 đa dạng    │
│                                 │
│  ✅ 13 ảnh SELECTED             │
│  ❌ 6 ảnh loại (dư thừa)        │
└─────────────┬───────────────────┘
              │
              ▼
      Fine-tune với 13 ảnh
```
