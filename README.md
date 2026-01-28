# 🔍 Scene Graph Generation với Reinforcement Learning

> Hệ thống phát hiện đối tượng và dự đoán quan hệ (Scene Graph Generation) kết hợp học tăng cường (Deep Q-Network) để giải quyết vấn đề **long-tail distribution** trong dữ liệu.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Tính năng chính](#-tính-năng-chính)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cấu hình](#-cấu-hình)
- [Kết quả](#-kết-quả)
- [Hệ thống Phân loại An toàn 3 Tầng](#-hệ-thống-phân-loại-an-toàn-3-tầng)
- [Cập nhật và cải tiến](#-cập-nhật-và-cải-tiến)

---

## 🎯 Giới thiệu

### Vấn đề
- **Long-tail distribution**: Trong các dataset Scene Graph (như Visual Genome), một số quan hệ xuất hiện rất thường xuyên (head classes: "on", "has", "wearing") trong khi đa số quan hệ xuất hiện rất hiếm (tail classes: "riding", "playing with", "looking at").
- Model thường học tốt head classes nhưng kém với tail classes.

### Giải pháp
Sử dụng **Reinforcement Learning (DQN)** để:
1. Tự động sinh dữ liệu synthetic cho các quan hệ hiếm (tail)
2. Điều chỉnh trọng số reward ưu tiên tail classes
3. Cải thiện cân bằng và hiệu suất tổng thể

---

## 🏗️ Kiến trúc hệ thống

### Tổng quan kiến trúc

Hệ thống bao gồm 2 luồng chính:

1. **Video Processing Pipeline**: Xử lý video với Safety Analysis
   - Input: Video file
   - Output: Video đã annotate + JSON thống kê an toàn

2. **RL Training Pipeline**: Training model với Reinforcement Learning
   - Input: Images + Relationships
   - Output: Trained model + Metrics

### Sơ đồ luồng hoạt động chi tiết

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 1: VRD PIPELINE (Visual Relationship Detection)              │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  [1] Input Image                                                                            │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  YOLO v11 Detection                                                          │           │
│  │  • Phát hiện bounding boxes                                                  │           │
│  │  • ROI features extraction                                                   │           │
│  │  • Global context vector                                                     │           │
│  └──────────────────────────────────────────────────────────────────────────────┘  
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  ReClip (Region Clip)                                                        │           │
│  │   • Gán nhãn cho các Boundingbox (classficication)                           │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  RelTR (Relationship Transformer)                                            │           │
│  │  • Transformer-based Scene Graph Generation                                  │           │
│  │  • Dự đoán triplets: (Subject, Predicate, Object)                            │           │
│  │  • Geometric + semantic relationship prediction                              │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  Scene Graph Output                                                          │           │
│  │  • Objects: [{bbox, class, confidence, features}]                            │           │
│  │  • Relationships: [{subject, relation, object, confidence}]                  │           │
│  │  • Long-tail analysis: Tính tần suất quan hệ                                 │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ├──────────────────────────────────────────────────────────────────────┐             │
│        │                                                                      │             │
│        │  [Video Mode - Safety Analysis Enabled]                              │             │
│        │        │                                                             │             │
│        │        ▼                                                             │             │
│        │  ┌──────────────────────────────────────────────────────────────┐   │             │
│        │  │  Safety Classifier (3-Tier System) ⭐                         │   │             │
│        │  │  • Tầng 1: White/Black/Gray List                              │   │             │
│        │  │    - White List: An toàn (SAFE)                               │   │             │
│        │  │    - Black List: Nguy hiểm (DANGEROUS)                        │   │             │
│        │  │    - Gray List: Không rõ → Chuyển Tầng 2                      │   │             │
│        │  │  • Tầng 2: LLM Semantic Reasoning                             │   │             │
│        │  │    - Hỏi OpenAI/Gemini: "Hành động này nguy hiểm không?"     │   │             │
│        │  │    - Cache responses trong llm_safety_cache.json              │   │             │
│        │  │    - Parse: SAFE/SUSPICIOUS/DANGEROUS                         │   │             │
│        │  │  • Tầng 3: Local Rules Database                               │   │             │
│        │  │    - Kiểm tra local_safety_rules.json (ưu tiên cao nhất)     │   │             │
│        │  │    - Học từ phản hồi người dùng (Human-in-the-loop)            │   │             │
│        │  │  • Output: Safety Level + Confidence + Explanation             │   │             │
│        │  └──────────────────────────────────────────────────────────────┘   │             │
│        │        │                                                             │             │
│        │        ▼                                                             │             │
│        │  ┌──────────────────────────────────────────────────────────────┐   │             │
│        │  │  Video Annotation & Alert Rendering                          │   │             │
│        │  │  • Vẽ relationships với mũi tên và text                     │   │             │
│        │  │  • Hiển thị cảnh báo an toàn:                                │   │             │
│        │  │    - DANGEROUS: Banner đỏ "NGUY HIỂM - Cần can thiệp ngay!"  │   │             │
│        │  │    - SUSPICIOUS: Banner cam "Cảnh báo nhẹ - Cần kiểm tra"    │   │             │
│        │  │  • Lưu thống kê vào *_summary.json                           │   │             │
│        │  │  • Output: Annotated video + JSON statistics                  │   │             │
│        │  └──────────────────────────────────────────────────────────────┘   │             │
│        │                                                                      │             │
│        └──────────────────────────────────────────────────────────────────────┘             │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                                        │
                        [Image Mode - RL Training]
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: REINFORCEMENT LEARNING TRAINING LOOP (DQN)                      │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [2.1] State Vector Construction (5D)                                        │           │
│  │  • detection_f1: F1 score từ detection model (normalized [0,1])              │           │
│  │  • relationship_f1: F1 score từ relationship model (normalized [0,1])        │           │
│  │  • reward: Reward hiện tại (tanh normalized)                                 │           │
│  │  • dataset_norm: Kích thước dataset / 50 (normalized)                        │           │
│  │  • epsilon: Exploration rate hiện tại (normalized)                           │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [2.2] DQN Agent Decision                                                    │           │
│  │  • Q-Network: Input(5) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(10)  │           │
│  │  • Epsilon-greedy: Exploration vs Exploitation                               │           │
│  │  • Action Space: Số biến thể prompt [1-10]                                   │           │
│  │  • Relationship-specific priorities:                                         │           │
│  │    - F1 thấp (<0.3) → sinh 3x variations                                     │           │
│  │    - F1 trung bình (0.3-0.7) → sinh 1-2x variations                          │           │
│  │    - F1 cao (>0.7) → sinh 0.5x variations                                    │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [2.3] Action Execution: Generate Synthetic Data                             │           │
│  │  • Input: Relationship triplets từ Scene Graph                               │           │
│  │  • Output: Số lượng variations cho mỗi relationship                          │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: GENERATIVE AI (Stable Diffusion + Annotation)                   │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [3.1] Prompt Generation                                                     │           │
│  │  • Relationship template mapping:                                            │           │
│  │    "holding" → "{subject} holding {object}"                                  │           │
│  │    "riding" → "{subject} riding {object}"                                    │           │
│  │  • Prompt variations:                                                        │           │
│  │    - Context: "in hands", "on street", "in room"                             │           │
│  │    - Quality: "high quality", "detailed", "realistic"                        │           │
│  │    - Lighting: "bright daylight", "soft lighting"                            │           │
│  │    - Background: "on the street", "in the park"                              │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [3.2] Stable Diffusion Image Generation                                     │           │
│  │  • Batch generation (batch_size=2) để tối ưu GPU                             │           │
│  │  • num_inference_steps: 50 (default)                                         │           │
│  │  • guidance_scale: 7.5 (default)                                             │           │
│  │  • Output: PIL Images                                                        │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [3.3] Quality Filter                                                        │           │
│  │  • Size/Aspect: min 512×512, max aspect 2.2                                  │           │
│  │  • Blur check: Laplacian variance ≥ 60                                       │           │
│  │  • Exposure: mean [20, 235], clip ratio ≤ 0.20                               │           │
│  │  • Duplicate: pHash comparison                                               │           │
│  │  • CLIP similarity: ≥ 0.23 với prompt                                        │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [3.4] Auto-Annotation (Priority Order)                                      │           │
│  │                                                                              │           │
│  │  Vấn đề: Stable Diffusion chỉ trả về pixels, không có bounding boxes         │           │
│  │  Giải pháp: Open-vocabulary detection để tự động tạo annotations             │           │
│  │                                                                              │           │
│  │  Flow:                                                                       │           │
│  │  1. Input: Synthetic image + relationship triplet (subject, relation, object)│           │
│  │  2. Extract text prompts: ["dog", "surfboard"] từ subject/object             │           │
│  │  3. Chạy detector với text prompts                                           │           │
│  │  4. Output: Objects với bbox [x1, y1, x2, y2] + class + confidence           │           │
│  │                                                                              │           │
│  │  Backend Priority (tự động chọn theo thứ tự):                                │           │
│  │                                                                              │           │
│  │  1. GroundingDINO (SOTA, chính xác nhất)                                     │           │
│  │     • Model: SwinT-OGC (Swin Transformer)                                    │           │
│  │     • Input format: Image + text prompt "dog . surfboard"                    │           │
│  │     • Thresholds: box_threshold=0.25, text_threshold=0.20                    │           │
│  │     • Output: Normalized coords [cx, cy, w, h] → convert to [x1, y1, x2, y2] │           │
│  │     • Auto-detect paths:                                                     │           │
│  │       - Config: GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py│           │
│  │       - Weights: weights/groundingdino_swint_ogc.pth                         │           │
│  │                                                                              │           │
│  │  2. OWL-ViT (Lightweight, HuggingFace)                                       │           │
│  │     • Model: google/owlvit-base-patch32 (tự động download từ HuggingFace)    │           │
│  │     • Input format: Image + text prompts ["a photo of a dog", "a photo of..."]│          │
│  │     • Threshold: box_threshold=0.25                                           │          │
│  │     • Output: Direct [x1, y1, x2, y2] coordinates                            │           │
│  │     • Fallback nếu GroundingDINO không có                                    │           │
│  │                                                                              │           │
│  │  3. YOLO + CLIP (Fallback)                                                   │           │
│  │     • Pipeline: YOLO detect → CLIP classify với open vocabulary              │           │
│  │     • Input: Image path                                                      │           │
│  │     • Matching: Fuzzy match labels với text prompts                          │           │
│  │     • Confidence: 0.7 nếu matched, 0.5 nếu không                             │           │
│  │     • Limited vocabulary (chỉ detect classes YOLO biết)                      │           │
│  │                                                                              │           │
│  │  4. Pseudo-bbox (Heuristic, low quality - LAST RESORT)                       │           │
│  │     • Chỉ dùng khi tất cả detectors đều fail                                 │           │
│  │     • Heuristics dựa trên relation type:                                     │           │
│  │       - "on"/"above"/"riding": Subject trên, Object dưới                     │           │
│  │       - "under"/"below": Subject dưới, Object trên                           │           │
│  │       - "holding"/"carrying": Subject lớn, Object nhỏ gần subject            │           │
│  │       - Default: Subject trái, Object phải                                   │           │
│  │     • Confidence: 0.3 (rất thấp)                                             │           │
│  │     • WARNING: Chất lượng thấp, chỉ dùng khi không còn lựa chọn              ;o│           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [3.5] Dataset Ingestion                                                     │           │
│  │  • Synthetic samples: {image_path, objects, relationships, metadata}         │           │
│  │  • Relationship inference: RelTR trên synthetic images                       │           │
│  │  • Fallback relationships nếu RelTR không detect được                        │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: MODEL TRAINING & EVALUATION                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [4.1] VRD Fine-tuning                                                       │           │
│  │  • Input: Dataset samples (original + synthetic)                             │           │
│  │  • Prepare VRD targets: entities + relationships                             │           │
│  │  • Training loop:                                                            │           │
│  │    - Forward pass với image tensor                                           │           │
│  │    - Compute loss: bbox_loss + giou_loss + rel_loss                          │           │
│  │    - Backward pass + optimizer step                                          │           │
│  │  • Checkpoint saving theo epoch                                              │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [4.2] Evaluation                                                            │           │
│  │  • Detection metrics: Precision, Recall, F1                                  │           │
│  │  • Relationship metrics: Precision, Recall, F1                               │           │
│  │  • mR@K metrics: mR@10, mR@20, mR@50, mR@100 (công bằng cho long-tail)       │           │
│  │  • Long-tail loss: Loss riêng cho các quan hệ hiếm                           │           │
│  │  • Diversity score: Đa dạng relation/object types                            │           │
│  │  • Consistency score: Độ ổn định predictions                                 │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [4.3] Reward Calculation                                                    │           │
│  │  • Base components:                                                          │           │ 
│  │    - Detection F1 × 0.25                                                     │           │
│  │    - Relationship F1 × 0.45 (cao nhất)                                       │           │
│  │    - Diversity × 0.15                                                        │           │
│  │    - Improvement × 0.05                                                      │           │
│  │  • Long-tail boost:                                                          │           │
│  │    tail_weight = 1 / sqrt(frequency + 1e-3)                                  │           │
│  │    relationship_score *= (1 + tail_weight)                                   │           │
│  │  • Final reward: Sum of all components                                       │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [4.4] DQN Update                                                            │           │
│  │  • Store experience: (state, action, reward, next_state)                     │           │
│  │  • Experience replay: Sample batch từ buffer (size=10,000)                   │           │
│  │  • Q-learning update:                                                        │           │
│  │    Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]                          │           │
│  │  • Target network update: Mỗi 20 steps                                       │           │
│  │  • Epsilon decay: ε ← ε × 0.995 (min=0.01)                                   │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        ▼                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐           │
│  │  [4.5] Experiment Tracking                                                   │           │
│  │  • Save metrics: JSON files                                                  │           │
│  │  • Save plots: Reward vs Epoch, Components breakdown                         │           │
│  │  • Save models: Checkpoints theo epoch                                       │           │
│  │  • Save synthetic images: experiments/exp_XXX/ai_images/                     │           │
│  │  • Metadata: Config, hyperparameters, timestamps                             │           │
│  └──────────────────────────────────────────────────────────────────────────────┘           │
│        │                                                                                    │
│        └──────────────────────────────────────────────────────────────────────┐             │
│                                                                               │             │
│                                                                               ▼             │
│                                                                    [Lặp lại từ Phase 2.1]   │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Tóm tắt luồng hoạt động

#### Luồng xử lý Video (VideoRelationPipeline)

1. **VRD Pipeline**: Input video frame → YOLO detection → Region Clip classification → RelTR relationship prediction → Scene Graph
2. **Safety Analysis** (nếu enabled):
   - **Tầng 1**: Kiểm tra White/Black List → Nếu match → Return SAFE/DANGEROUS
   - **Tầng 3**: Kiểm tra Local Rules (ưu tiên cao nhất) → Nếu có → Return từ rule
   - **Tầng 2**: Nếu không match → Hỏi LLM (OpenAI/Gemini) → Cache response
   - **Default**: Nếu LLM không hoạt động → Return SUSPICIOUS
3. **Video Annotation**: Vẽ relationships và cảnh báo an toàn lên frame
4. **Output**: Video đã annotate + JSON thống kê an toàn

#### Luồng RL Training (Reinforcement Learning)

1. **VRD Pipeline**: Input image → YOLO detection → Region Clip classification → RelTR relationship prediction → Scene Graph
2. **RL State**: Xây dựng state vector 5D từ metrics hiện tại
3. **DQN Decision**: Agent quyết định số lượng variations cần sinh cho mỗi relationship
4. **GenAI Generation**: Stable Diffusion sinh ảnh từ relationship triplets với prompt variations
5. **Quality Filter**: Lọc ảnh chất lượng thấp (blur, exposure, duplicate, CLIP similarity)
6. **Auto-Annotation**: GroundingDINO/OWL-ViT tự động tạo bounding boxes cho synthetic images
7. **Dataset Ingestion**: Thêm synthetic samples vào training dataset
8. **Model Training**: Fine-tune RelTR với dataset mới (hỗ trợ multiple epochs)
9. **Evaluation**: Tính metrics (Detection F1, Relationship F1, mR@K, Long-tail Loss, Diversity, Consistency)
10. **Reward Calculation**: Tính reward với long-tail boost cho quan hệ hiếm
11. **DQN Update**: Cập nhật Q-network từ experience replay
12. **Loop**: Lặp lại từ bước 2 cho đến khi đạt convergence hoặc max epochs

---

## ✨ Tính năng chính

### 1. Object Detection (YOLO + CLIP)
- YOLO v11 phát hiện bounding box
- CLIP re-classify với open vocabulary
- ROI features extraction

### 2. Relationship Prediction (RelTR)
- Transformer-based Scene Graph Generation
- Dự đoán triplets: (Subject, Predicate, Object)

### 3. Reinforcement Learning (DQN)
- **State vector (5D)**: detection_f1, relationship_f1, reward, dataset_norm, epsilon
- **Action space**: Số biến thể prompt sinh ảnh [1-10]
- **Reward function**: Detection + Relationship + Diversity + Consistency + Improvement
- **Long-tail boost**: Trọng số `1/sqrt(freq)` cho quan hệ hiếm
- **Multiple epochs training**: Hỗ trợ training nhiều epochs trên toàn bộ dataset tích lũy
- **Dataset input**: Cho phép chọn/nhập dataset từ thư mục ảnh khi training

### 4. Synthetic Data Generation
- Stable Diffusion sinh ảnh từ relationship triplets
- Quality Filter: blur, exposure, duplicate, CLIP similarity
- Auto-Annotation: GroundingDINO/OWL-ViT cho bbox

### 5. Experiment Management
- Lưu trữ metrics, plots, models theo experiment
- So sánh và export kết quả

### 6. Video Processing với Safety Analysis
- Xử lý video frame-by-frame với YOLO + RelTR
- Phân tích relationships và đánh giá mức độ nguy hiểm
- Hiển thị cảnh báo trực quan trên video (banner đỏ/cam)
- Xuất thống kê an toàn trong JSON

### 7. Hệ thống Phân loại An toàn 3 Tầng ⭐ MỚI
- **Tầng 1**: White/Black/Gray List - Phân loại nhanh các hành động đã biết
- **Tầng 2**: LLM Semantic Reasoning - Sử dụng OpenAI/Gemini để đánh giá ngữ nghĩa
- **Tầng 3**: Local Rules Database - Học từ phản hồi người dùng (Human-in-the-loop)
- Tích hợp tự động với `VideoRelationPipeline`
- Cache LLM responses để tiết kiệm chi phí API

---

## 🔧 Cài đặt

### Yêu cầu hệ thống
- Python 3.10+
- CUDA 11.8+ (khuyến nghị GPU với ≥8GB VRAM)
- 16GB RAM

### Bước 1: Clone repository
```bash
git clone <repository-url>
cd yolov11
```

### Bước 2: Tạo virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### Bước 3: Cài đặt dependencies cơ bản
```bash
pip install -r requirements.txt
```

### Bước 4: Cài đặt PyTorch với CUDA
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Bước 5: Cài đặt Stable Diffusion
```bash
pip install diffusers accelerate transformers
```

### Bước 6: Cài đặt CLIP
```bash
pip install git+https://github.com/openai/CLIP.git
```

### Bước 7: Cài đặt Auto-Annotation (chọn 1 trong 2)

**Option A: OWL-ViT (Dễ cài, khuyên dùng cho người mới)**
```bash
pip install transformers
# Model sẽ tự download khi chạy lần đầu (~1.5GB)
```

**Option B: GroundingDINO (Chính xác nhất, SOTA)**
```bash
# Clone và cài đặt
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..

# Download weights
mkdir -p weights
wget -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### Bước 8: Cài đặt thêm (tùy chọn)
```bash
# Duplicate image detection
pip install imagehash

# Sentence similarity
pip install sentence-transformers

# Environment variables cho Safety System
pip install python-dotenv
```

### Bước 9: Cấu hình API Key cho Safety System (Tùy chọn)

Nếu muốn sử dụng LLM Safety Analyzer (Tầng 2), cần cấu hình API key:

**Cách 1: Tạo file `.env` (Khuyến nghị)**
```bash
# Tạo file .env ở thư mục gốc
# Windows
type nul > .env

# Linux/Mac
touch .env
```

Mở file `.env` và thêm:
```
OPENAI_API_KEY=your_openai_api_key_here
# hoặc
GEMINI_API_KEY=your_gemini_api_key_here
```

**Cách 2: Environment variables**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

**Lưu ý**: Nếu không có API key, hệ thống vẫn hoạt động nhưng sẽ mặc định về "SUSPICIOUS" cho các hành động không biết (không hỏi LLM).

### Bước 10: Download model weights

```bash
# YOLO weights (nếu chưa có)
# Đặt file fine-tune.pt vào thư mục gốc

# RelTR checkpoint (nếu chưa có)
# Đặt file checkpoint.pth vào thư mục gốc
```

---

## 🚀 Sử dụng

### Chạy ứng dụng Console
```bash
python app_console.py
```

Menu:
1. Chọn ảnh để xử lý
2. Chạy pipeline phát hiện vật thể và mối quan hệ
3. Tải lại dữ liệu JSON
4. **Chạy RL Training** ← Học tăng cường
   - Chọn dataset: Sử dụng dataset hiện tại / Chọn thư mục ảnh / Bỏ qua
   - Nhập số epochs để training
5. Tạo dữ liệu synthetic
6. Đánh giá kết quả training
7. Quản lý Experiments
8. Tiếp tục RL Training từ experiment trước
9. Thoát

### Chạy ứng dụng GUI
```bash
python app.py
```

### Chạy detection đơn lẻ
```bash
python detect_objects.py <path_to_image>
```

### Chạy relationship prediction
```bash
python boundingbox_objects.py --yolo_json converted_bboxes.json --img_path <image> --resume checkpoint.pth
```

### Chạy video với Safety Analysis

```python
from video_relation_pipeline import VideoRelationPipeline

# Khởi tạo pipeline với safety classifier enabled
pipeline = VideoRelationPipeline(
    safety_classifier_enabled=True  # Bật hệ thống cảnh báo
)

# Xử lý video
result = pipeline.process_video(
    video_path="home_video.mp4",
    output_dir="video_outputs",
    frame_stride=2  # Xử lý mỗi 2 frames
)

# Kết quả:
# - Video đã annotate: video_outputs/home_video_relations.avi
# - Thống kê: video_outputs/home_video_summary.json
```

**Xem thống kê an toàn**:
```python
import json

with open("video_outputs/home_video_summary.json", "r", encoding="utf-8") as f:
    stats = json.load(f)
    alerts = stats["safety_alerts"]
    print(f"Tổng cảnh báo: {alerts['total_alerts']}")
    print(f"Nguy hiểm: {alerts['dangerous_count']}")
    print(f"Nghi ngờ: {alerts['suspicious_count']}")
```

---

## 📁 Cấu trúc dự án

```
yolov11/
├── app.py                      # GUI Application (Tkinter)
├── app_console.py              # Console Application
├── detect_objects.py           # YOLO + CLIP detection
├── boundingbox_objects.py      # RelTR relationship inference
├── convert_yolo_to_reltr.py    # Data format conversion
├── video_relation_pipeline.py  # Video processing
│
├── RL/                         # Reinforcement Learning module
│   ├── reinforcement_learning.py   # DQN Agent chính
│   ├── ai_images_generator.py      # Stable Diffusion + Quality Filter
│   ├── auto_annotator.py           # GroundingDINO/OWL-ViT annotation
│   ├── rl_enhancement.py           # RL integration with app
│   ├── experiment_manager.py       # Experiment tracking
│   ├── experiment_viewer.py        # View experiment results
│   ├── experience_manager.py       # Replay buffer management
│   ├── model_manager.py            # Model checkpointing
│   ├── training_evaluator.py       # Evaluation metrics
│   ├── data_augmentation.py        # Data augmentation
│   │
│   ├── safety_classifier.py        # ⭐ Safety System: 3-tier classifier
│   ├── llm_safety_analyzer.py      # ⭐ Safety System: LLM integration
│   ├── local_rules_db.py           # ⭐ Safety System: Local rules DB
│   ├── safety_config.json          # ⭐ Safety System: Configuration
│   └── README_SAFETY_SYSTEM.md     # ⭐ Safety System: Documentation
│
├── models/                     # Model definitions
│   ├── reltr.py                # RelTR model
│   ├── transformer.py          # Transformer layers
│   ├── backbone.py             # Backbone networks
│   └── ...
│
├── util/                       # Utilities for RelTR
│   ├── box_ops.py
│   └── misc.py
│
├── utils/                      # Utilities for YOLO
│   ├── general.py
│   ├── metrics.py
│   └── ...
│
├── tools/                      # Visualization tools
│   ├── reltr_rl_examples.py
│   └── yolo_heatmap.py
│
├── demo/                       # Demo images
├── data/                       # Dataset configs
├── experiments/                # Experiment outputs
├── templates/                  # HTML templates
│
├── fine-tune.pt               # YOLO weights
├── checkpoint.pth             # RelTR checkpoint
├── reltr_finetuned.pth        # Fine-tuned RelTR
├── requirements.txt           # Python dependencies
├── .env                       # API keys (tạo file này, không commit)
├── llm_safety_cache.json      # LLM cache (tự động tạo)
├── local_safety_rules.json   # Local safety rules (tự động tạo)
└── README.md                  # This file
```

---

## ⚙️ Cấu hình

### DQN Hyperparameters
| Parameter | Giá trị | Mô tả |
|-----------|---------|-------|
| Learning Rate | 0.001 | AdamW optimizer |
| Gamma | 0.95 | Discount factor |
| Epsilon | 0.9 → 0.01 | Exploration rate (decay=0.995) |
| Batch Size | 32 | Experience replay batch |
| Buffer Size | 10,000 | Replay buffer capacity |
| Target Update | 20 steps | Target network update interval |

### Reward Weights (Base)
| Component | Weight | Mô tả |
|-----------|--------|-------|
| Detection | 0.25 | F1 score của detection |
| Relationship | 0.45 | F1 score của relationship (cao nhất) |
| Diversity | 0.15 | Đa dạng về relation/object types |
| Consistency | 0.10 | Độ ổn định predictions |
| Improvement | 0.05 | Xu hướng cải thiện |

### Long-tail Boost
```python
tail_weight = 1 / sqrt(frequency + 1e-3)  # Normalize to sum=1
relationship_score *= (1 + tail_weight)   # Boost rare relations
```

### Quality Filter Thresholds
| Check | Threshold | Mô tả |
|-------|-----------|-------|
| Min Size | 512×512 | Kích thước tối thiểu |
| Max Aspect | 2.2 | Tỉ lệ tối đa |
| Blur Variance | 60 | Laplacian variance |
| Exposure Mean | [20, 235] | Độ sáng trung bình |
| CLIP Similarity | 0.23 | Bám sát prompt |

---

## 📊 Kết quả

### Metrics theo dõi
- **Detection Metrics**: Precision, Recall, F1 cho object detection
- **Relationship Metrics**: Precision, Recall, F1 cho relationship prediction
- **mR@K Metrics**: Mean Recall@K (mR@10, mR@20, mR@50, mR@100) - đánh giá công bằng cho long-tail
- **Long-tail Loss**: Loss riêng cho các quan hệ hiếm (weighted by tail_weights)
- **Reward**: Tổng reward từ DQN
- **Diversity Score**: Đa dạng synthetic data
- **Consistency Score**: Độ ổn định predictions

### Visualizations
- Reward vs Epoch plot
- Reward components breakdown
- Action frequency histogram
- Q-values heatmap

### Experiment outputs
```
experiments/
└── exp_001/
    ├── ai_images/          # Synthetic images
    ├── dataset/            # Training samples
    ├── logs/               # Training logs
    ├── metrics/            # JSON metrics
    ├── models/             # Saved checkpoints
    ├── plots/              # Visualization plots
    └── metadata.json       # Experiment config
```

---

## 🛡️ Hệ thống Phân loại An toàn 3 Tầng

Hệ thống phân loại an toàn được tích hợp vào `VideoRelationPipeline` để phát hiện và cảnh báo các hành động nguy hiểm trong video, đặc biệt là trong môi trường gia đình có trẻ em.

### Kiến trúc 3 Tầng

#### Sơ đồ luồng xử lý Safety System

```
Relationship từ RelTR: {subject, relation, object}
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  SafetyClassifier.classify()                          │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Tầng 3: Local Rules Database (Ưu tiên cao nhất)      │
│  • Kiểm tra local_safety_rules.json                   │
│  • Nếu có rule → Return level từ rule                 │
│  • Nếu không → Tiếp tục                               │
└────────────────────────────────────────────────────────┘
         │
         ├─ Có rule? → Return (SAFE/SUSPICIOUS/DANGEROUS)
         │
         └─ Không có rule? ▼
┌────────────────────────────────────────────────────────┐
│  Tầng 1: White/Black/Gray List                       │
│  • Kiểm tra Black List → Nếu match → DANGEROUS        │
│  • Kiểm tra White List → Nếu match → SAFE             │
│  • Không match → Gray List → Chuyển Tầng 2            │
└────────────────────────────────────────────────────────┘
         │
         ├─ Match Black? → Return DANGEROUS
         ├─ Match White? → Return SAFE
         │
         └─ Không match? ▼
┌────────────────────────────────────────────────────────┐
│  Tầng 2: LLM Semantic Reasoning                       │
│  • Kiểm tra cache (llm_safety_cache.json)             │
│  • Nếu có cache → Return từ cache                     │
│  • Nếu không → Gọi LLM API (OpenAI/Gemini)            │
│    - Build prompt: "Hành động này nguy hiểm không?"   │
│    - Parse response: SAFE/SUSPICIOUS/DANGEROUS        │
│    - Lưu vào cache                                     │
│  • Nếu LLM không hoạt động → Default SUSPICIOUS      │
└────────────────────────────────────────────────────────┘
         │
         ▼
Return: (SafetyLevel, confidence, explanation)
```

#### Tầng 1: White/Black/Gray List
- **White List**: Các hành động an toàn (ví dụ: `person sitting on chair`)
- **Black List**: Các hành động nguy hiểm (ví dụ: `child holding knife`)
- **Gray List**: Các hành động không rõ ràng → Chuyển sang Tầng 2

#### Tầng 2: LLM Semantic Reasoning
- Sử dụng OpenAI/Gemini để đánh giá ngữ nghĩa
- Hỏi LLM: "Hành động này có nguy hiểm không?"
- Cache responses trong `llm_safety_cache.json` để tiết kiệm chi phí API
- Hỗ trợ nhiều model: `gpt-3.5-turbo`, `gpt-4o`, `gpt-5.2`, `gemini-pro`, etc.

#### Tầng 3: Local Rules Database
- Học từ phản hồi người dùng (Human-in-the-loop)
- **Ưu tiên cao nhất** - kiểm tra trước cả white/black list
- Lưu trong `local_safety_rules.json`
- Tự động track `usage_count` cho mỗi rule

### Sử dụng

```python
from video_relation_pipeline import VideoRelationPipeline

# Bật safety classifier
pipeline = VideoRelationPipeline(
    safety_classifier_enabled=True
)

# Xử lý video
result = pipeline.process_video("video.mp4")
```

### Cấu hình

File `RL/safety_config.json` chứa:
- `white_list`: Danh sách hành động an toàn
- `black_list`: Danh sách hành động nguy hiểm
- `llm_enabled`: Bật/tắt LLM
- `llm_provider`: "openai" hoặc "gemini"
- `llm_model`: Model name (ví dụ: "gpt-3.5-turbo")

### Thêm quy tắc cục bộ

```python
from RL.local_rules_db import LocalRulesDatabase
from RL.llm_safety_analyzer import SafetyLevel

db = LocalRulesDatabase()
db.add_rule(
    subject="child",
    relation="playing with",
    object_name="toy",
    level=SafetyLevel.SAFE,
    source="user_feedback"
)
```

### Xem tài liệu chi tiết

Xem file [`RL/README_SAFETY_SYSTEM.md`](RL/README_SAFETY_SYSTEM.md) để biết thêm chi tiết về:
- Cách hoạt động của từng tầng
- Cấu hình chi tiết
- Ví dụ sử dụng
- Luồng xử lý

---

## 🔬 Chi tiết kỹ thuật

### State Vector (5 chiều)
```python
state = [
    detection_f1,      # F1 score detection (normalized)
    relationship_f1,   # F1 score relationship (normalized)
    reward,            # Current reward (tanh normalized)
    dataset_norm,      # Dataset size / 50 (normalized)
    epsilon            # Current exploration rate
]
```

### Q-Network Architecture
```
Input(5) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(10)
```

### Auto-Annotation Priority và Chi tiết Implementation

**File chính**: `RL/auto_annotator.py`

**Vấn đề**: Stable Diffusion chỉ trả về pixels, không có bounding boxes. Cần tự động tạo annotations cho synthetic images.

**Giải pháp**: Sử dụng Open-Vocabulary Detectors với priority order, tự động fallback nếu backend trước fail.

#### 1. GroundingDINO (Ưu tiên cao nhất - SOTA)
**Function**: `_annotate_groundingdino()` trong `RL/auto_annotator.py`

**Cách hoạt động**:
```python
# 1. Load image và convert sang tensor
image_source, image_tensor = load_image(image_path)

# 2. Tạo text prompt từ relationship triplet
# Input: relationship = {"subject": "dog", "relation": "riding", "object": "surfboard"}
# Output: text_prompt = "dog . surfboard"  # Format: "subject . object"

# 3. Predict với GroundingDINO
boxes, logits, phrases = predict(
    model=groundingdino_model,
    image=image_tensor,
    caption=text_prompt,
    box_threshold=0.25,   # Minimum confidence for boxes
    text_threshold=0.20,  # Minimum confidence for text matching
    device="cuda"
)

# 4. Convert normalized coords [cx, cy, w, h] → absolute [x1, y1, x2, y2]
cx, cy, bw, bh = box.tolist()
x1 = int((cx - bw / 2) * width)
y1 = int((cy - bh / 2) * height)
x2 = int((cx + bw / 2) * width)
y2 = int((cy + bh / 2) * height)

# 5. Clamp to image bounds
x1 = max(0, min(x1, width - 1))
y1 = max(0, min(y1, height - 1))
x2 = max(x1 + 1, min(x2, width))
y2 = max(y1 + 1, min(y2, height))
```

**Ưu điểm**:
- ✅ Open-vocabulary: Detect bất kỳ object nào từ text prompt
- ✅ Chính xác cao (State-of-the-Art)
- ✅ Hỗ trợ nhiều objects trong một prompt
- ✅ Model: SwinT-OGC (Swin Transformer backbone)

**Yêu cầu cài đặt**:
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
# Download weights
wget -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

**Auto-detect paths** (tự động tìm trong code):
- Config: `GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`
- Weights: `weights/groundingdino_swint_ogc.pth` hoặc `~/.cache/groundingdino/`

#### 2. OWL-ViT (Fallback 1 - Lightweight)
**Function**: `_annotate_owlvit()` trong `RL/auto_annotator.py`

**Cách hoạt động**:
```python
# 1. Load model từ HuggingFace (tự động download lần đầu)
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
model.to(device).eval()

# 2. Prepare inputs với text prompts
# Format: "a photo of a {object}" cho mỗi prompt
texts = [["a photo of a dog", "a photo of a surfboard"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# 3. Predict
with torch.no_grad():
    outputs = model(**inputs)

# 4. Post-process (có sẵn trong processor)
target_sizes = torch.tensor([[height, width]], device=device)
results = processor.post_process_object_detection(
    outputs, threshold=0.25, target_sizes=target_sizes
)[0]

# 5. Extract boxes (đã là [x1, y1, x2, y2] format)
for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
    x1, y1, x2, y2 = box.tolist()
    class_name = text_prompts[label.item()]  # Map label index to prompt
```

**Ưu điểm**:
- ✅ Dễ cài: Chỉ cần `pip install transformers`
- ✅ Tự động download model (~1.5GB) từ HuggingFace
- ✅ Lightweight hơn GroundingDINO
- ✅ Không cần config files

**Nhược điểm**:
- ⚠️ Độ chính xác thấp hơn GroundingDINO
- ⚠️ Cần format text: "a photo of a {object}"

#### 3. YOLO + CLIP (Fallback 2 - Limited Vocabulary)
**Function**: `_annotate_yolo_clip()` trong `RL/auto_annotator.py`

**Cách hoạt động**:
```python
# 1. YOLO detect objects (sử dụng pipeline có sẵn)
detected_objects, yolo_labels, original_image, feature_map, global_context = \
    detection_pipeline.detect_objects(image_path)

# 2. CLIP classify với open vocabulary
classified_results = detection_pipeline.classify_with_clip(
    detected_objects, yolo_labels
)
# Output: [(label, bbox), ...] ví dụ: [("dog", [x1, y1, x2, y2]), ...]

# 3. Fuzzy match với text prompts
for label, bbox in classified_results:
    label_lower = label.strip().lower()
    matched = any(
        prompt.lower() in label_lower or label_lower in prompt.lower()
        for prompt in text_prompts  # ["dog", "surfboard"]
    )
    confidence = 0.7 if matched else 0.5
```

**Ưu điểm**:
- ✅ Sử dụng pipeline có sẵn (`detect_objects.py`)
- ✅ Không cần cài thêm dependencies
- ✅ CLIP hỗ trợ open vocabulary (một phần)

**Nhược điểm**:
- ⚠️ Limited vocabulary (chỉ detect classes YOLO biết)
- ⚠️ Fuzzy matching có thể không chính xác
- ⚠️ Confidence thấp hơn (0.5-0.7)

#### 4. Pseudo-bbox (Last Resort - Heuristic, Low Quality)
**Function**: `_create_pseudo_annotations()` trong `RL/auto_annotator.py`

**Cách hoạt động**:
```python
# Heuristics dựa trên relation type
subject = relationship.get('subject', 'unknown')
relation = relationship.get('relation', 'unknown')
obj = relationship.get('object', 'unknown')

if relation in ['on', 'above', 'over', 'riding']:
    # Subject trên, Object dưới
    subject_bbox = [width*0.3, height*0.1, width*0.7, height*0.45]
    object_bbox = [width*0.2, height*0.5, width*0.8, height*0.9]
    
elif relation in ['under', 'below']:
    # Subject dưới, Object trên
    subject_bbox = [width*0.2, height*0.5, width*0.8, height*0.9]
    object_bbox = [width*0.3, height*0.1, width*0.7, height*0.45]
    
elif relation in ['holding', 'carrying', 'using']:
    # Subject lớn, Object nhỏ gần subject
    subject_bbox = [width*0.2, height*0.1, width*0.7, height*0.9]
    object_bbox = [width*0.5, height*0.3, width*0.8, height*0.6]
    
else:
    # Default: Subject trái, Object phải
    subject_bbox = [width*0.05, height*0.2, width*0.45, height*0.8]
    object_bbox = [width*0.55, height*0.2, width*0.95, height*0.8]

# Confidence cố định = 0.3 (rất thấp)
objects = [
    {'class': subject, 'bbox': subject_bbox, 'confidence': 0.3, 'source': 'pseudo'},
    {'class': obj, 'bbox': object_bbox, 'confidence': 0.3, 'source': 'pseudo'}
]
```

**Lưu ý**:
- ⚠️ **WARNING**: Chất lượng rất thấp (confidence = 0.3)
- ⚠️ Chỉ dùng khi tất cả detectors đều fail
- ⚠️ Bbox được tạo từ heuristics, không phải từ ảnh thực tế
- ⚠️ Có thể không chính xác với layout phức tạp

### Code Flow trong `_ingest_synthetic_samples()`

**File**: `RL/reinforcement_learning.py` - dòng 1007-1034

```python
# Trong hàm _ingest_synthetic_samples()
for synthetic_data in synthetic_samples:
    original_relationship = data.get('original_relationship')
    # {"subject": "dog", "relation": "riding", "object": "surfboard"}
    
    # 1. Gọi AutoAnnotator (singleton pattern)
    annotator = get_annotator()  # Tự động chọn backend tốt nhất
    
    # 2. Annotate từ relationship
    annotation_result = annotator.annotate_from_relationship(
        image_path, original_relationship
    )
    
    # 3. annotation_result structure:
    # {
    #   'image_path': 'experiments/exp_001/ai_images/image_001.jpg',
    #   'width': 512, 'height': 512,
    #   'objects': [
    #     {
    #       'class': 'dog',
    #       'bbox': [100, 150, 300, 400],  # [x1, y1, x2, y2]
    #       'confidence': 0.85,
    #       'source': 'groundingdino'  # hoặc 'owlvit', 'yolo_clip', 'pseudo'
    #     },
    #     {
    #       'class': 'surfboard',
    #       'bbox': [200, 300, 450, 500],
    #       'confidence': 0.78,
    #       'source': 'groundingdino'
    #     }
    #   ],
    #   'annotation_backend': 'groundingdino',
    #   'num_detected': 2
    # }
    
    # 4. Sử dụng objects để build RelTR training targets
    sample = {
        'image_path': annotation_result['image_path'],
        'objects': annotation_result['objects'],
        'relationships': [...],  # Được tạo ở bước tiếp theo
        'annotation_backend': annotation_result['annotation_backend']
    }
```

### Relationship Generation (Tạo Relationships từ Objects)

**Vấn đề**: Sau khi auto-annotation tạo được objects với bbox, cần tạo relationships (subject-relation-object triplets) để train RelTR.

**Giải pháp**: Hệ thống sử dụng **3 phương pháp theo thứ tự ưu tiên**, tự động fallback nếu phương pháp trước fail.

**File**: `RL/reinforcement_learning.py` - `_ingest_synthetic_samples()` (dòng 1040-1071)

#### Flow tổng quan:

```
Synthetic Image + Objects (từ Auto-Annotation)
         │
         ▼
┌────────────────────────────────────────┐
│ Method 1: Build from Original          │
│ Match objects với original_relationship│
└────────────────────────────────────────┘
         │
         ├─ Success? → Use relationships
         │
         └─ Fail? ▼
┌────────────────────────────────────────┐
│ Method 2: RelTR Inference              │
│ Dùng RelTR model để predict            │
└────────────────────────────────────────┘
         │
         ├─ Success? → Use relationships
         │
         └─ Fail? ▼
┌────────────────────────────────────────┐
│ Method 3: Fallback                    │
│ Tạo trực tiếp từ original_relationship │
└────────────────────────────────────────┘
```

#### Method 1: Build from Original (Ưu tiên cao nhất)

**Function**: `_build_relationship_from_original()` - dòng 753-774

**Cách hoạt động**:
```python
def _build_relationship_from_original(objects, original_relationship):
    # Input:
    # - objects: [{'class': 'dog', 'bbox': [...]}, {'class': 'surfboard', 'bbox': [...]}]
    # - original_relationship: {'subject': 'dog', 'relation': 'riding', 'object': 'surfboard'}
    
    # 1. Tìm index của subject và object trong objects list
    subject_idx = _find_object_index(objects, original_relationship.get('subject', ''))
    object_idx = _find_object_index(objects, original_relationship.get('object', ''))
    
    # 2. Nếu tìm thấy cả 2, tạo relationship
    if subject_idx is not None and object_idx is not None:
        return [{
            'subject': objects[subject_idx].get('class'),  # 'dog'
            'relation': original_relationship.get('relation'),  # 'riding'
            'object': objects[object_idx].get('class'),  # 'surfboard'
            'confidence': 1.0,  # High confidence vì match với original
            'source': 'original'
        }]
    return []  # Fail nếu không match được
```

**Object Matching Logic** (`_find_object_index()` - dòng 729-751):

Hệ thống sử dụng **3-level fuzzy matching** để tìm object:

```python
def _find_object_index(objects, class_name):
    normalized = _normalize_label(class_name)  # Lowercase, strip
    
    # Level 1: Exact match
    for idx, obj in enumerate(objects):
        if _normalize_label(obj.get('class')) == normalized:
            return idx  # ✅ Found
    
    # Level 2: Synonym matching
    synonyms = _get_label_synonyms(class_name)
    # Ví dụ: 'person' → ['man', 'woman', 'people', 'human', 'boy', 'girl', ...]
    for idx, obj in enumerate(objects):
        if _normalize_label(obj.get('class')) in synonyms:
            return idx  # ✅ Found
    
    # Level 3: Partial matching (substring)
    for idx, obj in enumerate(objects):
        obj_label = _normalize_label(obj.get('class'))
        if normalized in obj_label or obj_label in normalized:
            return idx  # ✅ Found (ví dụ: 'dog' in 'doggy')
    
    return None  # ❌ Not found
```

**Synonym Dictionary** (`_LABEL_SYNONYMS` - dòng 698-710):
```python
_LABEL_SYNONYMS = {
    'person': ['man', 'woman', 'people', 'human', 'boy', 'girl', 'child', 'adult'],
    'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bike', 'bicycle'],
    'animal': ['dog', 'cat', 'horse', 'bird', 'cow', 'sheep'],
    # ... và nhiều synonyms khác
}
```

**Ưu điểm**:
- ✅ Chính xác cao (match với original relationship)
- ✅ Confidence = 1.0 (tin cậy nhất)
- ✅ Nhanh (không cần chạy model)

**Nhược điểm**:
- ⚠️ Chỉ hoạt động nếu objects được detect match với original
- ⚠️ Nếu auto-annotation fail hoặc detect sai class → không tạo được relationship

#### Method 2: RelTR Inference (Fallback 1)

**Function**: `_run_reltr_inference()` → `_decode_relationships()` - dòng 911-1287

**Cách hoạt động**:
```python
def _run_reltr_inference(image_tensor, objects, global_context, image_size):
    # 1. Load RelTR model
    model, _ = _ensure_relationship_model()
    
    # 2. Prepare inputs
    samples = nested_tensor_from_tensor_list([image_tensor])
    context_tensor = _prepare_global_context_tensor(global_context)
    
    # 3. Run inference
    model.eval()
    with torch.no_grad():
        if context_tensor is not None:
            outputs = model(samples, global_context=context_tensor)
        else:
            outputs = model(samples)
    
    # 4. Decode relationships từ outputs
    return _decode_relationships(outputs, objects, image_size)
```

**Relationship Decoding** (`_decode_relationships()` - dòng 1180-1287):

RelTR decode relationships theo **2 strategies**:

**Strategy 1: Geometric Matching (Ưu tiên)** - dòng 1211-1258

```python
# Sử dụng sub_boxes và obj_boxes từ RelTR outputs
if use_geometric:  # Nếu có sub_boxes và obj_boxes
    sub_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(sub_boxes) * scale
    obj_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(obj_boxes) * scale
    
    for idx in range(rel_scores.shape[0]):
        # 1. Lấy relationship prediction
        rel_vector = rel_scores[idx]
        rel_conf, rel_idx = rel_vector.max(dim=0)
        relation_name = RELATION_CLASSES[rel_idx]  # 'riding', 'on', ...
        
        # 2. Match predicted boxes với detected objects bằng IoU
        subj_iou_vals = box_ops.box_iou(sub_boxes_xyxy[idx], object_boxes)[0]
        obj_iou_vals = box_ops.box_iou(obj_boxes_xyxy[idx], object_boxes)[0]
        
        subj_iou, subj_idx = subj_iou_vals.max(dim=0)
        obj_iou, obj_idx = obj_iou_vals.max(dim=0)
        
        # 3. Filter: IoU phải >= 0.05
        if subj_iou >= 0.05 and obj_iou >= 0.05:
            confidence = rel_conf * max(subj_iou, 0.05) * max(obj_iou, 0.05)
            relationships.append({
                'subject': objects[subj_idx].get('class'),
                'relation': relation_name,
                'object': objects[obj_idx].get('class'),
                'confidence': confidence,
                'source': 'model'
            })
```

**Strategy 2: Pair-wise Fallback** - dòng 1262-1287

```python
# Nếu không có geometric boxes, tạo tất cả pairs
keep = rel_scores.max(-1).values > 0.4  # Filter confidence > 0.4
filtered = rel_scores[keep] if keep.any() else rel_scores

pair_cursor = 0
for i in range(total_objects):
    for j in range(i + 1, total_objects):  # Tất cả pairs
        vector = filtered[pair_cursor % num_queries]
        rel_idx = int(vector.argmax().item())
        confidence = float(vector.max().item())
        relation_name = RELATION_CLASSES[rel_idx]
        
        relationships.append({
            'subject': objects[i].get('class'),
            'relation': relation_name,
            'object': objects[j].get('class'),
            'confidence': confidence,
            'source': 'model_fallback'
        })
        pair_cursor += 1
```

**Ưu điểm**:
- ✅ Tự động predict relationships từ ảnh
- ✅ Không cần original_relationship
- ✅ Có thể tạo nhiều relationships (không chỉ 1)

**Nhược điểm**:
- ⚠️ Phụ thuộc vào chất lượng RelTR model
- ⚠️ Có thể predict sai nếu model chưa được train tốt
- ⚠️ Chậm hơn Method 1 (cần chạy inference)

#### Method 3: Fallback (Last Resort)

**Code**: dòng 1060-1071

**Cách hoạt động**:
```python
# Nếu cả 2 methods trên đều fail
if not relationships and original_relationship:
    fallback_relationship = {
        'subject': original_relationship.get('subject', 'unknown'),
        'relation': original_relationship.get('relation', 'unknown'),
        'object': original_relationship.get('object', 'unknown'),
        'confidence': 0.5,  # Low confidence
        'source': 'fallback'
    }
    relationships = [fallback_relationship]
```

**Lưu ý**:
- ⚠️ **WARNING**: Chất lượng thấp (confidence = 0.5)
- ⚠️ Chỉ dùng khi không thể match objects hoặc RelTR inference fail
- ⚠️ Relationship này có thể không chính xác với objects thực tế trong ảnh

### Ví dụ Flow hoàn chỉnh

```python
# Input: Synthetic image từ Stable Diffusion
original_relationship = {
    'subject': 'dog',
    'relation': 'riding',
    'object': 'surfboard'
}

# Step 1: Auto-Annotation
annotation_result = annotator.annotate_from_relationship(image_path, original_relationship)
# Output: objects = [
#   {'class': 'dog', 'bbox': [100, 150, 300, 400], 'confidence': 0.85},
#   {'class': 'surfboard', 'bbox': [200, 300, 450, 500], 'confidence': 0.78}
# ]

# Step 2: Relationship Generation
# Method 1: Build from Original
relationships = _build_relationship_from_original(objects, original_relationship)
# ✅ Success! Output: [{
#   'subject': 'dog',
#   'relation': 'riding',
#   'object': 'surfboard',
#   'confidence': 1.0,
#   'source': 'original'
# }]

# Nếu Method 1 fail (ví dụ: detect sai class 'puppy' thay vì 'dog'):
# Method 2: RelTR Inference
relationships = _run_reltr_inference(image_tensor, objects, ...)
# Output: [{
#   'subject': 'dog',  # hoặc 'puppy' nếu detect sai
#   'relation': 'riding',  # RelTR predict
#   'object': 'surfboard',
#   'confidence': 0.75,  # Confidence từ model
#   'source': 'model'
# }]

# Nếu cả 2 methods đều fail:
# Method 3: Fallback
relationships = [{
    'subject': 'dog',
    'relation': 'riding',
    'object': 'surfboard',
    'confidence': 0.5,
    'source': 'fallback'
}]
```

### So sánh 3 Methods

| Method | Confidence | Accuracy | Speed | Khi nào dùng |
|--------|-----------|----------|-------|--------------|
| **Build from Original** | 1.0 | Cao nhất | Nhanh nhất | Objects match với original |
| **RelTR Inference** | 0.4-0.9 | Trung bình-Cao | Chậm | Objects không match, cần predict |
| **Fallback** | 0.5 | Thấp | Instant | Tất cả methods khác fail |

### Best Practices

1. **Ưu tiên Method 1**: Đảm bảo auto-annotation chính xác để objects match với original
2. **Cải thiện RelTR**: Train RelTR tốt để Method 2 chính xác hơn
3. **Avoid Fallback**: Fallback chỉ nên dùng khi không còn lựa chọn
4. **Monitor source field**: Track `source` để biết relationship được tạo bằng method nào

### Backend Selection Logic

**File**: `RL/auto_annotator.py` - `_initialize_backend()`

```python
def _initialize_backend(self, config, checkpoint):
    # 1. Try GroundingDINO first
    if GROUNDINGDINO_AVAILABLE:
        try:
            # Auto-detect paths
            if config is None:
                candidates = [
                    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    "~/.cache/groundingdino/GroundingDINO_SwinT_OGC.py"
                ]
                # Tìm file đầu tiên tồn tại
            
            model = load_model(config, checkpoint, device)
            return "groundingdino"  # ✅ Success
        except Exception as e:
            print(f"GroundingDINO failed: {e}")
    
    # 2. Try OWL-ViT
    if OWLVIT_AVAILABLE:
        try:
            processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            return "owlvit"  # ✅ Success
        except Exception as e:
            print(f"OWL-ViT failed: {e}")
    
    # 3. Fallback to YOLO+CLIP
    if YOLO_CLIP_AVAILABLE:
        return "yolo_clip"  # ✅ Always available (uses existing pipeline)
    
    # 4. No backend available
    return "none"  # ❌ Will use pseudo-bbox only
```

### Cấu hình Thresholds

| Backend | Box Threshold | Text Threshold | Mô tả |
|---------|---------------|----------------|-------|
| **GroundingDINO** | 0.25 | 0.20 | Minimum confidence cho box detection và text matching |
| **OWL-ViT** | 0.25 | N/A | Minimum confidence cho object detection |
| **YOLO+CLIP** | N/A | N/A | Sử dụng confidence từ YOLO/CLIP (0.5-0.7) |
| **Pseudo** | N/A | N/A | Fixed confidence = 0.3 (rất thấp) |

**Có thể điều chỉnh**:
```python
annotator = AutoAnnotator(
    box_threshold=0.3,   # Tăng để strict hơn
    text_threshold=0.25  # Tăng để match text chính xác hơn
)
```

### Tối ưu hóa và Best Practices

1. **Singleton Pattern**: 
   - `get_annotator()` trả về instance duy nhất
   - Model chỉ load 1 lần, tái sử dụng cho tất cả images
   - Tiết kiệm memory và thời gian

2. **Lazy Loading**: 
   - Model chỉ được load khi cần thiết (lần đầu gọi `annotate()`)
   - Không load tất cả backends cùng lúc

3. **Auto-detect Paths**: 
   - Tự động tìm config và weights files
   - Hỗ trợ nhiều vị trí phổ biến

4. **Error Handling**: 
   - Graceful fallback nếu một backend fail
   - Log lỗi nhưng không crash toàn bộ pipeline

5. **Performance Tips**:
   - GroundingDINO: Chính xác nhất nhưng chậm nhất
   - OWL-ViT: Cân bằng tốt giữa speed và accuracy
   - YOLO+CLIP: Nhanh nhất nhưng ít chính xác
   - Pseudo: Instant nhưng chất lượng thấp

### RelTR Training Configuration
| Parameter | Giá trị | Mô tả |
|-----------|---------|-------|
| Training Epochs | 1 (default) | Số epochs train trên toàn bộ dataset mỗi episode |
| Learning Rate | 1e-5 | AdamW optimizer cho RelTR |
| Weight Decay | 1e-4 | L2 regularization |
| Batch Processing | Sequential | Train từng sample, accumulate gradients |

**Lưu ý**: Có thể tăng `reltr_training_epochs` (2-5) để model học tốt hơn trên dataset lớn.

---

## 🆕 Cập nhật và cải tiến

### Version mới nhất - Các tính năng đã thêm

#### 1. **Long-tail Loss Metric** ✅
- **Mô tả**: Metric mới để theo dõi hiệu suất trên các quan hệ hiếm (long-tail)
- **Cách tính**: Relationship loss được trọng số hóa bởi `tail_weights` cho các quan hệ hiếm
- **Lợi ích**: Đánh giá chính xác hơn về hiệu suất model trên long-tail distribution
- **Vị trí**: Được lưu trong `relationship_metrics` JSON với key `long_tail_loss`

```json
{
  "relationship_loss": 31.20,
  "long_tail_loss": 42.80,  // ← Metric mới
  "relationship_metrics": {
    "precision": 0.28,
    "recall": 0.47,
    "f1": 0.35
  }
}
```

##### Chi tiết Implementation: Các hàm tính Long-tail Loss

**File**: `RL/reinforcement_learning.py`

Hệ thống tính long-tail loss qua **3 bước chính**:

###### Bước 1: Tính Tail Weights (`_recompute_tail_weights()`)

**Function**: `_recompute_tail_weights()` - dòng 159-180

**Mục đích**: Tính trọng số cho từng loại quan hệ dựa trên tần suất xuất hiện trong dataset. Quan hệ càng hiếm → trọng số càng cao.

**Cách hoạt động**:
```python
def _recompute_tail_weights(self) -> None:
    """Tính trọng số cho các quan hệ hiếm (long-tail) dựa trên tần suất."""
    if not self.dataset_samples:
        self.tail_weights = {}
        return
    
    # 1. Đếm tần suất của từng quan hệ trong dataset
    freq: Dict[str, int] = {}
    for sample in self.dataset_samples:
        for rel in sample.get('relationships', []) or []:
            rel_name = self._normalize_label(rel.get('relation', ''))
            if not rel_name:
                continue
            freq[rel_name] = freq.get(rel_name, 0) + 1
    
    # Ví dụ: freq = {
    #   'on': 100,      # Quan hệ phổ biến (head)
    #   'has': 80,
    #   'riding': 5,    # Quan hệ hiếm (tail)
    #   'playing': 3
    # }
    
    if not freq:
        self.tail_weights = {}
        return
    
    # 2. Tính raw weights: 1/sqrt(freq) để ưu tiên lớp hiếm
    # Quan hệ hiếm (freq nhỏ) → weight lớn
    # Quan hệ phổ biến (freq lớn) → weight nhỏ
    raw_weights = {k: 1.0 / math.sqrt(v + 1e-3) for k, v in freq.items()}
    # Ví dụ: raw_weights = {
    #   'on': 1/sqrt(100) = 0.1,
    #   'has': 1/sqrt(80) = 0.112,
    #   'riding': 1/sqrt(5) = 0.447,   # ← Cao hơn
    #   'playing': 1/sqrt(3) = 0.577   # ← Cao nhất
    # }
    
    # 3. Normalize để tổng = 1.0
    total = sum(raw_weights.values()) or 1.0
    self.tail_weights = {k: v / total for k, v in raw_weights.items()}
    # Ví dụ: tail_weights = {
    #   'on': 0.08,      # Trọng số thấp
    #   'has': 0.09,
    #   'riding': 0.36,  # Trọng số cao (quan hệ hiếm)
    #   'playing': 0.47  # Trọng số cao nhất
    # }
```

**Công thức**:
```
raw_weight(relation) = 1 / sqrt(frequency + ε)
tail_weight(relation) = raw_weight(relation) / Σ(raw_weights)
```

**Ví dụ tính toán**:
```
Dataset có:
- "on": 100 lần → raw = 1/√100 = 0.1
- "has": 50 lần → raw = 1/√50 = 0.141
- "riding": 5 lần → raw = 1/√5 = 0.447
- "playing": 2 lần → raw = 1/√2 = 0.707

Tổng raw = 1.395
Normalize:
- "on": 0.1/1.395 = 0.072
- "has": 0.141/1.395 = 0.101
- "riding": 0.447/1.395 = 0.320  ← Cao hơn
- "playing": 0.707/1.395 = 0.507 ← Cao nhất
```

**Khi nào được gọi**:
- Sau khi `_ingest_synthetic_samples()` (dòng 1088)
- Sau khi `build_dataset_from_directory()` (dòng 1137)
- Mỗi khi dataset thay đổi

###### Bước 2: Tính Sample Tail Weight (`_get_sample_tail_weight()`)

**Function**: `_get_sample_tail_weight()` - dòng 2164-2195

**Mục đích**: Tính trọng số trung bình cho một training sample dựa trên các quan hệ trong sample đó.

**Cách hoạt động**:
```python
def _get_sample_tail_weight(self, target: Dict[str, Any]) -> float:
    """Tính tail weight cho một training sample."""
    if not self.tail_weights:
        return 0.0
    
    # Collect all relationship names từ dataset và tính average tail weight
    tail_weights_in_dataset = []
    for sample in self.dataset_samples:
        for rel in sample.get('relationships', []):
            rel_name = self._normalize_label(rel.get('relation', ''))
            if rel_name in self.tail_weights:
                tail_weights_in_dataset.append(self.tail_weights[rel_name])
    
    if not tail_weights_in_dataset:
        return 0.0
    
    # Return average tail weight
    # Đây là proxy measure: nếu dataset có nhiều quan hệ hiếm,
    # thì average weight sẽ cao
    return sum(tail_weights_in_dataset) / len(tail_weights_in_dataset)
```

**Ví dụ**:
```python
# Giả sử tail_weights = {
#   'on': 0.08,
#   'riding': 0.36,
#   'playing': 0.47
# }

# Dataset có:
# - Sample 1: relationships = [{'relation': 'on'}, {'relation': 'on'}]
# - Sample 2: relationships = [{'relation': 'riding'}]
# - Sample 3: relationships = [{'relation': 'playing'}]

# tail_weights_in_dataset = [0.08, 0.08, 0.36, 0.47]
# Average = (0.08 + 0.08 + 0.36 + 0.47) / 4 = 0.2475
```

**Lưu ý**: Hàm này sử dụng average weight của toàn bộ dataset làm proxy, vì không thể trực tiếp map từ RelTR target (chứa indices) sang relationship names.

###### Bước 3: Tính Long-tail Loss trong Training (`train_relationship_model()`)

**Function**: `train_relationship_model()` - dòng 2054-2162

**Mục đích**: Tính long-tail loss bằng cách trọng số hóa relationship loss theo tail weights.

**Cách hoạt động**:
```python
def train_relationship_model(self, synthetic_data, num_epochs: int = 1):
    """Train RelTR và tính long-tail loss."""
    # ... setup model, optimizer ...
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_tail_loss = 0.0  # ← Tích lũy long-tail loss
        tail_weighted_count = 0  # ← Đếm số samples có tail weight
        
        for i, (image_tensor, target, global_context) in enumerate(shuffled_samples):
            # 1. Forward pass và tính loss thông thường
            outputs = model(samples, global_context=context_tensor)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict[k] * weight_dict.get(k, 1.0) 
                      for k in loss_dict.keys() if k in weight_dict)
            
            loss.backward()
            sample_loss = float(loss.item())
            total_loss += sample_loss
            
            # 2. Tính long-tail loss: nhân loss với tail_weight
            tail_weight = self._get_sample_tail_weight(target)
            if tail_weight > 0:
                total_tail_loss += sample_loss * tail_weight  # ← Weighted loss
                tail_weighted_count += 1
        
        # 3. Tính average losses
        epoch_avg_loss = total_loss / len(shuffled_samples)
        epoch_tail_loss = total_tail_loss / max(tail_weighted_count, 1)
        # ↑ Chia cho số samples có tail weight, không phải tổng số samples
        
        # ... lưu losses ...
    
    # 4. Return average across all epochs
    average_loss = sum(all_epoch_losses) / len(all_epoch_losses)
    long_tail_loss = sum(all_epoch_tail_losses) / len(all_epoch_tail_losses)
    
    return average_loss, long_tail_loss
```

**Công thức**:
```
For each training sample i:
  sample_loss_i = RelTR_loss(sample_i)
  tail_weight_i = _get_sample_tail_weight(sample_i)
  
  if tail_weight_i > 0:
    tail_loss_i = sample_loss_i * tail_weight_i
    total_tail_loss += tail_loss_i
    tail_weighted_count += 1

long_tail_loss = total_tail_loss / tail_weighted_count
```

**Ví dụ tính toán**:
```python
# Giả sử có 3 samples:
# Sample 1: loss = 30.0, tail_weight = 0.08 (quan hệ "on" - phổ biến)
# Sample 2: loss = 35.0, tail_weight = 0.36 (quan hệ "riding" - hiếm)
# Sample 3: loss = 40.0, tail_weight = 0.47 (quan hệ "playing" - rất hiếm)

# Tính long-tail loss:
total_tail_loss = (30.0 * 0.08) + (35.0 * 0.36) + (40.0 * 0.47)
                 = 2.4 + 12.6 + 18.8
                 = 33.8

tail_weighted_count = 3
long_tail_loss = 33.8 / 3 = 11.27

# So sánh với relationship_loss thông thường:
relationship_loss = (30.0 + 35.0 + 40.0) / 3 = 35.0

# → long_tail_loss (11.27) < relationship_loss (35.0) 
# vì đã được normalize bởi tail_weights
```

**Lưu ý quan trọng**:
- Long-tail loss **không phải** là loss riêng biệt, mà là **weighted version** của relationship loss
- Chỉ tính cho các samples có `tail_weight > 0`
- Được chia cho số samples có tail weight, không phải tổng số samples
- Giá trị có thể nhỏ hơn relationship_loss vì đã được normalize

### Flow hoàn chỉnh

```
1. Dataset được build/ingest
   ↓
2. _recompute_tail_weights() được gọi
   ↓
3. tail_weights được tính: {relation: weight}
   ↓
4. train_relationship_model() được gọi
   ↓
5. For each training sample:
   a. Tính sample_loss (RelTR loss)
   b. Tính tail_weight = _get_sample_tail_weight(sample)
   c. Nếu tail_weight > 0:
      total_tail_loss += sample_loss * tail_weight
   ↓
6. long_tail_loss = total_tail_loss / tail_weighted_count
   ↓
7. Return (relationship_loss, long_tail_loss)
```

### Sử dụng Long-tail Loss

**Trong training metrics**:
```python
results = rl_system.train_episode(...)
# results['long_tail_loss'] = 42.80
```

**Trong JSON metrics**:
```json
{
  "relationship_loss": 31.20,
  "long_tail_loss": 42.80,
  "relationship_metrics": {
    "precision": 0.28,
    "recall": 0.47,
    "f1": 0.35
  }
}
```

**Ý nghĩa**:
- `long_tail_loss` cao → Model đang gặp khó khăn với quan hệ hiếm
- `long_tail_loss` giảm → Model đang học tốt hơn trên long-tail
- So sánh với `relationship_loss` để đánh giá sự chênh lệch giữa head và tail classes

#### 2. **Multiple Epochs Training** ✅
- **Mô tả**: Hỗ trợ training nhiều epochs trên toàn bộ dataset tích lũy
- **Cấu hình**: `self.reltr_training_epochs` (mặc định: 1, có thể tăng lên 3-5)
- **Tính năng**:
  - Shuffle samples giữa các epochs (trừ epoch đầu)
  - Tối ưu logging cho dataset lớn
  - Tính average loss across epochs
- **Cách sử dụng**:
  ```python
  # Trong code
  rl_agent.reltr_training_epochs = 3  # Train 3 epochs mỗi episode
  ```

#### 3. **Dataset Input cho RL Training** ✅
- **Mô tả**: Cho phép chọn/nhập dataset từ thư mục ảnh khi chạy RL training
- **Console App (`app_console.py`)**:
  ```
  📂 CHỌN DATASET ĐỂ TRAINING:
  1. Sử dụng dataset hiện tại (nếu đã có)
  2. Chọn thư mục chứa ảnh để build dataset
  3. Bỏ qua (sẽ dùng dataset từ relationships hiện tại)
  ```
- **GUI App (`app.py`)**:
  - Dialog chọn thư mục ảnh
  - Tự động build dataset từ thư mục được chọn
- **Lợi ích**: Linh hoạt hơn trong việc quản lý và sử dụng dataset

#### 4. **mR@K Metrics (Mean Recall@K)** ✅
- **Mô tả**: Metric đánh giá công bằng cho long-tail relationships
- **Cách tính**:
  1. Tính R@K cho từng loại quan hệ riêng lẻ
  2. Lấy trung bình của tất cả các R@K đó
- **Giá trị K**: mR@10, mR@20, mR@50, mR@100
- **Lợi ích**:
  - Đánh giá công bằng hơn so với R@K thông thường
  - Không bị ảnh hưởng bởi các quan hệ phổ biến (head classes)
  - Tiêu chuẩn trong Scene Graph Generation research
- **Vị trí**: Được lưu trong `relationship_metrics` JSON

```json
{
  "relationship_metrics": {
    "precision": 0.28,
    "recall": 0.47,
    "f1": 0.35,
    "mr@10": 0.1234,   // ← Metrics mới
    "mr@20": 0.2345,
    "mr@50": 0.3456,
    "mr@100": 0.4567
  }
}
```

### So sánh mR@K vs R@K

| Metric | Cách tính | Ưu điểm | Nhược điểm |
|--------|-----------|---------|------------|
| **R@K** | Recall tổng thể trên tất cả predictions | Đơn giản, dễ hiểu | Bị ảnh hưởng bởi head classes |
| **mR@K** | Mean của R@K cho từng relation type | Công bằng cho long-tail | Phức tạp hơn |

**Ví dụ**:
- Quan hệ "on": 100 samples, R@10 = 0.8
- Quan hệ "riding": 5 samples, R@10 = 0.2
- **R@10** = (80 + 1) / 105 = 0.77 (bị ảnh hưởng bởi "on")
- **mR@10** = (0.8 + 0.2) / 2 = 0.5 (công bằng hơn)

### Cấu trúc Metrics JSON mới

```json
{
  "experiment_id": "exp_001",
  "epoch": 1,
  "detection_loss": 21.09,
  "relationship_loss": 31.20,
  "long_tail_loss": 42.80,  // ← Mới
  "reward": 0.43,
  "detection_metrics": {
    "precision": 0.28,
    "recall": 0.47,
    "f1": 0.35
  },
  "relationship_metrics": {
    "precision": 0.28,
    "recall": 0.47,
    "f1": 0.35,
    "mr@10": 0.1234,   // ← Mới
    "mr@20": 0.2345,   // ← Mới
    "mr@50": 0.3456,   // ← Mới
    "mr@100": 0.4567   // ← Mới
  }
}
```

### Hướng dẫn sử dụng các tính năng mới

#### Sử dụng Multiple Epochs Training
```python
# Trong RL/reinforcement_learning.py hoặc sau khi khởi tạo
rl_agent = RelationshipReinforcementLearning(...)
rl_agent.reltr_training_epochs = 3  # Train 3 epochs mỗi episode
```

#### Chọn Dataset khi Training
```bash
# Console App
python app_console.py
# Chọn 4. Chạy RL Training
# Chọn 2 để chọn thư mục ảnh
# Nhập: D:/path/to/images

# GUI App
python app.py
# Click "RL Training"
# Chọn Yes → Browse thư mục ảnh
```

#### Xem mR@K Metrics
```python
# Metrics được tự động lưu trong JSON
# experiments/exp_XXX/metrics/training_metrics_epoch_XX.json

import json
with open('experiments/exp_001/metrics/training_metrics_epoch_01.json') as f:
    data = json.load(f)
    print(f"mR@10: {data['relationship_metrics']['mr@10']:.4f}")
    print(f"mR@20: {data['relationship_metrics']['mr@20']:.4f}")
    print(f"mR@50: {data['relationship_metrics']['mr@50']:.4f}")
    print(f"mR@100: {data['relationship_metrics']['mr@100']:.4f}")
```

#### 5. **Hệ thống Phân loại An toàn 3 Tầng** ✅
- **Mô tả**: Hệ thống phát hiện và cảnh báo các hành động nguy hiểm trong video
- **Tích hợp**: Tự động tích hợp với `VideoRelationPipeline`
- **3 Tầng**:
  1. White/Black/Gray List - Phân loại nhanh
  2. LLM Semantic Reasoning - Đánh giá ngữ nghĩa (OpenAI/Gemini)
  3. Local Rules Database - Học từ người dùng
- **Tính năng**:
  - Cache LLM responses để tiết kiệm chi phí
  - Hiển thị cảnh báo trực quan trên video (banner đỏ/cam)
  - Thống kê an toàn trong JSON output
  - Hỗ trợ thêm quy tắc cục bộ từ phản hồi người dùng
- **Cấu hình**: File `RL/safety_config.json`
- **Tài liệu**: Xem [`RL/README_SAFETY_SYSTEM.md`](RL/README_SAFETY_SYSTEM.md)

**Ví dụ sử dụng**:
```python
from video_relation_pipeline import VideoRelationPipeline

pipeline = VideoRelationPipeline(safety_classifier_enabled=True)
result = pipeline.process_video("video.mp4")

# Xem thống kê an toàn
import json
with open(result["summary"], "r") as f:
    stats = json.load(f)
    print(f"Cảnh báo: {stats['safety_alerts']['total_alerts']}")
```

---

## 📝 License

[MIT License](LICENSE)

---

## 👥 Tác giả

- Dự án đồ án tốt nghiệp
- Hệ thống Scene Graph Generation với Reinforcement Learning

---

## 🙏 Acknowledgments

- [YOLO](https://github.com/ultralytics/ultralytics) - Object Detection
- [CLIP](https://github.com/openai/CLIP) - Vision-Language Model
- [RelTR](https://github.com/yrcong/RelTR) - Scene Graph Generation
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Image Generation
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Open-Vocabulary Detection
- [OpenAI](https://openai.com/) - GPT Models cho Safety Analysis
- [Google Gemini](https://gemini.google.com/) - Gemini Models cho Safety Analysis

