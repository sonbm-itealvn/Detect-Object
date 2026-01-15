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
│  │  • YOLO v11 classification                                                   │           │
│  │  • ROI features extraction                                                   │           │
│  │  • Global context vector                                                     │           │
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
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
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
│  │  1. GroundingDINO (SOTA, chính xác nhất)                                     │           │
│  │     • Open-vocabulary detection                                              │           │
│  │     • Input: Image + relationship triplet                                    │           │
│  │     • Output: Bounding boxes + classes                                       │           │
│  │                                                                              │           │
│  │  2. OWL-ViT (Lightweight, HuggingFace)                                       │           │
│  │     • Fallback nếu GroundingDINO không có                                    │           │
│  │                                                                              │           │
│  │  3. YOLO + CLIP (Fallback)                                                   │           │
│  │     • Limited vocabulary                                                     │           │
│  │                                                                              │           │
│  │  4. Pseudo-bbox (Heuristic, low quality)                                     │           │
│  │     • Tạo bbox từ relationship nếu không detect được                         │           │
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
│  │  [4.1] RelTR Fine-tuning                                                     │           │
│  │  • Input: Dataset samples (original + synthetic)                             │           │
│  │  • Prepare RelTR targets: entities + relationships                           │           │
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
│  │  • mR@K metrics: mR@10, mR@20, mR@50, mR@100 (công bằng cho long-tail)      │           │
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

1. **VRD Pipeline**: Input image → YOLO detection → RelTR relationship prediction → Scene Graph
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
```

### Bước 9: Download model weights

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

### Chạy video demo
```bash
# Trong app.py hoặc app_console.py, chọn video demo option
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
│   └── data_augmentation.py        # Data augmentation
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

### Auto-Annotation Priority
1. **GroundingDINO** - SOTA open-vocabulary detector
2. **OWL-ViT** - Lightweight, HuggingFace
3. **YOLO+CLIP** - Fallback, limited vocabulary
4. **Pseudo-bbox** - Heuristic, low quality

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

