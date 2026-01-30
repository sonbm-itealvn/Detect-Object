<![CDATA[# 🔍 Scene Graph Generation với Reinforcement Learning

> Hệ thống phát hiện đối tượng và dự đoán quan hệ (Scene Graph Generation) kết hợp học tăng cường (DQN) để giải quyết vấn đề **long-tail distribution**. Hỗ trợ xử lý ảnh, video với phân tích an toàn, và tự động sinh dữ liệu synthetic.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Mục lục

| Phần | Mô tả |
|------|-------|
| [1. Giới thiệu](#-1-giới-thiệu) | Vấn đề, giải pháp, ứng dụng |
| [2. Kiến trúc](#-2-kiến-trúc-hệ-thống) | Tổng quan pipeline, sơ đồ luồng |
| [3. Tính năng](#-3-tính-năng-chính) | Object Detection, RelTR, RL, GenAI, Safety |
| [4. Cài đặt](#-4-cài-đặt) | Yêu cầu, các bước cài đặt |
| [5. Sử dụng](#-5-sử-dụng) | Console, GUI, API |
| [6. Cấu trúc dự án](#-6-cấu-trúc-dự-án) | Thư mục và files |
| [7. Cấu hình](#-7-cấu-hình) | DQN, Reward, Thresholds |
| [8. Kết quả & Metrics](#-8-kết-quả-và-metrics) | Detection, Relationship, RL, Safety |
| [9. Safety System](#-9-hệ-thống-an-toàn-3-tầng) | White/Black List, LLM, Local Rules |
| [10. Chi tiết kỹ thuật](#-10-chi-tiết-kỹ-thuật) | State Vector, Q-Network, Auto-Annotation |
| [11. Cập nhật](#-11-cập-nhật-và-cải-tiến) | Long-tail Loss, mR@K, Multiple Epochs |

---

## 🎯 1. Giới thiệu

### 1.1 Vấn đề

| Vấn đề | Mô tả |
|--------|-------|
| **Long-tail distribution** | Một số quan hệ xuất hiện rất thường xuyên (head: "on", "has") trong khi đa số rất hiếm (tail: "riding", "playing with") |
| **Model bias** | Model học tốt head classes nhưng kém với tail classes |
| **Thiếu dữ liệu** | Thiếu training data cho các quan hệ hiếm |

### 1.2 Giải pháp

Sử dụng **Reinforcement Learning (DQN)** để:

1. **Tự động sinh dữ liệu synthetic** - Stable Diffusion tạo ảnh cho quan hệ hiếm
2. **Auto-annotation** - GroundingDINO/OWL-ViT tạo bounding boxes tự động
3. **Reward optimization** - Trọng số reward ưu tiên tail classes (`1/sqrt(frequency)`)
4. **Adaptive training** - DQN agent quyết định số lượng variations cần sinh
5. **Safety analysis** - Hệ thống 3 tầng phân tích an toàn cho video

### 1.3 Ứng dụng

- **Scene Graph Generation**: Phát hiện objects và relationships trong ảnh/video
- **Video Safety Monitoring**: Phát hiện và cảnh báo hành động nguy hiểm
- **Data Augmentation**: Tự động sinh dữ liệu training cho quan hệ hiếm
- **Research**: Nghiên cứu về long-tail distribution trong SGG

---

## 🏗️ 2. Kiến trúc hệ thống

### 2.1 Tổng quan

Hệ thống gồm 2 luồng chính:

| Luồng | Input | Output |
|-------|-------|--------|
| **Video Processing** | Video file | Video annotated + JSON thống kê an toàn |
| **RL Training** | Images + Relationships | Trained model + Metrics |

### 2.2 Pipeline Diagram

![VRD Pipeline](docs/vrd_pipeline_diagram.png)

<details>
<summary><b>📊 Xem sơ đồ luồng chi tiết (Text)</b></summary>

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: VRD PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│  [Input Image]                                                  │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  YOLO v11 Detection                                      │   │
│  │  • Bounding boxes • ROI features • Global context        │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ReClip (Region Clip)                                    │   │
│  │  • Gán nhãn cho các Boundingbox (classification)         │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  RelTR (Relationship Transformer)                        │   │
│  │  • Dự đoán triplets: (Subject, Predicate, Object)        │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Scene Graph Output                                      │   │
│  │  • Objects: [{bbox, class, confidence, features}]        │   │
│  │  • Relationships: [{subject, relation, object}]          │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ├─── [Video Mode] ──→ Safety Classifier (3-Tier)         │
│       │                      └─→ Video Annotation & Alert       │
│       │                                                         │
│       └─── [Image Mode] ──→ Phase 2 (RL Training)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                PHASE 2: RL TRAINING LOOP (DQN)                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  State Vector (5D)                                       │   │
│  │  • detection_f1 • relationship_f1 • reward               │   │
│  │  • dataset_norm • epsilon                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  DQN Agent Decision                                      │   │
│  │  • Q-Network: 5 → 64 → 64 → 10                           │   │
│  │  • Action Space: Số biến thể prompt [1-10]               │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Action: Generate Synthetic Data                         │   │
│  │  • Số lượng variations cho mỗi relationship              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 3: GENERATIVE AI (Stable Diffusion)          │
├─────────────────────────────────────────────────────────────────┤
│  [Prompt Generation] → [Stable Diffusion] → [Quality Filter]   │
│       ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Auto-Annotation (Priority Order)                        │   │
│  │  1. GroundingDINO (SOTA)                                 │   │
│  │  2. OWL-ViT (Lightweight)                                │   │
│  │  3. YOLO + CLIP (Fallback)                               │   │
│  │  4. Pseudo-bbox (Last resort)                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ↓                                                         │
│  [Dataset Ingestion] → Add synthetic samples to training set    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 4: MODEL TRAINING & EVALUATION            │
├─────────────────────────────────────────────────────────────────┤
│  [VRD Fine-tuning] → [Evaluation] → [Reward Calculation]       │
│       ↓                                                         │
│  [DQN Update] → Experience Replay → Target Network Update       │
│       ↓                                                         │
│  [Loop back to Phase 2]                                         │
└─────────────────────────────────────────────────────────────────┘
```

</details>

<details>
<summary><b>🎬 Luồng xử lý Video (VideoRelationPipeline)</b></summary>

1. **VRD Pipeline**: Input video frame → YOLO → ReClip → RelTR → Scene Graph
2. **Safety Analysis** (nếu enabled):
   - **Tầng 1**: Kiểm tra White/Black List → Nếu match → Return SAFE/DANGEROUS
   - **Tầng 3**: Kiểm tra Local Rules (ưu tiên cao nhất) → Nếu có → Return từ rule
   - **Tầng 2**: Nếu không match → Hỏi LLM (OpenAI/Gemini) → Cache response
   - **Default**: Nếu LLM không hoạt động → Return SUSPICIOUS
3. **Video Annotation**: Vẽ relationships và cảnh báo an toàn lên frame
4. **Output**: Video đã annotate + JSON thống kê an toàn

</details>

<details>
<summary><b>🤖 Luồng RL Training</b></summary>

1. **VRD Pipeline**: Input image → YOLO → ReClip → RelTR → Scene Graph
2. **RL State**: Xây dựng state vector 5D từ metrics hiện tại
3. **DQN Decision**: Agent quyết định số lượng variations cần sinh
4. **GenAI Generation**: Stable Diffusion sinh ảnh với prompt variations
5. **Quality Filter**: Lọc ảnh chất lượng thấp
6. **Auto-Annotation**: GroundingDINO/OWL-ViT tạo bounding boxes
7. **Dataset Ingestion**: Thêm synthetic samples vào training dataset
8. **Model Training**: Fine-tune RelTR (hỗ trợ multiple epochs)
9. **Evaluation**: Tính metrics (F1, mR@K, Long-tail Loss)
10. **Reward Calculation**: Tính reward với long-tail boost
11. **DQN Update**: Cập nhật Q-network từ experience replay
12. **Loop**: Lặp lại cho đến khi đạt convergence

</details>

---

## ✨ 3. Tính năng chính

### 3.1 Object Detection (YOLO v11 + CLIP)

| Tính năng | Mô tả |
|-----------|-------|
| **YOLO v11** | Phát hiện bounding boxes với độ chính xác cao |
| **CLIP Re-classification** | Open-vocabulary classification |
| **ROI Features** | Trích xuất feature vectors từ bounding boxes |
| **Global Context** | Context vector cho toàn bộ ảnh |
| **Output** | `converted_bboxes.json` |

### 3.2 Relationship Prediction (RelTR)

| Tính năng | Mô tả |
|-----------|-------|
| **Architecture** | Transformer-based (dựa trên DETR) |
| **Relationship Types** | 51 loại: "on", "holding", "riding", "wearing", etc. |
| **Triplet Prediction** | (Subject, Predicate, Object) với confidence |
| **Output** | `relationships.json` |

### 3.3 Reinforcement Learning (DQN)

<details>
<summary><b>Xem chi tiết DQN</b></summary>

**State Vector (5D)**:
| Dimension | Mô tả |
|-----------|-------|
| `detection_f1` | F1 score từ detection model (normalized [0,1]) |
| `relationship_f1` | F1 score từ relationship model (normalized [0,1]) |
| `reward` | Reward hiện tại (tanh normalized) |
| `dataset_norm` | Kích thước dataset / 50 |
| `epsilon` | Exploration rate hiện tại |

**Action Space**: Số biến thể prompt sinh ảnh [1-10]

**Q-Network**: `Input(5) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(10)`

**Reward Function**:
| Component | Weight | Mô tả |
|-----------|--------|-------|
| Detection F1 | 0.25 | F1 score của detection |
| Relationship F1 | 0.45 | F1 score của relationship (cao nhất) |
| Diversity | 0.15 | Đa dạng relation/object types |
| Consistency | 0.10 | Độ ổn định predictions |
| Improvement | 0.05 | Xu hướng cải thiện |

**Long-tail Boost**: `tail_weight = 1/sqrt(frequency + 1e-3)`

</details>

### 3.4 Synthetic Data Generation (Stable Diffusion)

<details>
<summary><b>Xem chi tiết</b></summary>

**Stable Diffusion v1.5**: Text-to-image generation

**Prompt Engineering**: Template-based với variations (context, quality, lighting)

**Quality Filter**:
| Check | Threshold |
|-------|-----------|
| Min Size | 512×512 |
| Max Aspect | 2.2 |
| Blur Variance | ≥ 60 |
| Exposure Mean | [20, 235] |
| CLIP Similarity | ≥ 0.23 |

**Auto-Annotation Priority**:
1. GroundingDINO (SOTA)
2. OWL-ViT (Lightweight)
3. YOLO + CLIP (Fallback)
4. Pseudo-bbox (Last resort)

</details>

### 3.5 Video Processing với Safety Analysis

| Tính năng | Mô tả |
|-----------|-------|
| **Frame-by-frame** | Xử lý với stride (mặc định: 2) |
| **Object Tracking** | Track IDs qua các frames |
| **Safe Zone** | Phát hiện xâm nhập vùng an toàn 2m |
| **Annotation** | Bounding boxes, arrows, cảnh báo |
| **Output** | Video annotated + JSON statistics |

### 3.6 Safety System (3 Tầng)

| Tầng | Mô tả | Ưu tiên |
|------|-------|---------|
| **Tầng 3** | Local Rules Database (Human-in-the-loop) | Cao nhất |
| **Tầng 1** | White/Black/Gray List | Trung bình |
| **Tầng 2** | LLM Semantic Reasoning (OpenAI/Gemini) | Fallback |

---

## 🔧 4. Cài đặt

### 4.1 Yêu cầu hệ thống

- Python 3.10+
- CUDA 11.8+ (khuyến nghị GPU ≥8GB VRAM)
- 16GB RAM

### 4.2 Các bước cài đặt

<details>
<summary><b>Bước 1-4: Cơ bản</b></summary>

```bash
# 1. Clone repository
git clone <repository-url>
cd yolov11

# 2. Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Cài đặt PyTorch với CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

</details>

<details>
<summary><b>Bước 5-6: Stable Diffusion & CLIP</b></summary>

```bash
# 5. Cài đặt Stable Diffusion
pip install diffusers accelerate transformers

# 6. Cài đặt CLIP
pip install git+https://github.com/openai/CLIP.git
```

</details>

<details>
<summary><b>Bước 7: Auto-Annotation (chọn 1)</b></summary>

**Option A: OWL-ViT (Dễ cài, khuyên dùng)**
```bash
pip install transformers
# Model tự động download khi chạy lần đầu (~1.5GB)
```

**Option B: GroundingDINO (SOTA, chính xác nhất)**
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e . && cd ..

# Download weights
mkdir -p GroundingDINO/weights
wget -P GroundingDINO/weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

</details>

<details>
<summary><b>Bước 8-10: Tùy chọn</b></summary>

```bash
# 8. Dependencies bổ sung
pip install imagehash sentence-transformers python-dotenv

# 9. Cấu hình API Key cho Safety System
# Tạo file .env:
# OPENAI_API_KEY=your_key_here
# GEMINI_API_KEY=your_key_here

# 10. Download model weights
# - YOLO: Tự động download khi chạy
# - RelTR: Đặt checkpoint.pth vào thư mục gốc
```

</details>

---

## 🚀 5. Sử dụng

### 5.1 Console Application

```bash
python app_console.py
```

**Menu chính**:
1. Chọn ảnh để xử lý
2. Chạy pipeline phát hiện vật thể và mối quan hệ
3. Tải lại dữ liệu JSON
4. **Chạy RL Training** ← Học tăng cường
5. Tạo dữ liệu synthetic
6. Đánh giá kết quả training
7. Quản lý Experiments
8. Tiếp tục RL Training từ experiment trước
9. Thoát

### 5.2 GUI Application

```bash
python app.py
```

**Tính năng**: Select Image, Detect Objects, Video Demo, RL Training, Synthetic Data, Evaluate

### 5.3 Detection đơn lẻ

```bash
python detect_objects.py <path_to_image>
# Output: result.json, converted_bboxes.json
```

**Hỗ trợ Dual Models (COCO + Fire) 🔥:**
- Hệ thống tự động load và chạy 2 models song song nếu có
- COCO model: `fine-tune.pt` (80 classes)
- Fire model: `fire-model.pt`, `fire_detection.pt`, hoặc `fire.pt`
- Tự động merge detections với NMS
- Test: `python test_dual_models.py <image>`
- Xem chi tiết: [`DUAL_MODEL_SETUP.md`](DUAL_MODEL_SETUP.md)

### 5.4 Relationship prediction

```bash
python boundingbox_objects.py --yolo_json converted_bboxes.json --img_path <image> --resume checkpoint.pth --device cpu
# Output: relationships.json, output_<image_id>.jpg
```

### 5.5 Video với Safety Analysis

```python
from video_relation_pipeline import VideoRelationPipeline

pipeline = VideoRelationPipeline(safety_classifier_enabled=True)
result = pipeline.process_video("video.mp4", output_dir="video_outputs", frame_stride=2)

# Output:
# - video_outputs/video_relations.avi
# - video_outputs/video_summary.json
```

---

## 📁 6. Cấu trúc dự án

<details>
<summary><b>Xem cấu trúc thư mục đầy đủ</b></summary>

```
yolov11/
├── app.py                      # GUI Application (Tkinter)
├── app_console.py              # Console Application
├── detect_objects.py           # YOLO + CLIP detection
├── boundingbox_objects.py      # RelTR relationship inference
├── convert_yolo_to_reltr.py    # Data format conversion
├── video_relation_pipeline.py  # Video processing + Safety
│
├── RL/                         # Reinforcement Learning module
│   ├── reinforcement_learning.py   # DQN Agent
│   ├── ai_images_generator.py      # Stable Diffusion + Quality Filter
│   ├── auto_annotator.py           # GroundingDINO/OWL-ViT
│   ├── rl_enhancement.py           # RL integration
│   ├── experiment_manager.py       # Experiment tracking
│   ├── experiment_viewer.py        # View/compare experiments
│   ├── experience_manager.py       # Replay buffer
│   ├── model_manager.py            # Model checkpointing
│   ├── training_evaluator.py       # Evaluation metrics
│   ├── safety_classifier.py        # ⭐ Safety 3-tier
│   ├── llm_safety_analyzer.py      # ⭐ LLM integration
│   ├── local_rules_db.py           # ⭐ Local rules
│   ├── safety_config.json          # ⭐ Configuration
│   └── README_SAFETY_SYSTEM.md     # ⭐ Documentation
│
├── models/                     # Model definitions
│   ├── reltr.py, transformer.py, backbone.py, ...
│
├── util/, utils/               # Utilities
├── tools/                      # Visualization tools
├── GroundingDINO/              # GroundingDINO submodule
├── demo/, data/                # Demo images, Dataset configs
├── experiments/                # Experiment outputs
├── video_outputs/              # Video processing outputs
│
├── fine-tune.pt               # YOLO weights
├── checkpoint.pth             # RelTR checkpoint
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

</details>

---

## ⚙️ 7. Cấu hình

### 7.1 DQN Hyperparameters

| Parameter | Giá trị | Mô tả |
|-----------|---------|-------|
| Learning Rate | 0.001 | AdamW optimizer |
| Gamma | 0.95 | Discount factor |
| Epsilon | 0.9 → 0.01 | Exploration rate (decay=0.995) |
| Batch Size | 32 | Experience replay |
| Buffer Size | 10,000 | Replay buffer capacity |
| Target Update | 20 steps | Target network update |

### 7.2 Quality Filter Thresholds

| Check | Threshold | Mô tả |
|-------|-----------|-------|
| Min Size | 512×512 | Kích thước tối thiểu |
| Max Aspect | 2.2 | Tỉ lệ tối đa |
| Blur Variance | ≥ 60 | Độ sắc nét (Laplacian) |
| Exposure Mean | [20, 235] | Độ sáng trung bình |
| CLIP Similarity | ≥ 0.23 | Cosine similarity với prompt |

### 7.3 Auto-Annotation Thresholds

| Backend | Box Threshold | Text Threshold |
|---------|---------------|----------------|
| GroundingDINO | 0.25 | 0.20 |
| OWL-ViT | 0.25 | N/A |
| YOLO+CLIP | N/A | N/A (0.5-0.7) |
| Pseudo | N/A | Fixed 0.3 |

---

## 📊 8. Kết quả và Metrics

### 8.1 Metrics theo dõi

<details>
<summary><b>Detection Metrics</b></summary>

- **Precision**: Tỷ lệ objects được detect đúng
- **Recall**: Tỷ lệ objects thực tế được detect
- **F1 Score**: Harmonic mean của Precision và Recall
- **Bbox Loss**: Loss cho bounding box prediction
- **GIoU Loss**: Generalized IoU loss

</details>

<details>
<summary><b>Relationship Metrics</b></summary>

- **Precision, Recall, F1**: Tương tự detection
- **Relationship Loss**: Loss cho relationship prediction
- **Long-tail Loss**: Weighted loss cho quan hệ hiếm
- **mR@K**: Mean Recall@K (mR@10, mR@20, mR@50, mR@100)

</details>

<details>
<summary><b>RL Metrics</b></summary>

- **Reward**: Tổng reward từ DQN
- **Q-values**: Q-values cho mỗi action
- **Epsilon**: Exploration rate
- **Experience Buffer Size**: Số experiences trong buffer

</details>

<details>
<summary><b>Safety Metrics (Video)</b></summary>

- **Total Alerts**: Tổng số cảnh báo
- **Dangerous Count**: Số hành động nguy hiểm
- **Suspicious Count**: Số hành động nghi ngờ
- **Intrusion Count**: Số lần xâm nhập safe zone

</details>

### 8.2 Experiment Outputs

```
experiments/exp_001/
├── ai_images/          # Synthetic images
├── dataset/            # Training samples (JSON)
├── logs/               # Training logs
├── metrics/            # JSON metrics files
├── models/             # Saved checkpoints
├── plots/              # Visualization
└── metadata.json       # Config & hyperparameters
```

### 8.3 Metrics JSON Format

```json
{
  "experiment_id": "exp_001",
  "epoch": 1,
  "detection_loss": 21.09,
  "relationship_loss": 31.20,
  "long_tail_loss": 42.80,
  "reward": 0.43,
  "detection_metrics": {"precision": 0.28, "recall": 0.47, "f1": 0.35},
  "relationship_metrics": {
    "precision": 0.28, "recall": 0.47, "f1": 0.35,
    "mr@10": 0.12, "mr@20": 0.23, "mr@50": 0.34, "mr@100": 0.45
  }
}
```

---

## 🛡️ 9. Hệ thống An toàn 3 Tầng

Hệ thống phân loại an toàn được tích hợp vào `VideoRelationPipeline` để phát hiện và cảnh báo các hành động nguy hiểm.

### 9.1 Kiến trúc 3 Tầng

```
Relationship từ RelTR: {subject, relation, object}
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Tầng 3: Local Rules Database (Ưu tiên cao nhất)       │
│  • Kiểm tra local_safety_rules.json                    │
│  • Nếu có rule → Return level từ rule                  │
└─────────────────────────────────────────────────────────┘
         │ Không có rule?
         ▼
┌─────────────────────────────────────────────────────────┐
│  Tầng 1: White/Black/Gray List                         │
│  • Black List → DANGEROUS                              │
│  • White List → SAFE                                   │
│  • Không match → Gray List → Tầng 2                    │
└─────────────────────────────────────────────────────────┘
         │ Không match?
         ▼
┌─────────────────────────────────────────────────────────┐
│  Tầng 2: LLM Semantic Reasoning                        │
│  • Kiểm tra cache → Nếu có → Return từ cache           │
│  • Gọi LLM API (OpenAI/Gemini)                         │
│  • Parse response: SAFE/SUSPICIOUS/DANGEROUS           │
│  • Lưu vào cache                                       │
└─────────────────────────────────────────────────────────┘
         │
         ▼
Return: (SafetyLevel, confidence, explanation)
```

### 9.2 Sử dụng

```python
from video_relation_pipeline import VideoRelationPipeline

pipeline = VideoRelationPipeline(safety_classifier_enabled=True)
result = pipeline.process_video("video.mp4")
```

### 9.3 Thêm quy tắc cục bộ

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

### 9.4 Cấu hình

File `RL/safety_config.json`:
- `white_list`: Danh sách hành động an toàn
- `black_list`: Danh sách hành động nguy hiểm
- `llm_enabled`: Bật/tắt LLM
- `llm_provider`: "openai" hoặc "gemini"
- `llm_model`: "gpt-3.5-turbo", "gpt-4o", "gemini-pro"

> 📖 Xem chi tiết: [`RL/README_SAFETY_SYSTEM.md`](RL/README_SAFETY_SYSTEM.md)

---

## 🔬 10. Chi tiết kỹ thuật

### 10.1 State Vector (5D)

```python
state = [
    detection_f1,      # F1 score detection (normalized)
    relationship_f1,   # F1 score relationship (normalized)
    reward,            # Current reward (tanh normalized)
    dataset_norm,      # Dataset size / 50 (normalized)
    epsilon            # Current exploration rate
]
```

### 10.2 Q-Network Architecture

```
Input(5) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(10)
```

### 10.3 Auto-Annotation Chi tiết

<details>
<summary><b>1. GroundingDINO (SOTA)</b></summary>

**Function**: `_annotate_groundingdino()` trong `RL/auto_annotator.py`

```python
# 1. Load image và convert sang tensor
image_source, image_tensor = load_image(image_path)

# 2. Tạo text prompt: "dog . surfboard"
text_prompt = f"{subject} . {object}"

# 3. Predict
boxes, logits, phrases = predict(
    model=groundingdino_model,
    image=image_tensor,
    caption=text_prompt,
    box_threshold=0.25,
    text_threshold=0.20
)

# 4. Convert [cx, cy, w, h] → [x1, y1, x2, y2]
```

**Ưu điểm**: Open-vocabulary, SOTA accuracy, Multi-object support

**Cài đặt**:
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e .
wget -P weights https://...groundingdino_swint_ogc.pth
```

</details>

<details>
<summary><b>2. OWL-ViT (Lightweight)</b></summary>

**Function**: `_annotate_owlvit()` trong `RL/auto_annotator.py`

```python
# 1. Load model từ HuggingFace (auto-download)
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# 2. Prepare inputs
texts = [["a photo of a dog", "a photo of a surfboard"]]
inputs = processor(text=texts, images=image, return_tensors="pt")

# 3. Predict & post-process
outputs = model(**inputs)
results = processor.post_process_object_detection(outputs, threshold=0.25)
```

**Ưu điểm**: Dễ cài, auto-download, lightweight

</details>

<details>
<summary><b>3. YOLO + CLIP (Fallback)</b></summary>

**Function**: `_annotate_yolo_clip()` trong `RL/auto_annotator.py`

```python
# 1. YOLO detect
detected_objects, yolo_labels, ... = detection_pipeline.detect_objects(image_path)

# 2. CLIP classify
classified_results = detection_pipeline.classify_with_clip(detected_objects, yolo_labels)

# 3. Fuzzy match với text prompts
matched = any(prompt.lower() in label.lower() for prompt in text_prompts)
confidence = 0.7 if matched else 0.5
```

**Nhược điểm**: Limited vocabulary (chỉ classes YOLO biết)

</details>

<details>
<summary><b>4. Pseudo-bbox (Last Resort)</b></summary>

**Function**: `_create_pseudo_annotations()` trong `RL/auto_annotator.py`

Heuristics dựa trên relation type:
- `on/above/riding`: Subject trên, Object dưới
- `under/below`: Subject dưới, Object trên
- `holding/carrying`: Subject lớn, Object nhỏ gần subject
- Default: Subject trái, Object phải

**⚠️ WARNING**: Confidence = 0.3, chất lượng thấp

</details>

### 10.4 Relationship Generation

<details>
<summary><b>3 Methods theo thứ tự ưu tiên</b></summary>

**Method 1: Build from Original** (Ưu tiên cao nhất)
- Match objects với original_relationship
- Confidence = 1.0
- 3-level fuzzy matching: Exact → Synonym → Partial

**Method 2: RelTR Inference** (Fallback 1)
- Sử dụng RelTR model để predict
- Geometric matching hoặc Pair-wise fallback

**Method 3: Fallback** (Last Resort)
- Tạo trực tiếp từ original_relationship
- Confidence = 0.5

| Method | Confidence | Accuracy | Speed |
|--------|-----------|----------|-------|
| Build from Original | 1.0 | Cao nhất | Nhanh nhất |
| RelTR Inference | 0.4-0.9 | Trung bình-Cao | Chậm |
| Fallback | 0.5 | Thấp | Instant |

</details>

---

## 🆕 11. Cập nhật và cải tiến

### 11.1 Long-tail Loss Metric ✅

Metric mới để theo dõi hiệu suất trên các quan hệ hiếm.

<details>
<summary><b>Chi tiết Implementation</b></summary>

**Bước 1: Tính Tail Weights** (`_recompute_tail_weights()`)

```python
# Công thức
raw_weight(relation) = 1 / sqrt(frequency + ε)
tail_weight(relation) = raw_weight(relation) / Σ(raw_weights)

# Ví dụ:
# "on" (freq=100) → tail_weight ≈ 0.08
# "riding" (freq=5) → tail_weight ≈ 0.36
# "playing" (freq=2) → tail_weight ≈ 0.47
```

**Bước 2: Tính Sample Tail Weight** (`_get_sample_tail_weight()`)[-3:]

Average weight của toàn bộ dataset

**Bước 3: Tính Long-tail Loss trong Training**

```python
for each sample:
    sample_loss = RelTR_loss(sample)
    tail_weight = _get_sample_tail_weight(sample)
    if tail_weight > 0:
        total_tail_loss += sample_loss * tail_weight

long_tail_loss = total_tail_loss / tail_weighted_count
```

</details>

### 11.2 Multiple Epochs Training ✅

```python
rl_agent.reltr_training_epochs = 3  # Train 3 epochs mỗi episode
```

### 11.3 Dataset Input cho RL Training ✅

Console/GUI cho phép chọn thư mục ảnh để build dataset khi training.

### 11.4 mR@K Metrics ✅

Mean Recall@K - đánh giá công bằng cho long-tail:
- Tính R@K cho từng relation type riêng lẻ
- Lấy trung bình của tất cả R@K

| Metric | Cách tính | Ưu/Nhược |
|--------|-----------|----------|
| R@K | Recall tổng thể | Bị ảnh hưởng bởi head classes |
| mR@K | Mean của R@K per relation | Công bằng cho long-tail |

### 11.5 Safety System 3 Tầng ✅

Xem [Phần 9](#-9-hệ-thống-an-toàn-3-tầng).

---

## 📝 License

[MIT License](LICENSE)

---

## 👥 Tác giả

Dự án đồ án tốt nghiệp - Hệ thống Scene Graph Generation với Reinforcement Learning

---

## 🙏 Acknowledgments

- [YOLO](https://github.com/ultralytics/ultralytics) - Object Detection
- [CLIP](https://github.com/openai/CLIP) - Vision-Language Model
- [RelTR](https://github.com/yrcong/RelTR) - Scene Graph Generation
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Image Generation
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Open-Vocabulary Detection
- [OpenAI](https://openai.com/) - GPT Models cho Safety Analysis
- [Google Gemini](https://gemini.google.com/) - Gemini Models cho Safety Analysis
]]>
