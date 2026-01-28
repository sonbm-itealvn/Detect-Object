# 📚 Giải Thích Chi Tiết Kiến Trúc Hệ Thống và Công Nghệ

## 🎯 Tổng Quan Hệ Thống

Đây là một hệ thống **Scene Graph Generation** (Sinh đồ thị cảnh) kết hợp **Reinforcement Learning** để giải quyết vấn đề **long-tail distribution** trong dữ liệu quan hệ giữa các đối tượng. Hệ thống có thể:

1. **Phát hiện đối tượng** và **dự đoán quan hệ** giữa chúng trong ảnh/video
2. **Sinh dữ liệu synthetic** bằng AI để cải thiện hiệu suất trên các quan hệ hiếm
3. **Phân tích an toàn** trong video để cảnh báo các hành động nguy hiểm
4. **Học tăng cường** (RL) để tự động tối ưu quá trình sinh dữ liệu

---

## 🏗️ Kiến Trúc Tổng Thể

Hệ thống được chia thành **4 luồng chính**:

### 1. **VRD Pipeline** (Visual Relationship Detection)
- **Input**: Ảnh hoặc Video
- **Output**: Scene Graph (Objects + Relationships)
- **Mục đích**: Phát hiện đối tượng và quan hệ giữa chúng

### 2. **RL Training Pipeline** (Reinforcement Learning)
- **Input**: Scene Graph từ VRD Pipeline
- **Output**: Model đã được fine-tune + Metrics
- **Mục đích**: Cải thiện hiệu suất trên long-tail relationships

### 3. **Safety Analysis Pipeline** (Chỉ cho Video)
- **Input**: Relationships từ VRD Pipeline
- **Output**: Safety Level + Alerts
- **Mục đích**: Phát hiện và cảnh báo hành động nguy hiểm

### 4. **Synthetic Data Generation Pipeline**
- **Input**: Relationship triplets
- **Output**: Synthetic images với annotations
- **Mục đích**: Tạo dữ liệu training cho các quan hệ hiếm

---

## 🔧 Công Nghệ Sử Dụng

### 1. **Object Detection: YOLO v11**

**File**: `models/yolo.py`, `detect_objects.py`

**Công nghệ**:
- **YOLO v11** (Ultralytics): Model phát hiện đối tượng real-time
- **CLIP** (OpenAI): Re-classify với open vocabulary
- **ROI Features Extraction**: Trích xuất đặc trưng từ bounding boxes

**Chức năng**:
```python
# Luồng xử lý:
1. YOLO detect → Bounding boxes + Class labels
2. CLIP re-classify → Open vocabulary classification
3. ROI Features → Feature vectors cho từng object
4. Global Context → Context vector cho toàn bộ ảnh
```

**Output**: 
- `converted_bboxes.json`: Objects với bbox, class, confidence
- Feature maps và global context vector

---

### 2. **Relationship Prediction: RelTR (Relationship Transformer)**

**File**: `models/reltr.py`, `boundingbox_objects.py`

**Công nghệ**:
- **Transformer Architecture**: Dựa trên DETR (Detection Transformer)
- **Backbone**: ResNet hoặc Swin Transformer
- **Attention Mechanism**: Self-attention và cross-attention

**Kiến trúc RelTR**:
```
Input Image
    ↓
Backbone (ResNet/Swin) → Feature Maps
    ↓
Input Projection → Hidden Dimension
    ↓
Transformer Encoder → Entity Queries
    ↓
Transformer Decoder → Subject/Object Queries
    ↓
Output Heads:
  - Entity Classification + Bbox
  - Subject Classification + Bbox  
  - Object Classification + Bbox
  - Relationship Classification
```

**Chức năng**:
- Dự đoán **triplets**: (Subject, Predicate, Object)
- Ví dụ: `(dog, riding, surfboard)`
- 51 loại quan hệ: "on", "holding", "riding", "wearing", etc.

**Output**: 
- `relationships.json`: List các relationships với confidence

---

### 3. **Reinforcement Learning: Deep Q-Network (DQN)**

**File**: `RL/reinforcement_learning.py`

**Công nghệ**:
- **DQN Agent**: Deep Q-Network với experience replay
- **State Space**: 5 chiều (detection_f1, relationship_f1, reward, dataset_norm, epsilon)
- **Action Space**: [1-10] - Số lượng variations cần sinh cho mỗi relationship
- **Reward Function**: Weighted sum của Detection F1, Relationship F1, Diversity, Consistency, Improvement

**Q-Network Architecture**:
```python
Input(5) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(10)
```

**Luồng hoạt động**:
```
1. State Vector Construction (5D)
   - detection_f1: F1 score từ detection model
   - relationship_f1: F1 score từ relationship model
   - reward: Reward hiện tại
   - dataset_norm: Kích thước dataset / 50
   - epsilon: Exploration rate

2. DQN Agent Decision
   - Epsilon-greedy: Exploration vs Exploitation
   - Q-network: Predict Q-values cho mỗi action
   - Action: Số variations cần sinh [1-10]

3. Action Execution
   - Generate synthetic data với số lượng variations được chọn
   - Relationship-specific priorities:
     * F1 < 0.3 → sinh 3x variations
     * F1 0.3-0.7 → sinh 1-2x variations
     * F1 > 0.7 → sinh 0.5x variations

4. Reward Calculation
   - Base components:
     * Detection F1 × 0.25
     * Relationship F1 × 0.45 (cao nhất)
     * Diversity × 0.15
     * Consistency × 0.10
     * Improvement × 0.05
   - Long-tail boost:
     * tail_weight = 1 / sqrt(frequency + 1e-3)
     * relationship_score *= (1 + tail_weight)

5. DQN Update
   - Experience replay: Sample batch từ buffer (size=10,000)
   - Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
   - Target network update: Mỗi 20 steps
   - Epsilon decay: ε ← ε × 0.995 (min=0.01)
```

**Hyperparameters**:
- Learning Rate: 0.001 (AdamW)
- Gamma: 0.95 (Discount factor)
- Epsilon: 0.9 → 0.01 (Exploration rate)
- Batch Size: 32
- Buffer Size: 10,000
- Target Update: 20 steps

---

### 4. **Synthetic Data Generation: Stable Diffusion**

**File**: `RL/ai_images_generator.py`

**Công nghệ**:
- **Stable Diffusion v1.5**: Text-to-image generation
- **Diffusers Library**: HuggingFace implementation
- **Prompt Engineering**: Template-based prompt generation

**Luồng sinh ảnh**:
```
1. Relationship Triplet Input
   Input: {"subject": "dog", "relation": "riding", "object": "surfboard"}

2. Prompt Generation
   - Template mapping: "riding" → "{subject} riding {object}"
   - Base prompt: "dog riding surfboard"
   - Variations:
     * Context: "in hands", "on street", "in room"
     * Quality: "high quality", "detailed", "realistic"
     * Lighting: "bright daylight", "soft lighting"
     * Background: "on the street", "in the park"
   - Final prompt: "dog riding surfboard, high quality, bright daylight, on the street"

3. Stable Diffusion Generation
   - Batch generation (batch_size=2) để tối ưu GPU
   - num_inference_steps: 50
   - guidance_scale: 7.5
   - Output: PIL Images

4. Quality Filter
   - Size/Aspect: min 512×512, max aspect 2.2
   - Blur check: Laplacian variance ≥ 60
   - Exposure: mean [20, 235], clip ratio ≤ 0.20
   - Duplicate: pHash comparison
   - CLIP similarity: ≥ 0.23 với prompt

5. Output
   - List of PIL Images
   - Metadata: prompt, original_relationship, generation_timestamp
```

---

### 5. **Auto-Annotation: Open-Vocabulary Detection**

**File**: `RL/auto_annotator.py`

**Vấn đề**: Stable Diffusion chỉ trả về pixels, không có bounding boxes. Cần tự động tạo annotations.

**Giải pháp**: Sử dụng Open-Vocabulary Detectors với priority order:

#### **Priority 1: GroundingDINO** (SOTA, chính xác nhất)
- **Model**: SwinT-OGC (Swin Transformer)
- **Input**: Image + text prompt "dog . surfboard"
- **Output**: Normalized coords [cx, cy, w, h] → convert to [x1, y1, x2, y2]
- **Thresholds**: box_threshold=0.25, text_threshold=0.20

#### **Priority 2: OWL-ViT** (Lightweight, HuggingFace)
- **Model**: google/owlvit-base-patch32
- **Input**: Image + text prompts ["a photo of a dog", "a photo of a surfboard"]
- **Output**: Direct [x1, y1, x2, y2] coordinates
- **Threshold**: box_threshold=0.25

#### **Priority 3: YOLO + CLIP** (Fallback)
- **Pipeline**: YOLO detect → CLIP classify với open vocabulary
- **Matching**: Fuzzy match labels với text prompts
- **Confidence**: 0.7 nếu matched, 0.5 nếu không
- **Limited vocabulary**: Chỉ detect classes YOLO biết

#### **Priority 4: Pseudo-bbox** (Last Resort, Low Quality)
- **Heuristics** dựa trên relation type:
  - "on"/"above"/"riding": Subject trên, Object dưới
  - "under"/"below": Subject dưới, Object trên
  - "holding"/"carrying": Subject lớn, Object nhỏ gần subject
  - Default: Subject trái, Object phải
- **Confidence**: 0.3 (rất thấp)

**Auto-detect Logic**:
```python
def _initialize_backend():
    # 1. Try GroundingDINO first
    if GROUNDINGDINO_AVAILABLE:
        try:
            model = load_model(config, checkpoint, device)
            return "groundingdino"
        except:
            pass
    
    # 2. Try OWL-ViT
    if OWLVIT_AVAILABLE:
        try:
            processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            return "owlvit"
        except:
            pass
    
    # 3. Fallback to YOLO+CLIP
    if YOLO_CLIP_AVAILABLE:
        return "yolo_clip"
    
    # 4. No backend available
    return "none"  # Will use pseudo-bbox only
```

---

### 6. **Safety Analysis: 3-Tier System**

**File**: `RL/safety_classifier.py`, `RL/llm_safety_analyzer.py`, `RL/local_rules_db.py`

**Công nghệ**:
- **Tier 1**: White/Black/Gray List (Pattern matching)
- **Tier 2**: LLM Semantic Reasoning (OpenAI/Gemini)
- **Tier 3**: Local Rules Database (Human-in-the-loop)

**Luồng xử lý**:
```
Relationship từ RelTR: {subject, relation, object}
    ↓
┌─────────────────────────────────────────┐
│ Tầng 3: Local Rules Database            │
│ (Ưu tiên cao nhất)                      │
│ • Kiểm tra local_safety_rules.json      │
│ • Nếu có rule → Return level từ rule    │
└─────────────────────────────────────────┘
    ↓ (Nếu không có rule)
┌─────────────────────────────────────────┐
│ Tầng 1: White/Black/Gray List          │
│ • Black List → DANGEROUS                │
│ • White List → SAFE                     │
│ • Không match → Gray List → Tầng 2      │
└─────────────────────────────────────────┘
    ↓ (Nếu Gray List)
┌─────────────────────────────────────────┐
│ Tầng 2: LLM Semantic Reasoning          │
│ • Kiểm tra cache (llm_safety_cache.json)│
│ • Nếu không có → Gọi LLM API            │
│   - OpenAI: gpt-3.5-turbo, gpt-4o       │
│   - Gemini: gemini-pro                  │
│ • Parse response: SAFE/SUSPICIOUS/DANGEROUS│
│ • Lưu vào cache                         │
└─────────────────────────────────────────┘
    ↓
Return: (SafetyLevel, confidence, explanation)
```

**Ví dụ**:
- Input: `{"subject": "child", "relation": "holding", "object": "knife"}`
- Tầng 1: Match Black List → `DANGEROUS` (confidence: 0.95)
- Output: Banner đỏ "NGUY HIỂM - Cần can thiệp ngay!"

---

### 7. **Video Processing Pipeline**

**File**: `video_relation_pipeline.py`

**Công nghệ**:
- **OpenCV**: Video I/O và frame processing
- **Object Tracking**: Track IDs cho objects qua các frames
- **Safe Zone Monitor**: Phát hiện xâm nhập vùng an toàn 2m

**Luồng xử lý video**:
```
1. Video Input
   - Load video file (mp4, avi, mov, mkv)
   - Frame extraction với stride (mặc định: 2)

2. Frame Processing (cho mỗi frame)
   a. YOLO Detection → Objects với bbox
   b. Object Tracking → Track IDs
   c. RelTR Inference → Relationships
   d. Safety Analysis (nếu enabled) → Safety levels
   e. Safe Zone Monitor → Intrusions

3. Annotation Rendering
   - Vẽ bounding boxes cho objects
   - Vẽ arrows cho relationships
   - Hiển thị cảnh báo an toàn:
     * DANGEROUS: Banner đỏ
     * SUSPICIOUS: Banner cam
   - Vẽ safe zone polygon

4. Output
   - Annotated video: video_outputs/{video_name}_relations.avi
   - Summary JSON: video_outputs/{video_name}_summary.json
     * Objects statistics
     * Relationships statistics
     * Safety alerts
```

---

## 📊 Luồng Hoạt Động Chi Tiết

### **Luồng 1: Image Processing (Ảnh đơn)**

```
Input Image
    ↓
[detect_objects.py]
    ├─ YOLO Detection → Bounding boxes
    ├─ CLIP Re-classify → Open vocabulary
    ├─ ROI Features → Feature vectors
    └─ Global Context → Context vector
    ↓
[convert_yolo_to_reltr.py]
    └─ Convert format → converted_bboxes.json
    ↓
[boundingbox_objects.py]
    ├─ Load RelTR model
    ├─ Prepare inputs (image tensor, objects, context)
    ├─ RelTR Inference → Relationships
    └─ Save → relationships.json
    ↓
Output:
    - output_{image_id}.jpg (annotated image)
    - converted_bboxes.json (objects)
    - relationships.json (relationships)
```

### **Luồng 2: RL Training**

```
1. Initialization
   - Load detection model (YOLO)
   - Load relationship model (RelTR)
   - Initialize DQN agent
   - Build dataset từ relationships hiện tại

2. Episode Loop (mỗi epoch)
   a. State Construction
      - Tính metrics: detection_f1, relationship_f1
      - Build state vector (5D)
   
   b. DQN Decision
      - Epsilon-greedy: Exploration vs Exploitation
      - Q-network: Predict action (số variations)
   
   c. Synthetic Data Generation
      - Chọn relationships cần sinh (ưu tiên long-tail)
      - Generate images với Stable Diffusion
      - Quality filter
      - Auto-annotation (GroundingDINO/OWL-ViT)
      - Build relationships từ annotations
   
   d. Dataset Ingestion
      - Thêm synthetic samples vào dataset
      - Recompute tail_weights
   
   e. Model Training
      - Fine-tune RelTR với dataset mới
      - Multiple epochs (mặc định: 1, có thể tăng)
      - Compute losses: bbox_loss, giou_loss, rel_loss, long_tail_loss
   
   f. Evaluation
      - Detection metrics: Precision, Recall, F1
      - Relationship metrics: Precision, Recall, F1, mR@K
      - Diversity score
      - Consistency score
   
   g. Reward Calculation
      - Weighted sum với long-tail boost
      - Store experience: (state, action, reward, next_state)
   
   h. DQN Update
      - Experience replay
      - Q-learning update
      - Target network update (mỗi 20 steps)
      - Epsilon decay

3. Experiment Tracking
   - Save metrics: JSON files
   - Save plots: Reward vs Epoch, Components breakdown
   - Save models: Checkpoints theo epoch
   - Save synthetic images: experiments/exp_XXX/ai_images/
```

### **Luồng 3: Video Processing với Safety**

```
Input Video
    ↓
[VideoRelationPipeline.process_video()]
    ↓
For each frame (với stride):
    ├─ YOLO Detection → Objects
    ├─ Object Tracking → Track IDs
    ├─ RelTR Inference → Relationships
    ├─ Safety Classifier (nếu enabled)
    │   ├─ Tầng 3: Local Rules
    │   ├─ Tầng 1: White/Black List
    │   └─ Tầng 2: LLM Analysis
    ├─ Safe Zone Monitor → Intrusions
    ├─ Annotation Rendering
    │   ├─ Draw bounding boxes
    │   ├─ Draw relationship arrows
    │   ├─ Draw safety alerts (banner đỏ/cam)
    │   └─ Draw safe zone polygon
    └─ Write frame to output video
    ↓
Output:
    - Annotated video
    - Summary JSON (objects, relationships, safety alerts)
```

---

## 🧩 Các Module Chính

### **1. Detection Module** (`detect_objects.py`)
- **Chức năng**: Phát hiện đối tượng với YOLO + CLIP
- **Input**: Image path
- **Output**: Objects với bbox, class, confidence, features

### **2. Relationship Module** (`boundingbox_objects.py`)
- **Chức năng**: Dự đoán quan hệ giữa các đối tượng
- **Input**: Image + Objects (JSON)
- **Output**: Relationships (triplets)

### **3. RL Module** (`RL/reinforcement_learning.py`)
- **Chức năng**: DQN agent để tối ưu synthetic data generation
- **Components**:
  - Q-Network
  - Experience Replay Buffer
  - Reward Function
  - Training Loop

### **4. Generator Module** (`RL/ai_images_generator.py`)
- **Chức năng**: Sinh ảnh synthetic từ relationships
- **Components**:
  - Stable Diffusion Pipeline
  - Prompt Engineering
  - Quality Filter

### **5. Annotator Module** (`RL/auto_annotator.py`)
- **Chức năng**: Tự động tạo annotations cho synthetic images
- **Backends**: GroundingDINO, OWL-ViT, YOLO+CLIP, Pseudo

### **6. Safety Module** (`RL/safety_classifier.py`)
- **Chức năng**: Phân loại an toàn 3 tầng
- **Components**:
  - White/Black/Gray List
  - LLM Analyzer
  - Local Rules DB

### **7. Video Module** (`video_relation_pipeline.py`)
- **Chức năng**: Xử lý video với tracking và safety analysis
- **Components**:
  - Frame Processing
  - Object Tracking
  - Safe Zone Monitor
  - Annotation Rendering

### **8. App Modules**
- **GUI App** (`app.py`): Tkinter-based GUI
- **Console App** (`app_console.py`): Command-line interface

---

## 🔄 Data Flow

### **Training Data Flow**:
```
Original Images
    ↓
VRD Pipeline → Relationships
    ↓
RL Agent → Action (số variations)
    ↓
Stable Diffusion → Synthetic Images
    ↓
Auto-Annotator → Annotations (bbox + relationships)
    ↓
Dataset Ingestion → Training Samples
    ↓
RelTR Training → Updated Model
    ↓
Evaluation → Metrics
    ↓
Reward Calculation → DQN Update
```

### **Inference Data Flow**:
```
Input (Image/Video)
    ↓
YOLO Detection → Objects
    ↓
RelTR Inference → Relationships
    ↓
Safety Analysis (nếu video) → Safety Levels
    ↓
Output (Annotated Image/Video + JSON)
```

---

## 📈 Metrics và Đánh Giá

### **Detection Metrics**:
- Precision, Recall, F1
- Bbox Loss, GIoU Loss

### **Relationship Metrics**:
- Precision, Recall, F1
- **mR@K** (Mean Recall@K): mR@10, mR@20, mR@50, mR@100
- Relationship Loss
- **Long-tail Loss**: Weighted loss cho quan hệ hiếm

### **RL Metrics**:
- Reward (tổng hợp từ các components)
- Q-values
- Action frequency
- Experience buffer size

### **Safety Metrics**:
- Total alerts
- Dangerous count
- Suspicious count
- Safe count

---

## 🎯 Điểm Mạnh của Hệ Thống

1. **Giải quyết Long-tail Problem**: RL tự động sinh dữ liệu cho quan hệ hiếm
2. **Open-vocabulary**: CLIP và GroundingDINO hỗ trợ detect bất kỳ object nào
3. **Safety Analysis**: 3-tier system với LLM reasoning
4. **Auto-annotation**: Tự động tạo annotations cho synthetic images
5. **Experiment Tracking**: Quản lý và so sánh experiments
6. **Multiple Epochs Training**: Hỗ trợ training nhiều epochs
7. **Video Support**: Xử lý video với tracking và safety alerts

---

## 🚀 Cách Sử Dụng

### **1. Image Processing**:
```bash
python app_console.py
# Chọn 1: Chọn ảnh
# Chọn 2: Chạy pipeline
```

### **2. RL Training**:
```bash
python app_console.py
# Chọn 4: Chạy RL Training
# Nhập số epochs
```

### **3. Video Processing**:
```python
from video_relation_pipeline import VideoRelationPipeline

pipeline = VideoRelationPipeline(safety_classifier_enabled=True)
result = pipeline.process_video("video.mp4")
```

---

## 📝 Kết Luận

Hệ thống này là một **pipeline hoàn chỉnh** từ detection → relationship prediction → synthetic data generation → RL training → safety analysis. Nó kết hợp nhiều công nghệ SOTA:

- **YOLO v11**: Object detection
- **RelTR**: Scene graph generation
- **Stable Diffusion**: Image generation
- **GroundingDINO/OWL-ViT**: Open-vocabulary detection
- **DQN**: Reinforcement learning
- **LLM (OpenAI/Gemini)**: Semantic reasoning

Hệ thống được thiết kế để giải quyết vấn đề **long-tail distribution** trong Scene Graph Generation, một vấn đề phổ biến trong computer vision research.

