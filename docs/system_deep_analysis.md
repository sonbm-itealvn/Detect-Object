# Phân Tích Chi Tiết Từng Dòng Code — Hệ Thống VRD + RL

> Tài liệu này giải thích **cực kỳ chi tiết** từng thành phần, từng hàm, từng công thức toán trong hệ thống. Mỗi phần đều có: (1) mục đích, (2) input/output, (3) thuật toán bên trong, (4) ví dụ số cụ thể.

---

## MỤC LỤC CHI TIẾT

- [PHẦN A: DETECTION PIPELINE](#phần-a-detection-pipeline)
- [PHẦN B: RELATIONSHIP PREDICTION](#phần-b-relationship-prediction)
- [PHẦN C: RL CORE — DQN AGENT](#phần-c-rl-core--dqn-agent)
- [PHẦN D: UNCERTAINTY ESTIMATOR (MC DROPOUT)](#phần-d-uncertainty-estimator-mc-dropout)
- [PHẦN E: ACTIVE LEARNING SELECTOR](#phần-e-active-learning-selector)
- [PHẦN F: APPROXIMATION ALGORITHM](#phần-f-approximation-algorithm)
- [PHẦN G: IMAGE GENERATOR + AUTO-ANNOTATION](#phần-g-image-generator--auto-annotation)
- [PHẦN H: REWARD FUNCTION CHI TIẾT](#phần-h-reward-function-chi-tiết)
- [PHẦN I: TRAINING EPISODE — TỪNG BƯỚC](#phần-i-training-episode--từng-bước)
- [PHẦN J: APPLICATION LAYER](#phần-j-application-layer)

---

# PHẦN A: DETECTION PIPELINE

**File**: `detect_objects.py`

## A.1. Mục đích

Nhận ảnh đầu vào → Trả về danh sách vật thể với: tên class, bounding box (x1,y1,x2,y2), feature vector, và global context vector.

## A.2. Các model được load khi khởi động

| Model | Vai trò | Cách load |
|-------|---------|-----------|
| **YOLO** (`fine-tune.pt`) | Phát hiện bbox + class (80 class COCO) | `YOLO(_resolve_yolo_weights())` |
| **Fire YOLO** (tùy chọn) | Phát hiện lửa/khói | `YOLO(fire_model_weights)` nếu có file |
| **CLIP** (`ViT-B/32`) | Phân loại open-vocabulary (~200+ class) | `clip.load("ViT-B/32")` |

## A.3. Backbone Feature Capture

```python
_BACKBONE_LAYER_INDEX = 9  # Layer SPPF trong YOLO backbone
_BACKBONE_STRIDE = int(yolo_model.model.model[-1].stride[-1].item())
```

- Đăng ký `forward_hook` vào layer thứ 9 (SPPF) của YOLO backbone
- Mỗi lần YOLO forward, feature map tự động được lưu vào `_feature_map_queue`
- Feature map shape: `(1, C, H/stride, W/stride)` — ví dụ `(1, 512, 20, 20)` cho ảnh 640×640

## A.4. Hàm `detect_objects(image_source)` — Chi tiết

**Input**: đường dẫn ảnh, PIL Image, hoặc numpy array

**Luồng thực thi:**

```
Step 1: Load ảnh → cv2 numpy array
Step 2: Chạy YOLO COCO model → coco_results (boxes, classes, confidences)
Step 3: (Nếu có) Chạy Fire model → fire_results
Step 4: Merge detections bằng NMS
Step 5: Lấy feature_map từ hook
Step 6: Tính global_context = GAP(feature_map) → L2 normalize
Step 7: Với mỗi detection:
        - Bỏ qua nếu bbox < 20×20 pixel
        - Crop vùng bbox + padding 10px
        - Lưu vào detected_objects
```

**Output**: `(detected_objects, yolo_labels, original_image, feature_map, global_context)`

## A.5. Merge Detections — NMS

```python
def _merge_detections(coco_results, fire_results, iou_threshold=0.5):
```

- Gộp tất cả detections từ COCO + Fire model vào 1 list
- Chuyển thành tensor: `boxes_tensor (N, 4)`, `scores_tensor (N,)`
- Gọi `torchvision_nms(boxes, scores, iou_threshold=0.5)`
- NMS loại bỏ box trùng lặp: nếu 2 box có IoU > 0.5, giữ box có confidence cao hơn
- Trả về danh sách detections đã merge

## A.6. Hàm `classify_with_clip()` — Batch Processing

**Tối ưu quan trọng**: Text features được pre-compute 1 lần khi module load:

```python
text_inputs = clip.tokenize(label_texts).to(device)
with torch.no_grad():
    _precomputed_text_features = clip_model.encode_text(text_inputs)
    _precomputed_text_features = _precomputed_text_features / _precomputed_text_features.norm(dim=-1, keepdim=True)
```

**Batch inference** (thay vì từng ảnh 1):

```python
batch_tensor = torch.stack(image_batch).to(device)  # Stack tất cả crops
image_features = clip_model.encode_image(batch_tensor)  # 1 lần forward cho tất cả
similarities = (image_features @ text_features.T).softmax(dim=-1)  # Cosine similarity
```

**Logic quyết định nhãn:**

```
if nhãn_YOLO thuộc important_labels AND nhãn_CLIP ≠ nhãn_YOLO:
    → giữ nhãn YOLO (tin YOLO hơn cho class đã biết)
elif CLIP_confidence < 0.3:
    → giữ nhãn YOLO (CLIP không chắc chắn)
else:
    → dùng nhãn CLIP (CLIP tự tin hơn)
```

## A.7. ROI Feature Extraction

```python
def extract_roi_features(feature_map, boxes, image_shape):
```

**Mục đích**: Trích xuất vector đặc trưng riêng cho mỗi vật thể (dùng cho relationship prediction).

**Thuật toán:**

1. Tính scale + padding do YOLO letterboxing
2. Map bbox từ tọa độ ảnh gốc → tọa độ feature map
3. Dùng `roi_align(feature_map, rois, output_size=(7,7))` — Bilinear interpolation pooling
4. `adaptive_avg_pool2d(pooled, (1,1))` → flatten → L2 normalize

**Output**: List các vector `(512,)` — mỗi vector đại diện cho 1 vật thể

## A.8. Global Context Vector

```python
def _compute_global_context(feature_map):
    pooled = feature_map.mean(dim=(2, 3))   # Global Average Pooling
    pooled = F.normalize(pooled, p=2, dim=1) # L2 normalize
    return pooled.squeeze(0).cpu().tolist()
```

- Nén toàn bộ feature map `(1, 512, H, W)` → vector `(512,)` bằng GAP
- Vector này đại diện cho **ngữ cảnh tổng thể** của ảnh (cảnh đường phố, trong nhà, v.v.)
- Được dùng làm input bổ sung cho RelTR model

---

# PHẦN B: RELATIONSHIP PREDICTION

**File**: `boundingbox_objects.py`

## B.1. RelTR Model

**Kiến trúc** (config từ `_build_reltr_args()`):

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| backbone | ResNet-50 | Trích xuất visual features |
| enc_layers | 6 | Số Transformer encoder layers |
| dec_layers | 6 | Số Transformer decoder layers |
| hidden_dim | 256 | Kích thước hidden state |
| nheads | 8 | Số attention heads |
| dim_feedforward | 2048 | Kích thước FFN |
| dropout | 0.1 | Dropout rate |
| num_entities | 100 | Số entity queries |
| num_triplets | 200 | Số relationship queries |
| position_embedding | sine | Positional encoding kiểu sin/cos |

**51 loại quan hệ** (`RELATION_CLASSES`):

```
__background__, above, across, against, along, and, at, attached to,
behind, belonging to, between, carrying, covered in, covering, eating,
flying in, for, from, growing on, hanging from, has, holding, in,
in front of, laying on, looking at, lying on, made of, mounted on, near,
of, on, on back of, over, painted on, parked on, part of, playing,
riding, says, sitting on, standing on, to, under, using, walking in,
walking on, watching, wearing, wears, with
```

## B.2. Hàm `_decode_relationships()` — 2 chiến lược

**Chiến lược 1: Geometric Matching** (ưu tiên)

```
1. Lấy sub_boxes, obj_boxes từ RelTR output
2. Chuyển từ cxcywh → xyxy, scale theo kích thước ảnh
3. Với mỗi relationship query:
   a. Tính IoU giữa sub_box và tất cả detected objects → tìm subject
   b. Tính IoU giữa obj_box và tất cả detected objects → tìm object
   c. Nếu cả 2 IoU > 0.05:
      confidence = rel_confidence × subj_iou × obj_iou
      → thêm relationship
```

**Chiến lược 2: Pairwise Fallback** (khi geometric thất bại)

```
1. Filter queries: giữ queries có max_score > 0.4
2. Với mỗi cặp (object_i, object_j) trong detected objects:
   a. Lấy relation vector từ filtered queries (round-robin)
   b. argmax → relation index → relation name
   c. → thêm relationship
```

## B.3. Spatial Validation

Kiểm tra tính hợp lý vật lý:

```python
SEMANTIC_CORRECTIONS = [
    {
        "subject": ANIMAL_CLASSES | PERSON_CLASSES,
        "object": TRANSPORT_CLASSES,
        "wrong_relations": {"wearing", "wears", "has", "holding", "carrying"},
        "correct_relation": "riding",
    },
    ...
]
```

**Ví dụ**: RelTR predict "person wearing skateboard" → sửa thành "person riding skateboard" vì logic: người không thể "mặc" ván trượt.

---

# PHẦN C: RL CORE — DQN AGENT

**File**: `RL/reinforcement_learning.py`, class `RelationshipReinforcementLearning`

## C.1. Khởi tạo (Constructor)

Khi tạo instance, hệ thống khởi tạo:

```python
self.memory = deque(maxlen=10000)    # Replay buffer
self.epsilon = 0.9                    # Exploration rate ban đầu
self.epsilon_decay = 0.995            # Mỗi epoch: ε *= 0.995
self.epsilon_min = 0.01               # Không giảm dưới 1%

self.action_space = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Số ảnh sinh/relationship
self.state_dim = 5                    # Vector trạng thái 5 chiều
self.gamma = 0.95                     # Discount factor
self.batch_size = 32                  # Batch size cho Q-learning
self.target_update_interval = 20      # Sync target network mỗi 20 steps

# Hai mạng Q:
self.q_network = MLP(5 → 64 → 64 → 10)         # Online network
self.target_network = MLP(5 → 64 → 64 → 10)     # Target network (copy)
self.q_optimizer = AdamW(q_network.params, lr=1e-3)
```

## C.2. Q-Network Architecture

```python
def _build_q_network(self, input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64),    # 5 → 64
        nn.ReLU(),
        nn.Linear(64, 64),           # 64 → 64
        nn.ReLU(),
        nn.Linear(64, output_dim),   # 64 → 10
    )
```

**Tổng tham số**: 5×64 + 64 + 64×64 + 64 + 64×10 + 10 = **5,194 parameters**

**Ý nghĩa mỗi output**: Q(s, a=k) = giá trị kỳ vọng khi chọn action k (sinh k ảnh/relationship) ở trạng thái s.

## C.3. State Vector — Xây dựng chi tiết

```python
def _build_state_vector(self, metrics):
    detection_f1 = metrics.get('detection_f1', 0.0)
    relationship_f1 = metrics.get('relationship_f1', 0.0)
    
    # Fallback nếu không có F1: chuyển loss → pseudo-F1
    if detection_f1 == 0.0 and 'detection_loss' in metrics:
        detection_f1 = max(0.0, 1.0 - tanh(detection_loss / 5.0))
    
    reward_value = tanh(reward / 1.0)
    dataset_norm = tanh(dataset_size / 50.0)
    epsilon_value = tanh(epsilon / 1.0)
    
    state = [detection_f1, relationship_f1, reward_value, dataset_norm, epsilon_value]
```

**Ví dụ số cụ thể:**

Giả sử epoch 3: detection_f1=0.65, relationship_f1=0.40, reward=0.55, dataset_size=12, epsilon=0.85

```
state[0] = 0.65                         (detection F1 trực tiếp)
state[1] = 0.40                         (relationship F1 trực tiếp)
state[2] = tanh(0.55/1.0) = 0.4999      (reward normalized)
state[3] = tanh(12/50.0) = 0.2342       (dataset size normalized)
state[4] = tanh(0.85/1.0) = 0.6913      (epsilon normalized)

→ state = [0.650, 0.400, 0.500, 0.234, 0.691]
```

## C.4. Action Selection — Epsilon-Greedy

```python
def _select_action(self, state):
    if random.random() < self.epsilon:      # Explore
        action_index = random.randrange(10)  # Random từ 0-9
    else:                                    # Exploit
        q_values = self.q_network(state.unsqueeze(0))
        action_index = q_values.argmax(dim=1).item()
    
    action_value = self.action_space[action_index]  # Map index → giá trị
    return action_index, action_value
```

**Ví dụ:**

- Epoch 1: ε = 0.9 → 90% random, 10% dùng Q-network
- Epoch 20: ε = 0.9 × 0.995^20 = 0.814 → 81% random
- Epoch 100: ε = 0.9 × 0.995^100 = 0.544 → 54% random
- Epoch 460: ε ≈ 0.01 → 1% random, 99% dùng Q-network (hầu hết exploit)

## C.5. Q-Network Update — Bellman Equation

```python
def _optimize_q_network(self):
    if len(self.memory) < self.batch_size:
        return None  # Chưa đủ experience
    
    # Sample random batch từ replay buffer
    batch = random.sample(self.memory, 32)
    
    # Unpack batch
    states = stack([exp[0] for exp in batch])       # (32, 5)
    actions = tensor([exp[1] for exp in batch])      # (32,)
    rewards = tensor([exp[2] for exp in batch])      # (32,)
    next_states = stack([exp[3] for exp in batch])   # (32, 5)
    dones = tensor([exp[4] for exp in batch])        # (32,)
    
    # Tính Q-values hiện tại cho actions đã chọn
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    # → (32,) — Q(s, a) cho mỗi experience
    
    # Tính target values dùng TARGET network (không phải online network!)
    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1).values
        target_values = rewards + 0.95 * next_q_values * (1 - dones)
    # → (32,) — r + γ * max_a' Q_target(s', a')
    
    # MSE loss
    loss = MSE(q_values, target_values)
    
    # Backprop với gradient clipping
    loss.backward()
    clip_grad_norm_(q_network.parameters(), max_norm=1.0)
    optimizer.step()
```

**Ví dụ số:**

Một experience: `(state=[0.65, 0.40, 0.50, 0.23, 0.69], action=3, reward=0.55, next_state=[0.70, 0.45, 0.55, 0.30, 0.65], done=False)`

```
Q(s, a=3) = q_network(state)[3] = 0.42   (giá trị hiện tại)
max_a' Q_target(s', a') = 0.60            (giá trị tương lai tốt nhất)
target = 0.55 + 0.95 × 0.60 × (1-0) = 0.55 + 0.57 = 1.12

loss = (0.42 - 1.12)² = 0.49
→ gradient update: Q(s, a=3) sẽ tăng lên về phía 1.12
```

## C.6. Target Network Sync

```python
self.learn_step_counter += 1
if self.learn_step_counter % 20 == 0:
    self.target_network.load_state_dict(self.q_network.state_dict())
```

**Tại sao cần 2 mạng?** Nếu dùng cùng 1 mạng, target `r + γ max Q(s', a')` thay đổi liên tục khi Q-network update → training không ổn định. Target network "đóng băng" 20 steps → target ổn định → convergence tốt hơn.

## C.7. Long-Tail Weight Computation

```python
def _recompute_tail_weights(self):
    # Đếm tần suất mỗi relation trong dataset
    freq = {}
    for sample in self.dataset_samples:
        for rel in sample.get('relationships', []):
            rel_name = normalize(rel.get('relation', ''))
            freq[rel_name] = freq.get(rel_name, 0) + 1
    
    # Inverse sqrt frequency
    raw_weights = {k: 1.0 / sqrt(v + 1e-3) for k, v in freq.items()}
    
    # Normalize
    total = sum(raw_weights.values())
    self.tail_weights = {k: v / total for k, v in raw_weights.items()}
```

**Ví dụ:**

| Relation | freq | 1/√freq | Normalized |
|----------|------|---------|------------|
| on | 50 | 0.141 | 0.067 |
| near | 30 | 0.183 | 0.087 |
| holding | 10 | 0.316 | 0.151 |
| riding | 3 | 0.577 | 0.276 |
| flying in | 1 | 1.000 | 0.478 |

→ "flying in" (freq=1) nhận weight **7× lớn hơn** "on" (freq=50). Điều này đảm bảo model không bỏ quên các quan hệ hiếm.

## C.8. Per-Relationship Performance Tracking

```python
def _update_relationship_performance(self, relationship, f1_score):
    rel_key = f"{subject}|{relation}|{object}".lower()
    
    if rel_key not in self.relationship_performance:
        self.relationship_performance[rel_key] = {
            'f1_scores': deque(maxlen=20),  # Lưu 20 F1 gần nhất
            'avg_f1': 0.0,
            'min_f1': 1.0,
            'max_f1': 0.0,
            'generation_count': 0,
            'last_improvement': 0.0,
        }
    
    perf = self.relationship_performance[rel_key]
    perf['f1_scores'].append(f1_score)
    perf['avg_f1'] = mean(perf['f1_scores'])
    
    # Tính improvement so với lần trước
    if len(perf['f1_scores']) > 1:
        perf['last_improvement'] = f1_score - previous_f1
```

Mỗi relationship được theo dõi riêng → biết chính xác relationship nào đang yếu → Active Learning sẽ ưu tiên.

---

# PHẦN D: UNCERTAINTY ESTIMATOR (MC DROPOUT)

**File**: `RL/uncertainty_estimator.py`

## D.1. Nguyên lý MC Dropout

**Bài toán**: Model RelTR predict "person riding horse" với confidence 0.75. Nhưng liệu 0.75 có đáng tin không? Model có thể vô tình "chắc chắn sai" (overconfident).

**Giải pháp MC Dropout**: 
- Bật dropout khi inference → mỗi lần forward, một tập con neurons khác bị tắt
- Forward T=10 lần → 10 kết quả khác nhau
- Nếu 10 kết quả giống nhau → model thực sự chắc chắn
- Nếu 10 kết quả khác nhau → model thực ra không chắc chắn

## D.2. Enable/Disable MC Dropout

```python
def enable_mc_dropout(self, model):
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()  # Bật dropout
            if self.dropout_rate is not None:
                module.p = self.dropout_rate  # Override rate nếu cần
```

**Quan trọng**: Bình thường `model.eval()` tắt hết dropout. Ở đây ta **chỉ bật dropout** mà giữ nguyên BatchNorm ở eval mode → không ảnh hưởng statistics.

## D.3. Main Estimation Function — Từng bước

```python
def estimate_relationship_uncertainty(self, image_tensor, objects, ...):
    # Bước 1: Bật MC Dropout
    dropout_modules = self.enable_mc_dropout(self.model)
    
    # Bước 2: Forward T=10 lần
    all_predictions = []
    all_confidences = []
    
    for t in range(10):
        samples = nested_tensor_from_tensor_list([image_tensor])
        outputs = self.model(samples)           # Forward với dropout random
        relationships = decode_fn(outputs, ...)  # Decode ra relationships
        
        all_predictions.append(relationships)
        confs = [r.get('confidence', 0) for r in relationships]
        all_confidences.append(confs)
    
    # Bước 3: Tắt MC Dropout
    self.disable_mc_dropout(self.model)
    
    # Bước 4: Tính metrics
    entropy = self._compute_predictive_entropy(all_predictions)
    mutual_info = self._compute_mutual_information(all_predictions, all_confidences)
    variation_ratio = self._compute_variation_ratio(all_predictions)
    mean_conf = mean(all flat confidences)
    
    # Bước 5: Combined score
    uncertainty_score = 0.30*entropy + 0.30*mutual_info + 0.20*variation_ratio + 0.20*(1-mean_conf)
```

## D.4. Predictive Entropy — Giải thích chi tiết

**Mục đích**: Đo tổng uncertainty (bao gồm cả noise trong data VÀ thiếu kiến thức model).

```python
def _compute_predictive_entropy(self, all_predictions):
    # Tạo "fingerprint" cho mỗi prediction
    # Ví dụ: "person|riding|horse" là 1 fingerprint
    
    fingerprint_counts = Counter()
    total_predictions = 0
    
    for predictions in all_predictions:  # 10 forward passes
        for pred in predictions:
            fp = f"{pred['subject']}|{pred['relation']}|{pred['object']}"
            fingerprint_counts[fp] += 1
            total_predictions += 1
    
    # Tính entropy
    entropy = 0
    for count in fingerprint_counts.values():
        p = count / total_predictions
        entropy -= p * log(p)
    
    # Normalize bởi log(|unique outcomes|)
    entropy /= log(len(fingerprint_counts))
```

**Ví dụ số:**

10 forward passes cho ra:

| Pass | Predictions |
|------|------------|
| 1-6 | person riding horse |
| 7-8 | person on horse |
| 9 | person near horse |
| 10 | man riding horse |

Fingerprint counts: `{person|riding|horse: 6, person|on|horse: 2, person|near|horse: 1, man|riding|horse: 1}`

Total = 10

```
H = -(6/10)log(6/10) - (2/10)log(2/10) - (1/10)log(1/10) - (1/10)log(1/10)
  = -(0.6×(-0.51)) - (0.2×(-1.61)) - (0.1×(-2.30)) - (0.1×(-2.30))
  = 0.306 + 0.322 + 0.230 + 0.230
  = 1.088

Normalized: 1.088 / log(4) = 1.088 / 1.386 = 0.785
```

→ Entropy = 0.785 — khá cao, model không chắc chắn lắm.

## D.5. Mutual Information (BALD) — Giải thích chi tiết

**Mục đích**: Tách riêng **epistemic uncertainty** (thiếu data) khỏi **aleatoric uncertainty** (noise tự nhiên).

```
I[y, θ] = H[ȳ] − E_θ[H[y|θ]]
         = Total uncertainty − Data uncertainty
         = Epistemic uncertainty (model thiếu knowledge)
```

**Tại sao quan trọng?** 
- Nếu I cao → model thiếu knowledge → **sinh thêm data sẽ giúp**
- Nếu I thấp nhưng H cao → uncertainty do noise → **sinh thêm data KHÔNG giúp**

```python
def _compute_mutual_information(self, all_predictions, all_confidences):
    # H[ȳ] = Predictive entropy (đã tính)
    predictive_entropy = self._compute_predictive_entropy(all_predictions)
    
    # E[H[y|θ]] = Trung bình entropy MỖI forward pass riêng lẻ
    per_pass_entropies = []
    for t, predictions in enumerate(all_predictions):
        confs = all_confidences[t]
        pass_entropy = 0
        for c in confs:
            pass_entropy -= c * log(c)
        pass_entropy /= log(len(confs))  # Normalize
        per_pass_entropies.append(pass_entropy)
    
    expected_entropy = mean(per_pass_entropies)
    
    # BALD = H[ȳ] - E[H[y|θ]]
    mutual_info = max(0, predictive_entropy - expected_entropy)
```

## D.6. Variation Ratio — Giải thích chi tiết

```python
def _compute_variation_ratio(self, all_predictions):
    T = len(all_predictions)  # 10
    
    # Tạo fingerprint cho MỖI forward pass (tập hợp relationships)
    pass_fingerprints = []
    for predictions in all_predictions:
        fps = frozenset(fingerprint(pred) for pred in predictions)
        pass_fingerprints.append(fps)
    
    # Đếm fingerprint phổ biến nhất
    fp_counter = Counter(pass_fingerprints)
    mode_count = fp_counter.most_common(1)[0][1]
    
    variation_ratio = 1 - mode_count / T
```

**Ví dụ**: 10 passes, 7 cho cùng tập relationships, 3 cho tập khác:

```
VR = 1 - 7/10 = 0.3
```

→ 30% passes cho kết quả khác biệt — mức uncertainty trung bình.

## D.7. Uncertainty Reduction Tracking

```python
def compute_uncertainty_reduction(self, current_uncertainties):
    # So sánh với lần đo trước
    reductions = []
    for key, current_val in current_uncertainties.items():
        prev_val = self._previous_uncertainties.get(key, current_val)
        if prev_val > 0:
            reduction = (prev_val - current_val) / prev_val
            reductions.append(reduction)
    
    self._previous_uncertainties = dict(current_uncertainties)
    return mean(reductions)  # ∈ [-1, 1]
```

**Ví dụ**: Epoch 2 uncertainty = 0.7, Epoch 3 uncertainty = 0.5

```
reduction = (0.7 - 0.5) / 0.7 = 0.286
```

→ 28.6% giảm uncertainty — training đang hiệu quả!

---

# PHẦN E: ACTIVE LEARNING SELECTOR

**File**: `RL/active_learning.py`

## E.1. Mục đích

Thay vì sinh đều ảnh cho tất cả relationships (uniform sampling), Active Learning **ưu tiên** relationships mà model yếu nhất → tối đa hóa information gain.

## E.2. Score Relationships — Từng bước

```python
def score_relationships(self, relationships, evaluation_samples, 
                        relationship_performance, tail_weights, ...):
    
    # Bước 1: Estimate uncertainty cho evaluation samples
    uncertainty_results = self.uncertainty_estimator.estimate_batch(
        evaluation_samples, max_samples=15
    )
    # → {0: {uncertainty_score: 0.8, ...}, 1: {uncertainty_score: 0.3, ...}, ...}
    
    # Bước 2: Với mỗi relationship
    for rel in relationships:
        # 2a: Lấy uncertainty score (trung bình từ relevant samples)
        uncertainty_score = self._get_relationship_uncertainty(rel, uncertainty_results, ...)
        
        # 2b: Lấy performance gap
        avg_f1 = relationship_performance[rel_key].get('avg_f1', 0.0)
        performance_score = 1.0 - avg_f1   # F1 thấp → score cao
        
        # 2c: Lấy tail weight
        tail_score = tail_weights.get(relation_name, 0.0)
        
        # 2d: Combined acquisition score
        acquisition_score = 0.40 * uncertainty_score 
                          + 0.35 * performance_score 
                          + 0.25 * min(tail_score, 1.0)
    
    # Bước 3: Sort theo acquisition score (cao → thấp)
    scored_relationships.sort(key=lambda x: x['acquisition_score'], reverse=True)
```

## E.3. Matching Uncertainty → Relationship

```python
def _get_relationship_uncertainty(self, rel_key, rel, uncertainty_results, evaluation_samples):
    relevant_scores = []
    subject = rel.get('subject', '').lower()
    obj = rel.get('object', '').lower()
    
    for sample_idx, unc_metrics in uncertainty_results.items():
        sample = evaluation_samples[sample_idx]
        
        # Kiểm tra sample có liên quan đến relationship không
        orig_rel = sample.get('original_relationship', {})
        if orig_rel.get('subject', '').lower() == subject or \
           orig_rel.get('object', '').lower() == obj:
            relevant_scores.append(unc_metrics.get('uncertainty_score', 0.5))
    
    if relevant_scores:
        return mean(relevant_scores)
    
    # Fallback: trung bình uncertainty tất cả samples
    return mean(all_scores) if all_scores else 0.5
```

## E.4. Budget Allocation — Ví dụ cụ thể

**Input**: 3 relationships, total_budget = 21

| Relationship | α(r) |
|-------------|------|
| person\|riding\|horse | 0.42 |
| dog\|sitting on\|chair | 0.65 |
| person\|holding\|phone | 0.35 |

**Phase 1**: Mỗi rel nhận min_per_rel=1 → `{horse: 1, chair: 1, phone: 1}`, remaining = 18

**Phase 2**: Proportional allocation

```
total_score = 0.42 + 0.65 + 0.35 = 1.42

horse: floor(0.42/1.42 × 18) = floor(5.32) = 5
chair: floor(0.65/1.42 × 18) = floor(8.24) = 8
phone: floor(0.35/1.42 × 18) = floor(4.44) = 4
allocated = 17, remaining = 1
```

**Phase 2b**: Largest fractional part: horse=0.32, chair=0.24, phone=0.44 → phone nhận thêm 1

**Kết quả**: `{horse: 1+5=6, chair: 1+8=9, phone: 1+4+1=6}` → Tổng: 21

→ "dog sitting on chair" nhận **nhiều nhất** (9 ảnh) vì acquisition score cao nhất (model yếu nhất ở relationship này).

---

# PHẦN F: APPROXIMATION ALGORITHM

**File**: `RL/approximation_algorithm.py`

## F.1. Bài toán

Sau khi sinh ảnh (ví dụ 30 ảnh), cần chọn subset tốt nhất (ví dụ 21 ảnh) sao cho: **đa dạng + chất lượng cao + đại diện tốt**.

Đây là bài toán **NP-hard** (tổ hợp C(30,21) = 30 triệu tổ hợp). Greedy Submodular giải xấp xỉ trong O(n²×k).

## F.2. Feature Extraction — 7 chiều

```python
def _extract_sample_features(self, sample):
    features = np.zeros(7)
    
    # [0-1]: Normalized bbox center
    features[0] = mean_center_x / 1000.0
    features[1] = mean_center_y / 1000.0
    
    # [2]: Bbox area ratio
    features[2] = min(mean_area / (1000*1000), 1.0)
    
    # [3]: Relationship type index (hashed)
    features[3] = encode("riding") / 100.0  # ví dụ: 5/100 = 0.05
    
    # [4]: Subject class index
    features[4] = encode("person") / 100.0  # ví dụ: 0/100 = 0.00
    
    # [5]: Object class index  
    features[5] = encode("horse") / 100.0   # ví dụ: 3/100 = 0.03
    
    # [6]: Average annotation confidence
    features[6] = mean_confidence           # ví dụ: 0.85
```

## F.3. Marginal Gain — 3 thành phần

```python
def _compute_marginal_gain(self, current_indices, candidate_idx, all_features, all_samples):
    # 1. DIVERSITY GAIN
    # = min distance từ candidate đến tất cả elements đã chọn trong S
    if S rỗng:
        diversity_gain = 1.0  # Mọi sample đều đa dạng
    else:
        diversity_gain = min(distance(candidate, s) for s in S)
    
    # 2. QUALITY GAIN
    # = trung bình: annotation confidence + backend quality + completeness + has_bbox
    quality_gain = mean(
        avg_object_confidence,    # ví dụ: 0.85
        backend_quality,          # GroundingDINO=0.9, pseudo=0.3
        relationship_completeness, # has subject+relation+object = 1.0
        has_valid_bboxes,         # 0.8 nếu có
    )
    
    # 3. REPRESENTATIVENESS GAIN
    # = tỷ lệ unselected samples mà candidate là nearest neighbor MỚI
    count = 0
    for u in unselected_samples:
        if distance(candidate, u) < min_distance(u, S):
            count += 1
    representativeness_gain = count / len(unselected)
    
    # Weighted sum
    total = 0.5 * diversity + 0.3 * quality + 0.2 * representativeness
```

## F.4. Euclidean Distance (Normalized)

```python
def _compute_distance(features_a, features_b):
    diff = features_a - features_b
    dist = sqrt(sum(diff²))
    return min(dist / 2.65, 1.0)  # Max distance cho 7-dim unit vector ≈ √7 ≈ 2.65
```

## F.5. Greedy Selection — Ví dụ cụ thể

Pool = 5 samples, budget = 3:

```
Step 1: S = {} → Tính marginal gain cho tất cả 5 candidates
        gains = [0.72, 0.68, 0.81, 0.55, 0.79]
        Best: index 2 (gain=0.81)
        S = {2}

Step 2: S = {2} → Tính marginal gain cho 4 remaining
        gains = [0.45, 0.51, -, 0.30, 0.62]
        Best: index 4 (gain=0.62)
        S = {2, 4}

Step 3: S = {2, 4} → Tính marginal gain cho 3 remaining
        gains = [0.38, 0.42, -, -, 0.29]  (diminishing returns!)
        Best: index 1 (gain=0.42)
        S = {2, 4, 1}

→ Selected: [sample_2, sample_4, sample_1]
  Total gain: 0.81 + 0.62 + 0.42 = 1.85
  Guarantee: ≥ 63.2% × OPT
```

## F.6. Early Stopping

```python
if best_gain < 1e-6 and step > effective_budget * 0.5:
    break  # Marginal gain quá nhỏ → dừng sớm
```

Diminishing returns: sau khi chọn đủ nhiều, thêm sample mới gần như không tăng giá trị → dừng sớm tiết kiệm thời gian.

## F.7. Deduplication

```python
def filter_redundant_samples(self, samples, min_distance=0.08):
    kept = [0]  # Luôn giữ sample đầu tiên
    
    for i in range(1, len(samples)):
        is_unique = True
        for j in kept:
            if distance(features[i], features[j]) < 0.08:
                is_unique = False  # Quá giống → loại
                break
        if is_unique:
            kept.append(i)
```

---

# PHẦN G: IMAGE GENERATOR + AUTO-ANNOTATION

## G.1. Stable Diffusion Pipeline

**File**: `RL/ai_images_generator.py`

### Khởi tạo với tối ưu

```python
# Model: runwayml/stable-diffusion-v1-5
self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=float16)

# Tối ưu (CUDA):
pipe.enable_xformers_memory_efficient_attention()  # 2× speedup
pipe.enable_vae_slicing()                           # Giảm VRAM
pipe.unet.to(memory_format=torch.channels_last)    # Better memory layout
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")  # torch.compile
```

### Prompt Engineering

Từ relationship `(person, holding, phone)`:

```
Base: "person holding phone in hands"

Variation 1: "person holding phone in hands, high quality, bright daylight, on the street"
Variation 2: "person holding phone in hands, photorealistic, studio lighting, in a room, candid"
Variation 3: "person holding phone in hands, 4k, golden hour, in a park, during daytime, full shot"
```

Mỗi variation kết hợp random: quality + lighting + background + (70% time/weather) + (50% style) + (40% composition).

### Quality Filter — 5 bước

```
Check 1: Size ≥ 512×512, aspect ≤ 2.2
Check 2: Blur (Laplacian variance) ≥ 60
Check 3: Exposure mean ∈ [20, 235], clip_ratio ≤ 20%
Check 4: Duplicate (pHash) — không trùng ảnh đã sinh
Check 5: CLIP similarity (image↔prompt) ≥ 0.23
```

Nếu fail bất kỳ check nào → ảnh bị reject, không đưa vào training.

## G.2. Auto-Annotation

**File**: `RL/auto_annotator.py`

**Vấn đề**: Stable Diffusion chỉ sinh pixels, KHÔNG có bounding boxes → không thể dùng trực tiếp cho training.

**Giải pháp**: Dùng Open-Vocabulary Detector để tự động detect bbox.

### Thứ tự ưu tiên:

```
1. GroundingDINO (SOTA, open-vocab) — quality = 0.9
   Input: ảnh + text "dog . surfboard"
   Output: boxes, scores, phrases
   
2. OWL-ViT (lighter, HuggingFace) — quality = 0.7
   Input: ảnh + ["a photo of a dog", "a photo of a surfboard"]
   Output: boxes, scores, labels

3. YOLO+CLIP (fallback) — quality = 0.8
   Dùng pipeline detect_objects.py
   
4. Pseudo-Annotation (last resort) — quality = 0.3
   Heuristic: subject bên trái/trên, object bên phải/dưới
   WARNING: Chất lượng rất thấp!
```

### Pseudo-Annotation Heuristics

```python
if relation in ['on', 'above', 'riding']:
    subject_bbox = [30%W, 10%H, 70%W, 45%H]  # Trên
    object_bbox  = [20%W, 50%H, 80%W, 90%H]  # Dưới
elif relation in ['holding', 'carrying']:
    subject_bbox = [20%W, 10%H, 70%W, 90%H]  # Lớn
    object_bbox  = [50%W, 30%H, 80%W, 60%H]  # Nhỏ, gần subject
else:  # default
    subject_bbox = [5%W, 20%H, 45%W, 80%H]   # Bên trái
    object_bbox  = [55%W, 20%H, 95%W, 80%H]  # Bên phải
```

---

# PHẦN H: REWARD FUNCTION CHI TIẾT

**File**: `RL/reinforcement_learning.py`, method `calculate_reward()`

## H.1. Overview: 6 Thành Phần + Dynamic Weights

```
R_total = w_det × s_det + w_rel × s_rel + w_div × s_div 
        + w_con × s_con + w_imp × s_imp + w_unc × s_unc

→ R_scaled = sigmoid(scaling_factor × (R_total − 0.5))
```

## H.2. Dynamic Weights — Cơ chế thích nghi

```python
def _calculate_dynamic_weights(self, det_score, rel_score, div_score, con_score):
    base_weights = {
        'detection': 0.25,      # Quan trọng
        'relationship': 0.45,   # Quan trọng nhất
        'diversity': 0.15,
        'consistency': 0.10,
        'improvement': 0.05,
    }
    
    # Tính deviation so với baseline
    det_deviation = abs(det_score - baseline['detection'])
    rel_deviation = abs(rel_score - baseline['relationship'])
    ...
    
    # Thành phần nào kém → tăng weight
    adjustment_factor = 0.2
    adjusted = {
        'detection': 0.25 + 0.2 × det_deviation,
        'relationship': 0.45 + 0.2 × rel_deviation,
        ...
    }
    
    # Normalize tổng = 1.0
    total = sum(adjusted.values())
    normalized = {k: v/total for k, v in adjusted.items()}
```

Sau đó, **uncertainty_reduction** nhận cố định **10%**, các weight còn lại scale xuống:

```python
scale_factor = 0.9 / sum(existing_weights)
for k in existing_weights:
    existing_weights[k] *= scale_factor
weights['uncertainty_reduction'] = 0.10
```

**Ví dụ số:**

Baseline: det=0.3, rel=0.7. Hiện tại: det=0.1, rel=0.8.

```
det_deviation = |0.1 - 0.3| = 0.2   → Detection kém xa baseline
rel_deviation = |0.8 - 0.7| = 0.1   → Relationship tốt hơn baseline

adjusted_det = 0.25 + 0.2×0.2 = 0.29   (tăng)
adjusted_rel = 0.45 + 0.2×0.1 = 0.47   (tăng ít hơn)

→ w_det tăng vì detection đang kém → hệ thống tập trung cải thiện detection
```

## H.3. Baseline Performance — EWMA Update

```python
# Sau mỗi epoch, cập nhật baseline dùng EWMA (α=0.3)
baseline['detection'] = 0.3 × recent_avg + 0.7 × previous_baseline
```

**EWMA (Exponentially Weighted Moving Average)**: Ưu tiên kinh nghiệm gần đây nhưng vẫn nhớ lịch sử → baseline thích nghi dần dần, không nhảy đột ngột.

## H.4. Reward Scaling — Sigmoid

```python
def _apply_reward_scaling(self, raw_reward):
    return 1.0 / (1.0 + exp(-scaling_factor × (raw_reward - 0.5)))
```

**Tại sao sigmoid?**
- Map reward về khoảng (0, 1)
- Reward quanh 0.5 → sensitivity cao (phân biệt tốt)
- Reward cực đoan → bão hòa (tránh extreme values)

**Scaling factor** thích nghi:

```python
scaling_factor = 1.0 + (1.0 - min(reward_variance, 1.0))
```

- Variance thấp (performance ổn định) → scaling_factor cao → sigmoid dốc hơn → phân biệt rõ hơn
- Variance cao (performance dao động) → scaling_factor thấp → sigmoid mềm hơn → robust hơn

## H.5. Spatial Diversity — 3 thuật toán con

### Position Diversity (Spatial Clustering Analysis)

```python
# Normalize center points
for bbox in bboxes:
    center_x = (x1+x2)/2 / avg_width
    center_y = (y1+y2)/2 / avg_height

# Tính variance
x_variance = Σ(x - x_mean)² / n
y_variance = Σ(y - y_mean)² / n

position_diversity = tanh((x_variance + y_variance) × 4)
```

**Ví dụ**: Tất cả objects ở góc trái trên → variance thấp → diversity thấp. Objects rải đều ảnh → variance cao → diversity cao.

### Size Diversity (Coefficient of Variation)

```python
# CV = std / mean
area_cv = std(areas) / mean(areas)
aspect_ratio_cv = std(aspect_ratios) / mean(aspect_ratios)

size_diversity = 0.6 × min(area_cv, 2)/2 + 0.4 × min(ar_cv, 3)/3
```

### Coverage Diversity (Grid-Based + Entropy)

```python
# Chia ảnh thành grid 4×4 = 16 cells
# Đếm cells được cover bởi ít nhất 1 bbox
coverage_ratio = len(covered_cells) / 16

# Shannon Entropy — distribution đều hay lệch?
for cell in cell_counts:
    p = count / total_bboxes
    entropy -= p × log2(p)
normalized_entropy = entropy / log2(16)

coverage_diversity = 0.7 × coverage_ratio + 0.3 × normalized_entropy
```

**Ví dụ**: 10 bboxes cover 12/16 cells, entropy=3.2:

```
coverage_ratio = 12/16 = 0.75
normalized_entropy = 3.2 / 4.0 = 0.80
coverage_diversity = 0.7×0.75 + 0.3×0.80 = 0.525 + 0.240 = 0.765
```

---

# PHẦN I: TRAINING EPISODE — TỪNG BƯỚC

**File**: `RL/reinforcement_learning.py`, method `train_episode()`

## Bước 1: Sinh ảnh (hoặc dùng ảnh đã sinh)

```python
# Nếu có relationship_plan từ Active Learning:
for rel in original_relationships:
    num_variations = relationship_plan.get(rel_key, default)
    generated_images = generator.generate_from_relationship(rel, num_variations)
    synthetic_data.extend(generated_images)
```

## Bước 2: Approximation Algorithm — Subset Selection

```python
if len(synthetic_data) > 3:
    # 2a: Loại bỏ duplicates
    filtered = subset_selector.filter_redundant_samples(synthetic_data, min_distance=0.08)
    
    # 2b: Chọn 70% pool (budget)
    budget = max(3, int(len(filtered) × 0.7))
    if len(filtered) > budget:
        synthetic_data, stats = subset_selector.select_optimal_subset(filtered, budget)
```

## Bước 3: Ingest Synthetic Samples

```python
ingested = self._ingest_synthetic_samples(synthetic_data)
# Với mỗi sample:
#   1. Lưu ảnh ra disk (nếu là PIL Image)
#   2. Auto-annotate (GroundingDINO → OWL-ViT → YOLO+CLIP → Pseudo)
#   3. Build relationships (original → RelTR inference → fallback)
#   4. Thêm vào dataset_samples
#   5. Recompute tail_weights
```

## Bước 4: Train Detection Model (YOLO Fine-tune)

```python
def train_detection_model(self, synthetic_data):
    dataset_dir = self._prepare_detection_dataset()
    # Tạo YOLO dataset structure: images/train, labels/train, dataset.yaml
    
    epochs = max(1, min(5, num_trained_epochs + 1))  # 1-5 epochs
    detection_model.train(
        data=dataset.yaml,
        epochs=epochs,
        imgsz=640,
        batch=4,
        workers=0,
    )
```

## Bước 5: Train Relationship Model (RelTR Fine-tune)

```python
def train_relationship_model(self, synthetic_data, num_epochs=1):
    model, criterion = self._ensure_relationship_model()
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        for image_tensor, target, global_context in prepared_samples:
            # Forward
            outputs = model(samples, global_context=context)
            
            # Loss = weighted sum of criterion components
            loss_dict = criterion(outputs, targets)
            loss = Σ loss_dict[k] × weight_dict[k]
            
            # Long-tail weighted loss
            tail_weight = get_sample_tail_weight(target)
            total_tail_loss += loss × tail_weight
            
            loss.backward()
        
        optimizer.step()
```

## Bước 6: Calculate Reward (đã giải thích ở Phần H)

## Bước 7: Update Epsilon

```python
self.epsilon = max(0.01, self.epsilon × 0.995)
```

## Bước 8: Save Best Model

```python
if reward > self.training_history['best_reward']:
    self.save_model_state(reward, detection_loss, relationship_loss)
    # Lưu: detection_model, relationship_model, q_network, target_network,
    #       q_optimizer, reltr_optimizer, epsilon, training_history
```

## Bước 9: DQN Update

```python
rl_loss = self._finalize_rl_step(
    state, action_index, reward, detection_loss, relationship_loss,
    done=done, detection_metrics=..., relationship_metrics=...
)
# → Build next_state → Remember → Optimize Q-network → Sync target
```

---

# PHẦN J: APPLICATION LAYER

**File**: `app.py`, `RL/rl_enhancement.py`

## J.1. Class `ObjectDetectionApp` — Main GUI

**Khởi tạo:**

```python
self.rl_enhancement = AppReinforcementLearning(self)
self.training_evaluator = TrainingEvaluator()
self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Sentence similarity
```

## J.2. Detection Pipeline Flow (khi bấm "Detect Objects")

```
1. run_pipeline_thread() → tạo thread mới
2. run_pipeline():
   a. subprocess: python detect_objects.py image_path
   b. subprocess: python convert_yolo_to_reltr.py result.json
   c. subprocess: python boundingbox_objects.py --yolo_json ... --img_path ... --resume checkpoint.pth
   d. Hiển thị ảnh kết quả
   e. Load JSON → hiển thị objects + relationships
   f. Vẽ bbox + mũi tên trên ảnh
```

## J.3. RL Training Flow (khi bấm "RL Training")

```
1. Dialog chọn dataset (Yes/No/Cancel)
2. Nếu Yes: chọn thư mục ảnh → build_dataset_from_directory()
3. setup_reinforcement_learning() — nếu chưa init
4. run_reinforcement_learning(epochs=5):
   Với mỗi epoch:
   a. decide_action() → DQN + Active Learning
   b. generate_ai_images_for_epoch() → Stable Diffusion
   c. experiment_manager.save_ai_images()
   d. train_episode() → [Phần I ở trên]
   e. Lưu metrics, progress
5. Tạo plots + grid ảnh
6. Finalize experiment
```

## J.4. Class `AppReinforcementLearning` — Tầng trung gian

**Vai trò**: Bridge giữa GUI (`app.py`) và RL core (`reinforcement_learning.py`).

```python
class AppReinforcementLearning:
    def __init__(self, app_instance):
        self.app = app_instance
        self.generator = RelationshipImageGenerator()     # Stable Diffusion
        self.augmentation = RelationshipDataAugmentation() # Data augmentation
        self.rl_system = None                              # Lazy init
        self.experiment_manager = ExperimentManager()      # Quản lý experiments
        self.experience_manager = ExperienceManager()      # Replay buffer
```

**setup_reinforcement_learning()**: Đọc `relationships.json` → khởi tạo `RelationshipReinforcementLearning` với paths tới YOLO weights, RelTR checkpoint, images, JSON files.

**run_reinforcement_learning(epochs=5)**: Vòng lặp chính gọi `decide_action()` → `generate_ai_images_for_epoch()` → `train_episode()` → lưu metrics.

---

# TỔNG KẾT: BẢNG SO SÁNH CÁC THÀNH PHẦN

| Thành phần | Độ phức tạp | Input | Output | Thuật toán chính |
|-----------|-------------|-------|--------|-----------------|
| Detection | O(n) per image | Ảnh | Objects + bbox + features | YOLO + NMS + CLIP + RoIAlign |
| Relationship | O(n²) per image | Objects + features | Relationship triplets | Transformer + IoU matching |
| DQN Agent | O(1) per action | State (5D) | Action (1-10) | Epsilon-greedy + Bellman |
| Uncertainty | O(T×n) per image | Image + model | Uncertainty scores | MC Dropout (T=10 passes) |
| Active Learning | O(n×m) | Relationships + uncertainty | Generation plan | Acquisition function + proportional allocation |
| Approx. Algorithm | O(n²×k) | Candidate pool | Optimal subset | Greedy submodular (63.2% OPT) |
| Image Gen | O(25 steps) per image | Text prompt | PIL Image | Stable Diffusion v1.5 |
| Auto-Annotate | O(n) per image | Image + text prompts | Objects + bbox | GroundingDINO / OWL-ViT |
| Reward | O(n×m) | All data | Scalar reward | 6-component adaptive + sigmoid |

| Thành phần | Math nặng nhất | Guarantee |
|-----------|---------------|-----------|
| MC Dropout | Predictive Entropy, BALD | Approximate Bayesian inference |
| Active Learning | Acquisition Function | Maximize information gain |
| Greedy Submodular | Marginal gain optimization | f(S) ≥ (1-1/e)×OPT |
| DQN | Bellman equation | Convergence to Q* (with conditions) |
| Dynamic Weights | EWMA baseline + sigmoid scaling | Adaptive to performance changes |
