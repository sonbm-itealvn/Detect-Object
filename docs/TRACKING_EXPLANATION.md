# Giải thích Thuật toán Tracking trong Pipeline

## Tổng quan

Pipeline sử dụng **ByteTrack** - một thuật toán tracking đa object (MOT - Multi-Object Tracking) được tích hợp sẵn trong YOLOv11/Ultralytics.

## 1. Thuật toán cốt lõi: ByteTrack

### 1.1. Kiến trúc ByteTrack

ByteTrack là một thuật toán tracking **không cần re-identification model**, chỉ dựa vào:
- **Detection confidence**
- **IoU (Intersection over Union)** matching
- **Kalman Filter** để dự đoán vị trí

### 1.2. Quy trình Tracking

```
Frame t-1                    Frame t
  ↓                            ↓
[Track 1] ──IoU Match──> [Detection A] → Track 1 (updated)
[Track 2] ──IoU Match──> [Detection B] → Track 2 (updated)
[Track 3] ──No Match──>   [Detection C] → Track 3 (lost)
                          [Detection D] → New Track 4
```

### 1.3. Các bước chính

1. **Detection**: YOLO phát hiện objects trong frame hiện tại
2. **Prediction**: Kalman Filter dự đoán vị trí của tracks từ frame trước
3. **Matching**: So khớp detections với predicted tracks bằng IoU
4. **Update**: Cập nhật tracks với detections khớp
5. **Track Management**: Tạo tracks mới, xóa tracks mất

## 2. Phương trình Dự đoán Chuyển động: Kalman Filter

### 2.1. State Vector

Kalman Filter mô hình hóa object bằng **state vector** 8 chiều:

```
x = [cx, cy, s, r, vx, vy, vs, vr]ᵀ

Trong đó:
- cx, cy: Tọa độ center của bounding box
- s: Scale (diện tích = width × height)
- r: Aspect ratio (width/height)
- vx, vy: Vận tốc theo x, y
- vs: Vận tốc scale
- vr: Vận tốc aspect ratio
```

### 2.2. Phương trình Dự đoán (Prediction Step)

**State Prediction:**
```
x̂ₖ|ₖ₋₁ = F × x̂ₖ₋₁|ₖ₋₁
```

**Covariance Prediction:**
```
Pₖ|ₖ₋₁ = F × Pₖ₋₁|ₖ₋₁ × Fᵀ + Q
```

**Trong đó:**
- `F`: State transition matrix (mô hình chuyển động)
- `Q`: Process noise covariance (nhiễu hệ thống)
- `P`: Error covariance matrix

**State Transition Matrix F:**
```
F = [1  0  0  0  dt  0   0   0 ]   ← Constant Velocity Model
    [0  1  0  0  0   dt  0   0 ]
    [0  0  1  0  0   0   dt  0 ]
    [0  0  0  1  0   0   0   dt]
    [0  0  0  0  1   0   0   0 ]
    [0  0  0  0  0   1   0   0 ]
    [0  0  0  0  0   0   1   0 ]
    [0  0  0  0  0   0   0   1 ]
```

### 2.3. Phương trình Cập nhật (Update Step)

**Kalman Gain:**
```
K = Pₖ|ₖ₋₁ × Hᵀ × (H × Pₖ|ₖ₋₁ × Hᵀ + R)⁻¹
```

**State Update:**
```
x̂ₖ|ₖ = x̂ₖ|ₖ₋₁ + K × (zₖ - H × x̂ₖ|ₖ₋₁)
```

**Covariance Update:**
```
Pₖ|ₖ = (I - K × H) × Pₖ|ₖ₋₁
```

**Trong đó:**
- `H`: Observation matrix (chuyển state → measurement)
- `R`: Measurement noise covariance
- `zₖ`: Measurement từ detection (cx, cy, s, r)
- `I`: Identity matrix

### 2.4. Observation Matrix H

```
H = [1  0  0  0  0  0  0  0]   ← Chỉ quan sát được position, không quan sát velocity
    [0  1  0  0  0  0  0  0]
    [0  0  1  0  0  0  0  0]
    [0  0  0  1  0  0  0  0]
```

## 3. Tại sao có thể Tracking được?

### 3.1. Temporal Consistency (Tính nhất quán theo thời gian)

**Nguyên lý**: Objects di chuyển liên tục, không nhảy cóc giữa các frame.

```
Frame t-1:  [Object A] tại (100, 200)
Frame t:    [Object A] tại (105, 200)  ← Di chuyển 5 pixels
Frame t+1:  [Object A] tại (110, 200)  ← Di chuyển tiếp 5 pixels
```

Kalman Filter **dự đoán** vị trí tiếp theo dựa trên vận tốc hiện tại.

### 3.2. IoU Matching (So khớp bằng IoU)

**IoU (Intersection over Union)** đo độ trùng lặp giữa 2 bounding boxes:

```
IoU = Area(Intersection) / Area(Union)
```

**Matching Strategy:**
1. Tính IoU giữa **predicted track** và **detection**
2. Nếu IoU > threshold (thường 0.3-0.5) → Match
3. Sử dụng **Hungarian Algorithm** để tối ưu matching

### 3.3. Track Management

**Track States:**
- **Tracked**: Track đang được theo dõi (matched với detection)
- **Lost**: Track mất (không match trong N frames)
- **Removed**: Track bị xóa (lost quá lâu)

**ByteTrack đặc biệt:**
- Xử lý cả **high-confidence** và **low-confidence** detections
- Low-confidence detections có thể match với lost tracks
- Giảm ID switches (đổi ID nhầm)

## 4. Implementation trong Pipeline

### 4.1. Code Flow

```python
# video_relation_pipeline.py, dòng 415-421
stream = self.yolo_model.track(
    source=video_path,
    tracker=self.tracker_config,  # "bytetrack.yaml"
    stream=True,
    persist=True,  # ← Quan trọng: giữ track IDs qua các frame
    verbose=False,
)

# Mỗi frame, result chứa:
# - result.boxes.xyxy: Bounding boxes
# - result.boxes.id: Track IDs (từ Kalman Filter + Matching)
# - result.boxes.conf: Confidence scores
```

### 4.2. Track ID Assignment

```python
# video_relation_pipeline.py, dòng 552-556
track_ids = []
if result.boxes.id is not None:
    track_ids = result.boxes.id.int().cpu().tolist()
else:
    track_ids = [None] * len(boxes)
```

**Track ID được gán tự động bởi ByteTrack:**
- Object mới → ID mới (tăng dần: 1, 2, 3, ...)
- Object cũ → Giữ nguyên ID
- Object mất → ID được giữ trong N frames (thường 30-60 frames)

### 4.3. Sử dụng Track ID trong Relationships

```python
# video_relation_pipeline.py, dòng 329-330
'subject_track_id': objects[int(subj_idx)].get('track_id'),
'object_track_id': objects[int(obj_idx)].get('track_id'),
```

**Lợi ích:**
- Relationships có thể được **theo dõi qua các frame**
- Tránh nhầm lẫn khi có nhiều objects cùng class
- Hỗ trợ temporal analysis (phân tích theo thời gian)

## 5. Tại sao ByteTrack hiệu quả?

### 5.1. Không cần Re-ID Model

- **DeepSORT** cần Re-ID model (tốn tài nguyên)
- **ByteTrack** chỉ dùng IoU + Kalman Filter (nhanh, nhẹ)

### 5.2. Xử lý Low-Confidence Detections

- Detections confidence thấp vẫn được dùng để match với lost tracks
- Giảm false negatives (bỏ sót objects)

### 5.3. Simple but Effective

- Thuật toán đơn giản, dễ implement
- Hiệu quả cao trên nhiều datasets (MOT17, MOT20)

## 6. Ví dụ Minh họa

### Scenario: Tracking một người đi bộ

```
Frame 1:
  Detection: [person] tại (100, 200), confidence=0.9
  → Tạo Track ID=1, state=[100, 200, 10000, 0.5, 0, 0, 0, 0]

Frame 2:
  Kalman Prediction: Track 1 → (105, 200)  ← Dự đoán dựa trên vận tốc
  Detection: [person] tại (106, 200), confidence=0.85
  IoU(Track 1, Detection) = 0.92 > 0.5 → Match!
  → Update Track 1: state=[106, 200, 10000, 0.5, 6, 0, 0, 0]
  → Track ID=1 (giữ nguyên)

Frame 3:
  Kalman Prediction: Track 1 → (112, 200)
  Detection: [person] tại (110, 200), confidence=0.8
  IoU(Track 1, Detection) = 0.88 > 0.5 → Match!
  → Update Track 1: state=[110, 200, 10000, 0.5, 4, 0, 0, 0]
  → Track ID=1 (giữ nguyên)

Frame 4:
  Kalman Prediction: Track 1 → (114, 200)
  Detection: [person] tại (115, 200), confidence=0.3  ← Low confidence!
  IoU(Track 1, Detection) = 0.85 > 0.5 → Match! (ByteTrack vẫn match)
  → Update Track 1: state=[115, 200, 10000, 0.5, 5, 0, 0, 0]
  → Track ID=1 (giữ nguyên)
```

## 7. Kết luận

**Tracking hoạt động được vì:**

1. **Kalman Filter** dự đoán vị trí tiếp theo dựa trên chuyển động
2. **IoU Matching** so khớp detections với predicted tracks
3. **Temporal Consistency** - objects không nhảy cóc giữa frames
4. **Track Management** - quản lý vòng đời của tracks

**Trong pipeline:**
- YOLO phát hiện objects
- ByteTrack gán và duy trì Track IDs
- Track IDs được dùng để liên kết relationships qua các frame
- Hỗ trợ phân tích temporal và safety monitoring

---

**Tài liệu tham khảo:**
- ByteTrack Paper: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
- Kalman Filter: "An Introduction to the Kalman Filter" (Greg Welch & Gary Bishop)
- Ultralytics YOLO Tracking: https://docs.ultralytics.com/modes/track/

