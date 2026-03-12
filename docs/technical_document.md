# TÀI LIỆU KỸ THUẬT CHI TIẾT — Hệ Thống Phát Hiện Quan Hệ Thị Giác (VRD)

> Tài liệu trình bày **đầy đủ các công thức toán học** và **giải thích chi tiết từng thành phần** trong toàn bộ hệ thống Visual Relationship Detection Pipeline, bao gồm Inference Pipeline và Data Loop (Reinforcement Learning).

---

## MỤC LỤC

1. [Bảng Ký Hiệu](#1-bảng-ký-hiệu)
2. [Object Detection — YOLOv11](#2-object-detection--yolov11)
3. [Zero-shot Classification — CLIP](#3-zero-shot-classification--clip)
4. [ROI Feature Extraction — RoIAlign](#4-roi-feature-extraction--roialign)
5. [Scene Graph Generation — RelTR](#5-scene-graph-generation--reltr)
6. [Spatial-Semantic Validation](#6-spatial-semantic-validation)
7. [LLM Open-Vocabulary Enhancement](#7-llm-open-vocabulary-enhancement)
8. [Video Processing Pipeline](#8-video-processing-pipeline)
9. [Safety Classification System](#9-safety-classification-system)
10. [Deep Q-Network (DQN) Agent](#10-deep-q-network-dqn-agent)
11. [Adaptive Reward Function](#11-adaptive-reward-function)
12. [MC Dropout Uncertainty Estimation](#12-mc-dropout-uncertainty-estimation)
13. [Active Learning Acquisition](#13-active-learning-acquisition)
14. [Greedy Submodular Maximization](#14-greedy-submodular-maximization)
15. [Synthetic Data Generation & Auto-Annotation](#15-synthetic-data-generation--auto-annotation)

---

## 1. Bảng Ký Hiệu

| Ký hiệu | Ý nghĩa | Miền giá trị |
|---|---|---|
| $I$ | Ảnh đầu vào | $\mathbb{R}^{H \times W \times 3}$ |
| $\mathcal{O} = \{o_i\}_{i=1}^{N}$ | Tập đối tượng phát hiện được | $o_i = (b_i, c_i, s_i)$ |
| $b_i = (x_1, y_1, x_2, y_2)$ | Bounding box tọa độ pixel | $\mathbb{R}^4$ |
| $c_i$ | Nhãn lớp (class label) | $\{1, \ldots, C\}$ |
| $s_i$ | Điểm tin cậy (confidence score) | $[0, 1]$ |
| $\mathcal{R} = \{r_k\}$ | Tập quan hệ (relationships) | $r_k = (s_k, p_k, o_k, \sigma_k)$ |
| $(s_k, p_k, o_k)$ | Bộ ba: (chủ thể, vị từ, đối tượng) | |
| $\sigma_k$ | Điểm tin cậy của quan hệ | $[0, 1]$ |
| $F \in \mathbb{R}^{C' \times H' \times W'}$ | Feature map từ backbone | |
| $g \in \mathbb{R}^{C'}$ | Vector ngữ cảnh toàn cục | |
| $d_{\text{model}}$ | Chiều ẩn (hidden dimension) = 256 | $\mathbb{N}$ |
| $Q(s, a; \theta)$ | Hàm giá trị Q | $\mathbb{R}$ |
| $\varepsilon$ | Tỷ lệ khám phá (exploration rate) | $[0, 1]$ |
| $\gamma$ | Hệ số chiết khấu (discount factor) = 0.95 | $(0, 1)$ |
| $U(r)$ | Điểm bất định (uncertainty score) | $[0, 1]$ |
| $f(S)$ | Hàm mục tiêu submodular | $\mathbb{R}_{\geq 0}$ |

---

## 2. Object Detection — YOLOv11

**File nguồn:** `detect_objects.py`

### 2.1 Hàm phát hiện đối tượng

$$\mathcal{D}: \mathbb{R}^{H \times W \times 3} \rightarrow \mathcal{P}(\mathcal{B} \times \mathcal{C} \times [0,1])$$

**Giải thích từng thành phần:**
- $\mathbb{R}^{H \times W \times 3}$: Không gian ảnh đầu vào có chiều cao $H$, chiều rộng $W$, 3 kênh màu (RGB)
- $\mathcal{B} = \{(x_1, y_1, x_2, y_2) \in \mathbb{R}^4 : x_1 < x_2, y_1 < y_2\}$: Không gian bounding box hợp lệ (góc trên-trái phải nhỏ hơn góc dưới-phải)
- $\mathcal{C} = \{1, \ldots, C\}$: Tập nhãn lớp ($C = 80$ cho COCO)
- $\mathcal{P}(\cdot)$: Tập lũy thừa (power set) — kết quả là tập hợp con của tất cả bộ ba khả thi

### 2.2 Kiến trúc Feature Pyramid

**Neck (FPN + PAN)** kết hợp features đa tầng:

$$F_{\text{fpn}}^l = \text{Conv}(\text{Upsample}(F^{l+1}) \oplus F^l), \quad l \in \{3, 4, 5\}$$

**Giải thích:**
- $F^l$: Feature map tại tầng (scale) $l$ từ backbone
- $\text{Upsample}(F^{l+1})$: Tăng kích thước spatial của feature map tầng trên (resolution thấp hơn) lên gấp đôi
- $\oplus$: Phép nối kênh (channel concatenation)
- $\text{Conv}(\cdot)$: Tích chập để hòa trộn thông tin từ 2 tầng
- Mục đích: Kết hợp thông tin ngữ nghĩa cấp cao (từ tầng sâu) với thông tin vị trí chi tiết (từ tầng nông)

### 2.3 Giải mã Bounding Box (Anchor-Free)

Tại mỗi vị trí grid $(i, j)$ ở scale $l$, head dự đoán vector:

$$\hat{y}_{i,j,l} = (\hat{t}_x, \hat{t}_y, \hat{t}_w, \hat{t}_h, \hat{p}_{\text{obj}}, \hat{p}_{c_1}, \ldots, \hat{p}_{c_C})$$

**Giải mã tọa độ:**

$$b_x = 2\sigma(\hat{t}_x) - 0.5 + c_x$$
$$b_y = 2\sigma(\hat{t}_y) - 0.5 + c_y$$
$$b_w = (2\sigma(\hat{t}_w))^2 \cdot a_w$$
$$b_h = (2\sigma(\hat{t}_h))^2 \cdot a_h$$

**Giải thích từng thành phần:**
- $\hat{t}_x, \hat{t}_y$: Giá trị thô (raw predictions) cho tọa độ tâm
- $\sigma(\cdot)$: Hàm sigmoid $\sigma(x) = \frac{1}{1+e^{-x}}$, nén giá trị về $(0, 1)$
- $c_x, c_y$: Offset của grid cell (vị trí góc trên-trái của ô lưới)
- $2\sigma(\hat{t}_x) - 0.5$: Cho phép tâm dịch chuyển trong khoảng $(-0.5, 1.5)$ so với grid cell → linh hoạt hơn
- $a_w, a_h$: Kích thước anchor (mặc định cho scale tương ứng)
- $(2\sigma(\hat{t}_w))^2$: Bình phương để đảm bảo tỷ lệ kích thước luôn dương

**Điểm tin cậy cuối cùng:**

$$s_{\text{final}}(i, j, l) = \sigma(\hat{p}_{\text{obj}}) \cdot \max_{c \in \mathcal{C}} \sigma(\hat{p}_c)$$

- $\sigma(\hat{p}_{\text{obj}})$: Xác suất có vật thể tại vị trí đó
- $\max_c \sigma(\hat{p}_c)$: Xác suất cao nhất trong tất cả lớp
- Nhân hai giá trị: Chỉ tin cậy cao khi **cả** objectness **và** class probability đều cao

### 2.4 Intersection over Union (IoU)

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}$$

**Giải thích:**
- $|A \cap B|$: Diện tích vùng giao nhau giữa hai box
- $|A \cup B|$: Diện tích vùng hợp nhất (loại bỏ phần trùng)
- Thuộc tính: $0 \leq \text{IoU} \leq 1$; IoU = 1 khi hai box hoàn toàn trùng khớp; IoU = 0 khi không giao nhau

### 2.5 Non-Maximum Suppression (NMS)

Loại bỏ các detection trùng lặp. Với ngưỡng $\theta_{\text{nms}} = 0.45$:

1. Sắp xếp detections theo $s_i$ giảm dần
2. Chọn detection có score cao nhất $d^*$, thêm vào kết quả
3. Loại bỏ mọi detection $d$ cùng class mà $\text{IoU}(\text{box}(d), \text{box}(d^*)) > \theta_{\text{nms}}$
4. Lặp lại cho đến khi hết detections

**Ý nghĩa:** Giữ lại detection tốt nhất cho mỗi vật thể, loại bỏ các box gần trùng.

### 2.6 Dual-Model Detection

$$\mathcal{D}_{\text{merged}}(I) = \mathcal{D}_{\text{COCO}}(I) \cup \mathcal{D}_{\text{fire}}(I)$$

- $\mathcal{D}_{\text{COCO}}$: Model phát hiện 80 lớp COCO (người, xe, đồ vật...)
- $\mathcal{D}_{\text{fire}}$: Model chuyên biệt phát hiện lửa/khói (2 lớp)
- Kết quả merge qua NMS, ưu tiên fire detection khi overlap (an toàn trước)

### 2.7 YOLO Loss Function

$$\mathcal{L}_{\text{YOLO}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$

**CIoU Loss** (Complete IoU):

$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

$$v = \frac{4}{\pi^2}\left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2, \quad \alpha = \frac{v}{(1 - \text{IoU}) + v}$$

**Giải thích từng thành phần:**
- $1 - \text{IoU}$: Phạt khi IoU thấp (box không trùng)
- $\rho^2(b, b^{gt})$: Khoảng cách Euclidean bình phương giữa tâm box dự đoán và ground-truth
- $c^2$: Bình phương đường chéo của smallest enclosing box (hộp nhỏ nhất chứa cả 2 box)
- $\frac{\rho^2}{c^2}$: Phạt khi tâm 2 box cách xa nhau
- $v$: Đo sai lệch tỷ lệ khung hình (aspect ratio) giữa prediction và ground-truth
- $\alpha$: Trọng số thích nghi — khi IoU thấp, $\alpha$ nhỏ (tập trung sửa vị trí trước); khi IoU cao, $\alpha$ tăng (tinh chỉnh aspect ratio)

---

## 3. Zero-shot Classification — CLIP

**File nguồn:** `detect_objects.py` → `classify_with_clip()`

### 3.1 Kiến trúc CLIP Dual Encoder

- Image encoder: $f_I: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^d$ (ViT-B/32, $d = 512$)
- Text encoder: $f_T: \Sigma^* \rightarrow \mathbb{R}^d$ (Transformer, $d = 512$)

**Giải thích:** Hai encoder ánh xạ ảnh và văn bản vào **cùng một không gian vector 512 chiều**, nơi các cặp ảnh-văn bản liên quan có vector gần nhau.

### 3.2 Cosine Similarity

$$\text{sim}(I, T) = \frac{f_I(I)^\top f_T(T)}{\|f_I(I)\|_2 \cdot \|f_T(T)\|_2}$$

**Giải thích:**
- $f_I(I)^\top f_T(T)$: Tích vô hướng (dot product) giữa 2 vector embedding
- $\|f_I(I)\|_2$: Chuẩn L2 (độ dài) của vector ảnh
- Chia cho tích 2 chuẩn: Chuẩn hóa về $[-1, 1]$, chỉ đo góc giữa 2 vector
- $\text{sim} = 1$: Hoàn toàn giống nhau; $\text{sim} = 0$: Trực giao (không liên quan)

### 3.3 Softmax Temperature Scaling

$$p(c_k | x_{\text{roi}}) = \frac{\exp(s_k / \tau)}{\sum_{j=1}^{K} \exp(s_j / \tau)}$$

**Giải thích:**
- $s_k$: Cosine similarity giữa ảnh ROI và text prompt của lớp $k$
- $\tau$: Tham số nhiệt độ (temperature) — $\tau$ nhỏ → phân phối nhọn (tập trung); $\tau$ lớn → phân phối phẳng (đều)
- $\exp(\cdot)$: Hàm mũ, đảm bảo giá trị dương
- Mẫu số: Tổng tất cả $\exp(s_j / \tau)$, đảm bảo $\sum_k p(c_k) = 1$ (phân phối xác suất hợp lệ)

### 3.4 Contrastive Pre-training Loss

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_i, T_j)/\tau)} + \log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_j, T_i)/\tau)}\right]$$

**Giải thích:**
- $N$: Kích thước batch (số cặp ảnh-văn bản)
- Phần 1 (image→text): Với mỗi ảnh $I_i$, tối đa hóa similarity với text đúng $T_i$ so với tất cả text khác $T_j$
- Phần 2 (text→image): Đối xứng — với mỗi text, tối đa similarity với ảnh đúng
- $-\log$: Cross-entropy loss, phạt khi cặp đúng có similarity thấp

---

## 4. ROI Feature Extraction — RoIAlign

**File nguồn:** `detect_objects.py` → `_extract_roi_features()`

### 4.1 Feature Map Hooking

Hook vào SPPF layer (layer 9) của YOLO backbone:

$$F \in \mathbb{R}^{C' \times H' \times W'}, \quad C' = 512, \quad H' = H/32, \quad W' = W/32$$

**Giải thích:** Feature map $F$ là biểu diễn nén của ảnh gốc, mỗi pixel trên $F$ "nhìn" vùng $32 \times 32$ pixel trên ảnh gốc. 512 kênh mang thông tin ngữ nghĩa phong phú.

### 4.2 RoIAlign — Bilinear Interpolation

**Bước 1.** Scale bounding box sang feature map:

$$b'_x = b_x \cdot \frac{W'}{W}, \quad b'_y = b_y \cdot \frac{H'}{H}$$

**Bước 2.** Chia ROI thành $k \times k$ bins ($k = 7$):

$$\Delta_x = \frac{x_2' - x_1'}{k}, \quad \Delta_y = \frac{y_2' - y_1'}{k}$$

**Bước 3.** Bilinear interpolation tại mỗi điểm sample $(x, y)$:

$$F(x, y) = \sum_{(i,j) \in \mathcal{N}(x,y)} F[i, j] \cdot \max(0, 1 - |x - i|) \cdot \max(0, 1 - |y - j|)$$

**Giải thích:**
- $\mathcal{N}(x,y)$: 4 pixel láng giềng gần nhất trên feature map
- $\max(0, 1 - |x - i|)$: Trọng số giảm tuyến tính theo khoảng cách — pixel gần hơn đóng góp nhiều hơn
- **Ưu điểm so với RoIPool:** Không làm tròn (quantize) tọa độ → không mất thông tin vị trí (sub-pixel accuracy)

**Bước 4.** Average pooling → output $v_{\text{roi}} \in \mathbb{R}^{C' \times k \times k}$

### 4.3 Global Context Vector

$$g = \text{GAP}(F) = \frac{1}{H' \times W'}\sum_{h=1}^{H'}\sum_{w=1}^{W'} F[:, h, w] \in \mathbb{R}^{C'}$$

**Giải thích:**
- GAP = Global Average Pooling: Lấy trung bình tất cả vị trí spatial
- Kết quả: Vector 512 chiều đại diện cho **ngữ cảnh tổng thể** của cả ảnh (cảnh đường phố, trong nhà, ...)
- Được truyền vào RelTR để cung cấp thông tin bối cảnh cho việc dự đoán quan hệ

---

## 5. Scene Graph Generation — RelTR

**File nguồn:** `models/reltr.py`, `models/transformer.py`, `models/matcher.py`

### 5.1 Scene Graph

$$G = (\mathcal{V}, \mathcal{E})$$

- $\mathcal{V} = \{v_i = (b_i, c_i)\}$: Các node (đối tượng)
- $\mathcal{E} = \{e_k = (v_s, p_k, v_o)\}$: Các cạnh (quan hệ giữa subject và object)

### 5.2 Positional Encoding (2D Sinusoidal)

$$\text{PE}_{(x, 2i)} = \sin\left(\frac{x}{10000^{2i/d}}\right), \quad \text{PE}_{(x, 2i+1)} = \cos\left(\frac{x}{10000^{2i/d}}\right)$$

**Giải thích:**
- $x$: Vị trí tọa độ trên feature map
- $i$: Chỉ số chiều (dimension index)
- $10000^{2i/d}$: Tần số giảm dần theo chiều → các chiều đầu mã hóa vị trí chi tiết, các chiều sau mã hóa vị trí tổng thể
- Kết hợp sin/cos: Giữ thông tin khoảng cách tương đối — $\text{PE}_{pos}^\top \text{PE}_{pos+k}$ chỉ phụ thuộc vào $k$

### 5.3 Multi-Head Self-Attention (MHSA)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Giải thích từng thành phần:**
- $Q$ (Query), $K$ (Key), $V$ (Value): Ba phép chiếu khác nhau của input
- $QK^\top$: Ma trận attention scores — đo mức độ "quan tâm" giữa mọi cặp token
- $\sqrt{d_k}$: Scaling factor ($d_k = d_{\text{model}}/n_{\text{heads}} = 256/8 = 32$) — ngăn gradient vanishing khi $d_k$ lớn
- $\text{softmax}$: Chuẩn hóa scores thành trọng số xác suất (tổng = 1)
- $V$: Nhân với trọng số → output là trung bình có trọng số của values
- **Multi-head:** Chạy $h = 8$ attention song song, mỗi head học pattern khác nhau, rồi nối (concat) lại

### 5.4 Feed-Forward Network (FFN)

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

- $W_1 \in \mathbb{R}^{256 \times 2048}$: Mở rộng chiều 8 lần
- $W_2 \in \mathbb{R}^{2048 \times 256}$: Nén lại về chiều gốc
- $\text{ReLU}(x) = \max(0, x)$: Phi tuyến, cho phép học các pattern phức tạp

### 5.5 Encoder Layer (Post-Norm)

$$q = k = x + \text{PE}$$
$$x' = \text{LayerNorm}(x + \text{Dropout}(\text{MHSA}(q, k, x)))$$
$$\text{output} = \text{LayerNorm}(x' + \text{Dropout}(\text{FFN}(x')))$$

**Giải thích:**
- $x + \text{PE}$: Cộng positional encoding vào query/key (không cộng vào value)
- Residual connection ($x + ...$): Giúp gradient chảy qua, tránh vanishing gradient
- LayerNorm: Chuẩn hóa để ổn định training
- Dropout: Regularization, ngăn overfitting

### 5.6 FrozenBatchNorm2d

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

- $\mu, \sigma^2$: Running statistics (trung bình, phương sai) — **cố định** không cập nhật
- $\gamma, \beta$: Affine parameters — cũng **cố định**
- Mục đích: Giữ batch statistics ổn định khi fine-tune, tránh sai lệch do batch size nhỏ

### 5.7 Hungarian Matching

Tìm phép gán tối ưu $\hat{\sigma} \in \mathfrak{S}_N$:

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}(\hat{y}_{\sigma(i)}, y_i)$$

**Entity Matching Cost:**

$$C_{\text{entity}}(i, j) = \lambda_{\text{cls}} C_{\text{focal}}(i, j) + \lambda_{\text{box}} \|b_i - b_j\|_1 + \lambda_{\text{giou}} (-\text{GIoU}(b_i, b_j))$$

**Giải thích:**
- $\mathfrak{S}_N$: Nhóm hoán vị của $N$ phần tử — tìm cách gán 1-1 giữa predictions và ground-truth
- $C_{\text{focal}}$: Chi phí phân loại (Focal Loss cost)
- $\|b_i - b_j\|_1$: Khoảng cách L1 giữa 2 box (tổng sai lệch tuyệt đối 4 tọa độ)
- $-\text{GIoU}$: Âm của Generalized IoU (GIoU cao = box trùng → chi phí thấp)
- Giải bằng thuật toán Hungarian $O(N^3)$

### 5.8 GIoU (Generalized IoU)

$$\text{GIoU}(A, B) = \text{IoU}(A, B) - \frac{|C \setminus (A \cup B)|}{|C|}$$

**Giải thích:**
- $C$: Hộp nhỏ nhất bao quanh cả $A$ và $B$ (smallest enclosing box)
- $|C \setminus (A \cup B)|$: Diện tích vùng trống trong $C$ mà không thuộc $A$ hay $B$
- Phạm vi: $\text{GIoU} \in [-1, 1]$
- **Ưu điểm:** Khi $A \cap B = \emptyset$ (không giao nhau), IoU = 0 và gradient = 0 (không học được). GIoU vẫn cho gradient ≠ 0 vì dịch box gần nhau làm $|C|$ giảm → GIoU tăng

### 5.9 Predicate Classification

$$\hat{p}_{\text{rel}} = \text{MLP}_{640 \to 256 \to (R+1)}([\text{sub}_{\text{ctx}} \;;\; \text{obj}_{\text{ctx}} \;;\; m])$$

**Giải thích:**
- $\text{sub}_{\text{ctx}} = h_{\text{sub}} + g' \in \mathbb{R}^{256}$: Biểu diễn subject + ngữ cảnh toàn cục
- $\text{obj}_{\text{ctx}} = h_{\text{obj}} + g' \in \mathbb{R}^{256}$: Biểu diễn object + ngữ cảnh
- $m \in \mathbb{R}^{128}$: Đặc trưng spatial mask (từ attention maps)
- $[;]$: Nối vector → vector 640 chiều
- $R = 51$: Số loại quan hệ (Visual Genome vocabulary)

---

## 6. Spatial-Semantic Validation

**File nguồn:** `boundingbox_objects.py`

### 6.1 Spatial Validation

Centroid: $c_s = \left(\frac{x_1^s+x_2^s}{2}, \frac{y_1^s+y_2^s}{2}\right)$

| Quan hệ | Điều kiện | Hành động |
|---|---|---|
| `riding/sitting on/standing on` | $c_s^y < c_o^y$ (subject phải ở trên object) | Swap subject ↔ object |
| `above` | $c_s^y > c_o^y$ (subject ở dưới — sai) | Swap |

### 6.2 Heuristic Fallback

$$p_{\text{heuristic}} = \begin{cases} \texttt{"near"} & \text{nếu } \text{IoU}(b_s, b_o) > 0.1 \\ \texttt{"above"} & \text{nếu } c_s^y < c_o^y - \delta_y \\ \texttt{"next to"} & \text{otherwise} \end{cases}$$

---

## 7. LLM Open-Vocabulary Enhancement

**File nguồn:** `RL/llm_relationship_predictor.py`, `RL/visual_features.py`

### 7.1 Union Box

$$\text{UnionBox}(A, B) = (\min(x_1^A, x_1^B), \min(y_1^A, y_1^B), \max(x_2^A, x_2^B), \max(y_2^A, y_2^B))$$

### 7.2 Gaze Vector

$$\vec{g} = \frac{c_o - c_s}{\|c_o - c_s\|_2} \in \mathbb{R}^2$$

- $c_s, c_o$: Tâm của subject và object
- Chuẩn hóa L2: Vector đơn vị chỉ hướng từ subject sang object

**Proximity:**

$$\text{proximity} = 1 - \frac{\|c_o - c_s\|_2}{\sqrt{H^2 + W^2}}$$

- Tử: Khoảng cách Euclidean giữa 2 tâm
- Mẫu: Đường chéo ảnh (khoảng cách lớn nhất có thể)
- Giá trị gần 1: Hai vật thể rất gần; gần 0: Rất xa

### 7.3 Gaussian Interaction Heatmap

$$G_s(x,y) = \exp\left(-\frac{(x-c_s^x)^2/(w_s/2)^2 + (y-c_s^y)^2/(h_s/2)^2}{2\sigma^2}\right)$$

$$H_{\text{gaussian}} = G_s \cdot G_o$$

- $G_s$: Phân phối Gaussian lấy tâm tại subject, "lan tỏa" theo kích thước bbox
- Nhân $G_s \cdot G_o$: Vùng interaction cao khi cả subject và object đều có mật độ cao

---

## 8. Video Processing Pipeline

**File nguồn:** `video_relation_pipeline.py`

### 8.1 Kalman Filter State

$$\mathbf{x}_t = [c_x, c_y, s, r, \dot{c}_x, \dot{c}_y, \dot{s}]^\top$$

- $c_x, c_y$: Tâm bounding box
- $s = \sqrt{wh}$: Scale
- $r = w/h$: Tỷ lệ khung hình
- $\dot{c}_x, \dot{c}_y, \dot{s}$: Vận tốc (đạo hàm theo thời gian)

**Prediction:** $\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}\mathbf{x}_{t-1}$

**Update:** $\mathbf{x}_t = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t(\mathbf{z}_t - \mathbf{H}\hat{\mathbf{x}}_{t|t-1})$

- $\mathbf{F}$: Ma trận chuyển trạng thái (state transition)
- $\mathbf{K}_t$: Kalman Gain — cân bằng giữa tin vào dự đoán và tin vào quan sát
- $\mathbf{z}_t$: Quan sát thực tế (detection bounding box)

---

## 9. Safety Classification System

### 9.1 3-Tier Architecture

**Tier 1:** White/Black List — $O(1)$ lookup
**Tier 2:** LLM Safety Analyzer — gọi GPT-4/Gemini
**Tier 3:** Local Rules Database — quy tắc theo ngữ cảnh

### 9.2 Safety Zone Violation

$$\text{ViolationCheck}(b, Z) = \text{IoU}(b, Z) > 0 \wedge \text{class}(b) \in \mathcal{C}_{\text{restricted}}$$

---

## 10. Deep Q-Network (DQN) Agent

**File nguồn:** `RL/reinforcement_learning.py`

### 10.1 MDP Formalization

$(\ mathcal{S}, \mathcal{A}, P, R, \gamma)$:
- $\mathcal{S}$: Không gian trạng thái
- $\mathcal{A} = \{1, 2, ..., 10\}$: Số ảnh sinh cho mỗi relationship
- $\gamma = 0.95$: Hệ số chiết khấu

### 10.2 ε-Greedy Action Selection

$$a_t = \begin{cases} \text{random} \sim \text{Uniform}(\mathcal{A}) & \text{xác suất } \varepsilon_t \\ \arg\max_a Q(\mathbf{s}_t, a; \theta) & \text{xác suất } 1 - \varepsilon_t \end{cases}$$

$$\varepsilon_{t+1} = \max(\varepsilon_{\min}, \varepsilon_t \cdot \varepsilon_{\text{decay}})$$

- $\varepsilon_0 = 1.0$: Ban đầu 100% khám phá ngẫu nhiên
- $\varepsilon_{\text{decay}} = 0.995$: Giảm 0.5% mỗi bước
- $\varepsilon_{\min} = 0.01$: Luôn giữ 1% khám phá

### 10.3 Bellman Optimality Equation

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a')\right]$$

**Loss function (MSE):**

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{B}}\left[\left(r + \gamma(1-d)\max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

**Giải thích:**
- $Q(s, a; \theta)$: Giá trị Q dự đoán bởi online network
- $\theta^-$: Tham số target network (cập nhật mỗi 10-20 steps)
- $r + \gamma(1-d)\max_{a'} Q(s', a'; \theta^-)$: Target value theo Bellman
- $(1-d)$: $d = 1$ nếu episode kết thúc → không có giá trị tương lai
- Tối thiểu hóa sai lệch bình phương giữa prediction và target

### 10.4 State Vector

$$\mathbf{s}_t = [\text{detection\_f1}, \text{relationship\_f1}, \tanh(r/1.0), \tanh(n/50), \tanh(\varepsilon/1.0)]$$

- $\tanh(\cdot)$: Nén giá trị về $(-1, 1)$, chuẩn hóa các thành phần về cùng scale

---

## 11. Adaptive Reward Function

**File nguồn:** `RL/reinforcement_learning.py` → `calculate_reward()`

### 11.1 Công thức tổng quát

$$R = \sigma\big(k \cdot (R_{\text{raw}} - 0.5)\big) = \frac{1}{1 + e^{-k(R_{\text{raw}} - 0.5)}}$$

$$R_{\text{raw}} = W_{\text{det}} S_{\text{det}} + W_{\text{rel}} S_{\text{rel}} + W_{\text{div}} S_{\text{div}} + W_{\text{cons}} S_{\text{cons}} + W_{\text{imp}} S_{\text{imp}} + W_{\text{unc}} S_{\text{unc}}$$

### 11.2 Điểm phát hiện — $S_{\text{det}}$

$$S_{\text{det}} = F1_{\text{det}} \cdot C_n \cdot B_{PR}$$

- $C_n = \tanh(\alpha \cdot \ln(n + 1))$: Hệ số tin cậy mẫu — khi $n$ (số mẫu) ít, $C_n$ nhỏ → giảm tác động (cold-start protection)
- $B_{PR} = 1 - |P - R|$: Phạt khi Precision và Recall chênh lệch lớn

### 11.3 Điểm quan hệ — $S_{\text{rel}}$

$$S_{\text{rel}} = (F1_{\text{rel}} \cdot C_n \cdot B_{PR}) \times (1 + W_{\text{tail}})$$

$$W_{\text{raw}}(r) = \frac{1}{\sqrt{\text{freq}(r) + \epsilon}}, \quad W_{\text{tail}}(r) = \frac{W_{\text{raw}}(r)}{\sum_{i} W_{\text{raw}}(i)}$$

- $W_{\text{tail}}$: Trọng số đuôi dài — quan hệ hiếm (freq thấp) nhận trọng số cao hơn → khuyến khích cải thiện quan hệ ít gặp

### 11.4 Điểm đa dạng — $S_{\text{div}}$

$$S_{\text{div}} = 0.4\, D_{\text{type}} + 0.4\, D_{\text{class}} + 0.2\, S_{\text{spatial}}$$

$$S_{\text{spatial}} = 0.4\, \tanh(\alpha \cdot \text{Var}_{\text{pos}}) + 0.3\, \frac{\sigma_{\text{size}}}{\mu_{\text{size}}} + 0.3\, \left(-\sum_i p_i \log(p_i + \epsilon)\right)$$

- $D_{\text{type}}$: Tỷ lệ loại quan hệ xuất hiện / tổng loại khả dụng
- $D_{\text{class}}$: Tỷ lệ lớp vật thể xuất hiện / tổng lớp
- $\text{Var}_{\text{pos}}$: Phương sai vị trí tâm bbox — giá trị cao = vật thể phân tán đều
- $\sigma/\mu$: Coefficient of Variation kích thước — giá trị cao = kích thước đa dạng
- $-\sum p_i \log p_i$: Shannon entropy trên grid 4×4 — cao khi bbox phủ đều các ô

### 11.5 Điểm nhất quán — $S_{\text{cons}}$

$$S_{\text{cons}} = 0.7 \cdot \frac{1}{1 + \sigma_{F1}} + 0.3 \cdot S_{\text{trend}}$$

- $\frac{1}{1+\sigma_{F1}}$: Nghịch đảo độ lệch chuẩn F1 — F1 ổn định (σ nhỏ) → điểm cao
- $S_{\text{trend}}$: Hệ số góc hồi quy tuyến tính của chuỗi F1 → slope dương (cải thiện) → điểm cao

### 11.6 Điểm cải thiện — $S_{\text{imp}}$

$$S_{\text{imp}} = 0.6 \cdot \tanh(F1_{\text{current}} - F1_{\text{base}}) + 0.4 \cdot S_{\text{trend}}$$

### 11.7 Điểm giảm uncertainty — $S_{\text{unc}}$

$$\rho = \frac{1}{|\mathcal{K}|}\sum_{k \in \mathcal{K}} \frac{U_{\text{prev},k} - U_{\text{current},k}}{U_{\text{prev},k}}$$

$$S_{\text{unc}} = 0.5 + 0.5\,\rho, \quad \text{clip về } [0, 1]$$

- $\rho > 0$: Uncertainty giảm (tốt) → $S_{\text{unc}} > 0.5$
- $\rho < 0$: Uncertainty tăng (xấu) → $S_{\text{unc}} < 0.5$

### 11.8 Trọng số thích nghi (Adaptive Weights)

$$W_k = \frac{W_k^0 + \alpha\, (b_k - S_k)}{\sum_j (W_j^0 + \alpha\, (b_j - S_j))}$$

- $W_k^0$: Trọng số khởi tạo (det: 0.25, rel: 0.45, div: 0.15, cons: 0.10, imp: 0.05, unc: 0.10)
- $b_k - S_k$: Chênh lệch so với baseline — thành phần kém hơn baseline nhận trọng số cao hơn
- Mẫu số: Chuẩn hóa tổng trọng số = 1

---

## 12. MC Dropout Uncertainty Estimation

**File nguồn:** `RL/uncertainty_estimator.py`

### 12.1 Nền tảng lý thuyết (Gal & Ghahramani, 2016)

Dropout tương đương xấp xỉ variational inference:

$$q(\mathbf{W}_l) = \prod_{i} [p \cdot \delta(w_{l,i}) + (1-p) \cdot \delta(w_{l,i} - m_{l,i})]$$

- $p$: Dropout rate
- $\delta$: Hàm Dirac — với xác suất $p$, weight bị tắt (= 0); xác suất $1-p$, weight = $m_{l,i}$ (learned)

### 12.2 Predictive Entropy

$$\mathcal{H}[\mathbf{y}|\mathbf{x}, \mathcal{D}] = -\sum_{k=1}^{K} \bar{p}_k \log \bar{p}_k, \quad \bar{p}_k = \frac{1}{T}\sum_{t=1}^{T} p_{t,k}$$

- $T = 10$ (hoặc 30): Số lần forward pass với dropout bật
- $\bar{p}_k$: Xác suất trung bình cho lớp $k$ qua $T$ lần
- Đo **tổng uncertainty** (aleatoric + epistemic)

### 12.3 BALD (Mutual Information)

$$\mathcal{I}[\mathbf{y}; \boldsymbol{\omega}|\mathbf{x}] = \underbrace{-\sum_k \bar{p}_k \log \bar{p}_k}_{\text{Predictive entropy}} + \underbrace{\frac{1}{T}\sum_{t=1}^{T}\sum_k p_{t,k}\log p_{t,k}}_{\text{Expected entropy}}$$

- Predictive entropy: Tổng uncertainty
- Expected entropy: Uncertainty trung bình **trong mỗi** forward pass (aleatoric)
- BALD = Hiệu = **Epistemic uncertainty** — phần có thể giảm bằng thêm data
- BALD ≥ 0 (theo Jensen's inequality vì entropy là hàm concave)

### 12.4 Variation Ratio

$$\text{VR} = 1 - \frac{|\{t : \arg\max_k p_{t,k} = \hat{c}\}|}{T}$$

- $\hat{c}$: Lớp được dự đoán nhiều nhất (mode)
- VR = 0: Tất cả $T$ lần đều đồng ý → chắc chắn
- VR → 1: Các lần forward cho kết quả khác nhau → không chắc chắn

### 12.5 Combined Score

$$U_{\text{combined}} = 0.30\,\hat{\mathcal{H}} + 0.30\,\hat{\mathcal{I}} + 0.20\,\text{VR} + 0.20\,(1 - \bar{c})$$

---

## 13. Active Learning Acquisition

**File nguồn:** `RL/active_learning.py`

### 13.1 Acquisition Function

$$\alpha(r_k) = 0.40 \cdot U(r_k) + 0.35 \cdot \Delta_{\text{perf}}(r_k) + 0.25 \cdot \tau(r_k)$$

- $U(r_k)$: MC Dropout uncertainty score cho relationship $r_k$
- $\Delta_{\text{perf}}(r_k) = 1 - \text{avg\_F1}(r_k)$: Performance gap — F1 thấp → score cao
- $\tau(r_k) = W_{\text{tail}}(r_k)$: Trọng số đuôi dài — quan hệ hiếm → ưu tiên

### 13.2 Budget Allocation

$$n_k = \text{round}\left(N_{\text{budget}} \cdot \frac{\alpha(r_k)}{\sum_{k'} \alpha(r_{k'})}\right)$$

- $N_{\text{budget}}$: Tổng số ảnh cần sinh (quyết định bởi DQN)
- $n_k$: Số ảnh phân bổ cho relationship $r_k$
- Tỷ lệ theo acquisition score: Relationship yếu nhất nhận nhiều ảnh nhất

---

## 14. Greedy Submodular Maximization

**File nguồn:** `RL/approximation_algorithm.py`

### 14.1 Submodularity

$$f(A \cup \{x\}) - f(A) \geq f(B \cup \{x\}) - f(B), \quad \forall A \subseteq B$$

**Diminishing returns:** Thêm phần tử vào tập nhỏ có lợi ích nhiều hơn thêm vào tập lớn.

### 14.2 Objective Function

$$f(S) = \underbrace{\lambda_{\text{div}} \sum_{c} \log(1 + |S \cap S_c|)}_{\text{Diversity}} + \underbrace{\lambda_{\text{qual}} \sum_{x \in S} q(x)}_{\text{Quality}} + \underbrace{\lambda_{\text{rep}} \cdot \text{Coverage}(S, \Omega)}_{\text{Representativeness}}$$

- **Diversity:** $\log(1 + |S \cap S_c|)$ — concave → submodular; khuyến khích chọn từ nhiều lớp khác nhau
- **Quality:** $q(x)$ — annotation confidence, image quality
- **Representativeness:** Phủ phân phối dữ liệu gốc

### 14.3 Marginal Gain

$$\Delta(x|S) = 0.5 \cdot d_{\min}(x, S) + 0.3 \cdot q(x) + 0.2 \cdot \text{repr}(x, S)$$

- $d_{\min}(x, S) = \min_{s \in S} \text{dist}(x, s)$: Khoảng cách Euclidean nhỏ nhất đến tập đã chọn — đa dạng
- $q(x)$: Chất lượng mẫu
- $\text{repr}(x, S)$: Tỷ lệ mẫu chưa chọn mà $x$ là nearest neighbor mới → đại diện

### 14.4 Approximation Guarantee

$$f(S_{\text{greedy}}) \geq \left(1 - \frac{1}{e}\right) \cdot f(S^*) \approx 0.632 \cdot f(S^*)$$

**Ý nghĩa:** Thuật toán greedy đạt ít nhất **63.2%** giá trị tối ưu — đảm bảo chất lượng ngay cả khi bài toán NP-hard.

---

## 15. Synthetic Data Generation & Auto-Annotation

### 15.1 Quality Filtering

$$\text{Accept}(I) = \text{sharpness}(I) > \theta_s \wedge \text{brightness} \in [\theta_l, \theta_h] \wedge \neg\text{IsDuplicate}(I)$$

### 15.2 Duplicate Detection

$$\text{IsDuplicate}(I) = \min_{J \in \mathcal{D}} \text{HammingDist}(\text{pHash}(I), \text{pHash}(J)) < \theta_{\text{dup}}$$

- pHash: Perceptual hashing — hash bất biến với biến đổi nhỏ
- Hamming distance: Số bit khác nhau giữa 2 hash

### 15.3 Data Augmentation

$$I' = T_{\text{aug}}(I) = T_1 \circ T_2 \circ \ldots \circ T_7(I)$$

| Phép biến đổi | Xác suất |
|---|---|
| RandomBrightnessContrast (±0.2) | 0.5 |
| HueSaturationValue (±20°) | 0.5 |
| RandomRotate90 | 0.3 |
| HorizontalFlip | 0.5 |
| RandomScale (±20%) | 0.5 |
| GaussNoise ($\sigma^2 \in [10, 50]$) | 0.3 |
| Blur (kernel ≤ 3) | 0.3 |

---

## Bảng Tổng Hợp Hyperparameters

| Tham số | Giá trị | File nguồn |
|---|---|---|
| $d_{\text{model}}$ (RelTR) | 256 | `transformer.py` |
| $n_{\text{heads}}$ | 8 | `transformer.py` |
| $d_{\text{ff}}$ | 2048 | `transformer.py` |
| $L_{\text{enc}}, L_{\text{dec}}$ | 6, 6 | `transformer.py` |
| $N_{\text{entity}}$ | 100 | `reltr.py` |
| $N_{\text{triplet}}$ | 200 | `reltr.py` |
| $\gamma$ (DQN) | 0.95 | `reinforcement_learning.py` |
| $\varepsilon_0 / \varepsilon_{\min} / \varepsilon_{\text{decay}}$ | 1.0 / 0.01 / 0.995 | `reinforcement_learning.py` |
| Batch size (DQN) | 32 | `reinforcement_learning.py` |
| Buffer capacity | 10,000 | `reinforcement_learning.py` |
| Target update freq | 10 steps | `reinforcement_learning.py` |
| MC Dropout passes $T$ | 10–30 | `uncertainty_estimator.py` |
| CLIP threshold | 0.65 | `detect_objects.py` |
| NMS IoU threshold | 0.45 | `detect_objects.py` |

---

## Tài Liệu Tham Khảo

1. Y. Cong et al., "RelTR: Relation Transformer for Scene Graph Generation," *ACM Multimedia*, 2022.
2. N. Carion et al., "End-to-End Object Detection with Transformers," *ECCV*, 2020.
3. G. Jocher et al., "Ultralytics YOLO," 2024.
4. A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," *ICML*, 2021.
5. K. He et al., "Mask R-CNN," *ICCV*, 2017.
6. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015.
7. Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation," *ICML*, 2016.
8. G. L. Nemhauser, L. A. Wolsey, and M. L. Fisher, "An analysis of approximations for maximizing submodular set functions," *Math. Programming*, 1978.
