# Hệ thống Phân loại An toàn 3 Tầng

## Tổng quan

Hệ thống phân loại an toàn 3 tầng được thiết kế để phát hiện và cảnh báo các hành động nguy hiểm trong video, đặc biệt là trong môi trường gia đình có trẻ em. Hệ thống tích hợp với `VideoRelationPipeline` để phân tích các relationships (Subject-Relation-Object) được phát hiện từ video và đánh giá mức độ nguy hiểm.

## Kiến trúc 3 Tầng

### Tầng 1: Phân loại 3 Trạng thái (White/Black/Gray List)

**Nguyên tắc**: "Cái gì chưa biết thì phải Cảnh giác"

**File**: `RL/safety_classifier.py`

Hệ thống kiểm tra relationships theo thứ tự ưu tiên:

1. **Danh sách trắng (White List)**: Các hành động sinh hoạt bình thường, an toàn
   - Ví dụ: `person sitting on chair`, `person watching tv`, `person eating food`
   - Kết quả: `SafetyLevel.SAFE` với confidence 0.9
   
2. **Danh sách đen (Black List)**: Các hành động nguy hiểm rõ ràng
   - Ví dụ: `child holding knife`, `child eating battery`, `person near fire`
   - Kết quả: `SafetyLevel.DANGEROUS` với confidence 0.95
  
3. **Danh sách xám (Gray List)**: Các hành động không rõ ràng → Chuyển sang Tầng 2
   - Ví dụ: `child putting in mouth coin`, `dog chew electric wire`
   - Nếu không có trong white/black list → Hỏi LLM (Tầng 2)

**Pattern Matching**: Hỗ trợ cả exact match và regex pattern matching cho linh hoạt hơn.

### Tầng 2: LLM Semantic Reasoning

**Mục đích**: Sử dụng LLM (OpenAI/Gemini) để suy luận ngữ nghĩa

**File**: `RL/llm_safety_analyzer.py`

**Cách hoạt động**:
1. RelTR xuất ra relationship: `{"subject": "dog", "relation": "chew", "object": "electric wire"}`
2. Hệ thống xây dựng prompt và gửi đến LLM:
   ```
   Bạn là một hệ thống phân tích an toàn cho gia đình. Hãy đánh giá mức độ nguy hiểm của hành động sau:
   
   Subject (Chủ thể): dog
   Relation (Hành động): chew
   Object (Đối tượng): electric wire
   
   Trong bối cảnh gia đình, đặc biệt là khi có trẻ em, hành động này có nguy hiểm không?
   
   Hãy trả lời CHỈ bằng một trong ba từ sau: SAFE, SUSPICIOUS, hoặc DANGEROUS
   ```
3. LLM trả lời: "DANGEROUS"
4. Hệ thống parse và trả về `SafetyLevel.DANGEROUS` với confidence 0.9

**Ưu điểm**: 
- Nhẹ (chỉ gửi text, không gửi video)
- Không cần lập trình quy tắc từ trước
- Có thể hiểu ngữ cảnh phức tạp
- **Cache**: Responses được cache trong `llm_safety_cache.json` để tránh gọi API nhiều lần

**Hỗ trợ Providers**:
- OpenAI: `gpt-3.5-turbo`, `gpt-4`, `gpt-4o`, `gpt-5.2`, etc.
- Gemini: `gemini-pro`, `gemini-1.5-pro`, etc.

**Model Compatibility**: Tự động xử lý các model mới (dùng `max_completion_tokens`) và model cũ (dùng `max_tokens`).

### Tầng 3: Human-in-the-loop (Local Rules Database)

**Mục đích**: Hệ thống "lớn lên" cùng gia đình

**File**: `RL/local_rules_db.py`

**Cách hoạt động**:
1. Khi có cảnh báo sai hoặc bỏ sót, người dùng phản hồi qua App
2. Hệ thống tự động lưu quy tắc vào `local_safety_rules.json`
3. Lần sau gặp hành động tương tự → **Ưu tiên cao nhất**, kiểm tra trước cả white/black list

**Ưu tiên kiểm tra** (theo thứ tự):
1. **Local Rules** (Tầng 3) - Ưu tiên cao nhất
2. Black List (Tầng 1)
3. White List (Tầng 1)
4. LLM Analysis (Tầng 2) - Nếu không có trong danh sách
5. Default: SUSPICIOUS - Nếu LLM không hoạt động

## Cấu trúc File

```
RL/
├── safety_config.json          # Cấu hình danh sách trắng/đen/xám, LLM settings
├── llm_safety_analyzer.py      # Tầng 2: LLM integration (OpenAI/Gemini)
├── safety_classifier.py         # Tầng 1: 3 trạng thái classifier (orchestrator)
├── local_rules_db.py           # Tầng 3: Local rules database (JSON-based)
├── README_SAFETY_SYSTEM.md     # Tài liệu này
│
├── llm_safety_cache.json       # Cache LLM responses (tự động tạo)
└── local_safety_rules.json     # Local rules database (tự động tạo)
```

**Tích hợp với Video Pipeline**:
- `video_relation_pipeline.py`: Sử dụng `SafetyClassifier` để phân tích relationships từ video

## Cách sử dụng

### 1. Cấu hình API Key

**Cách 1: Sử dụng file .env (Khuyến nghị)**

1. Tạo file `.env` ở thư mục gốc project (cùng cấp với `video_relation_pipeline.py`):
```bash
# Windows
type nul > .env

# Linux/Mac
touch .env
```

2. Mở file `.env` và điền API key của bạn:
```
OPENAI_API_KEY=your_openai_api_key_here
# hoặc
GEMINI_API_KEY=your_gemini_api_key_here
```

**Cách 2: Sử dụng environment variables**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"
# hoặc
$env:GEMINI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
export GEMINI_API_KEY="your-api-key-here"
```

**Cách 3: Truyền trực tiếp qua parameter (khi khởi tạo)**
```python
from RL.llm_safety_analyzer import LLMSafetyAnalyzer

analyzer = LLMSafetyAnalyzer(
    provider="openai",
    model="gpt-3.5-turbo",
    api_key="your-api-key-here"  # Truyền trực tiếp
)
```

**Lưu ý**: 
- File `.env` sẽ được tự động load khi khởi động (sử dụng `python-dotenv`)
- Ưu tiên: **parameter > .env file > environment variable**
- Nếu không có API key, LLM sẽ bị tắt và hệ thống mặc định về "SUSPICIOUS" cho các hành động không biết

### 2. Sử dụng trong Video Pipeline

**Tích hợp tự động**:

```python
from video_relation_pipeline import VideoRelationPipeline

# Khởi tạo pipeline với safety classifier enabled
pipeline = VideoRelationPipeline(
    reltr_checkpoint="reltr_finetuned.pth",
    safety_classifier_enabled=True  # Bật hệ thống cảnh báo
)

# Xử lý video
result = pipeline.process_video(
    video_path="video.mp4",
    output_dir="video_outputs",
    frame_stride=2  # Xử lý mỗi 2 frames
)

# Kết quả:
# - Video đã annotate: video_outputs/video_relations.avi
# - Thống kê: video_outputs/video_summary.json
```

**Cách hoạt động trong pipeline**:
1. Video được xử lý frame-by-frame
2. Mỗi frame: YOLO detect objects → RelTR predict relationships
3. Mỗi relationship được phân tích bởi `SafetyClassifier`:
   ```python
   level, confidence, explanation = classifier.classify(
       subject="child",
       relation="holding",
       object_name="knife"
   )
   # Returns: (SafetyLevel.DANGEROUS, 0.95, "Black list match")
   ```
4. Cảnh báo được vẽ lên frame (banner đỏ/cam ở trên cùng)
5. Thống kê được lưu vào JSON file

**Sử dụng trực tiếp SafetyClassifier**:

```python
from RL.safety_classifier import SafetyClassifier

# Khởi tạo
classifier = SafetyClassifier(
    config_path="RL/safety_config.json",
    llm_enabled=True,
    local_rules_enabled=True
)

# Phân loại một relationship
level, conf, exp = classifier.classify(
    subject="child",
    relation="eating",
    object_name="battery"
)
print(f"Level: {level.value}, Confidence: {conf:.2f}, Explanation: {exp}")

# Phân loại nhiều relationships cùng lúc
relationships = [
    {"subject": "person", "relation": "sitting on", "object": "chair"},
    {"subject": "child", "relation": "holding", "object": "knife"},
    {"subject": "dog", "relation": "chew", "object": "electric wire"}
]
classified = classifier.classify_batch(relationships)
for rel in classified:
    print(f"{rel['subject']} {rel['relation']} {rel['object']} -> {rel['safety_level']}")
```

### 3. Thêm quy tắc cục bộ (Human-in-the-loop)

```python
from RL.local_rules_db import LocalRulesDatabase
from RL.llm_safety_analyzer import SafetyLevel

# Khởi tạo database
db = LocalRulesDatabase(db_path="local_safety_rules.json")

# Thêm quy tắc từ phản hồi người dùng
db.add_rule(
    subject="child",
    relation="playing with",
    object_name="toy",
    level=SafetyLevel.SAFE,
    source="user_feedback",
    metadata={"user_id": "user123", "context": "normal play"}
)

# Thêm quy tắc nguy hiểm
db.add_rule(
    subject="child",
    relation="eating",
    object_name="medicine",
    level=SafetyLevel.DANGEROUS,
    source="user_feedback"
)

# Lấy quy tắc
rule = db.get_rule("child", "playing with", "toy")
if rule:
    print(f"Rule found: {rule['level']} (confidence: {rule.get('usage_count', 0)} uses)")

# Xem thống kê
stats = db.get_stats()
print(f"Total rules: {stats['total_rules']}")
print(f"Rules by level: {stats['rules_by_level']}")

# Xuất/nhập quy tắc
db.export_rules("backup_rules.json")
db.import_rules("backup_rules.json", merge=True)
```

**Lưu ý**: 
- Local rules có **ưu tiên cao nhất** - được kiểm tra trước cả white/black list
- Rules được lưu trong `local_safety_rules.json` (JSON format)
- Hệ thống tự động track `usage_count` cho mỗi rule

## Cấu hình

File `safety_config.json` chứa cấu hình đầy đủ:

```json
{
  "white_list": {
    "relationships": [
      {"subject": "person", "relation": "sitting on", "object": "chair"},
      {"subject": "person", "relation": "watching", "object": "tv"}
    ],
    "patterns": [
      {
        "subject_pattern": "person",
        "relation_pattern": "sitting|standing|walking",
        "object_pattern": "chair|floor|tv"
      }
    ]
  },
  "black_list": {
    "relationships": [
      {"subject": "child", "relation": "holding", "object": "knife"},
      {"subject": "child", "relation": "eating", "object": "battery"}
    ],
    "patterns": [
      {
        "subject_pattern": "child|person",
        "relation_pattern": "holding|eating|near",
        "object_pattern": "knife|gun|fire|medicine|battery"
      }
    ]
  },
  "gray_list_threshold": 0.3,
  "llm_enabled": true,
  "llm_provider": "openai",
  "llm_model": "gpt-5.2",
  "alert_levels": {
    "safe": {
      "color": [0, 255, 0],
      "message": "An toàn",
      "sound_alert": false
    },
    "suspicious": {
      "color": [255, 165, 0],
      "message": "Cảnh báo nhẹ - Cần kiểm tra",
      "sound_alert": true,
      "sound_type": "soft"
    },
    "dangerous": {
      "color": [255, 0, 0],
      "message": "NGUY HIỂM - Cần can thiệp ngay!",
      "sound_alert": true,
      "sound_type": "urgent"
    }
  }
}
```

**Các tham số**:

- `white_list`: Danh sách hành động an toàn (exact match + pattern matching)
- `black_list`: Danh sách hành động nguy hiểm (exact match + pattern matching)
- `gray_list_threshold`: Ngưỡng confidence cho danh sách xám (mặc định: 0.3)
- `llm_enabled`: Bật/tắt LLM analysis (true/false)
- `llm_provider`: "openai" hoặc "gemini"
- `llm_model`: Model name (ví dụ: "gpt-3.5-turbo", "gpt-4o", "gpt-5.2", "gemini-pro")
- `alert_levels`: Cấu hình màu sắc (RGB), thông điệp và sound alert cho từng mức độ

## Cảnh báo

Hệ thống sẽ hiển thị cảnh báo trên video với các mức độ:

### DANGEROUS (Đỏ) - Mức độ cao nhất
- **Banner**: Banner đỏ ở trên cùng frame (height: 100px, alpha: 0.7)
- **Thông điệp**: "NGUY HIỂM - Cần can thiệp ngay!" (font lớn, màu trắng, có outline)
- **Chi tiết**: Liệt kê tối đa 3 cảnh báo nguy hiểm dưới banner
- **Ví dụ**: `⚠️ child holding knife`, `⚠️ child eating battery`

### SUSPICIOUS (Cam) - Mức độ trung bình
- **Banner**: Banner cam ở trên cùng frame (height: 80px, alpha: 0.5)
- **Thông điệp**: "Cảnh báo nhẹ - Cần kiểm tra" (font vừa, màu trắng)
- **Chi tiết**: Liệt kê tối đa 2 cảnh báo nghi ngờ dưới banner
- **Ví dụ**: `⚠️ dog chew electric wire`, `⚠️ child putting in mouth coin`

### SAFE (Xanh) - Không hiển thị
- Không hiển thị cảnh báo trên video (chỉ log trong console)

**Font hỗ trợ tiếng Việt**: Hệ thống tự động tìm font Windows (Arial, Tahoma, Calibri) để hiển thị tiếng Việt đúng cách.

## Thống kê

Sau khi xử lý video, file `*_summary.json` sẽ chứa thống kê đầy đủ:

```json
{
  "objects": [
    {"label": "person", "count": 150},
    {"label": "chair", "count": 20}
  ],
  "relations": [
    {
      "subject": "person",
      "relation": "sitting on",
      "object": "chair",
      "count": 15
    }
  ],
  "safety_alerts": {
    "total_alerts": 10,
    "dangerous_count": 2,
    "suspicious_count": 8,
    "alerts": [
      {
        "subject": "child",
        "relation": "holding",
        "object": "knife",
        "level": "dangerous",
        "confidence": 0.95,
        "explanation": "Black list match",
        "frame": 42
      },
      {
        "subject": "dog",
        "relation": "chew",
        "object": "electric wire",
        "level": "suspicious",
        "confidence": 0.7,
        "explanation": "LLM: DANGEROUS",
        "frame": 85
      }
    ]
  }
}
```

## Luồng xử lý chi tiết

### Khi một relationship được phát hiện:

```
1. VideoRelationPipeline phát hiện relationship từ RelTR
   ↓
2. SafetyClassifier.classify() được gọi
   ↓
3. Kiểm tra Local Rules (Tầng 3) - Ưu tiên cao nhất
   ├─ Nếu có rule → Return level từ rule
   └─ Nếu không → Tiếp tục
   ↓
4. Kiểm tra Black List (Tầng 1)
   ├─ Nếu match → Return DANGEROUS
   └─ Nếu không → Tiếp tục
   ↓
5. Kiểm tra White List (Tầng 1)
   ├─ Nếu match → Return SAFE
   └─ Nếu không → Tiếp tục
   ↓
6. Hỏi LLM (Tầng 2) - Nếu enabled
   ├─ Kiểm tra cache trước
   ├─ Nếu có cache → Return từ cache
   ├─ Nếu không → Gọi LLM API
   │   ├─ Parse response (SAFE/SUSPICIOUS/DANGEROUS)
   │   └─ Lưu vào cache
   └─ Return level từ LLM
   ↓
7. Default: Return SUSPICIOUS (Gray List)
```

## Ví dụ sử dụng

### Ví dụ 1: Phân loại đơn giản

```python
from RL.safety_classifier import SafetyClassifier

classifier = SafetyClassifier()

# Test case 1: An toàn (white list)
level, conf, exp = classifier.classify("person", "sitting on", "chair")
# Output: (SafetyLevel.SAFE, 0.9, "White list match")

# Test case 2: Nguy hiểm (black list)
level, conf, exp = classifier.classify("child", "holding", "knife")
# Output: (SafetyLevel.DANGEROUS, 0.95, "Black list match")

# Test case 3: Hỏi LLM (không có trong danh sách)
level, conf, exp = classifier.classify("dog", "chew", "electric wire")
# Output: (SafetyLevel.DANGEROUS, 0.9, "LLM: DANGEROUS")
```

### Ví dụ 2: Tích hợp với video

```python
from video_relation_pipeline import VideoRelationPipeline

pipeline = VideoRelationPipeline(
    safety_classifier_enabled=True
)

# Xử lý video
result = pipeline.process_video("home_video.mp4")

# Xem thống kê
import json
with open(result["summary"], "r", encoding="utf-8") as f:
    stats = json.load(f)
    alerts = stats["safety_alerts"]
    print(f"Tổng cảnh báo: {alerts['total_alerts']}")
    print(f"Nguy hiểm: {alerts['dangerous_count']}")
    print(f"Nghi ngờ: {alerts['suspicious_count']}")
```

### Ví dụ 3: Thêm quy tắc từ phản hồi người dùng

```python
from RL.local_rules_db import LocalRulesDatabase
from RL.llm_safety_analyzer import SafetyLevel

db = LocalRulesDatabase()

# Người dùng báo: "Trẻ em chơi với đồ chơi là an toàn"
db.add_rule(
    subject="child",
    relation="playing with",
    object_name="toy",
    level=SafetyLevel.SAFE,
    source="user_feedback",
    metadata={"user_id": "parent_001", "timestamp": "2024-01-15"}
)

# Lần sau gặp "child playing with toy" → Tự động trả về SAFE (không cần hỏi LLM)
```

## Lưu ý quan trọng

1. **LLM API Key**: 
   - Cần API key để LLM hoạt động
   - Nếu không có, hệ thống mặc định về "SUSPICIOUS" cho các hành động không biết
   - Cache được lưu trong `llm_safety_cache.json` để tiết kiệm chi phí API

2. **Cache LLM**: 
   - Responses được cache theo key: `subject|relation|object` (lowercase)
   - Cache được lưu tự động sau mỗi lần gọi API
   - Có thể xóa cache để force refresh

3. **Ưu tiên kiểm tra**: 
   - Local Rules (Tầng 3) > Black List > White List > LLM > Default SUSPICIOUS
   - Local Rules có ưu tiên cao nhất để cho phép customization

4. **Pattern Matching**: 
   - Hỗ trợ regex pattern trong `safety_config.json`
   - Ví dụ: `"subject_pattern": "child|person"` sẽ match cả "child" và "person"

5. **Performance**: 
   - LLM cache giúp giảm số lần gọi API
   - Local rules được load một lần và cache trong memory
   - Video pipeline xử lý theo `frame_stride` để tối ưu tốc độ

6. **Error Handling**: 
   - Nếu LLM API fail → Fallback về SUSPICIOUS
   - Nếu config file không tồn tại → Sử dụng defaults
   - Tất cả errors được log nhưng không crash pipeline


