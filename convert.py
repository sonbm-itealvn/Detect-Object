import matplotlib.pyplot as plt
import numpy as np

# --- CẤU HÌNH ---
EPOCHS = 2000
TARGET_MAIN = 0.85   # Precision, Recall, mAP50
TARGET_MAP95 = 0.72  # Đã kéo cao lên (Mức cũ là 0.65)
NOISE_SCALE = 0.06   # Độ nhiễu để tạo màu xanh đậm

x = np.linspace(0, EPOCHS, EPOCHS)

# Hàm tạo dữ liệu chuẩn style YOLO
def create_yolo_data(start, end, type='loss'):
    # Tạo xu hướng chính (Trend)
    if type == 'loss':
        # Loss giảm nhanh rồi đi ngang
        trend = (start - end) * np.exp(-0.005 * x) + end
    else:
        # Metric tăng theo hàm logistic (Cong chữ S)
        # Điều chỉnh độ dốc để nó lên nhanh đoạn đầu
        trend = end / (1 + np.exp(-0.005 * (x - 250))) 
        trend = np.clip(trend, 0, end)
        
        # Hack đoạn đầu: cho tăng thẳng từ 0 để giống training thực tế
        ramp_up = 150
        trend[0:ramp_up] = np.linspace(0, trend[ramp_up], ramp_up)

    # Tạo nhiễu (Noise) - Màu xanh đậm
    noise = np.random.normal(0, NOISE_SCALE * (trend + 0.15), EPOCHS)
    
    # Cộng nhiễu
    raw_data = trend + noise
    raw_data = np.maximum(raw_data, 0) # Không âm
    
    # Tạo đường Smooth (màu cam) - Moving Average
    smooth_data = np.convolve(raw_data, np.ones(50)/50, mode='same')
    
    # Xử lý mép đường smooth cho đẹp
    smooth_data[0:25] = raw_data[0:25] 
    
    return raw_data, smooth_data

# --- TẠO DỮ LIỆU ---
# 1. Losses
box_train, box_train_s = create_yolo_data(1.4, 0.3, 'loss')
cls_train, cls_train_s = create_yolo_data(1.2, 0.2, 'loss')
dfl_train, dfl_train_s = create_yolo_data(1.1, 0.8, 'loss')

box_val, box_val_s = create_yolo_data(1.0, 0.3, 'loss')
cls_val, cls_val_s = create_yolo_data(4.0, 0.3, 'loss')
dfl_val, dfl_val_s = create_yolo_data(0.9, 0.75, 'loss')

# 2. Metrics 
prec, prec_s = create_yolo_data(0, TARGET_MAIN, 'metric')
rec, rec_s = create_yolo_data(0, TARGET_MAIN, 'metric')
map50, map50_s = create_yolo_data(0, TARGET_MAIN, 'metric')

# --> mAP50-95 (Đã tăng lên TARGET_MAP95)
map95, map95_s = create_yolo_data(0, TARGET_MAP95, 'metric')

# --- VẼ BIỂU ĐỒ ---
fig, axs = plt.subplots(2, 5, figsize=(20, 8), facecolor='white')

data_map = [
    (axs[0,0], box_train, box_train_s, 'train/box_loss'),
    (axs[0,1], cls_train, cls_train_s, 'train/cls_loss'),
    (axs[0,2], dfl_train, dfl_train_s, 'train/dfl_loss'),
    (axs[0,3], prec, prec_s, 'metrics/precision(B)'),
    (axs[0,4], rec, rec_s, 'metrics/recall(B)'),
    (axs[1,0], box_val, box_val_s, 'val/box_loss'),
    (axs[1,1], cls_val, cls_val_s, 'val/cls_loss'),
    (axs[1,2], dfl_val, dfl_val_s, 'val/dfl_loss'),
    (axs[1,3], map50, map50_s, 'metrics/mAP50(B)'),
    (axs[1,4], map95, map95_s, 'metrics/mAP50-95(B)'),
]

for ax, raw, smooth, title in data_map:
    # 1. RAW (Xanh đậm)
    ax.plot(x, raw, color='#283899', linewidth=0.6, alpha=0.8)
    
    # 2. SMOOTH (Cam/Vàng)
    ax.plot(x, smooth, color='#FF9933', linewidth=1.2)
    
    # 3. Style YOLO
    ax.set_title(title, fontsize=12, color='#333333', pad=10)
    ax.set_xlabel('epoch', fontsize=10, color='#666666')
    ax.grid(True, linestyle=':', alpha=0.6, color='#999999')
    
    # Xóa khung viền thừa
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    
    if 'metrics' in title:
        ax.set_ylim(0, 1.05)

plt.tight_layout()
# Chỉnh khoảng cách giữa các hình cho giống
plt.subplots_adjust(wspace=0.25, hspace=0.3)
plt.savefig('yolo_result_high_map95.png', dpi=300, bbox_inches='tight')
plt.show()