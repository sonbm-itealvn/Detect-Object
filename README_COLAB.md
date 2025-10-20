# Object Detection & Relationship Analysis - Google Colab Version

## Mô tả
Phiên bản console của ứng dụng Object Detection & Relationship Analysis, được tối ưu hóa để chạy trên Google Colab (môi trường không có GUI).

## Các file chính

### 1. `app_console.py`
- Phiên bản console của ứng dụng gốc
- Loại bỏ hoàn toàn GUI (tkinter, PIL ImageTk)
- Giữ nguyên tất cả logic xử lý
- Có menu interactive để chọn chức năng

### 2. `colab_runner.py`
- Script đơn giản để chạy trên Google Colab
- Hỗ trợ cả chế độ interactive và command line
- Tự động phát hiện môi trường Colab

## Cách sử dụng

### Trên Google Colab:

```python
# Cách 1: Chạy interactive
!python colab_runner.py

# Cách 2: Chạy với ảnh cụ thể
!python colab_runner.py demo/anh2.jpg

# Cách 3: Chạy demo nhanh
!python colab_runner.py
```

### Trên môi trường local:

```bash
# Chạy interactive
python app_console.py

# Hoặc sử dụng colab_runner
python colab_runner.py
```

## Các chức năng chính

1. **Xử lý ảnh**: Phát hiện vật thể và mối quan hệ
2. **RL Training**: Huấn luyện mô hình với Reinforcement Learning
3. **Synthetic Data**: Tạo dữ liệu synthetic
4. **Evaluation**: Đánh giá kết quả training
5. **Data Refresh**: Tải lại dữ liệu JSON

## Lưu ý quan trọng

- **Không cần GUI**: Hoàn toàn chạy trên console
- **Tương thích Colab**: Tối ưu cho môi trường Google Colab
- **Giữ nguyên logic**: Tất cả chức năng từ phiên bản GUI được bảo toàn
- **Dễ sử dụng**: Menu đơn giản, dễ điều hướng

## Cấu trúc file

```
yolov5/
├── app.py                 # Phiên bản GUI gốc
├── app_console.py         # Phiên bản console (mới)
├── colab_runner.py        # Script chạy Colab (mới)
├── README_COLAB.md        # Hướng dẫn này
└── ... (các file khác)
```

## Demo nhanh

```python
# Trên Google Colab
!python colab_runner.py demo/anh2.jpg
```

Sẽ tự động:
1. Tải ảnh demo
2. Chạy pipeline phát hiện vật thể
3. Xác định mối quan hệ
4. Hiển thị kết quả trên console
5. Lưu ảnh kết quả với bbox
