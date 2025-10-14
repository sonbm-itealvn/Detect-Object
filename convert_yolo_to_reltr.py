import json
import sys
import os

def convert_yolo_to_reltr(input_json="result.json", output_json="converted_bboxes.json"):
    # Kiểm tra file đầu vào có tồn tại không
    if not os.path.exists(input_json):
        print(f"❌ Không tìm thấy file JSON: {input_json}")
        return

    # Đọc dữ liệu từ result.json
    try:
        with open(input_json, 'r') as file:
            result_data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi đọc file JSON: {e}")
        return

    # Khởi tạo cấu trúc dữ liệu mới
    data_output = [{
        "image_id": "result_image.jpg",
        "objects": []
    }]

    # Chuyển đổi từng đối tượng
    for idx, item in enumerate(result_data):
        try:
            label = item.get('label', '').strip()
            bbox = item.get('bbox', [])

            # Kiểm tra tính hợp lệ của nhãn và bbox
            if not label:
                print(f"⚠️ Cảnh báo: Đối tượng thứ {idx+1} thiếu nhãn, bỏ qua.")
                continue

            if len(bbox) != 4 or not all(isinstance(x, (int, float)) for x in bbox):
                print(f"⚠️ Cảnh báo: Bbox của đối tượng '{label}' không hợp lệ, bỏ qua.")
                continue

            # Thêm đối tượng hợp lệ vào danh sách output
            data_output[0]["objects"].append({
                "class": label,
                "bbox": bbox,
                "feature": item.get("feature", [])
            })
        
        except Exception as e:
            print(f"❌ Lỗi xử lý đối tượng thứ {idx+1}: {e}")

    # Kiểm tra nếu không có đối tượng hợp lệ nào
    if not data_output[0]["objects"]:
        print("❌ Không có đối tượng nào hợp lệ để lưu.")
        return

    # Ghi dữ liệu ra file mới
    with open(output_json, 'w') as output_file:
        json.dump(data_output, output_file, indent=4)

    print(f"Done. Data converted to RelTR format and saved to {output_json}")

if __name__ == "__main__":
    # Nếu có tham số dòng lệnh, sử dụng file JSON từ đối số
    input_json = sys.argv[1] if len(sys.argv) > 1 else "result.json"
    convert_yolo_to_reltr(input_json)
