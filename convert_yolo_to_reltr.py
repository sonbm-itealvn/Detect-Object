import json
import sys
import os
from pathlib import Path


def _normalize_entry(entry):
    label = (entry.get('label') or entry.get('class') or '').strip()
    bbox = entry.get('bbox', [])
    feature = entry.get('feature', [])
    return label, bbox, feature


def convert_yolo_to_reltr(input_json="result.json", output_json="converted_bboxes.json"):
    if not os.path.exists(input_json):
        print(f"❌ Không tìm thấy file JSON: {input_json}")
        return

    try:
        with open(input_json, 'r', encoding='utf-8') as file:
            result_data = json.load(file)
    except json.JSONDecodeError as exc:
        print(f"❌ Lỗi đọc file JSON: {exc}")
        return

    entries = []
    if isinstance(result_data, dict):
        entries = [result_data]
    elif isinstance(result_data, list):
        entries = [item for item in result_data if isinstance(item, dict)]

    if not entries:
        print("⚠️ Không có dữ liệu hợp lệ để chuyển đổi.")
        return

    converted: list[dict] = []
    for entry_index, entry in enumerate(entries):
        raw_objects = entry.get('objects') or []
        if not raw_objects:
            print(f"⚠️ Bỏ qua entry {entry_index + 1}: không có đối tượng nào.")
            continue

        converted_objects = []
        for obj_index, obj in enumerate(raw_objects, start=1):
            label, bbox, feature = _normalize_entry(obj)
            if not label:
                print(f"⚠️ Đối tượng {obj_index} thiếu nhãn, bỏ qua.")
                continue
            if len(bbox) != 4 or not all(isinstance(x, (int, float)) for x in bbox):
                print(f"⚠️ BBox của '{label}' không hợp lệ, bỏ qua.")
                continue
            converted_objects.append({
                "class": label,
                "bbox": bbox,
                "feature": feature,
            })

        if not converted_objects:
            print(f"⚠️ Entry {entry_index + 1} không có đối tượng hợp lệ để lưu.")
            continue

        image_path = entry.get('image_path') or ''
        image_name = Path(image_path).name if image_path else f"image_{entry_index + 1}.jpg"

        converted.append({
            "image_id": image_name,
            "image_path": image_path,
            "objects": converted_objects,
            "global_context": entry.get('global_context', []),
        })

    if not converted:
        print("⚠️ Không có entry nào được ghi ra file.")
        return

    with open(output_json, 'w', encoding='utf-8') as output_file:
        json.dump(converted, output_file, indent=4)

    print(f"✅ Đã lưu dữ liệu chuyển đổi vào {output_json}")


if __name__ == "__main__":
    input_json = sys.argv[1] if len(sys.argv) > 1 else "result.json"
    output_json = sys.argv[2] if len(sys.argv) > 2 else "converted_bboxes.json"
    convert_yolo_to_reltr(input_json, output_json)
