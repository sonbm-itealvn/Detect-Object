import json
import os
from tqdm import tqdm

COCO_JSON = "instances_train2017.json"
IMG_DIR = "images/train2017"
OUT_LABEL_DIR = "labels/train"

os.makedirs(OUT_LABEL_DIR, exist_ok=True)

with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}

for ann in tqdm(coco["annotations"]):
    img = images[ann["image_id"]]
    w, h = img["width"], img["height"]

    x, y, bw, bh = ann["bbox"]
    xc = (x + bw / 2) / w
    yc = (y + bh / 2) / h
    bw /= w
    bh /= h

    label_path = os.path.join(
        OUT_LABEL_DIR,
        img["file_name"].replace(".jpg", ".txt")
    )

    with open(label_path, "a") as f:
        f.write(f"{ann['category_id'] - 1} {xc} {yc} {bw} {bh}\n")
