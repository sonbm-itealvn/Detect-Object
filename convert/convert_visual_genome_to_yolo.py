import json
import os
from tqdm import tqdm

VG_JSON = "annotations.json"
IMG_DIR = "images"
OUT_LABEL_DIR = "labels"

os.makedirs(OUT_LABEL_DIR, exist_ok=True)

with open(VG_JSON) as f:
    data = json.load(f)

label_map = {}
label_id = 0

for img in tqdm(data):
    img_id = img["image_id"]
    w, h = img["width"], img["height"]

    for obj in img["objects"]:
        name = obj["names"][0]

        if name not in label_map:
            label_map[name] = label_id
            label_id += 1

        x = obj["x"] / w
        y = obj["y"] / h
        bw = obj["w"] / w
        bh = obj["h"] / h

        xc = x + bw / 2
        yc = y + bh / 2

        out_file = os.path.join(OUT_LABEL_DIR, f"{img_id}.txt")
        with open(out_file, "a") as f:
            f.write(f"{label_map[name]} {xc} {yc} {bw} {bh}\n")

# Save names
with open("vg.names", "w") as f:
    for k in sorted(label_map, key=label_map.get):
        f.write(k + "\n")
