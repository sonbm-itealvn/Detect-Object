import json
import os
from tqdm import tqdm

GQA_JSON = "gqa_objects.json"
OUT_LABEL_DIR = "labels"

os.makedirs(OUT_LABEL_DIR, exist_ok=True)

with open(GQA_JSON) as f:
    data = json.load(f)

label_map = {}
label_id = 0

for img_id, objs in tqdm(data.items()):
    for obj in objs:
        name = obj["name"]

        if name not in label_map:
            label_map[name] = label_id
            label_id += 1

        xc = obj["x"]
        yc = obj["y"]
        w = obj["w"]
        h = obj["h"]

        out_file = os.path.join(OUT_LABEL_DIR, f"{img_id}.txt")
        with open(out_file, "a") as f:
            f.write(f"{label_map[name]} {xc} {yc} {w} {h}\n")

with open("gqa.names", "w") as f:
    for k in sorted(label_map, key=label_map.get):
        f.write(k + "\n")
