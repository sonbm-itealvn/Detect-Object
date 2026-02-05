import csv
import os
from tqdm import tqdm

ANNOTATION_CSV = "train-annotations-bbox.csv"
CLASS_DESC = "class-descriptions-boxable.csv"
IMG_DIR = "images/train"
OUT_LABEL_DIR = "labels/train"

os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# Map class name → id
class_map = {}
with open(CLASS_DESC) as f:
    for i, row in enumerate(csv.reader(f)):
        class_map[row[0]] = i

with open(ANNOTATION_CSV) as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader):
        img_id = row["ImageID"]
        label = class_map[row["LabelName"]]

        xmin = float(row["XMin"])
        xmax = float(row["XMax"])
        ymin = float(row["YMin"])
        ymax = float(row["YMax"])

        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        out_file = os.path.join(OUT_LABEL_DIR, img_id + ".txt")
        with open(out_file, "a") as f:
            f.write(f"{label} {xc} {yc} {w} {h}\n")
