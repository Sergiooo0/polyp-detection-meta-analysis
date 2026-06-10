from pathlib import Path
import shutil
import random

random.seed(42)

img_src = Path("test/images/train")
lbl_src = Path("test/labels/train")

img_dst = Path("test/images/train75")
lbl_dst = Path("test/labels/train75")

img_dst.mkdir(parents=True, exist_ok=True)
lbl_dst.mkdir(parents=True, exist_ok=True)

files = list(img_src.glob("*.png"))

sample = random.sample(files, int(len(files) * 0.75))

for img in sample:
    shutil.copy2(img, img_dst / img.name)

    label = lbl_src / f"{img.stem}.txt"
    if label.exists():
        shutil.copy2(label, lbl_dst / label.name)
