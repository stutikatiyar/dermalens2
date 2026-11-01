import os
import pandas as pd
import shutil

# 🔥 YOUR PATHS
base_dir = r"D:\Downloads\WebDev\dermalens\backend\data"
image_dir_1 = r"C:\Users\LUNALUX\Downloads\archive (4)\HAM10000_images_part_1"
image_dir_2 = r"C:\Users\LUNALUX\Downloads\archive (4)\HAM10000_images_part_2"
csv_path = r"C:\Users\LUNALUX\Downloads\archive (4)\HAM10000_metadata.csv"

# label mapping
mapping = {
    "mel": "melanoma",
    "nv": "nevus",
    "bcc": "bcc",
    "akiec": "akiec",
    "bkl": "bkl",
    "df": "df",
    "vasc": "vasc"
}

df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    img_id = row["image_id"] + ".jpg"
    label = mapping[row["dx"]]

    # 🔥 find image in either folder
    src1 = os.path.join(image_dir_1, img_id)
    src2 = os.path.join(image_dir_2, img_id)

    if os.path.exists(src1):
        src = src1
    elif os.path.exists(src2):
        src = src2
    else:
        continue  # skip if not found

    # 🔥 split (80/20)
    split = "train" if hash(img_id) % 5 != 0 else "val"

    dst = os.path.join(base_dir, split, label)
    os.makedirs(dst, exist_ok=True)

    shutil.copy(src, os.path.join(dst, img_id))