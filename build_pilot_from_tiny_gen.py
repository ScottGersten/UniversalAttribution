import os
import random
from datasets import load_dataset

print("Loading cached Tiny-GenImage...")
ds = load_dataset("TheKernel01/Tiny-GenImage")

base_dir = "dataset/GenImage"
train_manifest_dir = os.path.join(base_dir, "split1_pilot_train", "annotations")
val_manifest_dir = os.path.join(base_dir, "split1_pilot_val", "annotations")
os.makedirs(train_manifest_dir, exist_ok=True)
os.makedirs(val_manifest_dir, exist_ok=True)

try:
    gen_names = ds['train'].features['generator'].names
except AttributeError:
    gen_names = ['Real', 'ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD14', 'SD15', 'VQDM', 'Wukong']

image_records = {gen: {'ai': [], 'nature': []} for gen in gen_names}

print("Scanning existing images...")
for split in ds.keys():
    for i, item in enumerate(ds[split]):
        img = item['image']
        label_val = item['label']
        gen_name = gen_names[item['generator']]

        img_type = "nature" if label_val == 0 else "ai"
        save_dir = os.path.join(base_dir, gen_name, "val", img_type)
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{split}_{i}.png"
        filepath = os.path.join(save_dir, filename)

        if not os.path.exists(filepath):
            img.save(filepath)

        rel_path = filepath.replace("\\", "/")
        if not rel_path.startswith("./"):
            rel_path = "./" + rel_path

        image_records[gen_name][img_type].append(rel_path)

print("Building corrected manifests...")
train_lines, val_lines, out_lines = [], [], []

# 1. Real Images (Label 0)
real_imgs = image_records['Real']['nature']
random.shuffle(real_imgs)
real_split = int(len(real_imgs) * 0.8)
for p in real_imgs[:real_split]: train_lines.append(f"{p}\t0\n")
for p in real_imgs[real_split:]: val_lines.append(f"{p}\t0\n")

# 2. Known AI Images (Labels 1 to 4)
known_ai = {'Wukong': 1, 'Midjourney': 2, 'VQDM': 3, 'SD14': 4}
for gen_name, label_idx in known_ai.items():
    ai_imgs = image_records[gen_name]['ai']
    random.shuffle(ai_imgs)
    split_idx = int(len(ai_imgs) * 0.8)
    for p in ai_imgs[:split_idx]: train_lines.append(f"{p}\t{label_idx}\n")
    for p in ai_imgs[split_idx:]: val_lines.append(f"{p}\t{label_idx}\n")

# 3. Unknown AI Images (Labels 5 to 8)
unknown_ai = {'SD15': 5, 'BigGAN': 6, 'ADM': 7, 'GLIDE': 8}
for gen_name, label_idx in unknown_ai.items():
    ai_imgs = image_records[gen_name]['ai']
    for p in ai_imgs: out_lines.append(f"{p}\t{label_idx}\n")

with open(os.path.join(train_manifest_dir, "split1_pilot_train.txt"), "w") as f:
    f.writelines(train_lines)
with open(os.path.join(val_manifest_dir, "split1_pilot_val.txt"), "w") as f:
    f.writelines(val_lines)
with open(os.path.join(val_manifest_dir, "split1_pilot_val_out_all.txt"), "w") as f:
    f.writelines(out_lines)

print(f"Train: {len(train_lines)} | Val: {len(val_lines)} | Out-All: {len(out_lines)}")