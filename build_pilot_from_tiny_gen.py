import os
import random
from datasets import load_dataset

print("Loading cached Tiny-GenImage...")
ds = load_dataset("TheKernel01/Tiny-GenImage")

base_dir = "dataset/GenImage"
manifest_dir = os.path.join(base_dir, "split1_pilot", "annotations")
os.makedirs(manifest_dir, exist_ok=True)

# Attempt to pull the exact string names of the generators from Hugging Face metadata
try:
    gen_names = ds['train'].features['generator'].names
except AttributeError:
    # Fallback to standard GenImage alphabetical order if metadata is missing
    gen_names = ['ADM', 'BigGAN', 'Midjourney', 'VQDM', 'glide', 
                 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'wukong']

print(f"Mapped generators: {gen_names}")

# The required splits according to our YAML config
known_gens = ['wukong', 'Midjourney', 'VQDM', 'stable_diffusion_v_1_4']
unknown_gens = ['stable_diffusion_v_1_5', 'BigGAN', 'ADM', 'glide']

# Map known generators to the integer IDs expected by linear.py
# 0 is always real/nature. 1 through 4 are the known AI generators.
label_map = {'real': 0, 'wukong': 1, 'Midjourney': 2, 'VQDM': 3, 'stable_diffusion_v_1_4': 4}

train_lines, val_lines, out_lines = [], [], []

# Dictionary to hold the file paths we generate
image_records = {gen: {'ai': [], 'nature': []} for gen in gen_names}

print("Saving images to disk...")
# Process all images in the dataset
for split in ds.keys():
    for i, item in enumerate(ds[split]):
        img = item['image']
        label_val = item['label'] # 0 for nature, 1 for ai
        gen_name = gen_names[item['generator']]

        img_type = "nature" if label_val == 0 else "ai"
        save_dir = os.path.join(base_dir, gen_name, "val", img_type)
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{split}_{i}.png"
        filepath = os.path.join(save_dir, filename)

        # Save the image bytes to a real file
        if not os.path.exists(filepath):
            img.save(filepath)

        # Create the relative path format the manifests expect (./dataset/...)
        rel_path = filepath.replace("\\", "/")
        if not rel_path.startswith("./"):
            rel_path = "./" + rel_path

        image_records[gen_name][img_type].append(rel_path)

print("Building manifests...")
# Generate Train and Val splits for Known classes (80% train / 20% val)
for gen_name in known_gens:
    if gen_name not in image_records: continue
    
    ai_imgs = image_records[gen_name]['ai']
    nature_imgs = image_records[gen_name]['nature']
    
    random.shuffle(ai_imgs)
    random.shuffle(nature_imgs)
    
    ai_split = int(len(ai_imgs) * 0.8)
    nature_split = int(len(nature_imgs) * 0.8)
    
    label_idx = label_map[gen_name]
    
    # Write Train lines
    for p in ai_imgs[:ai_split]: train_lines.append(f"{p}\t{label_idx}\n")
    for p in nature_imgs[:nature_split]: train_lines.append(f"{p}\t0\n")
        
    # Write Val lines
    for p in ai_imgs[ai_split:]: val_lines.append(f"{p}\t{label_idx}\n")
    for p in nature_imgs[nature_split:]: val_lines.append(f"{p}\t0\n")

# Generate Out-of-Distribution split for Unknown classes
for gen_name in unknown_gens:
    if gen_name not in image_records: continue
    # Unknown classes only need their AI images dumped into the out_all file
    for p in image_records[gen_name]['ai']:
        out_lines.append(f"{p}\t{gen_name}\n")

# Save the text files
with open(os.path.join(manifest_dir, "split1_pilot_train.txt"), "w") as f:
    f.writelines(train_lines)
with open(os.path.join(manifest_dir, "split1_pilot_val.txt"), "w") as f:
    f.writelines(val_lines)
with open(os.path.join(manifest_dir, "split1_pilot_val_out_all.txt"), "w") as f:
    f.writelines(out_lines)

print(f"DONE! Manifests saved to {manifest_dir}")
print(f"Train: {len(train_lines)} | Val: {len(val_lines)} | Out-All: {len(out_lines)}")