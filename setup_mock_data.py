import os
from PIL import Image

base_dir = "./dataset/GenImage"
splits = ["split1_train", "split1_val", "split1_test"]
known = ['real', 'wukong', 'Midjourney', 'VQDM', 'stable_diffusion_v_1_4']
unknown = ['stable_diffusion_v_1_5', 'BigGAN', 'ADM', 'glide']

# 1. Create directories
for s in splits:
    os.makedirs(f"{base_dir}/{s}/annotations", exist_ok=True)
    os.makedirs(f"{base_dir}/{s}/images", exist_ok=True)

# 2. Generate sample images and integer-labeled manifests
def create_dummy_images(split_name, class_list, count=6, start_label=0):
    entries = []
    for cls_idx, cls_name in enumerate(class_list):
        label_id = start_label + cls_idx
        for i in range(count):
            img_rel_path = f"{split_name}/images/{cls_name}_{i}.jpg"
            img_disk_path = os.path.join(base_dir, img_rel_path)
            
            # Simple RGB color variation per class
            color = ((label_id * 45) % 255, (label_id * 35 + 60) % 255, (label_id * 75 + 110) % 255)
            img = Image.new("RGB", (224, 224), color=color)
            img.save(img_disk_path)
            
            clean_path = f"{base_dir}/{img_rel_path}".replace("\\", "/")
            # Tab delimiter with integer label
            entries.append(f"{clean_path}\t{label_id}\n")
    return entries

# 3. Create splits
# Known classes get indices 0..4
train_entries = create_dummy_images("split1_train", known, count=8, start_label=0)
val_entries = create_dummy_images("split1_val", known, count=4, start_label=0)

# Unknown classes get distinct negative or out-of-set indices (e.g. 5..8)
val_out_entries = create_dummy_images("split1_val", unknown, count=4, start_label=len(known))
test_out_entries = create_dummy_images("split1_test", unknown, count=4, start_label=len(known))

# 4. Write manifest files
with open(f"{base_dir}/split1_train/annotations/split1_train.txt", "w") as f:
    f.writelines(train_entries)

with open(f"{base_dir}/split1_val/annotations/split1_val.txt", "w") as f:
    f.writelines(val_entries)

with open(f"{base_dir}/split1_val/annotations/split1_val_out_all.txt", "w") as f:
    f.writelines(val_out_entries)

with open(f"{base_dir}/split1_test/annotations/split1_test_out_all.txt", "w") as f:
    f.writelines(test_out_entries)
