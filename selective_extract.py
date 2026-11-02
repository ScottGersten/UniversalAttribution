import os
import zipfile
from pathlib import Path

# Paths to your pilot manifests
manifest_paths = [
    "dataset/GenImage/split1_pilot/annotations/split1_pilot_val.txt",
    "dataset/GenImage/split1_pilot/annotations/split1_pilot_val_out_all.txt",
    "dataset/GenImage/split1_pilot/annotations/split1_pilot_train.txt",
]

# Collect set of required paths normalized
needed_files = set()
for mpath in manifest_paths:
    if os.path.exists(mpath):
        with open(mpath, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    # Strip leading './' if present
                    clean_path = parts[0].lstrip("./").replace("\\", "/")
                    needed_files.add(clean_path)

print(f"Total target images to extract: {len(needed_files)}")

# Directory where your downloaded val.zip files sit
zips_dir = Path("downloads")
output_root = Path(".")  # Will recreate the dataset/GenImage/... structure

for zip_file in zips_dir.glob("*.zip"):
    print(f"Scanning {zip_file.name}...")
    with zipfile.ZipFile(zip_file, "r") as zf:
        for member in zf.namelist():
            norm_member = member.replace("\\", "/")
            # Find if this file in the zip matches the end of any required path
            matched = [p for p in needed_files if p.endswith(norm_member) or norm_member.endswith(p)]
            if matched:
                target_dest = output_root / matched[0]
                target_dest.parent.mkdir(parents=True, exist_ok=True)
                with open(target_dest, "wb") as f_out:
                    f_out.write(zf.read(member))
                needed_files.remove(matched[0])

print(f"Extraction complete! Remaining missing files: {len(needed_files)}")