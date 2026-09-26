import os
import io
from PIL import Image, ImageFilter
import argparse

def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def apply_gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_downsample_upsample(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    low_res = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    return low_res.resize((w, h), Image.Resampling.BILINEAR)

def apply_degradation_tier(img: Image.Image, tier: str) -> Image.Image:
    """
    Tier 1: Mild compression (standard web sharing)
    Tier 2: Compounded moderate degradation (resizing + JPEG)
    Tier 3: Aggressive lossy degradation (blur + heavy downsampling + low-Q JPEG)
    """
    if tier == "tier1":
        return apply_jpeg_compression(img, quality=80)
    elif tier == "tier2":
        img_down = apply_downsample_upsample(img, scale=0.75)
        return apply_jpeg_compression(img_down, quality=60)
    elif tier == "tier3":
        img_blur = apply_gaussian_blur(img, radius=1.5)
        img_down = apply_downsample_upsample(img_blur, scale=0.50)
        return apply_jpeg_compression(img_down, quality=40)
    else:
        raise ValueError(f"Unknown tier: {tier}")

def degrade_manifest(input_manifest_path: str, output_manifest_path: str, output_image_dir: str, tier: str):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)

    with open(input_manifest_path, "r") as f:
        lines = f.readlines()

    degraded_lines = []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        img_path, label = parts[0], parts[1]

        if not os.path.exists(img_path):
            continue

        img_name = os.path.basename(img_path)
        out_img_name = f"{tier}_{img_name}"
        out_img_path = os.path.join(output_image_dir, out_img_name).replace("\\", "/")

        # Process and save degraded image
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            degraded_img = apply_degradation_tier(rgb_img, tier)
            degraded_img.save(out_img_path, format="JPEG")

        degraded_lines.append(f"{out_img_path}\t{label}\n")

    with open(output_manifest_path, "w") as f:
        f.writelines(degraded_lines)

    print(f"Generated {tier} manifest: {output_manifest_path} ({len(degraded_lines)} samples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate degraded test splits")
    parser.add_argument("--tier", choices=["tier1", "tier2", "tier3"], default="tier1")
    args = parser.parse_args()

    base_dir = "./dataset/GenImage"
    target_tier = args.tier

    # Perturb the validation known split
    degrade_manifest(
        input_manifest_path=f"{base_dir}/split1_val/annotations/split1_val.txt",
        output_manifest_path=f"{base_dir}/split1_{target_tier}/annotations/split1_{target_tier}.txt",
        output_image_dir=f"{base_dir}/split1_{target_tier}/images",
        tier=target_tier
    )

    # Perturb the validation unknown split
    degrade_manifest(
        input_manifest_path=f"{base_dir}/split1_val/annotations/split1_val_out_all.txt",
        output_manifest_path=f"{base_dir}/split1_{target_tier}/annotations/split1_{target_tier}_out_all.txt",
        output_image_dir=f"{base_dir}/split1_{target_tier}/images",
        tier=target_tier
    )