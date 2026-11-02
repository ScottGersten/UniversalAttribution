import os
from collections import defaultdict

def subsample_manifest(input_manifest, output_manifest, max_per_class=100):
    class_counts = defaultdict(int)
    kept_lines = []

    with open(input_manifest, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            img_path, label = parts[0], parts[1]
            if class_counts[label] < max_per_class:
                kept_lines.append(line)
                class_counts[label] += 1

    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    with open(output_manifest, 'w') as f:
        f.writelines(kept_lines)

    print(f"Created {output_manifest}: {len(kept_lines)} samples across classes {dict(class_counts)}")
    return kept_lines

if __name__ == "__main__":
    # In-distribution known validation
    subsample_manifest(
        input_manifest="dataset/GenImage/split1_val/annotations/split1_val.txt",
        output_manifest="dataset/GenImage/split1_pilot/annotations/split1_pilot_val.txt",
        max_per_class=100
    )

    # Out-of-distribution unknown validation
    subsample_manifest(
        input_manifest="dataset/GenImage/split1_val/annotations/split1_val_out_all.txt",
        output_manifest="dataset/GenImage/split1_pilot/annotations/split1_pilot_val_out_all.txt",
        max_per_class=100
    )