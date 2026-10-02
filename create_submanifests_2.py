import os
from collections import defaultdict

def create_train_val_from_single_manifest(input_manifest, train_out, val_out, train_max=200, val_max=100):
    """Splits a single manifest into train and val without overlap."""
    class_counts = defaultdict(int)
    train_lines = []
    val_lines = []

    with open(input_manifest, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            
            label = parts[1]
            
            # First, fill up the training quota for this class
            if class_counts[label] < train_max:
                train_lines.append(line)
                class_counts[label] += 1
            # Once training is full, fill up the validation quota
            elif class_counts[label] < (train_max + val_max):
                val_lines.append(line)
                class_counts[label] += 1

    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    with open(train_out, 'w') as f:
        f.writelines(train_lines)
    with open(val_out, 'w') as f:
        f.writelines(val_lines)

    print(f"Created {train_out}: {len(train_lines)} training samples.")
    print(f"Created {val_out}: {len(val_lines)} validation samples.")

def subsample_manifest(input_manifest, output_manifest, max_per_class=100):
    """Standard subsampler for the out-of-distribution file."""
    class_counts = defaultdict(int)
    kept_lines = []

    with open(input_manifest, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            label = parts[1]
            if class_counts[label] < max_per_class:
                kept_lines.append(line)
                class_counts[label] += 1

    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    with open(output_manifest, 'w') as f:
        f.writelines(kept_lines)

    print(f"Created {output_manifest}: {len(kept_lines)} OOD samples.")

if __name__ == "__main__":
    # 1. Generate BOTH Train and Val from the official validation file (Known Classes)
    create_train_val_from_single_manifest(
        input_manifest="dataset/GenImage/split1_val/annotations/split1_val.txt",
        train_out="dataset/GenImage/split1_pilot/annotations/split1_pilot_train.txt",
        val_out="dataset/GenImage/split1_pilot/annotations/split1_pilot_val.txt",
        train_max=200,  # 200 images per class for training
        val_max=100     # 100 different images per class for testing
    )

    # 2. Generate Out-of-distribution validation (Unknown Classes)
    subsample_manifest(
        input_manifest="dataset/GenImage/split1_val/annotations/split1_val_out_all.txt",
        output_manifest="dataset/GenImage/split1_pilot/annotations/split1_pilot_val_out_all.txt",
        max_per_class=100
    )