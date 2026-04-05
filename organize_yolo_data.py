import os
import shutil
import random
import glob


datasets_config = [
    {"name": "buffalo", "path": "datasets/buffalo", "class_map": {2: 2}},
    {"name": "elephant", "path": "datasets/elephant-dataset-yolov", "class_map": {3: 3}},
    {"name": "tiger", "path": "datasets/tiger", "class_map": {4: 4}},
    {"name": "wild_boar", "path": "datasets/Wild Boar Dataset", "class_map": {5: 5}},
    {"name": "fire_smoke", "path": "datasets/fire_smok", "class_map": {0: 0, 1: 1}}
]

class_names = ["smoke", "fire", "buffalo", "elephant", "tiger", "wild_boar"]
output_dir = "master_dataset"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

def find_dataset_pairs(dataset_path):
    """
    Dynamically finds image-label pairs inside chaotic folder structures seamlessly.
    """
    label_files = glob.glob(os.path.join(dataset_path, '**/*.txt'), recursive=True)
    pairs = []
    
    for label_file in label_files:
        if "readme" in label_file.lower() or "classes" in label_file.lower():
            continue
            
        # Common YOLO structures place images parallel to labels
        image_search_paths = [
            label_file.replace('.txt', '.jpg'),
            label_file.replace('.txt', '.png'),
            label_file.replace('.txt', '.jpeg'),
            label_file.replace('labels', 'images').replace('.txt', '.jpg'),
            label_file.replace('labels', 'images').replace('.txt', '.png'),
            label_file.replace('labels', 'images').replace('.txt', '.jpeg')
        ]
        
        img_path = None
        for p in image_search_paths:
            if os.path.exists(p):
                img_path = p
                break
                
        if img_path:
            pairs.append((img_path, label_file))
            
    return pairs

def main():
    print(f"[*] Creating unified YOLO master dataset at '{output_dir}'")
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    total_images_processed = 0

    for config in datasets_config:
        ds_name = config["name"]
        ds_path = config["path"]
        class_map = config["class_map"]
        
        print(f"\nProcessing subset: {ds_name} from '{ds_path}'...")
        
        if not os.path.exists(ds_path):
            print(f"   [!] Warning: Path {ds_path} does not exist, skipping subset.")
            continue
            
        pairs = find_dataset_pairs(ds_path)
        # Unique mapping only to prevent duplications if glob found it multiple times via symlinks
        pairs = list(set(pairs))
        print(f"   > Found {len(pairs)} image-label pairs.")
        
        random.seed(42)
        random.shuffle(pairs)
        
        train_size = int(len(pairs) * train_ratio)
        val_size = int(len(pairs) * val_ratio)
        
        splits = {
            "train": pairs[:train_size],
            "val": pairs[train_size:train_size+val_size],
            "test": pairs[train_size+val_size:]
        }
        
        for split, split_pairs in splits.items():
            for img_src, label_src in split_pairs:
                # Prefixing file to avoid collision if two datasets both have "001.jpg"
                base_name = f"{ds_name}_{os.path.basename(img_src)}"
                label_name = f"{ds_name}_{os.path.basename(label_src)}"
                
                img_dst = os.path.join(output_dir, "images", split, base_name)
                label_dst = os.path.join(output_dir, "labels", split, label_name)
                
                # Copy image securely
                shutil.copy(img_src, img_dst)
                
                # Rewrite label parsing old index to universal unified index!
                num_written = 0
                with open(label_src, 'r') as f_in, open(label_dst, 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            old_cls = int(parts[0])
                            if old_cls in class_map:
                                new_cls = class_map[old_cls]
                                parts[0] = str(new_cls)
                                f_out.write(" ".join(parts) + "\n")
                                num_written += 1
                                
                # If no valid labels were written (e.g. empty txt file or invalid classes mapping), remove it to prevent yolov8 empty warnings
                if num_written == 0:
                    os.remove(label_dst)
                                
        total_images_processed += len(pairs)
        print(f"   > Finished '{ds_name}' parsing successfully.")

    # Generate YOLO Training Configuration target data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as yf:
        yf.write("train: images/train\n")
        yf.write("val: images/val\n")
        yf.write("test: images/test\n\n")
        yf.write(f"nc: {len(class_names)}\n")
        yf.write(f"names: {class_names}\n")

    print(f"\n✅ Merge Successful! Processed {total_images_processed} files.")
    print(f"   Master YOLO directory organized at -> '{output_dir}'")
    print(f"   Hyperparam target configs at -> '{yaml_path}'")

if __name__ == "__main__":
    main()
