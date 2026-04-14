import os
import glob
import random
import shutil

dataset_path = r"d:\anomaly-detection\master_dataset"
excess_path = r"d:\anomaly-detection\master_dataset_excess"

print("Restoring any previously moved excess images...")
for split in ["train", "val", "test"]:
    excess_label_dir = os.path.join(excess_path, "labels", split)
    excess_image_dir = os.path.join(excess_path, "images", split)
    label_dir = os.path.join(dataset_path, "labels", split)
    image_dir = os.path.join(dataset_path, "images", split)
    
    if os.path.exists(excess_label_dir) and os.path.exists(label_dir):
        for f in os.listdir(excess_label_dir):
            shutil.move(os.path.join(excess_label_dir, f), os.path.join(label_dir, f))
    if os.path.exists(excess_image_dir) and os.path.exists(image_dir):
        for f in os.listdir(excess_image_dir):
            shutil.move(os.path.join(excess_image_dir, f), os.path.join(image_dir, f))

print("Cleaning up old duplicates...")
for split in ["train", "val", "test"]:
    label_dir = os.path.join(dataset_path, "labels", split)
    image_dir = os.path.join(dataset_path, "images", split)
    if not os.path.exists(label_dir):
        continue
    for txt_file in glob.glob(os.path.join(label_dir, "*_dup*.txt")):
        try:
            os.remove(txt_file)
        except OSError:
            pass
        name_no_ext = os.path.splitext(os.path.basename(txt_file))[0]
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            img_path = os.path.join(image_dir, name_no_ext + ext)
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except OSError:
                    pass

# Target number of images per class for each split based on the majority class
for split in ["train", "val", "test"]:
    label_dir = os.path.join(dataset_path, "labels", split)
    image_dir = os.path.join(dataset_path, "images", split)
    
    if not os.path.exists(label_dir):
        continue
        
    # 1. Read all labels and map image -> set of classes
    image_classes = {}
    class_images = {c: [] for c in range(6)}
    
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    for txt_file in txt_files:
        basename = os.path.basename(txt_file)
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        classes = set()
        for line in lines:
            if not line.strip(): continue
            try:
                c = int(line.split()[0])
                if c in class_images:
                    classes.add(c)
            except ValueError:
                pass
        
        image_classes[basename] = classes
        for c in classes:
            class_images[c].append(basename)
            
    if not txt_files: 
        continue
    
    # 2. find majority class count
    counts = {c: len(imgs) for c, imgs in class_images.items() if len(imgs) > 0}
    if not counts:
        continue
        
    max_c = max(counts, key=counts.get)
    max_count = counts[max_c]
    
    print(f"\n--- Split {split} ---")
    print(f"Max class is {max_c} with {max_count} images.")
    
    # 3. Duplicate images
    class_current_counts = counts.copy()
    
    for c in class_images:
        random.shuffle(class_images[c])
        
    sorted_classes = sorted(counts.keys(), key=lambda c: counts[c])
    
    duplicate_count = 0
    for c in sorted_classes:
        available_images = class_images[c]
        if not available_images:
            continue
            
        while class_current_counts[c] < max_count:
            basename = random.choice(available_images)
            name_no_ext = os.path.splitext(basename)[0]
            
            duplicate_count += 1
            new_basename = f"{name_no_ext}_dup{duplicate_count}.txt"
            
            txt_src = os.path.join(label_dir, basename)
            txt_dst = os.path.join(label_dir, new_basename)
            shutil.copy(txt_src, txt_dst)
            
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                img_src = os.path.join(image_dir, name_no_ext + ext)
                if os.path.exists(img_src):
                    img_dst = os.path.join(image_dir, f"{name_no_ext}_dup{duplicate_count}{ext}")
                    shutil.copy(img_src, img_dst)
                    break
                    
            for occ in image_classes[basename]:
                class_current_counts[occ] += 1
                    
    print(f"Summary after oversampling:")
    for c in range(6):
        if len(class_images[c]) > 0:
            print(f"  Class {c}: {class_current_counts[c]} images (original {counts.get(c, 0)})")
        
    print(f"Duplicated {duplicate_count} items to match majority class.")
    
print("\nDone balancing dataset by oversampling.")
