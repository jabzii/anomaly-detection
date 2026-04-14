import os
import glob
import random
import shutil

dataset_path = r"d:\anomaly-detection\master_dataset"
excess_path = r"d:\anomaly-detection\master_dataset_excess"

# Target number of images per class for each split based on the minority class (buffalo)

for split in ["train", "val", "test"]:
    label_dir = os.path.join(dataset_path, "labels", split)
    image_dir = os.path.join(dataset_path, "images", split)
    
    if not os.path.exists(label_dir):
        continue
        
    excess_label_dir = os.path.join(excess_path, "labels", split)
    excess_image_dir = os.path.join(excess_path, "images", split)
    
    os.makedirs(excess_label_dir, exist_ok=True)
    os.makedirs(excess_image_dir, exist_ok=True)
    
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
    
    # 2. find minority class count
    counts = {c: len(imgs) for c, imgs in class_images.items() if len(imgs) > 0}
    if not counts:
        continue
        
    min_c = min(counts, key=counts.get)
    min_count = counts[min_c]
    
    print(f"\n--- Split {split} ---")
    print(f"Min class is {min_c} with {min_count} images.")
    
    # 3. Select images
    selected_images = set()
    class_current_counts = {c: 0 for c in range(6)}
    
    # First, shuffle all lists to ensure randomness
    for c in class_images:
        random.shuffle(class_images[c])
        
    # We iteratively pick images for classes that haven't reached min_count
    # Order classes by rarity to pick images for rare classes first
    sorted_classes = sorted(counts.keys(), key=lambda c: counts[c])
    
    for c in sorted_classes:
        for basename in class_images[c]:
            if class_current_counts[c] >= min_count:
                break
                
            if basename not in selected_images:
                # check if adding this image would exceed the targets significantly?
                # since we are balancing, going a bit over min_count for other classes is unavoidable
                # due to co-occurrence, but we stop strictly when class c reaches min_count.
                selected_images.add(basename)
                for occ in image_classes[basename]:
                    class_current_counts[occ] += 1
                    
    print(f"Summary after selection:")
    for c in range(6):
        if len(class_images[c]) > 0:
            print(f"  Class {c}: {class_current_counts[c]} images selected (out of original {counts.get(c, 0)})")
        
    # 4. Move unselected images
    moved_count = 0
    for txt_file in txt_files:
        basename = os.path.basename(txt_file)
        if basename not in selected_images:
            # move txt
            txt_src = os.path.join(label_dir, basename)
            txt_dst = os.path.join(excess_label_dir, basename)
            shutil.move(txt_src, txt_dst)
            
            # move img
            name_no_ext = os.path.splitext(basename)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                img_src = os.path.join(image_dir, name_no_ext + ext)
                if os.path.exists(img_src):
                    img_dst = os.path.join(excess_image_dir, name_no_ext + ext)
                    shutil.move(img_src, img_dst)
                    break
            moved_count += 1
            
    print(f"Moved {moved_count} unselected items to {excess_path}")
    
print("\nDone bounding and moving excess images.")
