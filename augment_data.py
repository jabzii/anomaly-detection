import os
import cv2
import glob
import albumentations as A
import numpy as np

# Define paths
DATASET_PATH = "master_dataset"
IMAGES_DIR = os.path.join(DATASET_PATH, "images", "train")
LABELS_DIR = os.path.join(DATASET_PATH, "labels", "train")

# Define target augmentations per class to balance the dataset
# Based on current counts: buffalo(314), tiger(3089), wild_boar(6141), fire/smoke(11400), elephant(17159)
AUGMENTATION_MULTIPLIER = {
    0: 1,   # smoke
    1: 1,   # fire
    2: 30,  # buffalo (heavily augment due to extreme minority)
    3: 0,   # elephant (already has ~17k, no need to augment)
    4: 3,   # tiger
    5: 2    # wild boar
}

# Define the augmentation pipeline with bounding box support
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussianBlur(p=0.2),
    A.CLAHE(p=0.3),
    A.HueSaturationValue(p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

def read_yolo_labels(label_path):
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                # YOLO format: x_center, y_center, width, height
                bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                bboxes.append(bbox)
                class_labels.append(class_id)
    return bboxes, class_labels

def save_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            # Ensure coordinates are within [0, 1] bounds
            x_c = max(0.0, min(1.0, bbox[0]))
            y_c = max(0.0, min(1.0, bbox[1]))
            w = max(0.0, min(1.0, bbox[2]))
            h = max(0.0, min(1.0, bbox[3]))
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

def augment_dataset():
    print("Starting data augmentation...")
    label_files = glob.glob(os.path.join(LABELS_DIR, "*.txt"))
    # Filter out already augmented files to prevent recursive augmentation
    label_files = [f for f in label_files if "aug_" not in os.path.basename(f)]
    
    total_files = len(label_files)
    augmented_count = 0
    
    for idx, label_path in enumerate(label_files):
        if idx % 500 == 0:
            print(f"Processing image {idx}/{total_files}...")
            
        base_name = os.path.splitext(os.path.basename(label_path))[0]
        # Common image extensions
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            temp_path = os.path.join(IMAGES_DIR, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                img_ext = ext
                break
                
        if not img_path:
            continue
            
        bboxes, class_labels = read_yolo_labels(label_path)
        if not bboxes:
            continue
            
        # Determine the number of augmentations based on the rarest class in the image
        # This helps target minority classes that appear alongside majority classes
        max_multiplier = 0
        for cls_id in class_labels:
            if cls_id in AUGMENTATION_MULTIPLIER:
                max_multiplier = max(max_multiplier, AUGMENTATION_MULTIPLIER[cls_id])
                
        if max_multiplier == 0:
            continue
            
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Convert BGR to RGB for Albumentations
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for i in range(max_multiplier):
            try:
                # Apply augmentation
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']
                
                if not aug_bboxes:
                    continue # Skip if all bounding boxes were cut off
                    
                # Convert back to BGR for saving
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                
                # Save augmented image
                new_base_name = f"aug_{i}_{base_name}"
                new_img_path = os.path.join(IMAGES_DIR, new_base_name + img_ext)
                cv2.imwrite(new_img_path, aug_img_bgr)
                
                # Save augmented labels
                new_label_path = os.path.join(LABELS_DIR, new_base_name + ".txt")
                save_yolo_labels(new_label_path, aug_bboxes, aug_labels)
                
                augmented_count += 1
            except Exception as e:
                # Albumentations can occasionally fail on edge-case bounding boxes
                pass
                
    print(f"Data augmentation complete! Generated {augmented_count} new images.")

if __name__ == "__main__":
    augment_dataset()
