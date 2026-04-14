import os
import cv2
import shutil
from pathlib import Path

def move_corrupted(dataset_dir):
    dataset_path = Path(dataset_dir)
    image_dir = dataset_path / "images"
    label_dir = dataset_path / "labels"
    corrupt_dir = dataset_path.parent / "corrupted_data"
    corrupt_dir.mkdir(exist_ok=True)
    
    corrupted_count = 0
    for split in ["train", "val", "test"]:
        split_img_dir = image_dir / split
        split_lbl_dir = label_dir / split
        
        if not split_img_dir.exists():
            continue
            
        for img_path in split_img_dir.glob("*.*"):
            im = cv2.imread(str(img_path))
            if im is None:
                print(f"Corrupted image found: {img_path}")
                # Move image
                shutil.move(str(img_path), str(corrupt_dir / img_path.name))
                
                # Move corresponding label
                label_path = split_lbl_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.move(str(label_path), str(corrupt_dir / label_path.name))
                    
                corrupted_count += 1
                
    # Also remove cache files if any corrupted images were moved
    if corrupted_count > 0 or True: # Just clear cache to be safe
        for split in ["train", "val", "test"]:
            cache_path = label_dir / f"{split}.cache"
            if cache_path.exists():
                print(f"Removing cache file: {cache_path}")
                cache_path.unlink()
                
    print(f"Total corrupted images moved to {corrupt_dir}: {corrupted_count}")

if __name__ == "__main__":
    move_corrupted(r"D:\anomaly-detection\master_dataset")
