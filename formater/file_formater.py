import os
import shutil
import random

# Paths
source_dir = "datasets/buffalo_old"
output_dir = "datasets/buffalo"

os.makedirs(output_dir, exist_ok=True)
# Split ratio
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# Get all image files
images = [f for f in os.listdir(source_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

# Shuffle images
random.seed(42)
random.shuffle(images)

# Split
train_size = int(len(images) * train_ratio)
val_size = int(len(images) * val_ratio)
train_images = images[:train_size]
val_images = images[train_size:train_size + val_size]
test_images = images[train_size + val_size:]

def copy_files(image_list, split):
    for img in image_list:
        name_ext = os.path.splitext(img)
        label = name_ext[0] + ".txt"

        img_src = os.path.join(source_dir, img)
        label_src = os.path.join(source_dir, label)

        img_dst = os.path.join(output_dir, "images", split, img)
        label_dst = os.path.join(output_dir, "labels", split, label)

        # Copy image
        if os.path.exists(img_src):
            shutil.copy(img_src, img_dst)

        # Copy label if exists
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)

# Copy files
print(f"Copying {len(train_images)} training images...")
copy_files(train_images, "train")
print(f"Copying {len(val_images)} validation images...")
copy_files(val_images, "val")
print(f"Copying {len(test_images)} testing images...")
copy_files(test_images, "test")

# Generate data.yaml
data_yaml_path = os.path.join(output_dir, "data.yaml")
with open(data_yaml_path, 'w') as f:
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("\n")
    f.write("nc: 1\n")
    f.write("names: ['buffalo']\n")

print("✅ Dataset split and formatting completed!")