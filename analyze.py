import os
import glob
from collections import defaultdict
import json

dataset_path = r"d:\anomaly-detection\master_dataset"
results = {}

for split in ["train", "val", "test"]:
    label_dir = os.path.join(dataset_path, "labels", split)
    
    image_class_counts = defaultdict(int)
    filename_classes = defaultdict(int)
    
    for txt_file in glob.glob(os.path.join(label_dir, "*.txt")):
        basename = os.path.basename(txt_file)
        prefix = basename.split('_')[0]
        filename_classes[prefix] += 1
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            classes_in_file = set()
            for line in lines:
                if not line.strip(): continue
                try:
                    class_id = int(line.split()[0])
                    classes_in_file.add(class_id)
                except ValueError:
                    pass
            for c in classes_in_file:
                image_class_counts[c] += 1
                
    results[split] = {
        "filename": dict(filename_classes),
        "images_per_class": dict(image_class_counts)
    }

print(json.dumps(results, indent=2))
