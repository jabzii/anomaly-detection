import os
from pathlib import Path

def fix_labels(labels_path):
    print(f"Checking labels in {labels_path}...")
    fixed_count = 0
    total_count = 0
    
    for label_file in Path(labels_path).rglob("*.txt"):
        total_count += 1
        with open(label_file, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        modified = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            cls = parts[0]
            coords = []
            for coord in parts[1:]:
                val = float(coord)
                if val < 0.0:
                    val = 0.0
                    modified = True
                elif val > 1.0:
                    val = 1.0
                    modified = True
                coords.append(f"{val:.6f}")
            
            new_lines.append(f"{cls} {' '.join(coords)}")
        
        if modified:
            with open(label_file, "w") as f:
                f.write("\n".join(new_lines) + "\n")
            fixed_count += 1
            # print(f"Fixed: {label_file}")
            
    print(f"Total files: {total_count}, Fixed files: {fixed_count}")

if __name__ == "__main__":
    fix_labels("master_dataset/labels")
