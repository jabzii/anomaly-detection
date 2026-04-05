"""
YOLO label class-id remapper.

Examples:
  python formater/change_box_no.py \
	  --dataset "/mnt/myworld/dev/projects/anomaly-detection/datasets/tiger" \
	  --from-id 0 --to-id 4

  python formater/change_box_no.py \
	  --dataset "/mnt/myworld/dev/projects/anomaly-detection/datasets/elephant-dataset-yolov" \
	  --from-id 0 --to-id 3
"""

from __future__ import annotations

import argparse
from pathlib import Path


def remap_class_ids(dataset_path: Path, from_id: str, to_id: str) -> tuple[int, int]:
	"""Replace YOLO class id in all label text files under any 'labels' folder."""
	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

	label_dirs = [p for p in dataset_path.rglob("labels") if p.is_dir()]
	if not label_dirs:
		raise FileNotFoundError(f"No 'labels' directories found under: {dataset_path}")

	files_changed = 0
	lines_changed = 0

	for labels_dir in label_dirs:
		for file_path in labels_dir.rglob("*.txt"):
			lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()

			changed_here = 0
			updated_lines: list[str] = []
			for line in lines:
				if not line.strip():
					updated_lines.append(line)
					continue

				parts = line.split()
				if parts and parts[0] == from_id:
					parts[0] = to_id
					updated_lines.append(" ".join(parts))
					changed_here += 1
				else:
					updated_lines.append(line)

			if changed_here > 0:
				file_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
				files_changed += 1
				lines_changed += changed_here

	return files_changed, lines_changed


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Change YOLO class-id values in label files.")
	parser.add_argument("--dataset", required=True, help="Path to dataset root folder")
	parser.add_argument("--from-id", required=True, help="Current class id to replace (e.g. 0)")
	parser.add_argument("--to-id", required=True, help="New class id value (e.g. 5)")
	return parser


def main() -> None:
	args = build_parser().parse_args()
	dataset_path = Path(args.dataset).expanduser().resolve()

	files_changed, lines_changed = remap_class_ids(
		dataset_path=dataset_path,
		from_id=str(args.from_id),
		to_id=str(args.to_id),
	)

	print(
		f"Updated class id {args.from_id} -> {args.to_id} "
		f"in {lines_changed} lines across {files_changed} files"
	)


if __name__ == "__main__":
	main()

