"""
YOLO label class-id remapper.

Examples:
	# Automatic mode (no input arguments)
	python formater/change_box_no.py

	# Optional single-dataset mode
  python formater/change_box_no.py \
	  --dataset "/mnt/myworld/dev/projects/anomaly-detection/datasets/elephant-dataset-yolov" \
	  --from-id 0 --to-id 3
"""

from __future__ import annotations

import argparse
from pathlib import Path


# Runs automatically when no arguments are provided.
# Mapping target:
# smoke=0, fire=1, buffalo=2, elephant=3, tiger=4, wild_boar=5
DEFAULT_REMAP_JOBS: list[tuple[str, str, str]] = [
	("datasets/buffalo", "0", "2"),
	("datasets/elephant-dataset-yolov", "0", "3"),
	("datasets/tiger", "0", "4"),
	("datasets/Wild Boar Dataset", "0", "5"),
]


def get_project_root() -> Path:
	return Path(__file__).resolve().parents[1]


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
	parser = argparse.ArgumentParser(
		description=(
			"Change YOLO class-id values in label files. "
			"If no arguments are passed, default datasets are processed automatically."
		)
	)
	parser.add_argument("--dataset", help="Path to dataset root folder")
	parser.add_argument("--from-id", help="Current class id to replace (e.g. 0)")
	parser.add_argument("--to-id", help="New class id value (e.g. 5)")
	return parser


def main() -> None:
	args = build_parser().parse_args()

	# Single-dataset mode (all three arguments required)
	provided = [args.dataset is not None, args.from_id is not None, args.to_id is not None]
	if any(provided):
		if not all(provided):
			raise SystemExit("When using custom mode, provide --dataset, --from-id, and --to-id together.")

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
		return

	# Automatic mode (no arguments)
	project_root = get_project_root()
	total_files = 0
	total_lines = 0

	print("Running automatic class-id remap jobs...")
	for dataset_rel, from_id, to_id in DEFAULT_REMAP_JOBS:
		dataset_path = (project_root / dataset_rel).resolve()
		try:
			files_changed, lines_changed = remap_class_ids(dataset_path, from_id, to_id)
			total_files += files_changed
			total_lines += lines_changed
			print(
				f"[OK] {dataset_path}: {from_id} -> {to_id}, "
				f"{lines_changed} lines in {files_changed} files"
			)
		except FileNotFoundError as exc:
			print(f"[SKIP] {exc}")

	print(f"Done. Total updated: {total_lines} lines across {total_files} files")


if __name__ == "__main__":
	main()

