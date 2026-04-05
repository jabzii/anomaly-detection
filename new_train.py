from pathlib import Path
import os

from ultralytics import YOLO


def resolve_data_yaml() -> str:
    """Find the merged YOLO dataset config file."""
    project_root = Path(__file__).resolve().parent
    candidates = [
        project_root / "datasets" / "master_dataset" / "data.yaml",
        project_root / "master_dataset" / "data.yaml",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find data.yaml. Expected one of:\n"
        "- datasets/master_dataset/data.yaml\n"
        "- master_dataset/data.yaml\n"
        "Run organize_yolo_data.py first to generate the merged dataset."
    )


def main() -> None:
    model_name = os.getenv("YOLO_MODEL", "yolo11s.pt")
    epochs = int(os.getenv("EPOCHS", "60"))
    imgsz = int(os.getenv("IMGSZ", "640"))
    data_yaml = resolve_data_yaml()

    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
    )


if __name__ == "__main__":
    main()
