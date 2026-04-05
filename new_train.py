from ultralytics import YOLO

# 1. Load the model (yolo11s.pt)
# This loads pretrained weights for transfer learning
model = YOLO('yolo11s.pt')

# 2. Train the model
# Matches your parameters: data, 60 epochs, and 640 image size
results = model.train(
    data='/content/data.yaml', 
    epochs=60, 
    imgsz=640
)
