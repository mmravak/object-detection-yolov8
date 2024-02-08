from ultralytics import YOLO

# Load a OpenImagesv7-pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Display model information
model.info()

# Train the model on the dataset for 20 epochs
results = model.train(data='datasets\open-images-v7\dataset.yaml', epochs = 15, imgsz = 640)