from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

# Ensure the model uses CPU
model.to('cpu')

# Train the model
model.train(
    data='dataset',  # Path to your dataset configuration file
    epochs=10,  # Number of epochs
    imgsz=224,  # Image size
    augment=True,  # Use augmented data
)

model.export(format='onnx',simplify=True)


