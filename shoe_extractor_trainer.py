import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import yaml
from pre_process_traning_data import process

def main():
    # Configurations for training
    epochs = 400
    patience = 0
    batch = 4
    imgsz = 640

    # Directories and file paths
    DATASET_DIR = "shoes_segmentation-5"
    TRAINED_MODELS_DIR = "trained_models"
    TRAINED_MODEL_NAME = "shoe-extractor-model-" + str(epochs) + "-epochs"
    DATA_YAML_PATH = os.path.join(DATASET_DIR, "data.yaml")

    # Process training data
    process(os.path.join(DATASET_DIR, "train"))

    # Load the number of classes from the data.yaml file
    with open(DATA_YAML_PATH) as data_file:
        num_classes = str(yaml.safe_load(data_file)['nc'])

    # Load the model and pre-trained weights
    model = YOLO("yolov8n-seg.yaml")
    model = YOLO('yolov8n-seg.pt')
    model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")
    model.to('cuda') # comment out if no gpu

    results = model.train(data=DATA_YAML_PATH,
                        project=TRAINED_MODELS_DIR,
                        name=TRAINED_MODEL_NAME,
                        epochs=epochs,
                        patience=patience,
                        imgsz=imgsz,
                        batch=batch)

if __name__ == '__main__':
    main()
