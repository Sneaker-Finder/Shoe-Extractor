from ultralytics import YOLO
import os
from image_manipulation_tool import apply_masks_to_image, crop_to_640, resize_and_pad


def extract_shoe(img_path, model_name):
    model_path = os.path.join('trained_models', model_name, 'weights', 'best.pt')

    model = YOLO("yolov8n-seg.pt")
    model = YOLO(model_path)

    #crop_to_640(img_path, img_path)
    resized_img_path = "resized-"+img_path
    resize_and_pad(img_path, "resized-"+img_path)
    results = model(resized_img_path)         

    for result in results:
        masks = result.masks.data  # Masks object for segmentation masks outputs
        apply_masks_to_image(resized_img_path, masks, resized_img_path.split(".")[0] + ".png")

extract_shoe("dunk.jpg", 'shoe-extractor-model-200-epochs')