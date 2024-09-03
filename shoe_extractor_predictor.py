import ultralytics
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def extract_shoe(img_paths: str | list[str], model_name: str) -> list[ultralytics.engine.results.Results]:
    """ 
    Takes in a list of image path, 'img_paths', and the 'model_name' to run,
    it returns the results as a list.
    
    Precondition: Make sure that the images in img_paths has the same dimension 
                  as the data the model has been trained with. 
                  This is noted in the args.yaml file in the model directory. (imgsz)
    """
    model_path = os.path.join('trained_models', model_name, 'weights', 'best.pt')

    model = ultralytics.YOLO("yolov8n-seg.pt")
    model = ultralytics.YOLO(model_path)

    return model(img_paths)         


def apply_mask(image_path: str, coordinates: list[list[float]]) -> Image:
    """
    Apply a polygon mask to an image.

    Args:
        image_path (str): The path to the image file.
        coordinates (list of tuples): A list of [x, y] list representing the mask .

    Returns:
        Image: The image with the applied mask.
    """
    image = Image.open(image_path).convert("RGBA")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    polygon = [(int(tuple[0]), int(tuple[1])) for tuple in coordinates]
    draw.polygon(polygon, outline=1, fill=255)

    alpha = mask.copy()
    rgba_image = Image.new("RGBA", image.size)
    rgba_image = Image.composite(image, rgba_image, alpha)

    return rgba_image


def apply_mask_for_each_isntance(result) -> None:
    """
    Function work in progress, detailed implementations of specific features are to be done. ru o
    """
    mask_list = result.masks.xy
    result.show()
    for mask in mask_list:
        apply_mask("dunk.jpg", mask).show()


# Example usage
# results = extract_shoe("dunk.jpg", 'shoe-extractor-model-400-epochs')
# apply_mask_for_each_isntance(results[0])