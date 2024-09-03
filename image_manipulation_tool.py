from PIL import Image, ImageDraw
import numpy as np


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

    # Create a mask with the same size as the image
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Convert the coordinates from float to integers
    polygon = [(int(tuple[0]), int(tuple[1])) for tuple in coordinates]

    # Draw the polygon on the mask
    draw.polygon(polygon, outline=1, fill=255)

    # Create an alpha mask
    alpha = mask.copy()

    # Create a new image with an RGBA (4-channel) mode
    rgba_image = Image.new("RGBA", image.size)

    # Paste the image onto the new image using the alpha mask as the transparency mask
    rgba_image = Image.composite(image, rgba_image, alpha)

    return rgba_image