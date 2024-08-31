from PIL import Image, ImageOps
import numpy as np


def apply_masks_to_image(image_path, masks_tensor, output_path):
    image = Image.open(image_path).convert("RGBA")
    image_np = np.array(image)
    masks_np = masks_tensor.cpu().numpy()
    if masks_np.ndim > 2:
        combined_mask = np.any(masks_np > 0, axis=0).astype(np.uint8) * 255
    else:
        combined_mask = masks_np.astype(np.uint8) * 255
    image_np[:, :, 3] = combined_mask
    masked_image = Image.fromarray(image_np)
    masked_image.save(output_path, format="PNG")


def crop_to_nbyn(image_path: str, output_path:str, n:int) -> bool:
    """
    Given a path to an image, 'image_path', it crops the image to 'n' by 'n' size,
    and saves the cropped output to 'output_path'. If image is smaller than n by n, then return false. 
    After successful crop, return true.
    """
    image = Image.open(image_path)
    width, height = image.size
    if width < n or height < n: return False

    left = max((width - n) // 2, 0)
    upper = max((height - n) // 2, 0)
    right = min(left + n, width)
    lower = min(upper + n, height)

    cropped_image = image.crop((left, upper, right, lower))
    cropped_image.save(output_path)

    return True


def resize_and_pad(image_path, output_path, target_size=(640, 640)):
    image = Image.open(image_path)
    original_size = image.size
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    resized_image = image.resize(new_size, Image.LANCZOS)
    new_image = Image.new("RGB", target_size, (255, 255, 255))
    paste_position = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    new_image.paste(resized_image, paste_position)
    new_image.save(output_path)

