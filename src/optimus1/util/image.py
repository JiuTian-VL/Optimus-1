import base64
from io import BytesIO
from typing import List

import cv2
import numpy as np
from PIL import Image

REFLECTION_IMAGE_ROOT = "src/optimus1/memories/v1/reflection/img"


def img_lst2base64(img_lst: List[str]):
    return [img_path2base64(img_path) for img_path in img_lst]


def img_path2base64(img_path: str) -> str:
    """
    Convert an image file to a base64 encoded string.

    Args:
        img_path: The path to the image file.

    Returns:
        A base64 encoded string representation of the image.
    """
    with open(img_path, "rb") as f:
        img_data = f.read()
    base64_str = base64.b64encode(img_data)
    return base64_str.decode("utf-8")


def img2base64(img: np.ndarray) -> str:
    """
    Convert an image to a base64 encoded string.

    Args:
        img: The image to be converted.

    Returns:
        A base64 encoded string representation of the image.
    """
    img = Image.fromarray(img)  # type: ignore
    output_buffer = BytesIO()
    img.save(output_buffer, format="JPEG")  # type: ignore
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode("utf-8")


def base64_to_img(base64_str: str, img_file: str):
    img_data = base64.b64decode(base64_str)

    with open(img_file, "wb") as f:
        f.write(img_data)


def save_obs(array: np.ndarray, img_file_name: str):
    """
    Save an RGB image array to a file.

    Args:
        array (np.ndarray): The RGB image array to be saved.
        img_file_name (str): The name of the output image file.

    Returns:
        None
    """
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_file_name, array)
