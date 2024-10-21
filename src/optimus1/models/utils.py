import numpy as np
from PIL import Image


def image2MineRLArray(image_path: str):
    """https://minerl.io/docs/environments/handlers.html#observations"""
    image = Image.open(image_path).convert("RGB")
    array = np.array(image, dtype=np.uint8)
    return array
