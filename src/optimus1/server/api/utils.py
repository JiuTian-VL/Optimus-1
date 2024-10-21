import base64
import logging
import os
from typing import Any, Dict, List

import shortuuid

logger = logging.getLogger(__name__)


def base64lst2img_path(base64_lst: List[str] | None):
    """
    Convert a list of base64-encoded images to actual image files and save them.

    Args:
        base64_lst (List[str]): A list of base64-encoded images.

    Returns:
        List[str]: A list of image file names that were saved.

    """
    image_root = "api/imgs"
    image_file_names = []
    if base64_lst is None:
        return []

    for idx, image_byte in enumerate(base64_lst):
        uuid = shortuuid.uuid()
        imgdata = base64.b64decode(image_byte)
        image_file = os.path.join(image_root, f"{uuid}_{idx}.jpg")

        with open(image_file, "wb") as f:
            f.write(imgdata)
        # logger.info(f"Save image to {image_file}")

        image_file_names.append(image_file)
    return image_file_names


def base64_to_image(
    rgb_images: List[Dict[str, Any]],
    image_root: str = "api/imgs",
    task: str = "plan|action|reflection|replan",
    step: int = 0,
) -> List[str]:
    """
    Convert a list of base64-encoded images to actual image files and save them.

    Args:
        rgb_images (List[Dict[str, Any]]): A list of dictionaries containing base64-encoded images.
        image_root (str, optional): The root directory where the image files will be saved. Defaults to "api/imgs".
        task (str, optional): The task name used in the image file name. Defaults to "plan|action|reflection|replan".
        step (int, optional): The step number used in the image file name. Defaults to 0.

    Returns:
        List[str]: A list of image file names that were saved.

    """
    os.makedirs(image_root, exist_ok=True)
    task = task.replace(" ", "_")
    image = rgb_images[-1]
    uuid = shortuuid.uuid()[:5]

    image_byte = image["image"]
    imgdata = base64.b64decode(image_byte)
    image_file = os.path.join(image_root, f"{uuid}_{task}_{step}.jpg")

    with open(image_file, "wb") as f:
        f.write(imgdata)
    return [image_file]


def base64_to_image2(
    rgb_images: List[Dict[str, Any]], image_root: str = "api/imgs"
) -> List[str]:
    if not rgb_images:
        print("none images")
        return []  # 如果输入列表为空，则直接返回空列表

    uuid = shortuuid.uuid()
    last_image_file = ""  # 初始化变量来存储最后一张图像的文件名

    # 仅处理最后一张图片

    image = rgb_images[-1]  # 获取最后一张图像的数据
    image_byte = image["image"]
    imgdata = base64.b64decode(image_byte)

    # 构建文件名
    if "yaw" in image and "pitch" in image:
        yaw = image["yaw"]
        pitch = image["pitch"]
        last_image_file = os.path.join(image_root, f"{uuid}_{yaw}_{pitch}.jpg")
    else:
        last_image_file = os.path.join(image_root, f"{uuid}.jpg")

    # 打印并保存图像
    print("Save image to ", last_image_file)
    with open(last_image_file, "wb") as f:
        f.write(imgdata)

    # 由于函数定义返回一个列表，这里我们返回包含最后一张图像文件名的列表
    # 如果你只需要返回一个字符串，这里可以直接返回 last_image_file
    return [last_image_file]
