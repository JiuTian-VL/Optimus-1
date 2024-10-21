from typing import List, Union

import cv2
import numpy as np
import av

FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS = 20


def save_frames_as_video(
    frames: list,
    savefile_path: str,
    fps: int = FPS,
    to_bgr: bool = False,
    fx: float = 1.0,
    fy: float = 1.0,
) -> None:
    """Save a list of frames as a video to savefile_path"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    first = cv2.resize(frames[0], None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    out = cv2.VideoWriter(savefile_path, fourcc, fps, (first.shape[1], first.shape[0]))
    for frame in frames:
        frame = np.uint8(frame)
        if to_bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # type: ignore
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame,
            None,
            fx=fx,
            fy=fy,
            interpolation=cv2.INTER_LINEAR,
        )  # type: ignore
        out.write(frame)
    out.release()


def created_fitted_text_image(
    desired_width,
    text,
    thickness=2,
    background_color=(255, 255, 255),
    text_color=(0, 0, 0),
    height_padding=20,
):
    """Create an image with text fitted to the desired width."""
    font_scale = 0.1
    text_size, _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    text_width, _ = text_size
    pad = desired_width // 5
    while text_width < desired_width - pad:
        font_scale += 0.1
        text_size, _ = cv2.getTextSize(text, FONT, font_scale, thickness)
        text_width, _ = text_size
    image = np.zeros((text_size[1] + 2 * height_padding, desired_width, 3), dtype=np.uint8)
    image[:] = background_color
    org = ((image.shape[1] - text_width) // 2, image.shape[0] - height_padding)
    return cv2.putText(image, text, org, FONT, font_scale, text_color, thickness)


def create_video_frame(gameplay_pov, prompt):
    """Creates a frame for the generated video with the gameplay POV and the prompt text on the right side."""
    frame = cv2.cvtColor(gameplay_pov, cv2.COLOR_RGB2BGR)
    prompt_section = created_fitted_text_image(
        frame.shape[1] // 2,
        prompt,
        background_color=(0, 0, 0),
        text_color=(255, 255, 255),
    )
    pad_top_height = (frame.shape[0] - prompt_section.shape[0]) // 2
    pad_top = np.zeros((pad_top_height, prompt_section.shape[1], 3), dtype=np.uint8)
    pad_bottom_height = frame.shape[0] - pad_top_height - prompt_section.shape[0]
    pad_bottom = np.zeros((pad_bottom_height, prompt_section.shape[1], 3), dtype=np.uint8)
    prompt_section = np.vstack((pad_top, prompt_section, pad_bottom))
    frame = np.hstack((frame, prompt_section))
    return frame


def write_video(
    file_name: str,
    frames: Union[List[np.ndarray], bytes],
    width: int = 640,
    height: int = 360,
    fps: int = FPS,
) -> None:

    """Write video frames to video files. """
    with av.open(file_name, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        for frame in frames:
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
