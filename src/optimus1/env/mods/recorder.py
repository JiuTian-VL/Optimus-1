import logging
import os
import threading
import uuid
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig

from optimus1.util.thread import MultiThreadServerAPI
from optimus1.util.utils import get_time, save_bin
from optimus1.util.video import create_video_frame, save_frames_as_video, write_video

from .mod import Mod


class RecorderMod(Mod):
    video_frames = []
    export_video: bool = True
    output_video_path: str = ""
    output_video_name: str = ""
    with_prompt: bool = False

    video_sub_task: bool = False
    video_sub_task_frames = []

    action_frames = []
    export_action: bool = True

    action_sub_task: bool = False
    action_sub_task_frames = []

    _lock: threading.Lock = threading.Lock()

    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        super().__init__(cfg)
        self.logger = logger
        self.reset()

    def reset(self):
        self.export_video = self.cfg["video"]["save"]
        self.video_frames = []

        self.export_action = self.cfg["action"]["save"]
        self.action_frames = []

        if self.export_video:
            self.video_sub_task = self.cfg["video"]["sub_task"]
            self.video_sub_task_frames = []

            self.output_video_path = self.cfg["video"]["path"]
            self.output_video_name = self.cfg["video"]["name"]

            os.makedirs(self.output_video_path, exist_ok=True)
            self.with_prompt = False

        if self.export_action:
            self.action_sub_task = self.cfg["action"]["sub_task"]
            self.action_sub_task_frames = []

    def step(
        self,
        obs: Dict[str, Any],
        prompt: str | None = None,
        action: Dict[str, Any] | None = None,
    ):
        if self.export_video:
            if prompt:
                frame = create_video_frame(obs["pov"], prompt)
                self.with_prompt = True
            else:
                frame = obs["pov"].astype(np.uint8)
            self.video_frames.append(frame)
            self.video_sub_task_frames.append(frame)

        if self.export_action:
            self.action_frames.append(action)
            self.action_sub_task_frames.append(action)

    def _save(
        self,
        task: str,
        status: str,
        is_sub_task: bool = False,
    ) -> str | None:
        if self.export_video:
            # dir/{task}/{status}/{time}.mp4
            task = task.replace(" ", "_")
            video_dir = os.path.join(self.output_video_path, task, status)
            os.makedirs(video_dir, exist_ok=True)
            time = get_time()

            uid = str(uuid.uuid4())[:5]

            output_video_filepath = os.path.join(video_dir, f"{uid}_{time}.mp4")
            output_action_filepath = os.path.join(video_dir, f"{uid}_{time}.pkl")

            self.logger.info(
                f"[dark_violet]save video&action to {output_video_filepath}[/dark_violet]"
            )

            video_frames = (
                self.video_frames
                if is_sub_task is False
                else self.video_sub_task_frames
            )
            action_frames = (
                self.action_frames
                if is_sub_task is False
                else self.action_sub_task_frames
            )
            with self._lock:
                save_bin(action_frames, output_action_filepath)
                if self.with_prompt:
                    save_frames_as_video(video_frames, output_video_filepath)
                else:
                    write_video(output_video_filepath, video_frames)

                if is_sub_task:
                    self.video_sub_task_frames = []
                    self.action_sub_task_frames = []
            return output_video_filepath

    def save(self, task: str, status: str, is_sub_task: bool = False):
        thread = MultiThreadServerAPI(self._save, args=(task, status, is_sub_task))
        thread.start()
        return thread
