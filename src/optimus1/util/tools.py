import json
import os
from typing import Any, Dict, List

import cv2
from omegaconf import DictConfig

from .prompt import PlanList

# from helpers import Helper
from .video import FPS, create_video_frame, save_frames_as_video


class PlanManager:
    _completed_plan: PlanList

    def __init__(self, plans: PlanList) -> None:
        self._plan_queue = plans

        self._completed_plan = []

    @property
    def remain_plans(self) -> PlanList:
        return self._plan_queue

    @property
    def completed_plans(self) -> PlanList:
        return self._completed_plan

    @property
    def all(self) -> PlanList:
        return self._completed_plan + self._plan_queue

    @property
    def next_plan(self):
        if len(self._plan_queue) > 0:
            temp = self._plan_queue.pop(0)
            self._completed_plan.append(temp)
            return temp
        return None

    def insert_plan(self, new_plans: List[Dict[str, Any]] | None = None, is_replan: bool = False):
        if new_plans is None:
            return
        if is_replan and len(self._completed_plan) > 0:
            self._completed_plan.pop()
        self._plan_queue = new_plans + self._plan_queue

    def __len__(self):
        return len(self._plan_queue)


# TODO plan format
class PlanWindow:
    def __init__(self, plans: List[str]):
        self._plans = plans

    @property
    def all_plan(self):
        """
        Returns all the plans in the PlanWindow.

        Returns:
            List[str]: A list of plans.

        """
        return self._plans

    @property
    def next_task(self) -> str | None:
        """
        Returns the next task in the PlanWindow.

        Returns:
            str | None: The next task in the PlanWindow, or None if there are no more tasks.

        """
        return self._plans.pop(0)

    def set_latest_plan(self, new_plans: List[str]) -> None:
        """
        Sets the latest plans in the PlanWindow.

        Args:
            new_plans (List[str]): A list of new plans.

        """
        self._plans = new_plans

    def add_new_plan(self, new_plan: List[str]):
        """
        Adds a new plan to the PlanWindow.

        Args:
            new_plan (List[str]): A list of new plans to be added.

        """
        self._plans = new_plan + self._plans

    def __len__(self):
        """
        Returns the number of plans in the PlanWindow.

        Returns:
            int: The number of plans in the PlanWindow.

        """
        return len(self._plans)

    def __str__(self) -> str:
        """
        Returns a string representation of the PlanWindow.

        Returns:
            str: A string representation of the PlanWindow.

        """
        return "\n".join(self._plans)


# TODO: 移到别的地方
def run_interactive(cfg: DictConfig, env, get_action, *, logger=None):
    """Runs the agent in the MineRL env and allows the user to enter prompts to control the agent.
    Clicking on the gameplay window will pause the gameplay and allow the user to enter a new prompt.

    Typing 'reset agent' will reset the agent's state.
    Typing 'reset env' will reset the environment.
    Typing 'save video' will save the video so far (and ask for a video name). It will also save a json storing
        the active prompt at each frame of the video.
    """
    window_name = "STEVE-1 Gameplay (Click to Enter Prompt)"

    state = {"obs": None}

    output_video_dirpath = cfg["video"]["path"]
    os.makedirs(output_video_dirpath, exist_ok=True)

    video_frames = []
    frame_prompts = []
    helper = Helper(env)

    def handle_prompt():
        # Pause the gameplay and ask for a new prompt
        prompt = input("\n\nEnter a prompt:\n>").strip().lower()

        # Save the video so far if prompted
        if prompt == "save video":
            # Ask for a video name
            video_name = input("Enter a video name:\n>").strip().lower()

            # Save both the video and the prompts for each frame
            output_video_filepath = os.path.join(output_video_dirpath, f"{video_name}.mp4")
            prompts_for_frames_filepath = os.path.join(output_video_dirpath, f"{video_name}.json")
            print(f"Saving video to {output_video_filepath}...")
            save_frames_as_video(video_frames, output_video_filepath, fps=FPS)
            print(f"Saving prompts for frames to {prompts_for_frames_filepath}...")
            with open(prompts_for_frames_filepath, "w") as f:
                json.dump(frame_prompts, f)
            print("Done. Continuing gameplay with previous prompt...")
            return
        if "equip" in prompt or "craft" in prompt or "smelt" in prompt:
            helper.set_arguments(video_frames, prompt, logger=logger)
            # TODO: extract goal
            goal = prompt
            done, info = helper.step(prompt, goal)
            video_frames, steps = helper.get_frames_and_steps(prompt)
            if done:
                logger.info(f"[green]{prompt} Success[/green]!")
            else:
                logger.warning(f"[red]{prompt} failed... Beacuse {info}[/red]")
        else:
            while True:
                minerl_action = get_action(cfg["server"], state["obs"], task=prompt, step=env.num_steps)
                state["obs"], _, _, _ = env.step(minerl_action)

                frame = create_video_frame(state["obs"]["pov"], prompt)
                video_frames.append(frame)
                frame_prompts.append(prompt)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)

    def reset_env():
        print("\nResetting environment...")
        state["obs"] = env.reset()
        seed = cfg["env"]["seed"]
        if seed is not None:
            print(f"Setting seed to {seed}...")
            env.seed(seed)

    reset_env()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            handle_prompt()

    initial_frame = create_video_frame(state["obs"]["pov"], "Click to Enter a Prompt")  # type: ignore
    cv2.imshow(window_name, initial_frame)
    cv2.setMouseCallback(window_name, on_click)
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Close the window when 'q' is pressed
            break
