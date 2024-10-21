from typing import Any, Dict, List

import numpy as np
import requests
from omegaconf import DictConfig

from .image import img2base64, img_lst2base64
from .thread import MultiThreadServerAPI


class ServerAPI:
    @staticmethod
    def _reset(server_cfg: DictConfig):
        res = requests.get(
            f"{server_cfg['url']}:{server_cfg['port']}/reset",
            timeout=server_cfg["timeout"],
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to reset: {res.text}")

    @staticmethod
    def reset(server_cfg: DictConfig):
        thread = MultiThreadServerAPI(ServerAPI._reset, (server_cfg,))
        thread.start()
        return thread

    @staticmethod
    def get_retrieval(
        server_cfg: DictConfig,
        obs: Dict[str, Any],
        task2plan: str | None = "gp4 plan task",
        error_info: str | None = None,
    ) -> str:
        if task2plan is None:
            raise ValueError("task2plan is None")
        b = img2base64(obs["pov"])
        plan_type = "retrieval"
        data = {
            "rgb_images": [{"image": b}],
            "task_or_instruction": task2plan,
            "error_info": error_info,
            "type": plan_type,
        }
        res = requests.post(
            f"{server_cfg['url']}:{server_cfg['port']}/chat",
            json=data,
            timeout=server_cfg["timeout"],
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to get plan: {res.text}")
        plan = res.json()["response"]
        return plan

    @staticmethod
    def get_plan(
        server_cfg: DictConfig,
        obs: Dict[str, Any],
        task2plan: str | None = "gp4 plan task",
        error_info: str | None = None,
        example: str | None = None,
        graph: str | None = None,
        visual_info: str | None = None,
    ) -> str:
        if task2plan is None:
            raise ValueError("task2plan is None")
        b = img2base64(obs["pov"])
        plan_type = "plan" if error_info is None else "replan"
        data = {
            "rgb_images": [{"image": b}],
            "task_or_instruction": task2plan,
            "error_info": error_info,
            "example": example,
            "graph": graph,
            "type": plan_type,
            "visual_info": visual_info,
        }
        res = requests.post(
            f"{server_cfg['url']}:{server_cfg['port']}/chat",
            json=data,
            timeout=server_cfg["timeout"],
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to get plan: {res.text}")
        plan = res.json()["response"]
        # print(plan)
        return plan

    @staticmethod
    def get_action(
        server_cfg: Dict[str, Any],
        obs: Dict[str, Any],
        task: str | None,
        step: int = 0,
    ) -> Dict[str, np.ndarray] | List[Dict[str, np.ndarray]]:
        """
        Sends a request to a server to get an action based on the given observation and task.

        Args:
            server_cfg (Dict[str, Any]): A dictionary containing server configuration parameters.
            obs (Dict[str, Any]): The observation data.
            task (str): The task to perform.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the action to perform.

        Raises:
            RuntimeError: If the request to the server fails.
        """
        if task is None:
            raise ValueError("task is None")
        b = img2base64(obs["pov"])
        data = {
            "rgb_images": [{"image": b}],
            "task_or_instruction": task,
            "type": "action",
            "current_step": step,
        }
        res = requests.post(
            f"{server_cfg['url']}:{server_cfg['port']}/chat",
            json=data,
            timeout=server_cfg["timeout"],
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to get action: {res.text}")
        action = res.json()["response"]
        if isinstance(action, dict):
            for k, v in action.items():
                action[k] = np.array(v)
        elif isinstance(action, list):
            for ac in action:
                for k, v in ac.items():
                    ac[k] = np.array(v)
        return action

    @staticmethod
    def _get_reflection(
        server_cfg: DictConfig,
        obs: Dict[str, Any],
        done_imgs: List[str] | None = None,
        cont_imgs: List[str] | None = None,
        replan_imgs: List[str] | None = None,
        task2reflection: str | None = "gpt4 reflection task",
        step: int = 0,
    ) -> tuple[str, str]:
        if task2reflection is None:
            raise ValueError("task2reflection is None")

        # status = ["done", "continue", "replan"]

        b = img2base64(obs["pov"])

        data = {
            "rgb_images": [{"image": b}],
            "task_or_instruction": task2reflection,
            "type": "reflection",
            "current_step": step,
            "done_imgs": img_lst2base64(done_imgs) if done_imgs else None,
            "cont_imgs": img_lst2base64(cont_imgs) if cont_imgs else None,
            "replan_imgs": img_lst2base64(replan_imgs) if replan_imgs else None,
        }
        res = requests.post(
            f"{server_cfg['url']}:{server_cfg['port']}/chat",
            json=data,
            timeout=server_cfg["timeout"],
        )
        res.raise_for_status()
        # if res.status_code != 200:
        #     raise RuntimeError(f"Failed to get action: {res.text}")
        response = res.json()
        res, appendix = response["response"], response["appendix"]
        # assert response in status, f"Invalid response: {response}"
        return res, appendix

    @staticmethod
    def get_reflection(
        server_cfg: DictConfig,
        obs: Dict[str, Any],
        done_imgs: List[str] | None = None,
        cont_imgs: List[str] | None = None,
        replan_imgs: List[str] | None = None,
        task2reflection: str | None = "gpt4 reflection task",
        step: int = 0,
    ):
        thread = MultiThreadServerAPI(
            ServerAPI._get_reflection,
            (server_cfg, obs, done_imgs, cont_imgs, replan_imgs, task2reflection, step),
        )
        # thread.daemon = True
        thread.start()
        return thread
