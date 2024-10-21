import logging
import random
from typing import Any, Dict, List

import gym
import numpy as np
from omegaconf import DictConfig

from ..util.server_api import MultiThreadServerAPI

from .mods import RecorderMod, StatusMod, TaskCheckerMod


def random_ore(env, ORE_MAP, ypos: float, thresold: float = 0.9):
    prob = random.random()
    if prob <= thresold:
        return
    dy = random.randint(-5, -3)
    new_pos = int(ypos + dy)
    if 45 <= ypos <= 50:  # max: 6
        # coal_ore
        if ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos >= 45:
            ORE_MAP[new_pos] = "coal_ore"
            ORE_MAP[ypos] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:coal_ore".format(dy))
            print(f"coal ore at {new_pos}")
    elif 26 <= ypos <= 43:  # max: 17
        if ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos >= 26:
            ORE_MAP[new_pos] = "iron_ore"
            ORE_MAP[ypos] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:iron_ore".format(dy))
            print(f"iron ore at {new_pos}")

    elif 14 < ypos <= 26:
        if ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos >= 17:  # max: 10
            ORE_MAP[new_pos] = "gold_ore"
            ORE_MAP[ypos] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:gold_ore".format(dy))
            print(f"gold ore at {new_pos}")
        elif ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos <= 16:  # max:12
            ORE_MAP[new_pos] = "redstone_ore"
            ORE_MAP[ypos] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:redstone_ore".format(dy))
            print(f"redstone ore at {new_pos}")
    elif (
        ypos <= 14 and ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos >= 1
    ):  # max: 14
        ORE_MAP[new_pos] = "diamond_ore"
        ORE_MAP[ypos] = 1
        env.execute_cmd("/setblock ~ ~{} ~ minecraft:diamond_ore".format(dy))
        print(f"diamond ore at {new_pos}")


class BasaltTimeoutWrapper(gym.Wrapper):
    """Timeout wrapper specifically crafted for the BASALT environments"""

    def __init__(self, env):
        super().__init__(env)
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0

    def reset(self):
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0
        return super().reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.num_steps += 1
        if self.num_steps >= self.timeout:
            done = True
        return observation, reward, done, info


class CustomEnvWrapper(gym.Wrapper):
    _api_thread: MultiThreadServerAPI | None

    can_change_hotbar: bool = False
    can_open_inventory: bool = False

    cache: Dict[str, Any]

    logger: logging.Logger
    cfg: DictConfig

    _only_once: bool = False

    def __init__(self, env, cfg: DictConfig, logger: logging.Logger):
        super().__init__(env)
        self._current_task_finish = False

        self.cfg = cfg
        self.logger = logger

        self.record_mod = RecorderMod(cfg["record"], logger)
        self.status_mod = StatusMod(cfg, logger)
        self.task_checker_mod = TaskCheckerMod(cfg)

    @property
    def current_task_finish(self):
        return self._current_task_finish

    def reset(self):
        self.ORE_MAP = {}
        self._current_task_finish = False
        self._api_thread = None

        self.record_mod.reset()
        self.status_mod.reset()
        self.task_checker_mod.reset()

        self.cache = {}
        self.cache["task"] = ""
        self.cache["ypos"] = {}

        self._only_once = False

        # ====设置spawn point & env seed ==========
        commands: List[str] = self.cfg["commands"]

        obs = self.env.reset()
        if commands:
            for cmd in commands:
                self.env.execute_cmd(cmd)
        return obs

    def raw_step(self, action: Dict[str, Any]):
        if not self.can_change_hotbar:
            for i in range(9):
                action[f"hotbar.{i+1}"] = np.array(0)
        # ban drop(Q) action
        action["drop"] = 0
        # attack时不乱动
        if action["attack"] > 0:
            action["jump"] = action["left"] = action["right"] = np.array(0)
            action["sneak"] = action["sprint"] = np.array(0)
        observation, reward, done, info = self.env.step(action)
        self.record_mod.step(observation, None, action)
        self.status_mod.step(observation, action)

        info.update(self.status_mod.get_status())

        info["isGuiOpen"] = observation["isGuiOpen"]

        return observation, reward, done, info

    def step(
        self,
        action: Dict[str, Any],
        goal: tuple[str, int] | None = None,
        prompt: str | None = None,
    ):
        if not self.can_change_hotbar:
            for i in range(9):
                action[f"hotbar.{i+1}"] = np.array(0)
            action["use"] = np.array(0)
            action["inventory"] = np.array(0)

            hotbar = self.find_best_pickaxe()
            if hotbar:
                action[hotbar] = np.array(1)

        if not self.can_open_inventory:
            action["inventory"] = np.array(0)
        action["drop"] = np.array(0)

        observation, reward, done, info = self.env.step(action)

        if goal is not None and goal[0] != self.cache["task"]:
            self.task_checker_mod.reset(observation["inventory"])
            self.cache["task"] = goal[0]

        self.record_mod.step(observation, prompt, action)
        self.status_mod.step(observation, action)

        info.update(self.status_mod.get_status())

        ypos = self.status_mod.get_height()
        if ypos not in self.cache["ypos"]:
            self.cache["ypos"][ypos] = 0
        self.cache["ypos"][ypos] += 1
        if self.cache["ypos"][ypos] > 8000:
            self.logger.critical("Stuck....")
            self.env.execute_cmd("/kill")
            self.cache["ypos"] = {}
            self.cache["explore"] = 100

        if self._only_once:
            random_ore(self.env, self.ORE_MAP, ypos)
            self._only_once = False

        try:
            self._current_task_finish = self.task_checker_mod.step(
                observation["inventory"], goal
            )
        except Exception as e:
            print("Error ", e)
            self._current_task_finish = True

        if (
            goal
            and "iron_ore" in goal[0]
            and ypos < 25
            and self._current_task_finish is False
        ):
            self.logger.critical("Return to ground....")
            self.env.execute_cmd("/kill")
            self.cache["ypos"] = {}
            self.cache["explore"] = 100

        if self._current_task_finish:
            self.cache["task"] = ""
        info["isGuiOpen"] = observation["isGuiOpen"]

        self.cache["info"] = info
        return observation, reward, done, info

    def save_video(
        self,
        task: str,
        status: str,
        is_sub_task: bool = False,
    ):
        thread = self.record_mod.save(task, status, is_sub_task)
        return thread

    def inventory_change(self) -> bool:
        return self.status_mod.inventory_change

    def inventory_change_what(self):
        return self.status_mod.inventory_change_what()

    @property
    def api_thread(self) -> MultiThreadServerAPI | None:
        return self._api_thread

    @api_thread.setter
    def api_thread(self, thread: MultiThreadServerAPI | None) -> None:
        self._api_thread = thread

    def api_thread_get_result(self):
        assert self._api_thread is not None, "Need set api_thread first."
        return self._api_thread.get_result()

    def api_thread_is_alive(self) -> bool:
        assert self._api_thread is not None, "Need set api_thread first."
        return self._api_thread.is_alive()

    def _call_func(self, func_name: str):
        action = self.env.noop_action()
        action[func_name] = 1
        self.step(action)
        action[func_name] = 0
        for _ in range(5):
            self.step(action)

    def null_action(self):
        action = self.env.noop_action()
        self.env.step(action)

    def find_best_pickaxe(self):
        if "info" not in self.cache:
            return None
        height = self.cache["info"]["location_stats"]["ypos"]
        if height < 70:
            inventory_id = -1
            # find pickaxe
            inventory_id_diamond = self._find_in_inventory("diamond_pickaxe")
            inventory_id_iron = self._find_in_inventory("iron_pickaxe")
            inventory_id_stone = self._find_in_inventory("stone_pickaxe")
            inventory_id_wooden = self._find_in_inventory("wooden_pickaxe")

            if inventory_id_wooden:
                inventory_id = inventory_id_wooden
            if inventory_id_stone:
                inventory_id = inventory_id_stone
            if inventory_id_iron:
                inventory_id = inventory_id_iron
            if inventory_id_diamond:
                inventory_id = inventory_id_diamond
            if inventory_id == -1:
                return None
            if inventory_id >= 0 and inventory_id <= 8:
                return f"hotbar.{inventory_id+1}"
            else:
                pass
        return None

    def _find_in_inventory(self, item: str):
        inventory = self.cache["info"]["plain_inventory"]
        for slot, it in inventory.items():
            if it["type"] == item:
                return slot
        return None
