import logging

from rich.progress import Progress, TaskID

from .jarvis_craft_helper import *
from .jarvis_equip_helper import *
from .jarvis_smelt_helper import *


class Helper:
    def __init__(self, env):
        self.craft_helper = CraftHelper(env)
        self.equip_helper = EquipHelper(env)
        self.smelt_helper = SmeltHelper(env)

    def reset(self, task: str, pbar: Progress, task_id: TaskID, logger: logging.Logger):
        if "equip" in task:
            return self.equip_helper.set_arguments(task, pbar, task_id, logger)
        elif "craft" in task:
            return self.craft_helper.set_arguments(task, pbar, task_id, logger)
        elif "smelt" in task:
            return self.smelt_helper.set_arguments(task, pbar, task_id, logger)
        elif "replan" in task:
            return self.replan_helper.set_arguments(task, pbar, task_id, logger)

    def get_task_steps(self, task: str):
        if "equip" in task:
            return self.equip_helper.steps
        elif "craft" in task:
            return self.craft_helper.steps
        elif "smelt" in task:
            return self.smelt_helper.steps
        elif "replan" in task:
            return self.replan_helper.steps
        else:
            return -1

    def step(self, task: str, goal: tuple):
        if "equip" in task:
            return self.equip_helper.equip_item(goal[0])
        elif "craft" in task:
            return self.craft_helper.crafting(goal[0], goal[1])
        elif "smelt" in task:
            return self.smelt_helper.smelting(goal[0], goal[1])
        elif "replan" in task:
            if goal[0] == "tower":
                return self.replan_helper.build_tower()
            elif goal[0] == "land":
                return self.replan_helper.go_to_land()
        else:
            return False, "not support"
