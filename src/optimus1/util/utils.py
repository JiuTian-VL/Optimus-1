import datetime
import json
import pickle
import random
from typing import Any, Dict, List

import numpy as np
from omegaconf import DictConfig


def save_bin(data: Any, path: str):
    with open(path, "wb") as fp:
        pickle.dump(data, fp)


def read_bin(path: str) -> Any:
    with open(path, "rb") as fp:
        data = pickle.load(fp, encoding="bytes")
    return data


def get_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")


def get_evaluate_task(cfg: DictConfig) -> List[str]:
    id_map = {task["id"]: task for task in cfg["all_task"]}
    evaluate = cfg["evaluate"]
    if len(evaluate) == 0:
        tasks = [id_map[task["id"]]["instruction"] for task in cfg["all_task"]]
    else:
        tasks = [id_map[task]["instruction"] for task in evaluate]
    return tasks


def get_learning_task(cfg: DictConfig) -> List[str]:
    with open("memories/learning3/experience/experience.json", "r") as fi:
        learned = json.load(fi)
    learn = {}
    cnt = -300
    for k, v in learned.items():
        if "output_qty" in v and v["output_qty"] != -1:
            learn[k] = True
            cnt += 1
        else:
            learn[k] = False
    print(f"Already learn {cnt+300} tasks.")

    id_map = {task["id"]: task for task in cfg["all_task"]}
    learning = cfg["learning_tasks"]
    if len(learning) == 0:
        tasks = [id_map[task["id"]]["instruction"] for task in cfg["all_task"] if not learn[task["instruction"]]]
        random.shuffle(tasks)
        tasks = tasks[: 150 - cnt]
    else:
        tasks = [id_map[task]["instruction"] for task in learning]
    random.shuffle(tasks)
    return tasks


def give_ramdom_initial_items(env, task: str, initial_inventory: Dict[str, str | int]):
    env.execute_cmd("/clear")

    for item, num in initial_inventory.items():
        if num == -1:
            num = int(np.random.uniform(1, 65))
        env.execute_cmd(f"/give @p minecraft:{item} {num}")
        if "Smelt" in task or "smelt" in task:
            env.execute_cmd("/give @p minecraft:coal 10")

        print(f"Give {num} {item} to player")
