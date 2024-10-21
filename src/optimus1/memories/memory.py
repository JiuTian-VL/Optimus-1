import json
import logging
import os
import threading
from typing import Any, Dict, List

import shortuuid
from omegaconf import DictConfig
from thefuzz import process

from ..util.prompt import render_replan_example
from ..util.thread import MultiThreadServerAPI

from .graph import KnowledgeGraph

REPLAN_EXAMPLE_FORMAT = """
<task>: {}
<error>: {}
<replan>:
{}
"""


PLAN_EXAMPLE_FORMAT = """
<task>: {}.
<visual info>
{}
<craft graph>:
{}
<task planning>
{}
"""


class Memory:
    plan_dir_path: str
    reflection_dir_path: str
    replan_dir_path: str
    experience_dir_path: str | None = None

    root_path: str

    crafting_graph: KnowledgeGraph

    current_environment: str = ""

    all_reflection_memory: List[str]

    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger | None = None,
        life_long_learning: bool = False,
    ) -> None:
        self.cfg = cfg
        self.version = self.cfg["version"]

        self.logger = logger

        self.root_path = self.cfg["memory"]["path"]

        os.makedirs(self.root_path, exist_ok=True)

        self.plan_dir_path = os.path.join(
            self.root_path, self.cfg["memory"]["plan"]["path"]
        )
        os.makedirs(self.plan_dir_path, exist_ok=True)
        self.reflection_dir_path = os.path.join(
            self.root_path, self.cfg["memory"]["reflection"]["path"]
        )
        os.makedirs(self.reflection_dir_path, exist_ok=True)
        os.makedirs(os.path.join(self.reflection_dir_path, "img"), exist_ok=True)

        self.replan_dir_path = os.path.join(
            self.root_path, self.cfg["memory"]["replan"]["path"]
        )
        os.makedirs(self.replan_dir_path, exist_ok=True)

        self.life_long_learning = life_long_learning

        self.all_reflection_memory = [
            file
            for file in os.listdir(self.reflection_dir_path)
            if file.endswith(".json")
        ]

        self.crafting_graph = KnowledgeGraph(
            life_long_learning=life_long_learning,
        )

    def save_plan(
        self,
        task: str,
        visual_info: str,
        goal: str,
        status: str,
        planning: List[Dict[str, Any]],
        steps: int | float,
        video_file: MultiThreadServerAPI | None = None,
        environment: str = "none",
    ):
        thread = MultiThreadServerAPI(
            self._save_plan,
            args=(
                task,
                visual_info,
                goal,
                status,
                planning,
                steps,
                video_file,
                environment,
            ),
        )
        thread.start()
        return thread

    def _save_plan(
        self,
        task: str,
        visual_info: str,
        goal: str,
        status: str,
        planning: List[Dict[str, Any]],
        steps: int | float,
        video_file: MultiThreadServerAPI | None = None,
        environment: str = "none",
    ):
        assert status in [
            "success",
            "failed",
        ], "status should be one of success, failed"
        file_name = self.cfg["memory"]["plan"]["file"].replace(
            "<task>", task.replace(" ", "_")
        )

        memory_path = self.plan_dir_path.replace("<status>", status)
        os.makedirs(memory_path, exist_ok=True)

        memory_file = os.path.join(memory_path, file_name)

        if self.logger:
            self.logger.info(
                f"[hot_pink]store plan of {task} to {memory_file} :smile:[/hot_pink]"
            )
        with self._lock:
            if os.path.exists(memory_file):
                with open(memory_file, "r") as fp:
                    memory = json.load(fp)
            else:
                memory = {"plan": []}
        vf = ""
        if video_file is not None:
            video_file.join()
            vf = video_file.get_result()
        with self._lock, open(memory_file, "w") as fp:
            memory["plan"].append(
                {
                    "id": shortuuid.uuid(),
                    "environment": environment,
                    "visual_info": visual_info,
                    "goal": goal,
                    "video": vf,
                    "planning": planning,
                    "status": status,
                    "steps": steps,
                }
            )
            json.dump(memory, fp, indent=2)

    def save_reflection(
        self, task: str, env_name: str, reflection_type: str, img_old: str, img_new: str
    ):
        assert reflection_type in [
            "done",
            "continue",
            "replan",
        ], "reflection type should be one of done, continue, replan"

        file_name = self.cfg["memory"]["reflection"]["file"].replace(
            "<task>", task.replace(" ", "_")
        )

        dir = self.reflection_dir_path
        img_dir = os.path.join(dir, self.cfg["memory"]["reflection"]["img_path"])

        os.makedirs(dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        memory_file = os.path.join(dir, file_name)
        with self._lock:
            if self.logger:
                self.logger.info(
                    f"[hot_pink]store reflection of {task}:{reflection_type} to {memory_file} :smile:[/hot_pink]"
                )

            if os.path.exists(memory_file):
                with open(memory_file, "r") as fp:
                    memory = json.load(fp)
            else:
                memory = {}

            if env_name not in memory:
                memory[env_name] = {"done": [], "continue": [], "replan": []}

            memory[env_name][reflection_type].append([img_old, img_new])

            with open(memory_file, "w") as fp:
                json.dump(memory, fp, indent=2)

    def save_replan(
        self, task: str, error_info: str, new_planning: List[Dict[str, Any]]
    ):
        file_name = self.cfg["memory"]["replan"]["file"].replace(
            "<task>", task.replace(" ", "_")
        )

        dir = self.replan_dir_path
        os.makedirs(dir, exist_ok=True)

        memory_file = os.path.join(dir, file_name)
        with self._lock:
            if self.logger:
                self.logger.info(
                    f"store replan of {task} because of {error_info} to {memory_file} :smile:"
                )

            if os.path.exists(memory_file):
                with open(memory_file, "r") as fp:
                    memory = json.load(fp)
            else:
                memory = {}

            if error_info not in memory:
                memory[error_info] = []

            memory[error_info].append(new_planning)

            with open(memory_file, "w") as fp:
                json.dump(memory, fp, indent=2)

    def retrieve_plan(self, task: str):
        task = task.replace(" ", "_").lower()
        has_done = False

        def get_best_match_recipe(target: str, choices):
            res = process.extractOne(target, choices)
            return res[0]

        try:
            lst_dir = os.listdir(f"src/optimus1/memories/{self.version}/plan/success")
        except FileNotFoundError:
            # from stratch
            return None, False
        target = get_best_match_recipe(task + ".json", lst_dir)
        has_done = task + ".json" == target
        print(f"Find example: {target}")
        with open(
            os.path.join(f"src/optimus1/memories/{self.version}/plan/success", target),
            "r",
        ) as fi:
            data = json.load(fi)

        plan = data["plan"][0]["planning"]

        render_plan = {}
        for idx, p in enumerate(plan):
            render_plan[f"step {idx+1}"] = p

        goal = (
            plan[-1]["goal"][0]
            if "goal" not in data["plan"][0]
            else data["plan"][0]["goal"]
        )
        visual_info = data["plan"][0].get("visual_info", "None")

        examples = PLAN_EXAMPLE_FORMAT.format(
            target.replace(".json", "").replace("_", " "),
            visual_info,
            self.retrieve_graph(goal),
            json.dumps(render_plan),
        )
        return examples, has_done

    def retrieve_reflection(self, task: str):
        import random

        from thefuzz import process

        def get_best_match_recipe(target: str, choices):
            res = process.extractOne(target, choices)
            return res[0] if res is not None else None

        def add_dir(path_lst):
            return [os.path.join(self.reflection_dir_path, "img", p) for p in path_lst]

        similar_task = get_best_match_recipe(
            task.replace(" ", "_") + ".json", self.all_reflection_memory
        )
        if similar_task is None:
            return None, None, None
        with open(os.path.join(self.reflection_dir_path, similar_task), "r") as fi:
            similar_task_reflct = json.load(fi)
        envs = list(similar_task_reflct.keys())
        similar_envs = get_best_match_recipe(self.current_environment, envs)
        reflection = similar_task_reflct[similar_envs]
        done, cont, replan = (
            reflection["done"],
            reflection["continue"],
            reflection["replan"],
        )
        if len(done) > 0:
            done = random.choice(done)
        if len(cont) > 0:
            cont = random.choice(cont)
        if len(replan) > 0:
            replan = random.choice(replan)
        return add_dir(done), add_dir(cont), add_dir(replan)

    def retrieve_replan(self, task: str, error_info: str, number: int = 1) -> str:
        task = task.replace(" ", "_")

        if os.path.exists(self.replan_dir_path) is False:
            return ""

        examples = []
        for replan_exper in os.listdir(self.replan_dir_path):
            if task in replan_exper:
                with open(os.path.join(self.replan_dir_path, replan_exper), "r") as fi:
                    data = json.load(fi)
                if error_info in data:
                    result = data[error_info][0]
                    examples.append(
                        REPLAN_EXAMPLE_FORMAT.format(
                            task, error_info, render_replan_example(result)
                        )
                    )
                if len(examples) == number:
                    break
        return "\n".join(examples).strip()

    def retrieve_graph(self, item: str, number: int = 1) -> str:
        if (
            item == "iron_ore"
            or item == "logs"
            or item == "cobblestone"
            or item == "redstone"
            or item == "sand"
            or item == "coal_ore"
        ):
            return "Just mine it!"
        item = item.replace("logs", "log")
        return self.crafting_graph.compile(item.replace(" ", "_"), number)
