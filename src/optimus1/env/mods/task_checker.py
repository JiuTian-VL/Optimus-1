from typing import Any, Dict, List

from omegaconf import DictConfig

from .mod import Mod


class TaskCheckerMod(Mod):
    _cache: Dict[str, Any]

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self._cache = {"inventory": {}}

    def reset(self, inventory: Dict[str, Any] | None = None):
        if inventory is None:
            inventory = {}
        self._cache["inventory"] = inventory

    def step(self, inventory, goal: tuple[str, int] | None):
        if goal is None:
            return False
        item, number = goal
        item = self._expand_item(item)
        need_item = [[item, number]]
        return self._check_number(inventory, need_item)

    def _check_number(self, inventory: Dict[str, Any], need_item: List[list]) -> bool:
        # [ [["xx"], 1]] ]
        total = 0
        for [item_list, number] in need_item:
            s = 0
            p = 0
            for item in item_list:
                s += inventory[item]
                p += self._cache["inventory"][item]
            if s >= p + number:
                total += 1
        return total == len(need_item)

    def _expand_item(self, item: str) -> List[str]:
        # TODO: update check item list
        if "log" in item or "logs" in item:
            return [
                "acacia_log",
                "birch_log",
                "dark_oak_log",
                "jungle_log",
                "oak_log",
                "spruce_log",
            ]
        elif "plank" in item or "planks" in item:
            return [
                "acacia_planks",
                "birch_planks",
                "dark_oak_planks",
                "jungle_planks",
                "oak_planks",
                "dark_oak_planks",
                "spruce_planks",
            ]
        elif "redstone" in item:
            return ["redstone"]
        elif "stone" in item:
            return ["cobblestone"]
        elif "coal" in item:
            return ["coal"]
        else:
            return [item]
