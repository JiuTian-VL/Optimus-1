import json
import os
from typing import Any, Dict


def get_item_name(item_dict: dict) -> str:
    if "item" in item_dict:
        item = item_dict["item"][10:]
    elif "tag" in item_dict:
        item = item_dict["tag"][10:]
    if item == "stone_crafting_materials" or item == "stone_tool_materials":
        item = "cobblestone"
    return item


class KnowledgeGraph:
    _recipe_dir: str = "src/optimus1/helper/recipes"
    _all_recipes: Dict[str, Any]
    item_output_number: Dict[str, int]  # item_output_number[planks] = 4 合成4个planks

    item_number_graph: Dict[str, Dict[str, int]]

    def __init__(self, life_long_learning: bool = False):
        self.graph = {}
        self._all_recipes = {}
        self.item_output_number = {}
        self.item_number_graph = {}

        self._in_degree = {}

        self.sub_graph = {}

        if not life_long_learning:
            self._load_recipes()

    def _load_recipes(self):
        recipes = os.listdir(self._recipe_dir)
        for recipe in recipes:
            with open(os.path.join(self._recipe_dir, recipe)) as f:
                data = json.load(f)
            pattern = self._load_needed_items(data)
            result_item = self._extract_output_item(data)
            if result_item == "stick":
                pass
            if result_item not in self._all_recipes:
                self._all_recipes[result_item] = pattern

        for k, v in self._all_recipes.items():
            rec = {item: num for [item, num] in v}
            self.add_recipe(k, rec, self.item_output_number[k])

        self.add_recipe("planks", {"logs": 1}, output_qty=4)
        self.item_number_graph["logs"] = {"planks": 1}

    def _extract_output_item(self, recipe: dict) -> str:
        """
        extract recipe result item name
        "result": {
          "item": "minecraft:acacia_fence",
          "count": 3
        }
        "result": "minecraft:smooth_quartz",

        """
        if "result" in recipe:
            if isinstance(recipe["result"], dict):
                result = recipe["result"]["item"][10:]
            else:
                result = recipe["result"][10:]
        else:
            result = recipe["type"][10:]
        return result

    def _count_item_in_pattern(self, pattern: list, item: str):
        return sum([i == item for p in pattern for i in p])

    def _need_crafting_table(self, target_data: Dict) -> bool:
        if "pattern" in target_data:
            pattern = target_data.get("pattern", [])
            col_len = len(pattern)
            row_len = len(pattern[0])
            if col_len <= 2 and row_len <= 2:
                return False
            return True
        else:
            ingredients = target_data.get("ingredients", [])
            item_num = len(ingredients)
            if item_num <= 4:
                return False
            return True

    def _load_needed_items(self, recipe: dict):
        result = self._extract_output_item(recipe)
        self.item_output_number[result] = (
            recipe["result"]["count"]
            if "result" in recipe and "count" in recipe["result"]
            else 1
        )
        needed_items = []
        need_furnace = False

        if "pattern" in recipe:
            keys = recipe["key"]
            pattern = recipe["pattern"]
            for key, value in keys.items():
                item = get_item_name(value)
                num = self._count_item_in_pattern(pattern, key)
                if "ingot" in item or "smelting" in recipe["type"]:
                    need_furnace = True
                if item not in self.item_number_graph:
                    self.item_number_graph[item] = {}
                # use {num} {item} to craft {result}
                self.item_number_graph[item][result] = num
                needed_items.append([item, num])

            if self._need_crafting_table(recipe):
                needed_items.append(["crafting_table", 1])
            if need_furnace:
                needed_items.append(["furnace", 1])

        elif "ingredients" in recipe:
            ingredients = recipe["ingredients"]
            for item in ingredients:
                item_name = get_item_name(item)
                needed_items.append([item_name, 1])

            if self._need_crafting_table(recipe):
                needed_items.append(["crafting_table", 1])
        elif "ingredient" in recipe:
            ingredient = recipe["ingredient"]
            item_name = get_item_name(ingredient)
            if "ingot" in item_name or "smelting" in recipe["type"]:
                need_furnace = True

            if need_furnace:
                needed_items.append(["furnace", 1])

            needed_items.append([item_name, 1])
        return needed_items

    def add_recipe(self, product, ingredients, output_qty=1):
        self.graph[product] = {"ingredients": ingredients, "output_qty": output_qty}

    def get_recipe(self, item: str):
        return self.graph.get(item)

    def _get_recipe_with_number(self, item: str, number: int):
        recipe = self.get_recipe(item)
        if recipe is None:
            return {item: number}
        res = {}
        number = (recipe["output_qty"] + number - 1) // recipe["output_qty"]
        for vk, vv in recipe["ingredients"].items():
            if vk == "furnace":
                continue
            elif vk == "crafting_table":
                res[vk] = 1
            else:
                if vk not in res:
                    res[vk] = 0
                res[vk] += vv * number
        return res

    def _compile_steps(
        self,
        item: str,
        quantity=1,
        steps=None,
        summary=None,
        sub_graph=None,
        in_degree=None,
    ):
        if steps is None:
            steps = []
        if summary is None:
            summary = {}
        if sub_graph is None:
            sub_graph = {}
        if in_degree is None:
            in_degree = {}
        if item not in sub_graph:
            sub_graph[item] = {}
        if item == "gold_block":
            return steps

        recipe = self.get_recipe(item)
        if recipe is None:
            return steps
        if item not in summary:
            summary[item] = 0
        if item in ["crafting_table", "furnace"]:
            if summary[item] == 1:
                return steps
            quantity = 1
            summary[item] = 1
        else:
            summary[item] += quantity

        if item not in in_degree:
            in_degree[item] = 0

        produced_qty = recipe["output_qty"]
        runs_needed = (quantity + produced_qty - 1) // produced_qty
        for ingredient, qty_needed in recipe["ingredients"].items():
            in_degree[item] += 1
            if ingredient not in sub_graph:
                sub_graph[ingredient] = {}
            sub_graph[ingredient][item] = 1

            self._compile_steps(
                ingredient,
                qty_needed * runs_needed,
                steps,
                summary,
                sub_graph,
                in_degree,
            )
            if self.get_recipe(ingredient) is None:
                if ingredient not in summary:
                    summary[ingredient] = 0
                summary[ingredient] += qty_needed * runs_needed

        steps.append(
            f"Craft {runs_needed} {item} to make {runs_needed * produced_qty} using {self._pretty_dict(recipe['ingredients'])}"
        )

        return steps

    def _compile_base(self, summary: dict):
        import copy

        reduce_graph = copy.deepcopy(summary)
        assert id(summary) != id(reduce_graph)
        for item in summary:
            for item2 in summary:
                if item != item2:
                    if (
                        item not in self.item_number_graph
                    ):  # this item can not use to craft anything
                        if (
                            item in reduce_graph
                            and item != "coal_ore"
                            and item != "charcoal"
                        ):
                            reduce_graph.pop(item)
                        continue
                    if (
                        self.item_number_graph[item].get(item2, 0) != 0
                        and item2 in reduce_graph
                    ):
                        reduce_graph.pop(item2)
        res = {}
        for base, num in reduce_graph.items():
            self.dict_update(self._get_recipe_with_number(base, num), res)
        summary.update(res)
        return res

    def _compile_tools(self, sub_graph: dict, summary: dict, base: dict):
        has_tools = {
            "wooden_pickaxe": False,
            "stone_pickaxe": False,
            "iron_pickaxe": False,
        }

        def high_level_item(tool):
            if "wooden" in tool:
                return "cobblestone"
            elif "stone" in tool:
                return "iron_ore"

        for base_item, number in base.items():
            if "log" in base_item:
                continue
            elif "cobblestone" in base_item or "coal" in base_item:
                need_item = ["wooden_pickaxe"]
            elif "iron" in base_item:
                need_item = ["stone_pickaxe"]
            elif (
                "gold" in base_item or "diamond" in base_item or "redstone" in base_item
            ):
                need_item = ["iron_pickaxe", "stone_pickaxe", "wooden_pickaxe"]
            else:
                continue
            max_level_tool = need_item[0]
            for it in reversed(need_item):
                if has_tools[it]:
                    continue
                temp_sub_graph = {}
                temp_in_degree = {}
                temp_summary = {}
                self._compile_steps(
                    it,
                    1,
                    summary=temp_summary,
                    sub_graph=temp_sub_graph,
                    in_degree=temp_in_degree,
                )
                self._compile_base(temp_summary)
                if it == max_level_tool:
                    temp_sub_graph[it][base_item] = 1  # wooden_pickaxe -> stone
                else:
                    temp_sub_graph[it][high_level_item(it)] = 1

                self.dict_update(temp_summary, summary)
                self.merge_graph(temp_sub_graph, sub_graph)
                has_tools[it] = True
            sub_graph[max_level_tool][base_item] = 1
        return sub_graph, summary

    def _topo_sort(self, sub_graph, in_degree):
        from queue import Queue

        q = Queue()
        for k, v in in_degree.items():
            if v == 0 and k != "":
                q.put(k)
        res = []
        while not q.empty():
            cur = q.get()
            res.append(cur)
            for k in sub_graph[cur]:
                in_degree[k] -= 1
                if in_degree[k] == 0 and k != "":
                    q.put(k)
        return res

    def _pretty_dict(self, reicpe: dict):
        return ", ".join([f"{v} {k}" for k, v in reicpe.items()])

    def _pretty_result(
        self, summary: dict, base: dict, sub_graph: dict, in_degree: dict | None = None
    ):
        if in_degree is None:
            in_degree = {}

        for u, v in sub_graph.items():
            if u not in in_degree:
                in_degree[u] = 0
            for k in v:
                if k not in in_degree:
                    in_degree[k] = 0
                in_degree[k] += 1

        order = self._topo_sort(sub_graph, in_degree)

        res = [
            f"{idx+1}. {item.lower()}: need {summary[item] if summary[item] > 0 else '??'}"
            for idx, item in enumerate(order)
        ]
        return "\n".join(res)

    def compile(self, item: str, number: int = 1):
        self.sub_graph = {}
        self._in_degree = {}
        summary = {}
        self._compile_steps(
            item,
            number,
            summary=summary,
            sub_graph=self.sub_graph,
            in_degree=self._in_degree,
        )
        base = self._compile_base(summary)
        sub_graph = self.sub_graph
        sub_graph, summary = self._compile_tools(sub_graph, summary, base)
        base = self._compile_base(summary)

        return (
            f"craft {number} {item} summary:\n"
            + self._pretty_result(summary, base, sub_graph)
            + "\n"
        )

    def dict_update(self, src: dict, dst: dict):
        """add src to dst"""
        for k, v in src.items():
            if k not in dst and k != "":
                dst[k] = v
            else:
                if k == "crafting_table":
                    if dst[k] > 1:
                        dst["planks"] -= 4
                        dst[k] -= 1
                    continue
                elif k == "furnace":
                    if dst[k] > 1:
                        dst["cobblestone"] -= 8
                        dst[k] -= 1
                    continue
                elif "wooden_pickaxe" in k:
                    if dst[k] > 1:
                        dst["planks"] -= 3
                        dst["stick"] -= 2
                        dst[k] -= 1
                    continue
                elif "stone_pickaxe" in k:
                    if dst[k] > 1:
                        dst["cobblestone"] -= 3
                        dst["stick"] -= 2
                        dst[k] -= 1
                    continue
                elif "iron_pickaxe" in k:
                    if dst[k] > 1:
                        dst["iron_ingot"] -= 3
                        dst["stick"] -= 2
                        dst[k] -= 1
                dst[k] += v

    def merge_graph(self, graph_src: dict, graph_dst: dict):
        """merge src to dst"""
        import copy

        for u, v in graph_src.items():
            if u not in graph_dst:
                temp_v = copy.deepcopy(v)
                graph_dst[u] = temp_v
            else:
                graph_dst[u].update(v)
