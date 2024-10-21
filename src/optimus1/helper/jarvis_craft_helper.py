import json
import math
import os
import random
import re
from typing import Dict, Tuple

import cv2
import numpy as np
from thefuzz import process

from optimus1.util import create_video_frame, render_recipe

from .slot import *

MISSING_MATERIAL_FORMAT = 'missing material: {{"{}": {}}}'


class CraftHelper:
    root_path: str = "src/optimus1/helper"
    recipe_path: str = os.path.join(root_path, "recipes")
    tag_json_path: str = os.path.join(root_path, "tag_items.json")
    has_crafting_table: bool = False

    def __init__(
        self,
        env,
        sample_ratio: float = 0.5,
        inventory_slot_range: Tuple[int, int] = (0, 36),
        debug: bool = False,
        **kwargs,
    ) -> None:
        self.sample_ratio = sample_ratio
        self.inventory_slot_range = inventory_slot_range
        # self.outframes, self.outactions, self.outinfos = [], [], []
        # ===arguments===
        self.env = env
        self.pbar = None
        self.logger = None
        self.debug = debug
        self.steps = 0
        self.task = ""
        self.window_name = None

        tag_json_path = os.path.join(self.root_path, "tag_items.json")
        with open(tag_json_path) as file:
            self.tag_info = json.load(file)

        self.all_recipes = [f[:-5] for f in os.listdir(self.recipe_path)]

        self.reset(fake_reset=False)

    def get_best_match_recipe(self, target: str):
        res = process.extractOne(target, self.all_recipes)
        if self.logger:
            self.logger.warning(f"[red]fix: {target} -> {res[0]}[/red]")
        return res[0]

    def set_arguments(
        self, task: str, pbar=None, pbar_task_id=None, logger=None, window_name=None
    ):
        self.task = task
        self.pbar = pbar
        self.pbar_task_id = pbar_task_id
        self.steps = 0
        self.logger = logger
        self.window_name = window_name

    def reset(self, fake_reset=True):
        if not fake_reset:
            self.has_crafting_table = False
            self.current_gui_type = None
            self.crafting_slotpos = "none"
            self.resource_record = {
                f"resource_{x}": {"type": "none", "quantity": 0} for x in range(9)
            }
            self._null_action(1)
        else:
            self.outframes, self.outactions, self.outinfos = [], [], []

    def _assert(self, condition, message=None):
        if not condition:
            if self.info["isGuiOpen"]:
                self._call_func("inventory")
            self.current_gui_type = None
            self.crafting_slotpos = "none"
            self.resource_record = {
                f"resource_{x}": {"type": "none", "quantity": 0} for x in range(9)
            }

            raise AssertionError(message)

    def _step(self, action):
        self.obs, reward, terminated, self.info = self.env.step(action)
        if self.pbar:
            self.pbar.update(self.pbar_task_id, advance=1)

        self.steps += 1

        if terminated:
            raise RuntimeError("Timeout!")

        self.info["resource"] = self.resource_record

        if self.window_name:
            cv2.imshow(self.window_name, create_video_frame(self.obs["pov"], self.task))
            cv2.waitKey(1)

        return self.obs, reward, terminated, self.info

    # open inventory
    def open_inventory_wo_recipe(self):
        self._call_func("inventory")
        self.cursor = [WIDTH // 2, HEIGHT // 2]
        # update slot pos
        self.current_gui_type = "inventory_wo_recipe"
        self.crafting_slotpos = SLOT_POS_INVENTORY_WO_RECIPE

    # before opening crafting_table
    def pre_open_tabel(self, attack_num=20):
        action = self.env.noop_action()
        self.obs, _, _, self.info = self._step(action)
        height_1 = self.info["location_stats"]["ypos"]

        action["jump"] = 1
        self.obs, _, _, self.info = self._step(action)
        height_2 = self.info["location_stats"]["ypos"]

        self._null_action(1)
        if height_2 - height_1 > 0.419:
            pass
        else:
            """euip pickaxe"""
            self.obs, _, _, self.info = self._step(action)
            height = self.info["location_stats"]["ypos"]
            if height < 50:
                # find pickaxe
                labels = self.get_labels()
                inventory_id_diamond = self.find_in_inventory(
                    labels, "diamond_pickaxe", "item"
                )
                inventory_id_iron = self.find_in_inventory(
                    labels, "iron_pickaxe", "item"
                )
                inventory_id_stone = self.find_in_inventory(
                    labels, "stone_pickaxe", "item"
                )
                inventory_id_wooden = self.find_in_inventory(
                    labels, "wooden_pickaxe", "item"
                )

                if inventory_id_wooden:
                    inventory_id = inventory_id_wooden
                if inventory_id_stone:
                    inventory_id = inventory_id_stone
                if inventory_id_iron:
                    inventory_id = inventory_id_iron
                if inventory_id_diamond:
                    inventory_id = inventory_id_diamond

                if inventory_id != "inventory_0":
                    self.open_inventory_wo_recipe()

                    """clear inventory 0"""
                    labels = self.get_labels()
                    if labels["inventory_0"]["type"] != "none":
                        for i in range(9):
                            if "resource_" + str(i) in labels:
                                del labels["resource_" + str(i)]
                        inventory_id_none = self.find_in_inventory(labels, "none")
                        self.pull_item_all(
                            self.crafting_slotpos, "inventory_0", inventory_id_none
                        )

                    self.pull_item(
                        self.crafting_slotpos, inventory_id, "inventory_0", 1
                    )
                    self._call_func("inventory")
                    self.current_gui_type = None
                    self.crafting_slotpos = "none"
                    self._call_func("hotbar.1")

            action = self.env.noop_action()
            for i in range(2):
                action["camera"] = np.array([-88, 0])
                self.obs, _, _, self.info = self._step(action)

            action["camera"] = np.array([22, 0])
            self.obs, _, _, self.info = self._step(action)

            for i in range(5):
                action["camera"] = np.array([0, 60])
                self.obs, _, _, self.info = self._step(action)
                self._attack_continue(attack_num)

    # open crafting_table
    def open_crating_table_wo_recipe(self):
        self.pre_open_tabel()
        self._null_action(1)
        if self.info["isGuiOpen"]:
            self._call_func("inventory")
        self.open_inventory_wo_recipe()
        labels = self.get_labels()
        inventory_id = self.find_in_inventory(labels, "crafting_table")
        self._assert(inventory_id, MISSING_MATERIAL_FORMAT.format("crafting_table", 1))
        self.has_crafting_table = True
        _invenotry_id = int(inventory_id.split("_")[-1])
        if _invenotry_id >= 0 and _invenotry_id <= 8:
            self._call_func("inventory")
            self.current_gui_type = None
            self.crafting_slotpos = "none"
            self._call_func(f"hotbar.{_invenotry_id+1}")
        else:
            if inventory_id != "inventory_0":
                labels = self.get_labels()
                if labels["inventory_0"]["type"] != "none":
                    for i in range(9):
                        del labels["resource_" + str(i)]
                    inventory_id_none = self.find_in_inventory(labels, "none")
                    self.pull_item_all(
                        self.crafting_slotpos, "inventory_0", inventory_id_none
                    )
                self.pull_item(self.crafting_slotpos, inventory_id, "inventory_0", 1)

            self._call_func("inventory")
            self.current_gui_type = None
            self.crafting_slotpos = "none"

            self._call_func("hotbar.1")

        self._place_down()
        for i in range(5):
            self._call_func("use")
            if self.info["isGuiOpen"]:
                break
        self.cursor = [WIDTH // 2, HEIGHT // 2]
        self.current_gui_type = "crating_table_wo_recipe"
        self.crafting_slotpos = SLOT_POS_TABLE_WO_RECIPE

    # action wrapper
    def _call_func(self, func_name: str):
        action = self.env.noop_action()
        action[func_name] = 1
        for i in range(1):
            self.obs, _, _, self.info = self._step(action)
        action[func_name] = 0
        for i in range(5):
            self.obs, _, _, self.info = self._step(action)

    def _look_down(self):
        action = self.env.noop_action()
        self._null_action()
        for i in range(2):
            action["camera"] = np.array([88, 0])
            self.obs, _, _, self.info = self._step(action)

    def _look_up(self):
        action = self.env.noop_action()
        self._null_action()
        for i in range(2):
            action["camera"] = np.array([-88, 0])
            self.obs, _, _, self.info = self._step(action)

    def _jump(self):
        self._call_func("jump")

    def _place_down(self):
        self._look_down()
        self._attack_continue(2)
        self._jump()
        self._call_func("use")

    def turn_left(self):
        action = self.env.noop_action()
        self._null_action()
        # for i in range(2):
        action["camera"] = np.array([0, -45])
        self.obs, _, _, self.info = self._step(action)

    def turn_back(self):
        # 平滑移动
        times = 20
        dy = -180 / times
        for i in range(times):
            action = self.env.noop_action()
            # self._null_action()
            # action["jump"] = 1
            # for i in range(2):
            d1, d2 = self._random_turn()
            action["camera"] = np.array([d1, dy + d2])
            self.obs, _, _, self.info = self._step(action)
        self._null_action()

    def _random_turn(self):
        d1 = random.uniform(-10, 10)
        d2 = random.uniform(-10, 10)
        return d1, d2

    def jump_forward(self):
        action = self.env.noop_action()
        action["jump"] = 1
        action["forward"] = 1
        self.obs, _, _, self.info = self._step(action)
        self._null_action()

    def _use_item(self):
        self._call_func("use")

    def _select_item(self):
        self._call_func("attack")

    def _null_action(self, times=1):
        action = self.env.noop_action()
        for i in range(times):
            self.obs, _, _, self.info = self._step(action)

    # continue attack (retuen crafting table)
    def _attack_continue(self, times=1):
        action = self.env.noop_action()
        action["attack"] = 1
        for i in range(times):
            self.obs, _, _, self.info = self._step(action)

    # move
    def move_to_pos(self, x: float, y: float, speed: float = 20):
        camera_x = x - self.cursor[0]
        camera_y = y - self.cursor[1]
        distance = max(abs(camera_x), abs(camera_y))
        num_steps = int(random.uniform(5, 10) * math.sqrt(distance) / speed)
        if num_steps < 1:
            num_steps = 1
        for _ in range(num_steps):
            d1 = camera_x / num_steps
            d2 = camera_y / num_steps
            self.move_once(d1, d2)

    def random_move_or_stay(self):
        if np.random.uniform(0, 1) > 0.5:
            num_random = random.randint(2, 4)
            if random.uniform(0, 1) > 0.25:
                for i in range(num_random):
                    self.move_once(0, 0)
            else:
                for i in range(num_random):
                    d1 = random.uniform(-5, 5)
                    d2 = random.uniform(-5, 5)
                    self.move_once(d1, d2)
        else:
            pass

    def move_once(self, x: float, y: float):
        action = self.env.noop_action()
        action["camera"] = np.array([y * CAMERA_SCALER, x * CAMERA_SCALER])
        self.obs, _, _, self.info = self._step(action)
        self.cursor[0] += x
        self.cursor[1] += y

    def move_to_slot(self, SLOT_POS: Dict, slot: str):
        self._assert(slot in SLOT_POS, f"Error: slot: {slot}")
        x, y = SLOT_POS[slot]
        self.move_to_pos(x, y)

    # pull
    # select item_from, select item_to
    def pull_item_all(self, SLOT_POS: Dict, item_from: str, item_to: str) -> None:
        self.move_to_slot(SLOT_POS, item_from)
        self._null_action(1)
        self._select_item()
        self._null_action(1)
        self.move_to_slot(SLOT_POS, item_to)
        self._null_action(1)
        self._select_item()
        self._null_action(1)
        self.random_move_or_stay()

    def swap_item(self, SLOT_POS: Dict, item_from: str, item_to: str):
        self.move_to_slot(SLOT_POS, item_from)
        self._null_action(1)
        self._select_item()
        self._null_action(1)
        self.move_to_slot(SLOT_POS, item_to)
        self._null_action(1)
        self._select_item()
        self._null_action(1)
        self.move_to_slot(SLOT_POS, item_from)
        self._null_action(1)
        self._select_item()

    # select item_from, use n item_to
    def pull_item(
        self, SLOT_POS: Dict, item_from: str, item_to: str, target_number: int
    ) -> None:
        if "resource" in item_to:
            item = self.info["plain_inventory"][int(item_from.split("_")[-1])]
            self.resource_record[item_to] = item
        self.move_to_slot(SLOT_POS, item_from)
        self._null_action(1)
        self._select_item()
        self.move_to_slot(SLOT_POS, item_to)
        self._null_action(1)
        for i in range(target_number):
            self._use_item()
            self._null_action(1)
        self.random_move_or_stay()

    # use n item_to
    def pull_item_continue(
        self, SLOT_POS: Dict, item_to: str, item: str, target_number: int
    ) -> None:
        if "resource" in item_to:
            self.resource_record[item_to] = item
        self.move_to_slot(SLOT_POS, item_to)
        self._null_action(1)
        for i in range(target_number):
            self._use_item()
            self._null_action(1)
        self.random_move_or_stay()

    # select item_to
    def pull_item_return(
        self,
        SLOT_POS: Dict,
        item_to: str,
    ) -> None:
        self.move_to_slot(SLOT_POS, item_to)
        self._null_action(1)
        self._select_item()
        self._null_action(1)
        self.random_move_or_stay()

    # use n item_frwm, select item_to
    def pull_item_result(
        self, SLOT_POS: Dict, item_from: str, item_to: str, target_number: int
    ) -> None:
        self.move_to_slot(SLOT_POS, item_from)
        for i in range(target_number):
            self._use_item()
            self._null_action(1)
        self.move_to_slot(SLOT_POS, item_to)
        self._select_item()
        self._null_action(1)
        self.random_move_or_stay()

    # get all labels
    def get_labels(self, noop=True):
        if noop:
            self._null_action(1)
        result = {}
        # generate resource recording item labels
        for i in range(9):
            slot = f"resource_{i}"
            item = self.resource_record[slot]
            result[slot] = item

        # generate inventory item labels
        for slot, item in self.info["plain_inventory"].items():
            result[f"inventory_{slot}"] = item

        return result

    # crafting
    def crafting(self, target: str, target_num: int = 1):
        try:
            # is item/tag
            is_tag = False

            for key in self.tag_info:
                if key[10:] == target:
                    is_tag = True

            # open recipe one by one: only shapeless crafting like oak_planks
            if is_tag:
                enough_material = False
                enough_material_target = "none"
                item_list = self.tag_info["minecraft:" + target]

                require_item_and_number = {}
                is_example = True

                for item in item_list:
                    subtarget = item[10:]
                    try:
                        recipe_json_path = os.path.join(
                            self.recipe_path, subtarget + ".json"
                        )
                        with open(recipe_json_path) as file:
                            recipe_info = json.load(file)
                    except FileNotFoundError:
                        subtarget = self.get_best_match_recipe(subtarget)
                        recipe_json_path = os.path.join(
                            self.recipe_path, subtarget + ".json"
                        )
                        with open(recipe_json_path) as file:
                            recipe_info = json.load(file)
                    need_table = self.crafting_type(recipe_info)

                    # find materials(shapeless) like oak_planks
                    ingredients = recipe_info.get("ingredients")
                    random.shuffle(ingredients)
                    items = dict()
                    items_type = dict()

                    # clculate the amount needed and store <item, quantity> in items
                    for i in range(len(ingredients)):
                        if ingredients[i].get("item"):
                            item = ingredients[i].get("item")[10:]
                            item_type = "item"
                        else:
                            item = ingredients[i].get("tag")[10:]
                            item_type = "tag"
                        items_type[item] = item_type
                        if items.get(item):
                            items[item] += 1
                        else:
                            items[item] = 1

                    if recipe_info.get("result").get("count"):
                        iter_num = math.ceil(
                            target_num / int(recipe_info.get("result").get("count"))
                        )
                    else:
                        iter_num = target_num

                    enough_material_subtarget = True
                    for item, num_need in items.items():
                        labels = self.get_labels()
                        inventory_id = self.find_in_inventory(
                            labels, item, items_type[item]
                        )
                        if not inventory_id:
                            enough_material_subtarget = False
                            break
                        inventory_num = labels.get(inventory_id).get("quantity")
                        if num_need * iter_num > inventory_num:
                            enough_material_subtarget = False
                            break
                    if enough_material_subtarget:
                        enough_material = True
                        enough_material_target = subtarget
                        require_item_and_number[subtarget] = items
                    if is_example:
                        require_item_and_number["example"] = items
                        is_example = False

                if enough_material:
                    target = enough_material_target
                else:
                    self._assert(
                        0,
                        f"missing material: {render_recipe(require_item_and_number['example'])}",
                    )

            # if inventory is open by accident, close inventory
            self._null_action(1)
            if self.info["isGuiOpen"]:
                self._call_func("inventory")

            try:
                recipe_json_path = os.path.join(self.recipe_path, target + ".json")
                with open(recipe_json_path) as file:
                    recipe_info = json.load(file)
            except FileNotFoundError:
                target = self.get_best_match_recipe(target)
                recipe_json_path = os.path.join(self.recipe_path, target + ".json")
                with open(recipe_json_path) as file:
                    recipe_info = json.load(file)

            need_table = self.crafting_type(recipe_info)

            if need_table:
                self.open_crating_table_wo_recipe()
            else:
                self.open_inventory_wo_recipe()

            # crafting
            if recipe_info.get("result").get("count"):
                iter_num = math.ceil(
                    target_num / int(recipe_info.get("result").get("count"))
                )
            else:
                iter_num = target_num

            self.crafting_once(target, iter_num, recipe_info, target_num)

            # close inventory
            self._call_func("inventory")
            if need_table:
                self.return_crafting_table()
            self.current_gui_type = None
            self.crafting_slotpos = "none"

        except AssertionError as e:
            if self.has_crafting_table:
                self.return_crafting_table()
            return False, str(e)
        except RuntimeError as e:
            if self.has_crafting_table:
                self.return_crafting_table()
            return False, str(e)

        return True, None

    # return crafting table
    def return_crafting_table(self):
        self._look_down()
        labels = self.get_labels()
        table_info = self.find_in_inventory(labels, "crafting_table")
        tabel_exist = 0
        if table_info:
            tabel_exist = 1
            tabel_num = labels.get(table_info).get("quantity")

        done = 0
        for i in range(4):
            for i in range(10):
                self._attack_continue(8)
                labels = self.get_labels(noop=False)
                if tabel_exist:
                    table_info = self.find_in_inventory(labels, "crafting_table")
                    tabel_num_2 = labels.get(table_info).get("quantity")
                    if tabel_num_2 != tabel_num:
                        done = 1
                        break
                else:
                    table_info = self.find_in_inventory(labels, "crafting_table")
                    if table_info:
                        done = 1
                        break
            self._call_func("forward")
        # self._assert(done, "return crafting_table unsuccessfully")

    # judge crafting_table / inventory
    def crafting_type(self, target_data: Dict):
        if "pattern" in target_data:
            pattern = target_data.get("pattern")
            col_len = len(pattern)
            row_len = len(pattern[0])
            if col_len <= 2 and row_len <= 2:
                return False
            else:
                return True
        else:
            ingredients = target_data.get("ingredients")
            item_num = len(ingredients)
            if item_num <= 4:
                return False
            else:
                return True

    # search item in agent's inventory
    def find_in_inventory(
        self, labels: Dict, item: str, item_type: str = "item", path=None
    ):
        if path == None:
            path = []
        for key, value in labels.items():
            current_path = path + [key]
            if item_type == "item":
                if re.match(item, str(value)):
                    return current_path
                elif isinstance(value, dict):
                    result = self.find_in_inventory(
                        value, item, item_type, current_path
                    )
                    if result is not None:
                        return result[0]
            elif item_type == "tag":
                # tag info
                relative_path = os.path.join("tag_items.json")
                tag_json_path = os.path.join(self.root_path, relative_path)
                with open(tag_json_path) as file:
                    self.tag_info = json.load(file)

                item_list = self.tag_info["minecraft:" + item]
                for i in range(len(item_list)):
                    if re.match(item_list[i][10:], str(value)):
                        return current_path
                    elif isinstance(value, dict):
                        result = self.find_in_inventory(
                            value, item, item_type, current_path
                        )
                        if result is not None:
                            return result[0]
        return None

    # crafting once
    def crafting_once(
        self, target: str, iter_num: int, recipe_info: Dict, target_num: int
    ):
        # shaped crafting
        if "pattern" in recipe_info:
            self.crafting_shaped(target, iter_num, recipe_info)
        # shapless crafting
        else:
            self.crafting_shapeless(target, iter_num, recipe_info)

        # get result
        # Do not put the result in resource
        labels = self.get_labels()
        for i in range(9):
            del labels["resource_" + str(i)]

        result_inventory_id_1 = self.find_in_inventory(labels, target)

        if result_inventory_id_1:
            item_num = labels.get(result_inventory_id_1).get("quantity")
            if item_num + target_num < 60:
                self.pull_item_result(
                    self.crafting_slotpos, "result_0", result_inventory_id_1, iter_num
                )
                labels_after = self.get_labels()
                item_num_after = labels_after.get(result_inventory_id_1).get("quantity")

                if item_num == item_num_after:
                    result_inventory_id_2 = self.find_in_inventory(labels, "none")
                    self._assert(result_inventory_id_2, f"no space to place result")
                    self.pull_item_return(self.crafting_slotpos, result_inventory_id_2)
                    self._assert(
                        self.get_labels().get(result_inventory_id_2).get("type")
                        == target,
                        f"fail for unkown reason",
                    )
            else:
                result_inventory_id_2 = self.find_in_inventory(labels, "none")
                self._assert(result_inventory_id_2, f"no space to place result")
                self.pull_item_result(
                    self.crafting_slotpos, "result_0", result_inventory_id_2, iter_num
                )
                self._assert(
                    self.get_labels().get(result_inventory_id_2).get("type") == target,
                    f"fail for unkown reason",
                )
        else:
            result_inventory_id_2 = self.find_in_inventory(labels, "none")
            self._assert(result_inventory_id_2, f"no space to place result")
            self.pull_item_result(
                self.crafting_slotpos, "result_0", result_inventory_id_2, iter_num
            )
            self._assert(
                self.get_labels().get(result_inventory_id_2).get("type") == target,
                f"fail for unkown reason",
            )

        # clear resource
        self.resource_record = {
            f"resource_{x}": {"type": "none", "quantity": 0} for x in range(9)
        }

    # shaped crafting
    def crafting_shaped(self, target: str, iter_num: int, recipe_info: Dict):
        slot_pos = self.crafting_slotpos
        labels = self.get_labels()
        pattern = recipe_info.get("pattern")
        items = recipe_info.get("key")
        items = random_dic(items)
        # place each item in order
        for k, v in items.items():
            signal = k
            if v.get("item"):
                item = v.get("item")[10:]
                item_type = "item"
            else:
                item = v.get("tag")[10:]
                item_type = "tag"
            labels = self.get_labels()

            # clculate the amount needed
            num_need = 0
            for i in range(len(pattern)):
                for j in range(len(pattern[i])):
                    if pattern[i][j] == signal:
                        num_need += 1
            num_need = num_need * iter_num
            inventory_id = self.find_in_inventory(labels, item, item_type)
            self._assert(inventory_id, MISSING_MATERIAL_FORMAT.format(item, num_need))
            inventory_num = labels.get(inventory_id).get("quantity")
            self._assert(
                num_need <= inventory_num,
                MISSING_MATERIAL_FORMAT.format(item, num_need - inventory_num),
            )
            # place
            resource_idx = 0
            first_pull = 1
            if "table" in self.current_gui_type:
                type = 3
            else:
                type = 2
            for i in range(len(pattern)):
                resource_idx = i * type
                for j in range(len(pattern[i])):
                    if pattern[i][j] == signal:
                        if first_pull:
                            self.pull_item(
                                slot_pos,
                                inventory_id,
                                "resource_" + str(resource_idx),
                                iter_num,
                            )
                            first_pull = 0
                        else:
                            self.pull_item_continue(
                                slot_pos,
                                "resource_" + str(resource_idx),
                                item,
                                iter_num,
                            )
                    resource_idx += 1

            # return the remaining items
            if num_need < inventory_num:
                self.pull_item_return(slot_pos, inventory_id)

    # shapeless crafting
    def crafting_shapeless(self, target: str, iter_num: int, recipe_info: Dict):
        slot_pos = self.crafting_slotpos
        labels = self.get_labels()
        ingredients = recipe_info.get("ingredients")
        random.shuffle(ingredients)
        items = dict()
        items_type = dict()

        # clculate the amount needed and store <item, quantity> in items
        for i in range(len(ingredients)):
            if ingredients[i].get("item"):
                item = ingredients[i].get("item")[10:]
                item_type = "item"
            else:
                item = ingredients[i].get("tag")[10:]
                item_type = "tag"
            items_type[item] = item_type
            if items.get(item):
                items[item] += 1
            else:
                items[item] = 1

        # place each item in order
        resource_idx = 0
        for item, num_need in items.items():
            labels = self.get_labels()
            inventory_id = self.find_in_inventory(labels, item, items_type[item])
            # self._assert(inventory_id, f"not enough {item}")
            self._assert(
                inventory_id, MISSING_MATERIAL_FORMAT.format(item, num_need * iter_num)
            )
            inventory_num = labels.get(inventory_id).get("quantity")
            self._assert(
                num_need * iter_num <= inventory_num,
                MISSING_MATERIAL_FORMAT.format(
                    item, num_need * iter_num - inventory_num
                ),
            )

            # place
            num_need -= 1
            self.pull_item(
                slot_pos, inventory_id, "resource_" + str(resource_idx), iter_num
            )

            resource_idx += 1
            if num_need != 0:
                for i in range(num_need):
                    self.pull_item_continue(
                        slot_pos, "resource_" + str(resource_idx), item, iter_num
                    )
                    resource_idx += 1

            # return the remaining items
            num_need = (num_need + 1) * iter_num
            if num_need < inventory_num:
                self.pull_item_return(slot_pos, inventory_id)
