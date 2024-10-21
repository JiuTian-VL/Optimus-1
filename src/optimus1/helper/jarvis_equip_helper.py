import random
from typing import Tuple

from .jarvis_craft_helper import CraftHelper
from .slot import *


class EquipHelper(CraftHelper):
    def __init__(
        self,
        env,
        sample_ratio: float = 0.5,
        inventory_slot_range: Tuple[int, int] = (0, 36),
        **kwargs,
    ) -> None:
        super().__init__(env, sample_ratio, inventory_slot_range, **kwargs)

    """ find empty slot (there is no object) ids of inventory, num from 0-35, 
        if is the bottom bar, add it to empty_ids_bar number from 0-8 """

    def find_empty_box(self, inventory):
        empty_ids, empty_ids_bar = [], []
        for k, v in inventory.items():
            if v["type"] == "none":
                empty_ids.append(k)
                if k < 9:
                    empty_ids_bar.append(k)
        return empty_ids, empty_ids_bar

    """ equip item such as wooden_pickaxe """

    def equip_item(self, target_item: str):
        try:
            # check target_item is equippable before equip_item()
            # check is gui open and open gui
            if not self.info["isGuiOpen"]:
                self.open_inventory_wo_recipe()
            # check and get target_item's pos in inventory
            pos_id = self.find_in_inventory(self.get_labels(), target_item)
            # self._assert(pos_id, f"Can not find {target_item} in inventory")
            self._assert(pos_id, f'missing material: {{"{target_item}": 1}}')

            """ whether pickaxe in inventory or bar, move anyhow """
            _, empty_bar = self.find_empty_box(self.info["plain_inventory"])
            if len(empty_bar) == 0:
                result = [f"inventory_{i}" for i in range(9)]
                result = random.choice(result)
            else:
                result = "inventory_{}".format(random.choice(empty_bar))

            pos_bottom = result
            slot_pos = SLOT_POS_MAPPING[self.current_gui_type]
            self.pull_item(slot_pos, pos_id, pos_bottom, 1)

            # if bottom is fully, the item will be substitued with target_item

            self._call_func("inventory")  # close inventory
            hotbar_id = int(pos_bottom.split("_")[-1])
            self._call_func("hotbar.{}".format(hotbar_id + 1))
            self._attack_continue()

        except AssertionError as e:
            return False, str(e)

        return True, None
