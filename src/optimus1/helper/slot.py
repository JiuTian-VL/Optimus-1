import random

CAMERA_SCALER = 360.0 / 2400.0
WIDTH, HEIGHT = 640, 360

# compute slot position
KEY_POS_INVENTORY_WO_RECIPE = {
    "resource_slot": {
        "left-top": (329, 114),
        "right-bottom": (365, 150),
        "row": 2,
        "col": 2,
        "prefix": "resource",
        "start_id": 0,
    },
    "result_slot": {
        "left-top": (385, 124),
        "right-bottom": (403, 142),
        "row": 1,
        "col": 1,
        "prefix": "result",
        "start_id": 0,
    },
    "hotbar_slot": {
        "left-top": (239, 238),
        "right-bottom": (401, 256),
        "row": 1,
        "col": 9,
        "prefix": "inventory",
        "start_id": 0,
    },
    "inventory_slot": {
        "left-top": (239, 180),
        "right-bottom": (401, 234),
        "row": 3,
        "col": 9,
        "prefix": "inventory",
        "start_id": 9,
    },
    "recipe_slot": {
        "left-top": (336, 158),
        "right-bottom": (356, 176),
        "row": 1,
        "col": 1,
        "prefix": "recipe",
        "start_id": 0,
    },
}
KEY_POS_TABLE_WO_RECIPE = {
    "resource_slot": {
        "left-top": (261, 113),
        "right-bottom": (315, 167),
        "row": 3,
        "col": 3,
        "prefix": "resource",
        "start_id": 0,
    },
    "result_slot": {
        "left-top": (351, 127),
        "right-bottom": (377, 153),
        "row": 1,
        "col": 1,
        "prefix": "result",
        "start_id": 0,
    },
    "hotbar_slot": {
        "left-top": (239, 238),
        "right-bottom": (401, 256),
        "row": 1,
        "col": 9,
        "prefix": "inventory",
        "start_id": 0,
    },
    "inventory_slot": {
        "left-top": (239, 180),
        "right-bottom": (401, 234),
        "row": 3,
        "col": 9,
        "prefix": "inventory",
        "start_id": 9,
    },
    "recipe_slot": {
        "left-top": (237, 131),
        "right-bottom": (257, 149),
        "row": 1,
        "col": 1,
        "prefix": "recipe",
        "start_id": 0,
    },
}
KEY_POS_FURNACE_WO_RECIPE = {
    "resource_slot": {
        "left-top": (287, 113),
        "right-bottom": (303, 164),
        "row": 2,
        "col": 1,
        "prefix": "resource",
        "start_id": 0,
    },
    "result_slot": {
        "left-top": (345, 127),
        "right-bottom": (368, 152),
        "row": 1,
        "col": 1,
        "prefix": "result",
        "start_id": 0,
    },
    "hotbar_slot": {
        "left-top": (242, 236),
        "right-bottom": (401, 256),
        "row": 1,
        "col": 9,
        "prefix": "inventory",
        "start_id": 0,
    },
    "inventory_slot": {
        "left-top": (242, 178),
        "right-bottom": (401, 234),
        "row": 3,
        "col": 9,
        "prefix": "inventory",
        "start_id": 9,
    },
    "recipe_slot": {
        "left-top": (254, 132),
        "right-bottom": (272, 147),
        "row": 1,
        "col": 1,
        "prefix": "recipe",
        "start_id": 0,
    },
}


def COMPUTE_SLOT_POS(KEY_POS):
    result = {}
    for k, v in KEY_POS.items():
        left_top = v["left-top"]
        right_bottom = v["right-bottom"]
        row = v["row"]
        col = v["col"]
        prefix = v["prefix"]
        start_id = v["start_id"]
        width = right_bottom[0] - left_top[0]
        height = right_bottom[1] - left_top[1]
        slot_width = width // col
        slot_height = height // row
        slot_id = 0
        for i in range(row):
            for j in range(col):
                result[f"{prefix}_{slot_id + start_id}"] = (
                    left_top[0] + j * slot_width + (slot_width // 2),
                    left_top[1] + i * slot_height + (slot_height // 2),
                )
                slot_id += 1
    return result


SLOT_POS_INVENTORY_WO_RECIPE = COMPUTE_SLOT_POS(KEY_POS_INVENTORY_WO_RECIPE)
SLOT_POS_TABLE_WO_RECIPE = COMPUTE_SLOT_POS(KEY_POS_TABLE_WO_RECIPE)
SLOT_POS_FURNACE_WO_RECIPE = COMPUTE_SLOT_POS(KEY_POS_FURNACE_WO_RECIPE)
SLOT_POS_MAPPING = {
    "inventory_w_recipe": None,  # SLOT_POS_INVENTORY_W_RECIPE
    "inventory_wo_recipe": SLOT_POS_INVENTORY_WO_RECIPE,
    "crating_table_w_recipe": None,  # SLOT_POS_TABLE_W_RECIPE
    "crating_table_wo_recipe": SLOT_POS_TABLE_WO_RECIPE,
}


def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic
