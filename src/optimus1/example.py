wooden_pickaxe: str = """
<task>: craft a wooden pickaxe.
<visual info>
health bar: full
food bar: full
hotbar: empty
environment: forest
<craft graph>:
craft 1 wooden_pickaxe summary:
1. log: need 3
2. planks: need 9
3. stick: need 2
4. crafting_table: need 1
5. wooden_pickaxe: need 1
<task planning>
{
"step 1": {"task": "chop a tree", "goal": ["logs", 3]},
"step 2": {"task": "craft planks", "goal": ["planks", 9]},
"step 3": {"task": "craft stick", "goal": ["stick", 2]},
"step 4": {"task": "craft crafting table", "goal": ["crafting_table", 1]},
"step 5": {"task": "craft wooden pickaxe", "goal": ["wooden_pickaxe", 1]}
}
"""
stone_sword: str = """
<task>: craft a stone sword.
<visual info>
health bar: full
food bar: full
hotbar: empty
environment: forest
<craft graph>:
craft 1 stone_sword summary:
1. log: need 5
2. planks: need 15
3. stick: need 3
4. crafting_table: need 1
5. wooden_pickaxe: need 1
6. cobblestone: need 2
7. stone_sword: need 1
<task planning>
{
"step 1": {"task": "chop a tree", "goal": ["logs", 5]},
"step 2": {"task": "craft planks", "goal": ["planks", 15]},
"step 3": {"task": "craft stick", "goal": ["stick", 3]},
"step 4": {"task": "craft crafting table", "goal": ["crafting_table", 1]},
"step 5": {"task": "craft wooden pickaxe", "goal": ["wooden_pickaxe", 1]},
"step 6": {"task": "equip wooden pickaxe", "goal": ["wooden_pickaxe", 1]},
"step 7": {"task": "dig down and break down cobblestone", "goal": ["cobblestone",2]},
"step 8": {"task": "craft stone sword", "goal": ["stone_sword", 1]}
}
"""
iron_sword: str = """
<task>: craft a iron sword.
<visual info>
health bar: full
food bar: full
hotbar: empty
environment: forest
<craft graph>:
craft 1 iron_sword summary:
1. log: need 3
2. iron_ore: need 2
3. stone: need 8
4. planks: need 9
5. iron_ingot: need 2
6. stick: need 1
7. crafting_table: need 1
8. furnace: need 1
9. iron_sword: need 1
<task planning>
{
"step 1": {"task": "chop a tree", "goal": ["logs", 5]},
"step 2": {"task": "craft planks", "goal": ["planks", 16]},
"step 3": {"task": "craft stick", "goal": ["stick", 4]},
"step 4": {"task": "craft crafting table", "goal": ["crafting_table", 1]},
"step 5": {"task": "craft wooden pickaxe", "goal": ["wooden_pickaxe", 1]},
"step 6": {"task": "equip wooden pickaxe", "goal": ["wooden_pickaxe", 1]},
"step 7": {"task": "dig down and break down cobblestone", "goal": ["cobblestone",11]},
"step 8": {"task": "craft stone pickaxe", "goal": ["stone_pickaxe", 1]},
"step 9": {"task": "equip stone pickaxe", "goal": ["stone_pickaxe", 1]},
"step 10": {"task": "craft furnace", "goal": ["furnace", 1]},
"step 11": {"task": "dig down and break down iron ore", "goal": ["iron_ore", 2]},
"step 12": {"task": "smelt iron ore", "goal": ["iron_ingot", 2]},
"step 13": {"task": "craft iron sword", "goal": ["iron_sword", 1]}
}"""

golden_sword: str = """"
<task>: craft a golden sword.
<visual info>
health bar: full
food bar: full
hotbar: empty
environment: birch forest
<craft graph>:
craft 1 golden_sword summary:
1. log: need 9
2. planks: need 27
3. stick: need 7
4. crafting_table: need 1
5. wooden_pickaxe: need 1
6. cobblestone: need 19
7. coal_ore: need 5
8. furnace: need 1
9. stone_pickaxe: need 1
10. iron_ore: need 3
11. iron_ingot: need 3
12. iron_pickaxe: need 1
13. gold_block: need 1
14. gold_ingot: need 2
15. golden_sword: need 1
<task planning>
{
"step 1": {"task": "chop a tree", "goal": ["logs", 5]},
"step 2": {"task": "craft planks", "goal": ["planks", 16]},
"step 3": {"task": "craft stick", "goal": ["stick", 4]},
"step 4": {"task": "craft crafting table", "goal": ["crafting_table", 1]},
"step 5": {"task": "craft wooden pickaxe", "goal": ["wooden_pickaxe", 1]},
"step 6": {"task": "equip wooden pickaxe", "goal": ["wooden_pickaxe", 1]},
"step 7": {"task": "dig down and break down cobblestone", "goal": ["cobblestone",11]},
"step 8": {"task": "craft stone pickaxe", "goal": ["stone_pickaxe", 1]},
"step 9": {"task": "equip stone pickaxe", "goal": ["stone_pickaxe", 1]},
"step 10": {"task": "craft furnace", "goal": ["furnace", 1]},
"step 11": {"task": "dig down and break down iron ore", "goal": ["iron_ore", 3]},
"step 12": {"task": "smelt iron ore", "goal": ["iron_ingot", 3]},
"step 13": {"task": "craft iron pickaxe", "goal": ["iron_pickaxe", 1]},
"step 14": {"task": "equip iron pickaxe", "goal": ["iron_pickaxe", 1]},
"step 15": {"task": "dig down and mine gold", "goal": ["gold_ore", 1]},
"step 16": {"task": "smelt gold", "goal": ["gold_ingot", 1]},
"step 17": {"task": "craft golden sword", "goal": ["golden_sword", 1]}
}
"""

jukebox = """
<task>: craft a jukebox.
<visual info>
health bar: full
food bar: full
hotbar: empty
environment: birch forest
<craft graph>:
craft 1 jukebox summary:
1. log: need 10
2. planks: need 33
3. crafting_table: need 1
4. stick: need 6
5. wooden_pickaxe: need 1
6. cobblestone: need 11
7. stone_pickaxe: need 1
8. furnace: need 1
9. iron_ore: need 3
10. iron_ingot: need 3
11. iron_pickaxe: need 1
12. diamond_ore: need 1
13. diamond: need 1
14. jukebox: need 1
<task planning>
{
"step 1": {"task": "chop a tree", "goal": ["logs", 10]},
"step 2": {"task": "craft planks", "goal": ["planks", 33]},
"step 3": {"task": "craft stick", "goal": ["stick", 6]},
"step 4": {"task": "craft crafting table", "goal": ["crafting_table", 1]},
"step 5": {"task": "craft wooden pickaxe", "goal": ["wooden_pickaxe", 1]},
"step 6": {"task": "equip wooden pickaxe", "goal": ["wooden_pickaxe", 1]},
"step 7": {"task": "dig down and break down cobblestone", "goal": ["cobblestone",11]},
"step 8": {"task": "craft stone pickaxe", "goal": ["stone_pickaxe", 1]},
"step 9": {"task": "equip stone pickaxe", "goal": ["stone_pickaxe", 1]},
"step 10": {"task": "craft furnace", "goal": ["furnace", 1]},
"step 11": {"task": "dig down and break down iron ore", "goal": ["iron_ore", 3]},
"step 12": {"task": "smelt iron ore", "goal": ["iron_ingot", 3]},
"step 13": {"task": "craft iron pickaxe", "goal": ["iron_pickaxe", 1]},
"step 14": {"task": "equip iron pickaxe", "goal": ["iron_pickaxe", 1]},
"step 15": {"task": "dig down and mine diamond", "goal": ["diamond", 1]},
"step 16": {"task": "craft jukebox", "goal": ["jukebox", 1]}
}
"""
