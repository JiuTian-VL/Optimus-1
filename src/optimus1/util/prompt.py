import json
import re
from typing import Any, Dict, List

PlanList = List[Dict[str, Any]]


def render_gpt4_plan(plan: str, is_replan: bool = False) -> PlanList:
    plan = plan.replace("<task planning>:", "<task planning>").replace("**", "")
    sep_str = "<replan>:" if is_replan else "<task planning>"

    temp = plan.split(sep_str)[-1].strip()
    if "```json" in temp:
        temp = temp.split("```json")[1].strip().split("```")[0].strip()

    if "{{" in temp:
        temp = temp.replace("{{", "{").replace("}}", "}")

    r = temp.rfind("}")
    temp = temp[: r + 1]

    temp = json.loads(temp)

    sub_plans = [
        temp[step]
        for step in temp.keys()
        if "open" not in temp[step]["task"]
        and "place" not in temp[step]["task"]
        and "access" not in temp[step]["task"]
    ]

    for p in sub_plans:
        p["task"] = p["task"].replace("punch", "chop").replace("collect", "chop").replace("gather", "chop")

    return sub_plans


def render_reflection(reflection: str):
    """Environment: <Ocean>
    Situation: <Replan>
    Predicament: <In_water>
    """
    reflection = reflection.strip().replace(": ", ": <")

    matches = re.findall(r"<([^<]+)$", reflection, re.MULTILINE)
    rp = None
    if len(matches) == 3:
        rp = matches[2].split("/")[0].strip().lower()
    res = (
        matches[0].split("/")[0].replace(">", "").strip().lower().split(" ")[0].split("\n")[0],
        matches[1].split("/")[0].replace(">", "").strip().lower().split(" ")[0].split("\n")[0],
        rp,
    )
    return res


def render_recipe(recipe) -> str:
    lst = [f'"{k}": {v}' for k, v in recipe.items()]

    return "{" + ", ".join(lst) + "}"


def render_replan_example(replan: List[Dict[str, Any]]):
    res = {}
    for idx, plan in enumerate(replan):
        res[f"step {idx + 1}"] = plan
    return json.dumps(res)


if __name__ == "__main__":
    plan = """{
    "step 1": {"task": "punch a tree", "goal": ["logs", 3]},
    "step 2": {"task": "open inventory", "goal": ["inventory accessed", 1]},
    "step 3": {"task": "craft planks", "goal": ["planks", 12]},
    "step 4": {"task": "craft sticks", "goal": ["sticks", 4]},
    "step 5": {"task": "place crafting table", "goal": ["crafting_table placed", 1]},
    "step 6": {"task": "use crafting table", "goal": ["crafting_table used", 1]},
    "step 7": {"task": "craft wooden pickaxe", "goal": ["wooden_pickaxe", 1]}
}"""
    temp = json.loads(plan)
    print(temp["step 1"]["task"])
    sub_plans = [
        temp[step] for step in temp.keys() if "open" not in temp[step]["task"] and "place" not in temp[step]["task"]
    ]
    print(sub_plans)
    # print(render_reflection(plan))
