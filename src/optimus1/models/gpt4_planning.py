import base64
from typing import List

import openai

from .base_model import BasePlanningModel, BaseReflectionModel

client = openai.OpenAI(
    api_key="<your api key>",
    timeout=2000,
    max_retries=3,
)

## retrieval_prompt
retrieval_prompt = """
You are a MineCraft game expert and you can guide agents to complete complex tasks. For a given game screen and task, you need to complete <goal inference> and <visual inference>.
<goal inference>: According to the task, you need to infer the weapons, equipment, or materials required to complete the task.
<visual inference>: According to the game screen, you need to infer the following aspects: health bar, food bar, hotbar, environment.
I will give you an example as follow:
[Example]
<task>: craft a stone sword.
<goal inference>: stone sword
<visual inference>
health bar: full
food bar: full
hotbar: empty
environment: forest
Here is a game screen and task, you MUST output in example format.
<task>: {task}.
"""

plan_prompt = """You are a MineCraft game expert and you can guide agents to complete complex tasks. For a given game screen and task, you need to make a plan with the help of <visual info> and <craft graph>.
<visual info>: Consists of the following aspects: health bar, food bar, hotbar, environment. Based on the current visual information, you need to consider whether prequel steps needed to ensure that agent can complete the task.
<craft graph>: a top-down list of all the tools and materials needed to complete the task. 
I will give you an example of planning under specific visual conditions as follow:
[Example]
{example}
Here is a game screen and task, you MUST output in example format. Remember <task planning> MUST output in example format.
<task>: {task}
<visual info>: {visual}
<craft graph>: {graph}
"""

no_reflection_plan_prompt = """You are a MineCraft game expert and you can guide agents to complete complex tasks. For a given game screen and task, you need to make a plan.
I will give you an example of planning under specific visual conditions as follow:
[Example]
{example}
Here is a game screen and task, you MUST output in example format. Remember <task planning> MUST output in example format.
<task>: {task}
"""

### replan prompt ###
replan_prompt = """
You are a MineCraft game expert and you can guide agents to complete complex tasks. 
Agent is executing the <task>: {task1}, and agent meets <error>: {error}.
Based on <error> information and <knowledge>, replanning to allow the agent to successfully complete the <task>.

<knowledge>: The basic tools the agent can acquire come in multiple tiers based on your materials, and they include the pickaxe, the axe, and the shovel, for mining, respectively, stone-type, wood-type, and dirt & sand-type blocks. The six tiers are wood, stone, iron, gold, diamond, and netherite. For pickaxes in particular, wooden pickaxes can collect cobblestones and coal ore, stone pickaxe can collect iron ore, and more advanced ores require at least an iron pickaxe.

<logic inference>: a top-down list of all the tools and materials needed to complete the task. Then, summarise the materials required and their quantities from the bottom up.

You MUST focus on how to solve the <error>.
I will give you an example as follow:
[Example]
<logic inference>:
{logic1}
{example}
Here is a game screen and task, you MUST output in example format.Remember <replan> MUST output in example format.
<logic inference>:
{logic}
<task>: {task1}
<error>: {error}
"""

non_reflection_replan_prompt = """
You are a MineCraft game expert and you can guide agents to complete complex tasks. 
Agent is executing the <task>: {task1}, and agent meets <error>: {error}.
Based on <error> information and <knowledge>, replanning to allow the agent to successfully complete the <task>.

<knowledge>: The basic tools the agent can acquire come in multiple tiers based on your materials, and they include the pickaxe, the axe, and the shovel, for mining, respectively, stone-type, wood-type, and dirt & sand-type blocks. The six tiers are wood, stone, iron, gold, diamond, and netherite. For pickaxes in particular, wooden pickaxes can collect cobblestones and coal ore, stone pickaxe can collect iron ore, and more advanced ores require at least an iron pickaxe.

You MUST focus on how to solve the <error>.
I will give you an example as follow:
[Example]
<logic inference>:
{logic1}
{example}
Here is a game screen and task, you MUST output in example format.Remember <replan> MUST output in example format.
<task>: {task1}
<error>: {error}
"""

### reflection prompt ###
reflection_systerm = """
You are a MineCraft game expert and you can guide agents to complete complex tasks. Agent is executing the task: {task}.
Given two images about agent's state before executing the task and its current state, you should first detection the environment (forest, cave, ocean, etc.,) in which the agent is located, then determine whether the agent's current situation is done, continue, or replan.
<done>: Comparing the image before the task was performed, the current image reveals that the task is complete.
<continue>: Current image reveals that the task is NOT complete, but agent is in good state (good health, not hungry) with high likelihood to complete task.
<replan>: Current image reveals that the task is NOT complete, and agent is in bad state (bad health, or hungry) or situation (in danger, or in trouble), need for replanning. For replan, you need to further determine whether the agent's predicament is "drop_down" or "in_water". "drop_down" means that the agent has fallen into a cave or is trapped in a mountain or river, while "in_water" means that the agent is in the ocean and needs to return to land immediately.
"""
reflection_examples = """
I'll give you some examples to illustrate the different situations. Each example consists of two images, where the first image is the state of the agent before performing the task and the second image is the current state of the agent.\n[Examples]
"""
reflection_prompt = """
\nNow given two images about agent's state before executing the task and its current state, you MUST and ONLY output in following format:
Enviroment: <environment>
Situation: <situation>
(if situation is replan) Predicament: <predicament>
"""


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        a = base64.b64encode(image_file.read()).decode("utf-8")
    return a


def encode_image(image_path: list) -> tuple:
    a = _encode_image(image_path[0])
    b = _encode_image(image_path[1])
    return a, b


def is_path(path):
    if len(path) == 2:
        return True
    else:
        return False


class PlanningModel(BasePlanningModel, BaseReflectionModel):
    def __init__(self) -> None:
        super().__init__()

    def retrieve(
        self,
        task: str,
        image_path: str,
    ):
        image = _encode_image(image_path)

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": retrieval_prompt.format(task=task)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        },
                    ],
                },
            ],
            model="gpt-4o",
            max_tokens=2000,
        )
        return response.choices[0].message.content

    def planning(
        self,
        task: str,
        image_path: str,
        example: str | None = None,
        visual_info: str | None = None,
        graph: str | None = None,
    ):
        image = _encode_image(image_path)
        if visual_info is None and graph is None:
            prompt = no_reflection_plan_prompt.format(task=task, example=example)
        else:
            prompt = plan_prompt.format(
                task=task,
                example=example,
                visual=visual_info,
                graph=graph,
            )

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        },
                    ],
                },
            ],
            model="gpt-4o",
            max_tokens=2000,
        )
        return response.choices[0].message.content

    def replan(
        self,
        task: str,
        image_path: str,
        error_info: str | None = None,
        examples: str | None = None,
        graph_summary: str | None = None,
    ):
        image = _encode_image(image_path)

        logic1 = ""
        if examples is None or examples == "":
            logic1 = """craft 1 crafting_table summary:
1. log: need 1
2. planks: need 4
3. crafting_table: need 1"""
            examples = """<task>: craft wooden_pickaxe.
<error>: missing material: {"crafting_table": 1}.
<replan>: 
{
    "step 1": {"task": "chop tree", "goal": ["logs", 1]},
    "step 2": {"task": "craft planks", "goal": ["planks", 4]},
    "step 3": {"task": "craft crafting table", "goal": ["crafting_table", 1]
}
"""

        if logic1 == "":
            logic1 = graph_summary

        if graph_summary is None or graph_summary == "":
            prompt = non_reflection_replan_prompt.format(
                task1=task,
                logic1=logic1,
                example=examples.strip(),
                error=error_info,
            )
        else:
            prompt = replan_prompt.format(
                task1=task,
                logic1=logic1,
                logic=graph_summary.strip(),  # type: ignore
                example=examples.strip(),
                error=error_info,
            )

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        },
                    ],
                }
            ],
            model="gpt-4o",
            max_tokens=2000,
        )
        return response.choices[0].message.content

    def reflection(
        self,
        task: str,
        done_path: List[str],
        continue_path: List[str],
        replan_path: List[str],
        image_path: List[str],
    ):
        is_done, is_continue, is_replan = (
            is_path(done_path),
            is_path(continue_path),
            is_path(replan_path),
        )
        content = []
        content.append(
            {
                "type": "text",
                "text": reflection_systerm.format(task=task),
            }
        )
        if is_done or is_continue or is_replan:
            content.append(
                {
                    "type": "text",
                    "text": reflection_examples,
                }
            )
            if is_done:
                done_b, done_a = encode_image(done_path)
                content.append(
                    {
                        "type": "text",
                        "text": "\n<done>:\n",
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{done_b}"},
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{done_a}"},
                    }
                )
            if is_continue:
                continue_b, continue_a = encode_image(continue_path)
                content.append(
                    {
                        "type": "text",
                        "text": "\n<continue>:\n",
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{continue_b}"},
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{continue_a}"},
                    }
                )
            if is_replan:
                replan_b, replan_a = encode_image(replan_path)
                content.append(
                    {
                        "type": "text",
                        "text": "\n<replan>:\n",
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{replan_b}"},
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{replan_a}"},
                    }
                )
        image_b, image_a = encode_image(image_path)
        content.append(
            {
                "type": "text",
                "text": reflection_prompt,
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b}"},
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_a}"},
            }
        )

        result = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model="gpt-4o",
            max_tokens=2000,
        )
        return result.choices[0].message.content
