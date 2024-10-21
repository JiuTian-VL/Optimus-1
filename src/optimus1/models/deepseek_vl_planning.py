from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM

from .deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from .deepseek_vl.utils.io import load_pil_images

from .base_model import BasePlanningModel, BaseReflectionModel

plan_prompt = """ For a given game screen and task, you need to make a plan with the help of <visual info> and <craft graph>.
<visual info>: Consists of the following aspects: health bar, food bar, hotbar, environment. Based on the current visual information, you need to consider whether prequel steps needed to ensure that agent can complete the task.
<craft graph>: a top-down list of all the tools and materials needed to complete the task. 
I will give you an example of planning under specific visual conditions as follow:
[Example]
{example}

[Your turn]
Here is a game screen and task, you MUST output in example format. Remember <task planning> MUST output in example format.
<image_placeholder>
<task>: {task}
<visual info>: {visual}
<craft graph>: {graph}
"""
retrieve_prompt = """
 For a given game screen and task, you need to complete <goal inference> and <visual inference>.
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

[Your turn]
Here is a game screen and task, you MUST output in example format.
<image_placeholder>
<task>: {task}.
"""

no_reflection_plan_prompt = """ For a given game screen and task, you need to make a plan.
I will give you an example of planning under specific visual conditions as follow:
[Example]
{example}

[Your turn]
Here is a game screen and task, you MUST output in example format. Remember <task planning> MUST output in example format.
<image_placeholder>
<task>: {task}
"""

### replan prompt ###
replan_prompt = """
 
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

[Your turn]
Here is a game screen, you MUST output in example format.Remember <replan> MUST output in example format.
<logic inference>:
{logic}
<image_placeholder>
<error>: {error}
"""

non_reflection_replan_prompt = """
 
Agent is executing the <task>: {task1}, and agent meets <error>: {error}.
Based on <error> information and <knowledge>, replanning to allow the agent to successfully complete the <task>.

<knowledge>: The basic tools the agent can acquire come in multiple tiers based on your materials, and they include the pickaxe, the axe, and the shovel, for mining, respectively, stone-type, wood-type, and dirt & sand-type blocks. The six tiers are wood, stone, iron, gold, diamond, and netherite. For pickaxes in particular, wooden pickaxes can collect cobblestones and coal ore, stone pickaxe can collect iron ore, and more advanced ores require at least an iron pickaxe.

You MUST focus on how to solve the <error>.
I will give you an example as follow:
[Example]
<logic inference>:
{logic1}
{example}

[Your turn]
Here is a game screen, you MUST output in example format.Remember <replan> MUST output in example format.
<image_placeholder>
<error>: {error}
"""

### reflection prompt ###
reflection_systerm = """
 Agent is executing the task: {task}.
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
<image_placeholder> <image_placeholder>
Enviroment: <environment>
Situation: <situation>
(if situation is replan) Predicament: <predicament>
"""


def is_path(path):
    if len(path) == 2:
        return True
    else:
        return False


class PlanningModel(BasePlanningModel, BaseReflectionModel):
    IMAGE_TAG: str = "<image_placeholder>"

    def __init__(self, model_path: str, system_prompt: Optional[str] = None) -> None:
        # specify the path to the model
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer  # type: ignore

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()  # type: ignore

        self.set_system_prompt()

    def set_system_prompt(
        self,
        sys_prompt: str | None = " ",
    ):
        if sys_prompt:
            self.vl_chat_processor.system_prompt = sys_prompt  # type: ignore

    def retrieve(
        self,
        task: str,
        image_path: str,
    ):
        return self._inference(retrieve_prompt.format(task=task), image_path)

    def replan(
        self,
        task: str,
        image_path: str,
        error_info: str | None = None,
        examples: str | None = None,
        graph_summary: str | None = None,
    ):
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

        return self._inference(prompt, image_path)

    def planning(
        self,
        task: str,
        images: str | List[str],
        example: str | None = None,
        visual_info: str | None = None,
        graph: str | None = None,
    ):
        if visual_info is None and graph is None:
            prompt = no_reflection_plan_prompt.format(task=task, example=example)
        else:
            prompt = plan_prompt.format(
                task=task,
                example=example,
                visual=visual_info,
                graph=graph,
            )
        print(f"====\n{prompt}\n====")
        return self._inference(prompt, images)

    def _inference(self, instruction: str, images: str | List[str]) -> str:
        if isinstance(images, str):
            images = [images]

        conversation = [
            {
                "role": "User",
                "content": instruction,
                "images": images,
            },
            {"role": "Assistant", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversations=conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)  # type: ignore

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=2048,
            do_sample=False,
            use_cache=True,
        )

        answer = self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        return answer

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
        prompt = reflection_systerm.format(task=task)
        imgs = []
        if is_done or is_continue or is_replan:
            prompt += "\n" + reflection_examples
            if is_done:
                prompt += f"\n<done>:\n{self.IMAGE_TAG} {self.IMAGE_TAG}"
                imgs += done_path

            if is_continue:
                prompt += f"\n<continue>:\n{self.IMAGE_TAG} {self.IMAGE_TAG}"
                imgs += continue_path

            if is_replan:
                prompt += f"\n<replan>:\n{self.IMAGE_TAG} {self.IMAGE_TAG}"
                imgs += replan_path

        imgs += image_path
        prompt += reflection_prompt

        return self._inference(prompt, imgs)
