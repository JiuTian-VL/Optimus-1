import logging
from typing import List

import torch

from ..models.deepseek_vl_planning import PlanningModel as DeepSeekPlanningModel
from ..models.gpt4_planning import PlanningModel as GPT4PlanningModel
from ..models.steve_action_model import ActionModel as SteveActionModel

logger = logging.getLogger(__name__)


PLAN_MODEL_PATH = {
    "deepseek-vl": "deepseek-vl-7b-chat",
}


class Agent:
    gpt_v: bool = False

    def __init__(
        self,
        plan_with_gpt: bool = True,
        plan_model: str | None = None,
        in_model: str = "checkpoints/vpt/2x.model",
        in_weights: str = "checkpoints/steve1/steve1.weights",
        prior_weights: str = "checkpoints/steve1/steve1_prior.pt",
    ) -> None:
        self.plan_with_gpt = plan_with_gpt
        if plan_with_gpt:
            self.gpt_v = True
            self.plan_model = GPT4PlanningModel()
            logger.info("gpt4o as planning model")
            self.reflection_model = self.plan_model
        else:
            logger.info("Using DeepSeek-VL for planning.")
            if plan_model == "deepseek-vl":
                self.plan_model = DeepSeekPlanningModel(PLAN_MODEL_PATH[plan_model])
            else:
                raise ValueError(f"Unknown plan model: {plan_model}")
            self.reflection_model = self.plan_model

        logger.info("Using Steve-1 for action.")
        self.action_model = SteveActionModel(
            in_model=in_model,
            in_weights=in_weights,
            prior_weights=prior_weights,
        )

        logger.info("Agent initialized.")

    def retrieve(
        self,
        task: str,
        rgb_obs: str,
    ):
        """rgb_obs: [img1, img2, ...]"""
        assert self.plan_model is not None, "The plan model is not initialized."

        plans_retrieve = self.plan_model.retrieve(task, rgb_obs)
        return plans_retrieve

    def plan(
        self,
        task: str,
        rgb_obs: str,
        example: str | None = None,
        visual_info: str | None = None,
        graph: str | None = None,
    ):
        """rgb_obs: [img1, img2, ...]"""
        assert self.plan_model is not None, "The plan model is not initialized."
        if self.plan_with_gpt:
            if self.gpt_v:
                plan = self.plan_model.planning(
                    task, rgb_obs, example, visual_info, graph
                )
            else:
                plan = self.plan_model.planning(task, example)
        else:
            plan = self.plan_model.planning(task, rgb_obs, example, visual_info, graph)
        return plan

    def action(self, instruction: str, rgb_obs: List[str]):
        return self.action_model.action(instruction, rgb_obs)

    def replan(
        self,
        task: str,
        rgb_obs: str,
        error_info: str | None = None,
        examples: str | None = None,
        graph_summary: str | None = None,
    ):
        assert self.plan_model is not None, "The plan model is not initialized."
        if self.plan_with_gpt:
            if self.gpt_v:
                replan = self.plan_model.replan(
                    task, rgb_obs, error_info, examples, graph_summary
                )  # type: ignore
            else:
                replan = self.plan_model.replan(task, error_info)
        else:
            replan = self.plan_model.replan(
                task, rgb_obs, error_info, examples, graph_summary
            )
        return replan

    def reflection(
        self,
        task: str,
        old_obs: str,
        current_obs: str,
        done_img_path: List[str] | None = None,
        cont_img_path: List[str] | None = None,
        replan_img_path: List[str] | None = None,
    ):
        assert self.reflection_model is not None, "The plan model is not initialized."
        if done_img_path is None:
            done_img_path = []
        if cont_img_path is None:
            cont_img_path = []
        if replan_img_path is None:
            replan_img_path = []
        reflection = self.reflection_model.reflection(
            task, done_img_path, cont_img_path, replan_img_path, [old_obs, current_obs]
        )
        return reflection


class AgentFactory:
    _agent: Agent | None = None

    @staticmethod
    def get_agent(
        plan_with_gpt: bool = False,
        plan_model: str | None = None,
        in_model: str = "checkpoints/vpt/2x.model",
        in_weights: str = "checkpoints/steve1/steve1.weights",
        prior_weights: str = "checkpoints/steve1/steve1_prior.pt",
    ):
        if AgentFactory._agent is None:
            AgentFactory._agent = Agent(
                plan_with_gpt=plan_with_gpt,
                plan_model=plan_model,
                in_model=in_model,
                in_weights=in_weights,
                prior_weights=prior_weights,
            )
            AgentFactory._args = (
                plan_with_gpt,
                plan_model,
                in_model,
                in_weights,
                prior_weights,
            )
        return AgentFactory._agent

    @staticmethod
    def reset():
        if AgentFactory._agent is not None:
            del AgentFactory._agent
            torch.cuda.empty_cache()
            AgentFactory._agent = None
        return AgentFactory.get_agent(
            plan_with_gpt=AgentFactory._args[0],
            plan_model=AgentFactory._args[1],
            in_model=AgentFactory._args[2],
            in_weights=AgentFactory._args[3],
            prior_weights=AgentFactory._args[4],
        )
