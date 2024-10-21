from typing import Any, List

import torch

from .steve1.config import PRIOR_INFO
from .steve1.data.text_alignment.vae import load_vae_model
from .steve1.MineRLConditionalAgent import MineRLConditionalAgent
from .steve1.utils.embed_utils import get_prior_embed
from .steve1.utils.mineclip_agent_env_utils import (
    load_mineclip_wconfig,
    load_model_parameters,
)

from .base_model import BaseActionModel
from .utils import image2MineRLArray

FPS = 20


class ActionModel(BaseActionModel):
    def __init__(
        self,
        in_model: str = "checkpoints/vpt/2x.model",
        in_weights: str = "checkpoints/steve1/steve1.weights",
        prior_weights: str = "checkpoints/steve1/steve1_prior.pt",
        text_cond_scale: float = 6.0,
        visual_cond_scale: float = 7.0,
    ) -> None:
        self.in_model = in_model
        self.in_weights = in_weights
        self.prior_weights = prior_weights
        self.text_cond_scale = text_cond_scale
        self.visual_cond_scale = visual_cond_scale

        self.mineclip = load_mineclip_wconfig()
        self.prior = load_vae_model(PRIOR_INFO)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        self.agent_policy_kwargs, self.agent_pi_head_kwargs = load_model_parameters(
            in_model
        )

        self.agent = MineRLConditionalAgent(
            device=self.device,
            policy_kwargs=self.agent_policy_kwargs,
            pi_head_kwargs=self.agent_pi_head_kwargs,
        )
        self.agent.load_weights(in_weights)
        self.agent.reset(text_cond_scale)

    def _get_prompt_embed(self, prompt: str) -> Any:
        prompt_embed = get_prior_embed(prompt, self.mineclip, self.prior, self.device)
        self.prompt_embed = {prompt: prompt_embed}
        return self.prompt_embed

    def action(self, prompt: str, observation: List[str]):
        obs = image2MineRLArray(observation[-1])

        minerl_obs = {"pov": obs}
        prompt_embed = self._get_prompt_embed(prompt)

        for _, embed in prompt_embed.items():
            minerl_action = self.agent.get_action(minerl_obs, embed)

        for k, v in minerl_action.items():
            minerl_action[k] = v.tolist()
        minerl_action["ESC"] = [0]
        return minerl_action
