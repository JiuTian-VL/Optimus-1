import logging

import gym
from omegaconf import DictConfig

from .wrapper import CustomEnvWrapper

from .custom_env import *
from .inventory_agent_start import *
from .plain_inventory import *
from .wrapper import *

ALL_BIOMES = [
    "forest",
    "nether",
    "taiga",
    "the_end",
    "none",
    "swamp",
    "ocean",
    "mesa",
    "extreme_hills",
    "plains",
    "savanna",
    "beach",
    "jungle",
    "river",
    "desert",
    "mushroom",
    "icy",
]


def env_make(env_id: str, cfg: DictConfig, logger: logging.Logger) -> CustomEnvWrapper:
    """
    Create and return an environment based on the given `env_id`.

    Parameters:
    - env_id (str): The ID of the environment to create.

    Returns:
    - env: The created environment.

    """
    env = gym.make(env_id)
    env = CustomEnvWrapper(env, cfg, logger)
    return env


def register_custom_env(cfg: DictConfig) -> None:
    """
    Register a custom environment based on the provided configuration.

    Args:
        cfg (DictConfig): The configuration for the custom environment.

    Raises:
        AssertionError: If the specified biome is not found in the list of all biomes.

    Returns:
        None
    """
    biome = cfg["env"]["prefer_biome"]
    assert biome in ALL_BIOMES, f"Biome {biome} not found in {ALL_BIOMES}"

    CustomEnvSpec(
        env_name=cfg["env"]["name"],
        prefer_biome=biome,
        initial_inventory=cfg["env"]["initial_inventory"],
        max_mintues=cfg["env"]["max_minutes"],
    ).register()
