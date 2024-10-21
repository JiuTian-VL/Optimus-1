from typing import List, Optional, Sequence

from minerl.env import _fake, _singleagent
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.mc import ALL_ITEMS

from .chat_action import ChatAction
from .inventory_agent_start import CustomInventoryAgentStart
from .obversation_current_location import CustomObservationFromCurrentLocation
from .plain_inventory import PlainInventoryObservation
from .wrapper import BasaltTimeoutWrapper


def _custom_gym_entrypoint(
    env_spec: "CustomBaseEnvSpec",
    fake: bool = False,
):
    """Used as entrypoint for `gym.make`."""
    if fake:
        env = _fake._FakeSingleAgentEnv(env_spec=env_spec)
    else:
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)

    env = BasaltTimeoutWrapper(env)
    return env


CUSTOM_GYM_ENTRY_POINT = "src.optimus1.env.custom_env:_custom_gym_entrypoint"


class CustomBaseEnvSpec(HumanSurvival):
    LOW_RES_SIZE = 64
    HIGH_RES_SIZE = 1024

    def __init__(
        self,
        name,
        demo_server_experiment_name,
        max_episode_steps=2400,
        inventory: Sequence[dict] = (),
        preferred_spawn_biome: str = "plains",
    ) -> None:
        self.inventory = inventory  # Used by minerl.util.docs to construct Sphinx docs.
        self.preferred_spawn_biome = preferred_spawn_biome
        self.demo_server_experiment_name = demo_server_experiment_name
        super().__init__(
            name=name,
            max_episode_steps=max_episode_steps,
            # Hardcoded variables to match the pretrained models
            fov_range=[70, 70],
            resolution=(640, 360),
            gamma_range=[2, 2],
            guiscale_range=[1, 1],
            cursor_size_range=[16.0, 16.0],
        )

    def is_from_folder(self, folder: str) -> bool:
        # Implements abstractmethod.
        return folder == self.demo_server_experiment_name

    def _entry_point(self, fake: bool) -> str:
        # Don't need to inspect `fake` argument here because it is also passed to the
        # entrypoint function.
        return CUSTOM_GYM_ENTRY_POINT

    def create_observables(self):
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(ALL_ITEMS),
            PlainInventoryObservation(),
        ] + [
            handlers.EquippedItemObservation(
                items=ALL_ITEMS,
                mainhand=True,
                offhand=True,
                armor=True,
                _default="air",
                _other="air",
            ),
            handlers.ObservationFromLifeStats(),
            CustomObservationFromCurrentLocation(),
            handlers.IsGuiOpen(),
        ]

    def create_agent_start(self) -> List[handlers.Handler]:
        return super().create_agent_start() + [
            CustomInventoryAgentStart(self.inventory),  # type: ignore
            handlers.PreferredSpawnBiome(self.preferred_spawn_biome),
            handlers.DoneOnDeath(),
        ]

    def create_actionables(self):
        """
        Simple envs have some basic keyboard control functionality, but
        not all.
        """
        return super().create_actionables() + [ChatAction()]

    def create_agent_handlers(self) -> List[handlers.Handler]:
        return []

    def create_server_world_generators(self) -> List[handlers.Handler]:
        # TODO the original biome forced is not implemented yet. Use this for now.
        return [handlers.DefaultWorldGenerator(force_reset=True)]

    def create_server_quit_producers(self) -> List[handlers.Handler]:
        return [
            # handlers.ServerQuitFromTimeUp((self.max_episode_steps * mc.MS_PER_STEP)),  # type: ignore
            handlers.ServerQuitWhenAnyAgentFinishes(),
        ]

    def create_server_decorators(self) -> List[handlers.Handler]:
        return []

    def create_server_initial_conditions(self) -> List[handlers.Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=False),
            handlers.SpawningInitialCondition(allow_spawning=True),
        ]

    def get_blacklist_reason(self, npz_data: dict) -> Optional[str]:
        """
        Some saved demonstrations are bogus -- they only contain lobby frames.

        We can automatically skip these by checking for whether any snowballs
        were thrown.
        """
        # TODO(shwang): Waterfall demos should also check for water_bucket use.
        #               AnimalPen demos should also check for fencepost or fence gate use.
        # TODO Clean up snowball stuff (not used anymore)
        equip = npz_data.get("observation$equipped_items$mainhand$type")
        use = npz_data.get("action$use")
        if equip is None:
            return f"Missing equip observation. Available keys: {list(npz_data.keys())}"
        if use is None:
            return f"Missing use action. Available keys: {list(npz_data.keys())}"

        assert len(equip) == len(use) + 1, (len(equip), len(use))

        for i in range(len(use)):
            if use[i] == 1 and equip[i] == "snowball":
                return None
        return "BasaltEnv never threw a snowball"

    def create_mission_handlers(self):
        # Implements abstractmethod
        return ()

    def create_monitors(self):
        # Implements abstractmethod
        return ()

    def create_rewardables(self):
        # Implements abstractmethod
        return ()

    def determine_success_from_rewards(self, rewards: list) -> bool:
        """Implements abstractmethod.

        Basalt environment have no rewards, so this is always False."""
        return False

    def get_docstring(self):
        return self.__class__.__doc__


MINUTE = 20 * 60


class CustomEnvSpec(CustomBaseEnvSpec):
    """
    After spawning in a plains biome, explore and find a cave. When inside a cave, end
    the episode by setting the "ESC" action to 1.
    """

    def __init__(
        self,
        env_name: str,
        max_mintues: float = 2.0,
        prefer_biome: str = "forest",  # https://minecraft.fandom.com/wiki/Biome#Biome_IDs
        initial_inventory=[],
    ) -> None:
        super().__init__(
            name=env_name,
            demo_server_experiment_name=env_name,
            max_episode_steps=600 * MINUTE
            if max_mintues == -1
            else max_mintues * MINUTE,  # type: ignore
            preferred_spawn_biome=prefer_biome,
            inventory=initial_inventory,
        )
