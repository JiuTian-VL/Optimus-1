from typing import List

import numpy as np
from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.agent.observations.location_stats import (
    ObservationFromCurrentLocation,
)
from minerl.herobraine.hero.handlers.translation import (
    KeymapTranslationHandler,
)


class CustomObservationFromCurrentLocation(ObservationFromCurrentLocation):
    """
    Includes the current biome, how likely rain and snow are there, as well as the current light level, how bright the
    sky is, and if the player can see the sky.

    Also includes x, y, z, roll, and pitch
    """

    def xml_template(self) -> str:
        return str("""<ObservationFromFullStats/>""")

    def to_string(self) -> str:
        return "location_stats"

    def __init__(self):
        super().__init__()


class _FullStatsObservation(KeymapTranslationHandler):
    def to_hero(self, x) -> int:
        for key in self.hero_keys:
            x = x[key]
        return x

    def __init__(self, key_list: List[str], space=None, default_if_missing=None):
        if space is None:
            if key_list[0] == "achievement":
                space = spaces.Box(low=0, high=1, shape=(), dtype=int)
            else:
                space = spaces.Box(low=0, high=np.inf, shape=(), dtype=int)
        if default_if_missing is None:
            default_if_missing = np.zeros((), dtype=float)

        super().__init__(
            hero_keys=key_list,
            univ_keys=key_list,
            space=space,
            default_if_missing=default_if_missing,
        )

    def xml_template(self) -> str:
        return str("""<ObservationFromFullStats/>""")
