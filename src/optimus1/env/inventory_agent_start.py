from typing import Dict, List, Union
from minerl.herobraine.hero.handlers import InventoryAgentStart


class CustomInventoryAgentStart(InventoryAgentStart):
    """An inventory agentstart specification which
    just fills the inventory of the agent sequentially.
    """

    def __init__(self, inventory: List[Dict[str, Union[str, int]]]):
        """Creates a custom inventory agent start.

        For example:

            sias =  CustomInventoryAgentStart(
                [
                    {'type':'dirt', 'quantity':10, 'slot': 0},
                    {'type':'planks', 'quantity':5, 'slot':1},
                    {'type':'log', 'quantity':1, 'slot': 2},
                    {'type':'iron_ore', 'quantity':4, 'slot': 3}
                ]
            )
        """
        try:
            iag = {item["slot"]: item for item in inventory}
        except KeyError:
            raise ValueError("Each inventory item must have a 'slot' key.")
        super().__init__(iag)
