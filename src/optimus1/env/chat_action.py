import minerl.herobraine.hero.spaces as spaces
from minerl.herobraine.hero.handlers.agent.action import Action


class ChatAction(Action):
    """
    Handler which lets agents send Minecraft chat messages

    Note: this may currently be limited to the
    first agent sending messages (check Malmo for this)

    This can be used to execute MINECRAFT COMMANDS !!!

    Example usage:

    .. code-block:: python

        ChatAction()

    To summon a creeper, use this action dictionary:

    .. code-block:: json

        {"chat": "/summon creeper"}

    """

    def to_string(self):
        return "chat"

    def xml_template(self) -> str:
        return str("<ChatCommands> </ChatCommands>")

    def __init__(self):
        self._command = "chat"
        super().__init__(self.command, spaces.Text([1]))

    def from_universal(self, x):
        return []
