from abc import ABC, abstractmethod


class Monitor(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def to_string(self) -> str:
        return "Monitor"

    @abstractmethod
    def update(self):
        raise NotImplementedError("Not Implement")

    @abstractmethod
    def get_metric(self):
        raise NotImplementedError("Not Implement")

    @abstractmethod
    def __iadd__(self, other: "Monitor"):
        raise NotImplementedError
