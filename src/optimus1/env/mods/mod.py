from abc import ABC, abstractmethod

from omegaconf import DictConfig


class Mod(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def reset(self):
        raise NotImplementedError("Not Implement")

    @abstractmethod
    def step(self):
        raise NotImplementedError("Not Implement")
