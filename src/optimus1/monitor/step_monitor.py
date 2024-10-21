from typing import Dict

from .monitor import Monitor


class StepMonitor(Monitor):
    def __init__(self) -> None:
        super().__init__()
        self._sumary = dict()

    def to_string(self):
        return "StepMonitor"

    def update(self, task: str, step: int) -> None:
        if task not in self._sumary:
            self._sumary[task] = 0
        self._sumary[task] += step

    def get_metric(self) -> Dict[str, int]:
        return self._sumary

    def all_step(self) -> int:
        res = 0
        for v in self._sumary.values():
            res += v
        return res

    def __iadd__(self, other: "StepMonitor"):
        for k, v in other._sumary.items():
            if k not in self._sumary:
                self._sumary[k] = 0
            self._sumary[k] += v
        return self

    def get_steps(self, task: str) -> int:
        return self._sumary.get(task, 0)
