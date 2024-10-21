from typing import Dict

from .monitor import Monitor


class SuccessMonitor(Monitor):
    def __init__(self) -> None:
        super().__init__()
        self._sumary = dict()

    def to_string(self):
        return "SuccessMonitor"

    def update(self, task: str, success: bool) -> None:
        if task not in self._sumary:
            self._sumary[task] = 0
        self._sumary[task] += int(success)

    def get_metric(self) -> Dict[str, int]:
        return self._sumary

    def get_task_success_times(self, task: str) -> int:
        for k, v in self._sumary.items():
            if task.lower().replace(" ", "_") in k.replace(" ", "_"):
                return v
        return 0

    def __iadd__(self, other: "SuccessMonitor"):
        for k, v in other._sumary.items():
            if k not in self._sumary:
                self._sumary[k] = 0
            self._sumary[k] += v
        return self
