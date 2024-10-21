from typing import Any, Dict, List

from .monitor import Monitor
from .step_monitor import StepMonitor
from .success_monitor import SuccessMonitor


class Monitors:
    def __init__(self, monitors: List[Monitor]) -> None:
        self._monitors = monitors

    def update(self, task: str | None, success: bool = False, step: int = 1) -> None:
        if task is None:
            return
        for monitor in self._monitors:
            if isinstance(monitor, SuccessMonitor):
                monitor.update(task, success)
            if isinstance(monitor, StepMonitor):
                monitor.update(task, step)

    def get_success_times(self, task: str) -> int:
        for monitor in self._monitors:
            if isinstance(monitor, SuccessMonitor):
                return monitor.get_task_success_times(task)
        return -1

    def get_steps(self, task: str) -> int:
        for monitor in self._monitors:
            if isinstance(monitor, StepMonitor):
                return monitor.get_steps(task)
        return -1

    def all_steps(self) -> int:
        for monitor in self._monitors:
            if isinstance(monitor, StepMonitor):
                return monitor.all_step()
        return 0

    def get_metric(self):
        monitor = {monitor.__class__.__name__: monitor.get_metric() for monitor in self._monitors}
        # merge all monitor
        metric = dict()
        for k, v in monitor.items():
            for task_k, task_v in v.items():
                if task_k not in metric:
                    metric[task_k] = dict()
                metric[task_k].update({k: task_v})
        return metric

    def __iadd__(self, other_monitors: "Monitors"):
        for monitor in self._monitors:
            for other_monitor in other_monitors._monitors:
                if monitor.to_string() == other_monitor.to_string():
                    monitor += other_monitor  # type: ignore
        return self

    def reset(self, planning: List[Dict[str, Any]]):
        for prog, task in enumerate(planning):
            self.update(f"{task['task']}_{prog}", success=False, step=0)


if __name__ == "__main__":
    monitors = Monitors([SuccessMonitor(), StepMonitor()])
    monitors.update("a", success=False)
    monitors.update("a", success=True)
    monitors.update("b", False)
    monitors.update("c", False)
    monitors.update("d", False)
    monitors.update("e", False)
    monitors.update("b", True)

    other_monitors = Monitors([SuccessMonitor(), StepMonitor()])
    other_monitors.update("test", False)
    other_monitors.update("a", True)

    monitors += other_monitors
    print(monitors.get_metric())
