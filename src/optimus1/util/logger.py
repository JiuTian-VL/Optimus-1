import logging
import os
from typing import Dict

import hydra.core.hydra_config
from rich.logging import RichHandler


def get_logger(name) -> logging.Logger:
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    job_name = hydra.core.hydra_config.HydraConfig.get().job.name

    file_handler = logging.FileHandler(os.path.join(hydra_path, f"{job_name}.log"))
    rich_handler = RichHandler(markup=True)

    logging.basicConfig(format="[%(asctime)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.addHandler(rich_handler)
    log.addHandler(file_handler)
    log.propagate = False

    # log.info("Successfully create rich logger")
    return log


def pretty_result(task: str, metrics: Dict[str, Dict[str, int]], times: int = 10, steps: int = 0):
    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"Evaluate Task: {task} in {times} times, sum steps: {steps}")

    table.add_column("Sub-Task", justify="center", style="cyan", no_wrap=True)
    table.add_column("Success Times", justify="center", style="green")
    table.add_column("Success Rate", style="purple", justify="center")
    table.add_column("Avg Steps", style="blue", justify="center")
    table.add_column("Avg Time(s)", style="red", justify="center")

    for task, monitor in metrics.items():
        table.add_row(
            task,
            str(monitor["SuccessMonitor"]),
            f"{monitor['SuccessMonitor']/times:.2%}",
            f"{monitor['StepMonitor']/monitor['SuccessMonitor']:.2f}" if monitor["SuccessMonitor"] > 0 else "NaN",
            f"{monitor['StepMonitor']/monitor['SuccessMonitor']/20:.2f}" if monitor["SuccessMonitor"] > 0 else "NaN",
        )

    console = Console()
    console.print(table)
