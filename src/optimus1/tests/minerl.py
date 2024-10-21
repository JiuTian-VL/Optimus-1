import hydra
from omegaconf import DictConfig, OmegaConf

from optimus1.env import env_make, register_custom_env


from optimus1.util import (
    ServerAPI,
    get_evaluate_task,
    get_logger,
)


@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(cfg: DictConfig):
    register_custom_env(cfg)

    logger = get_logger(__name__)
    logger.info(OmegaConf.to_yaml(cfg))

    env = env_make(cfg["env"]["name"], cfg, logger)

    evaluate_tasks = get_evaluate_task(cfg)
    logger.info(f"Evaluate Tasks: {evaluate_tasks}")

    t = ServerAPI.reset(cfg["server"])
    logger.info("[red]env & server reset...[/red] ")
    env.reset()
    t.join()
    logger.info("[red]env reset done...[/red] ")

    env.close()

    logger.info("[green] MineRL test Pass.")


if __name__ == "__main__":
    main()
