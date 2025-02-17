import wandb
from omegaconf import OmegaConf, DictConfig


def wandb_init(
    run_name: str,
    cfg: DictConfig,
):
    if not cfg.logging.wandb_log:
        return

    run_config = OmegaConf.to_container(cfg=cfg)

    # We're using groups, so you'll have to select "goup by Group" in wandb dashboard
    wandb.init(
        entity=cfg.logging.wandb_entity,
        project=cfg.logging.wandb_project,
        name="training-loop",
        group=run_name,
        config=run_config,
        dir="tmp",
    )


def wandb_resume():
    ...
