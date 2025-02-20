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


def wandb_resume(
    run_id: str,
    cfg: DictConfig,
):
    """
    Resuming wandb runs is problematic, because if there were logs sent
    after the checkpoint you're resuming from, the new run will not override them.
    It will also annoy you with warning until it gets past the last step logged into wandb.
    """
    if not cfg.logging.wandb_log:
        return

    wandb.init(
        entity=cfg.logging.wandb_entity,
        project=cfg.logging.wandb_project,
        id=run_id,
        resume="must",
        dir="tmp",
    )
