import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from gpt2 import train as gpt2_train

load_dotenv()


@hydra.main(config_path="configs", config_name="gpt2_pretraining", version_base=None)
def main(cfg: DictConfig):
    if cfg.command == "init":
        gpt2_train.training_from_scratch(cfg=cfg)
    elif cfg.command == "tune":
        gpt2_train.model_tuning(tune_cfg=cfg)
    elif cfg.command == "resume":
        gpt2_train.resume_training(resume_cfg=cfg)
    else:
        raise ValueError("Incorrect command/config")


def load_config(
    config_name: str = "gpt2_pretraining",
    overrides: list[str] = None,
) -> DictConfig:
    """
    Use overrides like bash cli arguments, i.e.:

    overrides = ["dataset_config=eee", "other_param=downsample"]
    """
    with hydra.initialize(version_base=None, config_path="configs", job_name="repl"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

    return cfg


if __name__ == "__main__":
    main()
