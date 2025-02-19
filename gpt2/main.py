import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from gpt2 import train as gpt2_train
from gpt2.setup import hardware as hardware_setup

load_dotenv()


@hydra.main(config_path="configs", config_name="gpt2_pretraining", version_base=None)
def main(cfg: DictConfig):
    # 1. Stuf common for all commands
    device_setup = hardware_setup.setup_device(cfg)
    print("Device setup:", device_setup)

    if cfg.command == "init":
        gpt2_train.training_from_scratch(
            cfg=cfg,
            device_setup=device_setup,
        )
    elif cfg.command == "tune":
        # train.tune(cfg=cfg, device_setup=device_setup)
        ...
    elif cfg.command == "resume":
        gpt2_train.resume_training(
            resume_cfg=cfg,
            device_setup=device_setup,
        )
    else:
        raise ValueError("Incorrect command/config")


if __name__ == "__main__":
    main()
