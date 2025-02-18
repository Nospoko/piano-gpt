import hydra
from omegaconf import DictConfig

from gpt2 import train as gpt2_train
from gpt2.setup import hardware as hardware_setup


@hydra.main(config_path="configs", config_name="gpt2_pretraining", version_base=None)
def main(cfg: DictConfig):
    # 1. Stuf common for all commands
    device_setup = hardware_setup.setup_device(cfg)
    print("Device setup:", device_setup)

    # FIXME: Find a config design where this is not necessary
    if device_setup.is_ddp:
        # World_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert cfg.optimizer.gradient_accumulation_steps % device_setup.world_size == 0

        cfg.optimizer.gradient_accumulation_steps //= device_setup.world_size

    if cfg.command == "init":
        gpt2_train.training_from_scratch(
            cfg=cfg,
            device_setup=device_setup,
        )
    elif cfg.command == "tune":
        # train.tune(cfg=cfg, device_setup=device_setup)
        ...
    elif cfg.command == "resume":
        gpt2_train.resume_training(cfg=cfg, device_setup=device_setup)
    else:
        raise ValueError("Incorrect command/config")


if __name__ == "__main__":
    main()
