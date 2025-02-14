import os
import datetime
from typing import NamedTuple

import torch
from omegaconf import DictConfig
from torch.distributed import init_process_group


class DeviceSetup(NamedTuple):
    device: str
    is_ddp: bool
    is_master_process: bool

    local_rank: int
    world_size: int

    seed_offset: int


def setup_device(cfg: DictConfig) -> DeviceSetup:
    # TODO Is this the best way to check if we're in DDP mode?
    is_ddp = int(os.environ.get("RANK", -1)) != -1

    if is_ddp:
        init_process_group(
            backend=cfg.ddp.backend,
            timeout=datetime.timedelta(seconds=1800),
        )

        ddp_rank = int(os.environ["RANK"])
        is_master_process = ddp_rank == 0

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

        world_size = int(os.environ["WORLD_SIZE"])

        device_setup = DeviceSetup(
            device=device,
            is_ddp=True,
            is_master_process=is_master_process,
            seed_offset=ddp_rank,
            local_rank=local_rank,
            world_size=world_size,
        )
        return device_setup
    else:
        device_setup = DeviceSetup(
            device=cfg.system.device,
            is_ddp=False,
            is_master_process=True,
            seed_offset=0,
            world_size=1,
            local_rank=0,
        )
        return device_setup
