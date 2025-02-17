import os
import math
import datetime
from typing import NamedTuple
from contextlib import nullcontext

import torch
from omegaconf import DictConfig
from torch.distributed import init_process_group


class DeviceSetup(NamedTuple):
    device: str
    device_type: str

    is_ddp: bool
    is_master_process: bool

    local_rank: int
    world_size: int

    seed_offset: int

    ptdtype: torch.dtype
    autocast_ctx: nullcontext | torch.amp.autocast


def setup_device(cfg: DictConfig) -> DeviceSetup:
    # TODO Is this the best way to check if we're in DDP mode?
    is_ddp = int(os.environ.get("RANK", -1)) != -1

    if is_ddp:
        init_process_group(
            backend=cfg.ddp.backend,
            timeout=datetime.timedelta(seconds=1800),
        )

        ddp_rank = int(os.environ["RANK"])
        seed_offset = ddp_rank
        is_master_process = ddp_rank == 0

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

        world_size = int(os.environ["WORLD_SIZE"])
    else:
        is_ddp = False
        device = cfg.system.device
        local_rank = 0
        seed_offset = 0
        world_size = 1
        is_master_process = True

    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.system.dtype]

    device_type = "cuda" if "cuda" in device else "cpu"

    if "cpu" in device:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.amp.autocast(
            device_type=device_type,
            dtype=ptdtype,
        )

    device_setup = DeviceSetup(
        device=device,
        is_ddp=is_ddp,
        ptdtype=ptdtype,
        local_rank=local_rank,
        world_size=world_size,
        device_type=device_type,
        seed_offset=seed_offset,
        autocast_ctx=autocast_ctx,
        is_master_process=is_master_process,
    )

    n_workers = math.floor(cfg.system.data_workers / device_setup.world_size)
    torch.set_num_threads(n_workers)

    # TODO Move all 3 settings to config
    torch.manual_seed(1337 + device_setup.seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    return device_setup
