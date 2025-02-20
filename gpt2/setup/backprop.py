from typing import NamedTuple

import torch
from omegaconf import OmegaConf, DictConfig

from gpt2.setup.hardware import DeviceSetup
from gpt2.lr_scheduler import LearningRateScheduler, get_lr_scheduler


class BackpropSetup(NamedTuple):
    lr_scheduler: LearningRateScheduler
    grad_scaler: torch.amp.GradScaler
    optimizer: torch.optim.Optimizer


def setup_backprop(
    cfg: DictConfig,
    model: torch.nn.Module,
    device_setup: DeviceSetup,
) -> BackpropSetup:
    lr_config = OmegaConf.to_container(cfg=cfg.lr)
    lr_scheduler = get_lr_scheduler(lr_config=lr_config)

    enable_grad_scaler = cfg.system.dtype == "float16"
    grad_scaler = torch.amp.GradScaler(
        device=device_setup.device_type,
        enabled=enable_grad_scaler,
    )

    optimizer = configure_optimizers(
        model=model,
        learning_rate=lr_scheduler.get_lr(0),
        weight_decay=cfg.optimizer.weight_decay,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
    )

    backprop_setup = BackpropSetup(
        lr_scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
        optimizer=optimizer,
    )
    return backprop_setup


def configure_optimizers(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
) -> torch.optim.Optimizer:
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}

    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(
        params=optim_groups,
        lr=learning_rate,
        betas=betas,
        fused=True,
    )
    print("Using fused AdamW")

    return optimizer
