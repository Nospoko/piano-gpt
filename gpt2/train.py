import os
import time
import datetime

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig
from torch.distributed import destroy_process_group
from midi_tokenizers import ExponentialTimeTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

from data.musicality import MusicManager
from gpt2.model import GPT, estimate_mfu
from gpt2.setup.hardware import DeviceSetup
from gpt2.setup import datasets as data_setup
from gpt2.setup.datasets import DatasetsSetup
from gpt2.lr_scheduler import get_lr_scheduler
from gpt2.setup import logging as logging_setup
from gpt2.setup.backprop import BackpropSetup, setup_backprop


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


def resume_training(
    cfg: DictConfig,
    device_setup: DeviceSetup,
):
    checkpoint = torch.load(
        cfg.checkpoint_path,
        weights_only=False,
    )

    run_cfg: DictConfig = checkpoint["run_config"]

    # Load tokenizer
    tokenizer_desc = checkpoint["tokenizer_desc"]
    tokenizer = ExponentialTimeTokenizer.from_dict(tokenizer_desc)

    # TODO vocab and padding info should be checkpointed
    model_cfg = checkpoint["model_cfg"]
    model = GPT(
        config=model_cfg,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    state_dict = checkpoint["model"]

    # free up memory
    # checkpoint = None

    # This may be resolved by using the not-compiled model object:
    # https://discuss.pytorch.org/t/how-to-save-load-a-model-with-torch-compile/179739
    unwanted_prefix = "_orig_mod."
    for param_name, _ in list(state_dict.items()):
        if param_name.startswith(unwanted_prefix):
            fixed_name = param_name[len(unwanted_prefix) :]
            state_dict[fixed_name] = state_dict.pop(param_name)

    model.load_state_dict(state_dict)

    # TODO: I'm not convinved it's a good approach to have
    # optimizer setup as a model method
    optimizer = model.configure_optimizers(
        weight_decay=run_cfg.optimizer.weight_decay,
        learning_rate=run_cfg.lr.learning_rate,
        betas=(run_cfg.optimizer.beta1, run_cfg.optimizer.beta2),
        device_type=device_setup.device_type,
    )
    optimizer.load_state_dict(checkpoint["optimizer"])

    music_manager = MusicManager()
    if cfg.model_task == "next_token_prediction":
        datasets_setup = data_setup.next_token_prediction_setup(
            cfg=run_cfg,
            device_setup=device_setup,
            music_manager=music_manager,
        )
    elif cfg.model_task == "piano_task":
        datasets_setup = data_setup.piano_task_setup(
            cfg=run_cfg,
            device_setup=device_setup,
            music_manager=music_manager,
        )

    # learning rate decay scheduler
    lr_config = OmegaConf.to_container(cfg=cfg.lr)
    lr_scheduler = get_lr_scheduler(lr_config=lr_config)
    checkpoint_addons = {
        "tokenizer_desc": datasets_setup.tokenizer.to_dict(),
    }

    run_name = checkpoint["run_name"]

    enable_grad_scaler = cfg.system.dtype == "float16"
    grad_scaler = torch.amp.GradScaler(
        device=device_setup.device_type,
        enabled=enable_grad_scaler,
    )

    training_loop(
        cfg=cfg,
        model=model,
        run_name=run_name,
        optimizer=optimizer,
        grad_scaler=grad_scaler,
        device_setup=device_setup,
        lr_scheduler=lr_scheduler,
        checkpoint_addons=checkpoint_addons,
        val_loaders=datasets_setup.val_loaders,
        train_loader=datasets_setup.train_loader,
    )


def training_from_scratch(
    cfg: DictConfig,
    device_setup: DeviceSetup,
):
    music_manager = MusicManager()

    cfg.out_dir = to_absolute_path(cfg.out_dir_relative)
    if device_setup.is_master_process:
        os.makedirs(cfg.out_dir, exist_ok=True)

    if cfg.model_task == "next_token_prediction":
        datasets_setup = data_setup.next_token_prediction_setup(
            cfg=cfg,
            device_setup=device_setup,
            music_manager=music_manager,
        )
    elif cfg.model_task == "piano_task":
        datasets_setup = data_setup.piano_task_setup(
            cfg=cfg,
            device_setup=device_setup,
            music_manager=music_manager,
        )

    # init a new model from scratch
    print("Initializing a new model from scratch")

    # TODO I don't like two sources of model config: vocab size
    # and pad id should be merged with the rest of options somehow
    model = GPT(
        config=cfg.model,
        vocab_size=datasets_setup.tokenizer.vocab_size,
        pad_token_id=datasets_setup.tokenizer.pad_token_id,
    )
    model.to(device_setup.device)

    backprop_setup = setup_backprop(
        cfg=cfg,
        model=model,
        device_setup=device_setup,
    )

    if cfg.system.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    milion_params = model.get_num_params() / 1e6
    run_name = f"{milion_params:.0f}M-" f"{cfg.run_name_suffix}"

    # wrap model into DDP container
    if device_setup.is_ddp:
        model = DDP(model, device_ids=[device_setup.local_rank])

    if device_setup.is_master_process:
        logging_setup.wandb_init(
            run_name=run_name,
            cfg=cfg,
        )

    training_loop(
        cfg=cfg,
        model=model,
        run_name=run_name,
        device_setup=device_setup,
        backprop_setup=backprop_setup,
        datasets_setup=datasets_setup,
    )


def training_loop(
    run_name: str,
    cfg: DictConfig,
    model: torch.nn.Module,
    device_setup: DeviceSetup,
    backprop_setup: BackpropSetup,
    datasets_setup: DatasetsSetup,
):
    # TODO: Helper methods that we might want to refactor out
    def get_batch(split: str):
        if split == "train":
            return datasets_setup.train_loader.get_batch()
        else:
            return datasets_setup.val_loaders[split].get_batch()

    @torch.no_grad()
    def estimate_loss():
        splits = ["train"] + list(datasets_setup.val_loaders.keys())
        out = {}
        model.eval()
        for split in splits:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y, mask = get_batch(split)
                with device_setup.autocast_ctx:
                    logits, loss = model(X, Y, mask)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Fetch the very first batch before the loop starts
    X, Y, mask = get_batch("train")

    # number of iterations in the lifetime of this process
    # we count from 1 because of unknown reasons
    iter_num = 1

    total_tokens = 0
    running_mfu = -1.0
    best_val_loss = 1e9

    while True:
        t0 = time.time()

        # Get the scheduled learning rate for this step
        lr = backprop_setup.lr_scheduler.get_lr(it=iter_num)
        for param_group in backprop_setup.optimizer.param_groups:
            param_group["lr"] = lr

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        n_iter_tokens = 0
        for micro_step in range(cfg.optimizer.gradient_accumulation_steps):
            if device_setup.is_ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = micro_step == cfg.optimizer.gradient_accumulation_steps - 1

            with device_setup.autocast_ctx:
                n_iter_tokens += X.numel()
                logits, loss = model(X, Y, mask)

                # scale the loss to account for gradient accumulation
                loss = loss / cfg.optimizer.gradient_accumulation_steps

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, mask = get_batch("train")

            # backward pass, with gradient scaling if training in fp16
            backprop_setup.grad_scaler.scale(loss).backward()

        # clip the gradient
        if cfg.optimizer.grad_clip != 0.0:
            backprop_setup.grad_scaler.unscale_(backprop_setup.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)

        # step the optimizer and scaler if training in fp16
        backprop_setup.grad_scaler.step(backprop_setup.optimizer)
        backprop_setup.grad_scaler.update()

        # flush the gradients as soon as we can, no need for this memory anymore
        backprop_setup.optimizer.zero_grad(set_to_none=True)

        # End of deep learning \o/

        # Start of metrics and logs
        t_forward_backward = time.time() - t0

        # Count tokens
        tokens_in_step = n_iter_tokens * device_setup.world_size
        total_tokens += tokens_in_step

        # Common log
        if iter_num % cfg.logging.log_interval == 1 and device_setup.is_master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * cfg.optimizer.gradient_accumulation_steps
            mfu = estimate_mfu(
                model=model.module if device_setup.is_ddp else model,
                fwdbwd_per_iter=cfg.data.batch_size * cfg.optimizer.gradient_accumulation_steps,
                dt=t_forward_backward,
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            tps = tokens_in_step / t_forward_backward
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %X")
            if cfg.logging.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "loss/train_loss": lossf,
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                        "total_tokens": total_tokens,
                        "tps": tps,
                    },
                    step=iter_num,
                )
            print(
                f"{timestamp} iter {iter_num}: "
                f"loss {lossf:.4f}, "
                f"time {t_forward_backward:.2f}s, "
                f"mfu {running_mfu * 100:.2f}%, "
                f"tps {tps:.2f}"
            )

        # Eval log
        if iter_num % cfg.eval_interval == 1 and device_setup.is_master_process:
            losses = estimate_loss()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %X")
            print(
                f"{timestamp} iter {iter_num}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['full_val']:.4f}"
            )

            # Unwrap DataParallel
            state_dict = model.module.state_dict() if device_setup.is_ddp else model.state_dict()
            checkpoint = {
                "model": state_dict,
                "optimizer": backprop_setup.optimizer.state_dict(),
                "model_cfg": cfg.model,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "val_loss": losses["full_val"].item(),
                "train_loss": losses["train"].item(),
                "run_config": cfg,
                "total_tokens": total_tokens,
                "run_name": run_name,
                "tokenizer_desc": datasets_setup.tokenizer.to_dict(),
            }

            if cfg.logging.wandb_log:
                checkpoint |= {
                    "wandb_id": wandb.run.id,
                }

            if losses["full_val"] < best_val_loss:
                best_val_loss = losses["full_val"].item()
                checkpoint["best_val_loss"] = best_val_loss
                best_checkpoint_path = os.path.join(cfg.out_dir, run_name + "-best.pt")
                print(f"saving best checkpoint to {best_checkpoint_path}")
                torch.save(checkpoint, best_checkpoint_path)

            checkpoint_path = os.path.join(cfg.out_dir, run_name + "-last.pt")
            print(f"saving latest checkpoint to {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)

            if cfg.logging.wandb_log:
                validation_results = {
                    f"loss/val_{split}": loss.item()
                    for split, loss in losses.items()
                    if split not in ["train", "full_val"]
                }
                wandb.log(
                    {
                        "iter": iter_num,
                        "total_tokens": total_tokens,
                        "loss/train_loss_batch": losses["train"].item(),
                        "loss/best_val_loss": best_val_loss,
                        "loss/val_loss_batch": losses["full_val"].item(),
                        **validation_results,
                    },
                    step=iter_num,
                )

        # Loop end
        iter_num += 1

        if iter_num == cfg.optimizer.max_iters:
            break

    if device_setup.is_ddp:
        destroy_process_group()
