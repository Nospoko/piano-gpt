"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import math
import time
import datetime
from contextlib import nullcontext

import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig
from midi_tokenizers import ExponentialTimeTokenizer
from piano_dataset.piano_tasks import PianoTaskManager
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import wandb
import artifacts
from gpt2.model import GPT
from gpt2.lr_scheduler import LRScheduler
from data.piano_dataset import PianoDataset
from gpt2.dataloader import CyclicalDataLoader
from data.random_sampler import ValidationRandomSampler, MemoryEfficientRandomSampler
from gpt2.utils import (
    load_tokenizer,
    create_piano_datasets,
    create_augmented_dataset,
    create_tokenized_dataset,
    create_next_token_datasets,
)

load_dotenv()


def setup_device(cfg: DictConfig):
    if int(os.environ.get("RANK", -1)) != -1:
        init_process_group(backend=cfg.ddp.backend, timeout=datetime.timedelta(seconds=1800))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}", True
    return cfg.system.device, False


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


@hydra.main(config_path="configs", config_name="gpt2_pretraining", version_base=None)
def main(cfg: DictConfig):
    os.environ["TOKENIZERS_PARALLELISM"] = "1"  # for training BPE tokenizer

    device, ddp = setup_device(cfg=cfg)
    if ddp:
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed

        # World_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert cfg.optimizer.gradient_accumulation_steps % ddp_world_size == 0
        # TODO: maybe we should not change what is inside the config?
        cfg.optimizer.gradient_accumulation_steps //= ddp_world_size

    else:
        # If not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    if cfg.init_from == "scratch_next_token":
        if cfg.stage == "piano_task":
            raise NotImplementedError("piano_task stage cannot be run from scratch")
        special_tokens = artifacts.dataset_tokens + artifacts.composer_tokens
        tokenizer = load_tokenizer(
            cfg=cfg,
            special_tokens=special_tokens,
        )

        out_dir = to_absolute_path(cfg.out_dir)

        run_config = OmegaConf.to_container(cfg=cfg)
        # init a new model from scratch
        print("Initializing a new model from scratch")

        model_cfg = cfg.model
        model = GPT(
            config=model_cfg,
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
        )

    # First load checkpoint if init_from *midi_gpt2*
    elif "midi-gpt2" in cfg.init_from:
        # resume training from a checkpoint.
        ckpt_path = cfg.init_from
        checkpoint = torch.load(
            ckpt_path,
            map_location=device,
            weights_only=False,
        )
        checkpoint_cfg = OmegaConf.create(checkpoint["run_config"])
        # This will be saved to checkpoint
        run_config = OmegaConf.to_container(cfg=cfg)

        # FIXME Configs should not be modified, if your loading
        # a checkpoint, reuse its config
        cfg.system.dtype = checkpoint_cfg.system.dtype

        # TODO We'll need more elaborate configs to control piano tasks
        # For now let's see what will happen with the default setup

        # Let's reate piano_task_manager here for now, the special tokens will be needed for both stages
        # (to create tokenizers, unless we implement a better way of handling the special tokens)
        piano_task_manager = PianoTaskManager.load_default()
        if checkpoint["tokenizer_desc"]["name"] == "ExponentialTimeTokenizer":
            tokenizer = ExponentialTimeTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer_desc"])
            special_tokens = piano_task_manager.get_special_tokens()
            special_tokens.append(PianoDataset.generation_token)
            tokenizer.add_special_tokens(special_tokens=special_tokens)
        else:
            raise NotImplementedError(f"Unknown tokenizer: {cfg.tokenizer.class_name}")

        out_dir = to_absolute_path(cfg.out_dir)
        pad_token_id = tokenizer.token_to_id["<PAD>"]

        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from config
        model_cfg = checkpoint["model_cfg"]
        if cfg.model.dropout != model_cfg["dropout"]:
            model_cfg["dropout"] = cfg.model.dropout

        # create the model
        # context_size is how many tokens can we fit in the context
        # We have vocab size and block size that are computed right before pre-traning,
        # Maybe we should still save all model_args seperately from the initial configuration?
        model = GPT(
            config=model_cfg,
            vocab_size=tokenizer.vocab_size,
            pad_token_id=pad_token_id,
        )
        state_dict = checkpoint["model"]
        checkpoint = None  # free up memory

        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        state_dict = None
        # crop down the model block size if desired, using model surgery
        # Happens when finetuning on smaller context than the model was pretrained on
        if cfg.data.context_size < model.config.context_size:
            model.crop_context_size(cfg.data.context_size)
            # so that the checkpoint will have the right value
            model_cfg["context_size"] = cfg.data.context_size

    if cfg.stage == "next_token_pretraining":
        hf_dataset = create_tokenized_dataset(
            cfg=cfg,
            tokenizer=tokenizer,
        )
        datasets = create_next_token_datasets(
            hf_dataset=hf_dataset,
            cfg=cfg,
            tokenizer=tokenizer,
        )

    elif cfg.stage == "piano_task":
        if piano_task_manager is None:
            raise ValueError("Piano task manager required for piano task stage")
        hf_dataset = create_augmented_dataset(cfg)
        datasets = create_piano_datasets(
            hf_dataset=hf_dataset,
            cfg=cfg,
            tokenizer=tokenizer,
            piano_task_manager=piano_task_manager,
        )
    else:
        raise NotImplementedError(f"Unknown stage: {cfg.stage}")

    train_dataset = datasets["train_split"]

    # this contains full, bach, chopin, mozart splits
    val_datasets = datasets["validation_splits"]

    tokens_per_batch = cfg.data.batch_size * cfg.data.context_size
    tokens_per_iter = cfg.optimizer.gradient_accumulation_steps * ddp_world_size * tokens_per_batch
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    if cfg.stage != "next_token_pretraining":
        tokens_in_dataset = train_dataset.dataset.num_rows * train_dataset.context_size
        print(f"total tokens in the training dataset will be: {tokens_in_dataset:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    n_workers = math.floor(cfg.system.data_workers / ddp_world_size)
    torch.set_num_threads(n_workers)

    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.system.dtype]

    # TODO What's the difference?
    if device_type == "cpu":
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    print("Data splits used for validation:")
    for split_name, dataset in val_datasets.items():
        print(f"{split_name} split size: {len(dataset)}")

    train_sampler = MemoryEfficientRandomSampler(
        data_source=train_dataset,
        seed=4 + seed_offset,
    )
    # Create the loaders
    train_loader = CyclicalDataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=cfg.data.batch_size,
        pin_memory=device_type == "cuda",
        num_workers=cfg.system.data_workers // ddp_world_size,
        device=device,
    )
    val_loaders = {}
    # We need validation only for the master process - this is the only place it will be performed
    if master_process:
        for split_name, dataset in val_datasets.items():
            sampler = ValidationRandomSampler(
                n_records=len(dataset),
                seed=4,
                num_samples=cfg.data.batch_size * cfg.eval_iters,
            )
            val_loaders[split_name] = CyclicalDataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=cfg.data.batch_size,
                pin_memory=device_type == "cuda",
                num_workers=cfg.system.data_workers // ddp_world_size,
                device=device,
            )

    def get_batch(split: str):
        if split == "train":
            return train_loader.get_batch()
        else:
            return val_loaders[split].get_batch()

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    # TODO Does this mean that the training will not work for anything other than float16?
    enable_grad_scaler = cfg.system.dtype == "float16"
    grad_scaler = torch.cuda.amp.GradScaler(enabled=enable_grad_scaler)

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay=cfg.optimizer.weight_decay,
        learning_rate=cfg.lr.learning_rate,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        device_type=device_type,
    )

    # compile the model
    if cfg.system.compile:
        print("compiling the model... (takes a ~minute)")
        # unoptimized_model is never used ...
        # unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    milion_params = model.get_num_params() / 1e6
    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        splits = ["train"] + list(val_loaders.keys())
        out = {}
        model.eval()
        for split in splits:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y, mask = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y, mask)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    lr_scheduler = LRScheduler(scheduler_config=cfg.lr)

    run_name = f"midi-gpt2-{milion_params:.0f}M-{cfg.logging.wandb_run_name_suffix}-{cfg.logging.wandb_time_suffix}"
    # logging
    if cfg.logging.wandb_log and master_process:
        wandb.init(
            project=cfg.logging.wandb_project,
            name="training-loop",
            group=run_name,
            config=run_config,
            dir="tmp",
        )
        # define our custom x axis metric
        wandb.define_metric("total_tokens")
        # define which metrics will be plotted against it
        wandb.define_metric("train/loss_batch", step_metric="total_tokens")
        wandb.define_metric("val/loss_batch", step_metric="total_tokens")
        wandb.define_metric("train/loss", step_metric="total_tokens")
        wandb_link = wandb.run.get_url()

    # training loop
    # fetch the very first batch
    X, Y, mask = get_batch("train")

    t0 = time.time()
    # unwrap DDP container if needed
    raw_model = model.module if ddp else model
    running_mfu = -1.0

    # number of iterations in the lifetime of this process
    # we count from 1 because of unknown reasons
    iter_num = 1

    total_tokens = 0
    best_val_loss = 1e9

    while True:
        # determine and set the learning rate for this iteration
        if cfg.lr.decay_lr:
            lr = lr_scheduler.get_lr(it=iter_num)
        else:
            lr = cfg.lr.learning_rate

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        t00 = time.time()
        n_iter_tokens = 0
        for micro_step in range(cfg.optimizer.gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = micro_step == cfg.optimizer.gradient_accumulation_steps - 1
            with ctx:
                n_iter_tokens += X.numel()
                logits, loss = model(X, Y, mask)
                # scale the loss to account for gradient accumulation
                loss = loss / cfg.optimizer.gradient_accumulation_steps

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, mask = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            grad_scaler.scale(loss).backward()

        tokens_in_step = n_iter_tokens * ddp_world_size
        total_tokens += tokens_in_step

        # clip the gradient
        if cfg.optimizer.grad_clip != 0.0:
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)

        # step the optimizer and scaler if training in fp16
        grad_scaler.step(optimizer)
        grad_scaler.update()

        t_forward_backward = time.time() - t00
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.eval_interval == 1 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['full_val']:.4f}")
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_cfg": model_cfg,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "val_loss": losses["full_val"].item(),
                "train_loss": losses["train"].item(),
                "run_config": run_config,
                "wandb": wandb_link,
                "wandb_run_name": run_name,
                "wandb_id": wandb.run.id,
                "total_tokens": total_tokens,
                "tokenizer_desc": tokenizer.to_dict(),
            }

            if cfg.stage == "piano_task":
                checkpoint["piano_tasks_config"] = piano_task_manager.tasks_config

            if losses["full_val"] < best_val_loss:
                best_val_loss = losses["full_val"].item()
                checkpoint["best_val_loss"] = best_val_loss
                best_checkpoint_path = os.path.join(out_dir, run_name + ".pt")
                print(f"saving best checkpoint to {best_checkpoint_path}")
                torch.save(checkpoint, best_checkpoint_path)

            checkpoint_path = os.path.join(out_dir, run_name + "last.pt")
            print(f"saving latest checkpoint to {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)
            if cfg.logging.wandb_log:
                # TODO: this is ugly
                validation_results = {
                    f"val/{split}": loss.item() for split, loss in losses.items() if split not in ["train", "full_val"]
                }
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss_batch": losses["train"].item(),
                        "val/loss_batch": losses["full_val"].item(),
                        **validation_results,
                        "total_tokens": total_tokens,
                        "best_val_loss": best_val_loss,
                    },
                    step=iter_num,
                )

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        if iter_num % cfg.logging.log_interval == 1 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * cfg.optimizer.gradient_accumulation_steps
            mfu = raw_model.estimate_mfu(
                fwdbwd_per_iter=cfg.data.batch_size * cfg.optimizer.gradient_accumulation_steps,
                dt=dt,
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            tps = tokens_in_step / t_forward_backward
            if cfg.logging.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": lossf,
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                        "total_tokens": total_tokens,
                        "tps": tps,
                    },
                    step=iter_num,
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt:.2f}s, mfu {running_mfu*100:.2f}%, tps {tps:.2f}",
            )
        iter_num += 1

        if iter_num == cfg.optimizer.max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
