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
import wandb
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Sampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from data.dataset import MidiDataset
from gpt2.model import GPT, GPTConfig
from gpt2.utils import get_dataset_for_task
from data.memory_efficient_random_sampler import MemoryEfficientRandomSampler

load_dotenv()


class CyclicalDataLoader:
    def __init__(
        self,
        dataset: MidiDataset,
        sampler: Sampler,
        batch_size: int,
        shuffle: bool = False,
        pin_memory: bool = False,
        num_workers: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.device = device
        self.dataloader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=num_workers,
        )
        self.iterator = iter(self.dataloader)

    def get_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            # Reset the iterator when it's exhausted
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        x = batch["source_token_ids"].to(self.device, non_blocking=True)
        y = batch["target_token_ids"].to(self.device, non_blocking=True)
        mask = batch["target_mask"].to(self.device, non_blocking=True)
        return x, y, mask


def setup_device(cfg: DictConfig):
    if int(os.environ.get("RANK", -1)) != -1:
        init_process_group(backend=cfg.ddp.backend, timeout=datetime.timedelta(seconds=1800))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}", True
    return cfg.system.device, False


@hydra.main(config_path="configs", config_name="gpt2_pretraining", version_base=None)
def main(cfg: DictConfig):
    model_args = dict(
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        block_size=cfg.data.sequence_length,
        bias=cfg.model.bias,
        vocab_size=None,
        dropout=cfg.model.dropout,
    )  # start with model_args from command line

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
        cfg.optimizer.gradient_accumulation_steps //= ddp_world_size

    else:
        # If not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    # First load checkpoint if init_from midi_gpt2*
    if cfg.init_from.startswith("midi-gpt2"):
        # resume training from a checkpoint.
        ckpt_path = os.path.join("checkpoints/", cfg.init_from)
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        checkpoint_cfg = OmegaConf.create(checkpoint["config"])

        cfg.model = checkpoint_cfg.model
        cfg.tokenizer = checkpoint_cfg.tokenizer
        cfg.system.dtype = checkpoint_cfg.system.dtype

        train_dataset, val_dataset = get_dataset_for_task(cfg=cfg)
        out_dir = to_absolute_path(cfg.out_dir)
        tokenizer = train_dataset.tokenizer
        pad_token_id = tokenizer.token_to_id["<PAD>"]
        config = OmegaConf.to_container(cfg=cfg)
        # model init

        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from config
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]

        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(config=gptconf, pad_token_id=pad_token_id)
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

    elif cfg.init_from == "scratch":
        train_dataset, val_dataset = get_dataset_for_task(cfg=cfg)
        out_dir = to_absolute_path(cfg.out_dir)
        tokenizer = train_dataset.tokenizer
        pad_token_id = tokenizer.token_to_id["<PAD>"]
        config = OmegaConf.to_container(cfg=cfg)
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        model_args["vocab_size"] = tokenizer.vocab_size
        gptconf = GPTConfig(**model_args)
        model = GPT(config=gptconf, pad_token_id=pad_token_id)

    tokens_per_batch = cfg.data.batch_size * cfg.data.sequence_length
    tokens_per_iter = cfg.optimizer.gradient_accumulation_steps * ddp_world_size * tokens_per_batch
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    if cfg.task != "next_token_prediction":
        tokens_in_dataset = train_dataset.dataset.num_rows * train_dataset.sequence_length
        print(f"total tokens in the training dataset will be: {tokens_in_dataset:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    torch.set_num_threads(math.floor(cfg.system.data_workers / ddp_world_size))

    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    train_sampler = MemoryEfficientRandomSampler(
        data_source=train_dataset,
        seed=4 + seed_offset,
    )
    val_sampler = MemoryEfficientRandomSampler(
        data_source=val_dataset,
        seed=4 + seed_offset,
        num_samples=cfg.data.batch_size * cfg.eval_iters,
    )
    # Create the loaders
    train_loader = CyclicalDataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        pin_memory=device_type == "cuda",
        num_workers=cfg.system.data_workers // ddp_world_size,
        device=device,
    )

    val_loader = CyclicalDataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        pin_memory=device_type == "cuda",
        num_workers=cfg.system.data_workers // ddp_world_size,
        device=device,
    )

    def get_batch(split):
        if split == "train":
            return train_loader.get_batch()
        else:
            return val_loader.get_batch()

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the dataset
    meta_vocab_size = tokenizer.vocab_size

    print(f"found vocab_size = {meta_vocab_size} (inside {tokenizer.name})")

    # crop down the model block size if desired, using model surgery
    if cfg.data.sequence_length < model.config.block_size:
        model.crop_block_size(cfg.data.sequence_length)
        model_args["block_size"] = cfg.data.sequence_length  # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.system.dtype == "float16"))

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
        out = {}
        model.eval()
        for split in ["train", "val"]:
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
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < cfg.lr.warmup_iters:
            return cfg.lr.learning_rate * it / cfg.lr.warmup_iters

        # 2) if it > lr_decay_iters, return min learning rate
        if it > cfg.lr.lr_decay_iters:
            return cfg.lr.min_lr

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - cfg.lr.warmup_iters) / (cfg.lr.lr_decay_iters - cfg.lr.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return cfg.lr.min_lr + coeff * (cfg.lr.learning_rate - cfg.lr.min_lr)

    run_name = f"midi-gpt2-{milion_params:.0f}M-{cfg.logging.wandb_run_name_suffix}-{cfg.logging.wandb_time_suffix}"
    # logging
    if cfg.logging.wandb_log and master_process:
        wandb.init(project=cfg.logging.wandb_project, name=run_name, config=config)
        # define our custom x axis metric
        wandb.define_metric("total_tokens")
        # define which metrics will be plotted against it
        wandb.define_metric("train/loss_batch", step_metric="total_tokens")
        wandb.define_metric("val/loss_batch", step_metric="total_tokens")
        wandb.define_metric("train/loss", step_metric="total_tokens")
        wandb_link = wandb.run.get_url()

    total_tokens = 0
    # training loop
    X, Y, mask = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    iter_num = 1
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if cfg.lr.decay_lr else cfg.lr.learning_rate
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
            scaler.scale(loss).backward()

        tokens_in_step = n_iter_tokens * ddp_world_size
        total_tokens += tokens_in_step

        # clip the gradient
        if cfg.optimizer.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        t_forward_backward = time.time() - t00
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss.item(),
                    "train_loss": losses["train"].item(),
                    "config": config,
                    "wandb": wandb_link,
                    "total_tokens": total_tokens,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, run_name + ".pt"))

            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss.item(),
                "train_loss": losses["train"].item(),
                "config": config,
                "wandb": wandb_link,
                "total_tokens": total_tokens,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, run_name + "last.pt"))
            if cfg.logging.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss_batch": losses["train"],
                        "val/loss_batch": losses["val"],
                        "total_tokens": total_tokens,
                        "best_val_loss": best_val_loss,
                    },
                    step=iter_num,
                )

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        if local_iter_num % cfg.logging.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * cfg.optimizer.gradient_accumulation_steps
            mfu = raw_model.estimate_mfu(cfg.data.batch_size * cfg.optimizer.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            tps = tokens_in_step / t_forward_backward

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
        local_iter_num += 1

        if iter_num == cfg.optimizer.max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
