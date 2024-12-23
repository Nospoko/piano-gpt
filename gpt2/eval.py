import os
import math
from contextlib import nullcontext

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from piano_metrics.f1_piano import calculate_f1
from torch.utils.data import Sampler, DataLoader
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from data.dataset import MidiDataset
from gpt2.utils import get_dataset_for_task, get_model
from data.random_sampler import ValidationRandomSampler

load_dotenv()


class CyclicalDataLoader:
    def __init__(
        self,
        dataset: MidiDataset,
        sampler: Sampler,
        batch_size: int,
        pin_memory: bool = False,
        num_workers: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        self.dataset = dataset
        self.batch_size = batch_size
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
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        x = batch["source_token_ids"].to(self.device, non_blocking=True)
        y = batch["target_token_ids"].to(self.device, non_blocking=True)
        mask = batch["target_mask"].to(self.device, non_blocking=True)
        return x, y, mask


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    model_args = dict(
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        block_size=cfg.data.sequence_length,
        bias=cfg.model.bias,
        vocab_size=None,
        dropout=cfg.model.dropout,
    )

    device = cfg.system.device
    if cfg.init_from.startswith("midi-gpt2"):
        ckpt_path = os.path.join("checkpoints/", cfg.init_from)
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        checkpoint_cfg = OmegaConf.create(checkpoint["config"])

        cfg.model = checkpoint_cfg.model
        cfg.tokenizer = checkpoint_cfg.tokenizer
        cfg.system.dtype = checkpoint_cfg.system.dtype

        if cfg.tokenizer.name == "ExponentialTimeTokenizer":
            tokenizer = ExponentialTimeTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer"])
        else:
            tokenizer = AwesomeMidiTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer"])

        val_datasets = get_dataset_for_task(cfg=cfg, tokenizer=tokenizer)[1]

        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]

        model = get_model(
            checkpoint_cfg,
            pad_token_id=tokenizer.pad_token_id,
            model_args=model_args,
        )
        state_dict = checkpoint["model"]

        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(math.floor(cfg.system.data_workers))

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    val_samplers = [
        ValidationRandomSampler(
            data_source=dataset,
            seed=4,
            num_samples=cfg.data.batch_size * cfg.eval_iters,
        )
        for dataset in val_datasets
    ]

    val_loaders = [
        CyclicalDataLoader(
            dataset,
            sampler=sampler,
            batch_size=cfg.data.batch_size,
            pin_memory=device_type == "cuda",
            num_workers=cfg.system.data_workers,
            device=device,
        )
        for dataset, sampler in zip(val_datasets, val_samplers)
    ]

    if cfg.data.sequence_length < model.config.block_size:
        model.crop_block_size(cfg.data.sequence_length)
        model_args["block_size"] = cfg.data.sequence_length

    model.to(device)

    if cfg.system.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()

        splits = ["val", "bach", "chopin", "mozart"]
        for split, loader in zip(splits, val_loaders):
            losses = torch.zeros(cfg.eval_iters)
            f1s = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                f1_scores = torch.zeros(cfg.data.batch_size)
                X, Y, mask = loader.get_batch()
                with ctx:
                    logits, loss = model(X, Y, mask)
                predictions = logits.argmax(dim=-1)
                for b in range(cfg.data.batch_size):
                    token_ids = predictions[b]
                    y_token_ids = Y[b]

                    small_token_positions = (token_ids < 102).nonzero(as_tuple=True)[0]
                    if len(small_token_positions) >= 3:
                        control_position = small_token_positions[2].item()
                    else:
                        control_position = small_token_positions[1].item()
                    small_token_positions = (y_token_ids < 102).nonzero(as_tuple=True)[0]
                    if len(small_token_positions) >= 3:
                        y_control_position = small_token_positions[2].item()
                    else:
                        y_control_position = small_token_positions[1].item()
                    generated_tokens = token_ids[control_position:]
                    generated_df = tokenizer.decode(token_ids=generated_tokens.numpy())
                    original_df = tokenizer.decode(token_ids=y_token_ids[y_control_position:].numpy())
                    # Cropping because we have no EOS token
                    generated_df = generated_df[generated_df.start < original_df.end.max()]
                    f1_scores[b] = calculate_f1(
                        target_df=original_df,
                        generated_df=generated_df,
                        velocity_threshold=30,
                    )[0]
                f1s[k] = f1_scores.mean()
                losses[k] = loss.item()
                print(f1s[k], losses[k])
            out[split] = (losses.mean(), f1s.mean())
        return out

    metrics = estimate_loss()
    print(f"Validation losses for {cfg.init_from}:")
    for split, metrics in metrics.items():
        print(f"{split}: {metrics[0]:.4f}, {metrics[1]:.4f}")


if __name__ == "__main__":
    main()
