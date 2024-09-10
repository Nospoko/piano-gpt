"""
This script allows to compare models on the whole validation split of the dataset
and on chosen tasks.
"""

import os
from typing import Any
from contextlib import nullcontext

import hydra
import torch
import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from hydra.utils import to_absolute_path
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, DictConfig

from data.dataset import MidiDataset
from gpt2.model import GPT, GPTConfig
from gpt2.utils import load_tokenizer
from data.piano_dataset import PianoDataset
from data.next_token_dataset import NextTokenDataset
from data.piano_composer_dataset import PianoComposerDataset
from data.next_token_composer_dataset import NextTokenComposerDataset
from data.memory_efficient_random_sampler import MemoryEfficientRandomSampler

load_dotenv()


class ValidationDataLoader:
    def __init__(
        self,
        dataset: MidiDataset,
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
        sampler = MemoryEfficientRandomSampler(data_source=dataset, seed=4)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            sampler=sampler,
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
        prompt_lengths = batch["prompt_length"]
        return x, y, mask, prompt_lengths


def get_dataset_for_task(cfg: DictConfig) -> tuple[Any, Any]:
    task_to_dataset = {
        "next_token_prediction": prepare_next_token_dataset,
        "next_token_prediction_with_composer": prepare_next_token_composer_dataset,
        "multi": prepare_piano_dataset,
        "multi_with_composer": prepare_piano_composer_dataset,
    }
    prepare_function = task_to_dataset.get(cfg.task)
    if prepare_function:
        return prepare_function(cfg)
    raise ValueError(f"Unknown task: {cfg.task}")


def prepare_piano_composer_dataset(cfg: DictConfig) -> PianoComposerDataset:
    dataset_config = OmegaConf.to_container(cfg.dataset)
    dataset_path = to_absolute_path("./midi_datasets/AugmentedDataset")

    dataset = load_dataset(
        dataset_path,
        trust_remote_code=True,
        num_proc=cfg.system.data_workers,
        **dataset_config,
    )
    validation_split: Dataset = dataset["validation"]

    tokenizer = load_tokenizer(cfg)
    val_dataset = PianoComposerDataset(
        dataset=validation_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        tasks=cfg.tasks.list,
    )
    return val_dataset


def prepare_next_token_composer_dataset(cfg: DictConfig) -> NextTokenComposerDataset:
    validation_split = prepare_dataset_base(cfg, "MidiTokenizedDataset")
    tokenizer = load_tokenizer(cfg)
    val_dataset = NextTokenComposerDataset(
        dataset=validation_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    return val_dataset


def prepare_dataset_base(cfg: DictConfig, dataset_name: str) -> tuple[Dataset, Dataset]:
    dataset_config = OmegaConf.to_container(cfg.dataset)
    dataset_path = to_absolute_path(f"./midi_datasets/{dataset_name}")
    if dataset_name == "MidiTokenizedDataset":
        dataset_config["tokenizer_parameters"] = OmegaConf.to_container(cfg.tokenizer.tokenizer_parameters)

    dataset = load_dataset(
        dataset_path,
        trust_remote_code=True,
        num_proc=cfg.system.data_workers,
        **dataset_config,
    )
    validation_split: Dataset = dataset["validation"]
    return validation_split


def prepare_next_token_dataset(cfg: DictConfig) -> tuple[NextTokenDataset, NextTokenDataset]:
    validation_split = prepare_dataset_base(cfg, "MidiTokenizedDataset")
    tokenizer = load_tokenizer(cfg)
    val_dataset = NextTokenDataset(
        dataset=validation_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    return val_dataset


def prepare_piano_dataset(cfg: DictConfig) -> tuple[PianoDataset, PianoDataset]:
    dataset_config = OmegaConf.to_container(cfg.dataset)
    dataset_path = to_absolute_path("./midi_datasets/AugmentedDataset")

    dataset = load_dataset(
        dataset_path,
        trust_remote_code=True,
        num_proc=cfg.system.data_workers,
        **dataset_config,
    )
    validation_split: Dataset = dataset["validation"]

    tokenizer = load_tokenizer(cfg)
    val_dataset = PianoDataset(
        dataset=validation_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        tasks=cfg.tasks.list,
        num_proc=cfg.system.data_workers,
    )
    return val_dataset


@hydra.main(config_path="configs", config_name="eval", version_base=None)
@torch.no_grad()
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

    device = cfg.system.device
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

        val_dataset = get_dataset_for_task(cfg=cfg)
        tokenizer = val_dataset.tokenizer
        pad_token_id = tokenizer.pad_token_id
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

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    val_loader = ValidationDataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        pin_memory=device_type == "cuda",
        num_workers=cfg.system.data_workers // ddp_world_size,
        device=device,
    )

    def get_batch():
        return val_loader.get_batch()

    # attempt to derive vocab_size from the dataset
    meta_vocab_size = tokenizer.vocab_size

    print(f"found vocab_size = {meta_vocab_size} (inside {tokenizer.name})")

    # crop down the model block size if desired, using model surgery
    if cfg.data.sequence_length < model.config.block_size:
        model.crop_block_size(cfg.data.sequence_length)
        model_args["block_size"] = cfg.data.sequence_length  # so that the checkpoint will have the right value
    model.to(device)

    # compile the model
    if cfg.system.compile:
        print("compiling the model... (takes a ~minute)")
        # unoptimized_model is never used ...
        # unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    model.train()
    model.eval()
    accuracies = pd.DataFrame(columns=["pitch", "velocity", "start", "end"])
    target_prefix_tokens = 1 if cfg.task == "multi" else 2
    # Iterate whole dataset once
    for k in range(len(val_dataset) // cfg.data.batch_size):
        X, Y, mask, prompt_lengths = get_batch()
        with ctx:
            out = model.generate(
                X[:, : prompt_lengths[0] + target_prefix_tokens],
                max_new_tokens=cfg.data.sequence_length,
                temperature=0.1,
            )
        for generation, prompt_length, labels in zip(out, prompt_lengths, Y):
            output = generation.cpu().numpy()

            out_tokens = [tokenizer.vocab[token_id] for token_id in output]
            target_tokens = [tokenizer.vocab[token_id] for token_id in labels]

            generated_tokens = out_tokens[prompt_length:]
            true_tokens = target_tokens[prompt_length + 1 :]

            # Convert tokens back to notes
            generated_notes = tokenizer.untokenize(generated_tokens)
            true_notes = tokenizer.untokenize(true_tokens)
            generated_notes = generated_notes.iloc[: len(true_notes)]
            row = {
                "pitch": (generated_notes.pitch == true_notes.pitch).sum() / len(true_notes),
                "velocity": (generated_notes.velocity == true_notes.velocity).sum() / len(true_notes),
                "start": (abs(generated_notes.start - true_notes.start) < 0.01).sum() / len(true_notes),
                "end": (abs(generated_notes.end - true_notes.end) < 0.01).sum() / len(true_notes),
            }
            accuracies.loc[k] = row

        if k % 20 == 0:
            p_acc = accuracies.pitch.mean()
            v_acc = accuracies.velocity.mean()
            s_acc = accuracies.start.mean()
            e_acc = accuracies.end.mean()

            print(f"iter: {k}, accuracies - pitch: {p_acc} | velocity: {v_acc} | start: {s_acc} | end: {e_acc}")


if __name__ == "__main__":
    main()
