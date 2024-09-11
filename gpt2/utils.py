from typing import Any
from functools import partial

import torch
from hydra.utils import to_absolute_path
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, DictConfig
from midi_tokenizers import MidiTokenizer, ExponentialTimeTokenizer

from artifacts import special_tokens
from gpt2.model import GPT, GPTConfig
from data.piano_dataset import PianoDataset
from data.next_token_dataset import NextTokenDataset
from data.tokenizer_utils import load_tokenizer_if_exists
from data.piano_composer_dataset import PianoComposerDataset
from data.next_token_composer_dataset import NextTokenComposerDataset


def load_cfg(checkpoint: dict) -> DictConfig:
    train_config = checkpoint["config"]
    return OmegaConf.create(train_config)


def load_tokenizer(cfg: DictConfig):
    tokenizer_cfg = OmegaConf.to_container(cfg.tokenizer)
    tokenizer_parameters = tokenizer_cfg["parameters"]
    tokenizer_parameters |= {"special_tokens": special_tokens}

    if cfg.tokenizer.name == "AwesomeMidiTokenizer":
        return load_tokenizer_if_exists(tokenizer_cfg=tokenizer_cfg)
    else:
        return ExponentialTimeTokenizer(**tokenizer_parameters)


def initialize_model(
    cfg: DictConfig,
    checkpoint: dict,
    device: torch.device,
    pad_token_id: int = 0,
) -> GPT:
    """
    Initializes the GPT model using the given configurations and checkpoint.

    Parameters:
        cfg (DictConfig): The configuration object.
        dataset_config (dict): The dataset configuration.
        checkpoint (dict): The model checkpoint.
        device (torch.device): The device to load the model on.

    Returns:
        GPT: The initialized GPT model.
    """
    model_args = {
        "n_layer": cfg.model.n_layer,
        "n_head": cfg.model.n_head,
        "n_embd": cfg.model.n_embd,
        "block_size": cfg.data.sequence_length,
        "bias": cfg.model.bias,
        "vocab_size": None,
        "dropout": cfg.model.dropout,
    }

    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, pad_token_id=pad_token_id)
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model


def get_dataset_for_task(cfg: DictConfig, tokenizer: MidiTokenizer) -> tuple[Any, Any]:
    task_to_dataset = {
        "next_token_prediction": partial(prepare_next_token_datasets, tokenizer=tokenizer),
        "next_token_prediction_with_composer": partial(prepare_next_token_composer_datasets, tokenizer=tokenizer),
        "multi": partial(prepare_piano_dataset, tokenizer=tokenizer),
        "multi_with_composer": partial(prepare_piano_composer_dataset, tokenizer=tokenizer),
    }
    prepare_function = task_to_dataset.get(cfg.task)
    if prepare_function:
        return prepare_function(cfg)
    raise ValueError(f"Unknown task: {cfg.task}")


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
    train_split: Dataset = dataset["train"]
    validation_split: Dataset = dataset["validation"]
    validation_split.shuffle(seed=1337)

    if validation_split.num_rows > cfg.data.batch_size * cfg.eval_iters:
        validation_split = validation_split.select(range(cfg.data.batch_size * cfg.eval_iters))
    return train_split, validation_split


def prepare_next_token_composer_datasets(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
) -> tuple[NextTokenDataset, NextTokenDataset]:
    train_split, validation_split = prepare_dataset_base(cfg, "MidiTokenizedDataset")
    train_dataset = NextTokenComposerDataset(
        dataset=train_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    val_dataset = NextTokenComposerDataset(
        dataset=validation_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    return train_dataset, val_dataset


def prepare_next_token_datasets(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
) -> tuple[NextTokenDataset, NextTokenDataset]:
    train_split, validation_split = prepare_dataset_base(cfg, "MidiTokenizedDataset")
    train_dataset = NextTokenDataset(
        dataset=train_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    val_dataset = NextTokenDataset(
        dataset=validation_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    return train_dataset, val_dataset


def prepare_piano_dataset(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
) -> tuple[PianoDataset, PianoDataset]:
    dataset_config = OmegaConf.to_container(cfg.dataset)
    dataset_path = to_absolute_path("./midi_datasets/AugmentedDataset")

    dataset = load_dataset(
        dataset_path,
        trust_remote_code=True,
        num_proc=cfg.system.data_workers,
        **dataset_config,
    )
    train_split: Dataset = dataset["train"]
    validation_split: Dataset = dataset["validation"]

    train_dataset = PianoDataset(
        dataset=train_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        tasks=cfg.tasks.list,
    )
    val_dataset = PianoDataset(
        dataset=validation_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        tasks=cfg.tasks.list,
    )
    return train_dataset, val_dataset


def prepare_piano_composer_dataset(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
) -> tuple[PianoComposerDataset, PianoComposerDataset]:
    dataset_config = OmegaConf.to_container(cfg.dataset)
    dataset_path = to_absolute_path("./midi_datasets/AugmentedDataset")

    dataset = load_dataset(
        dataset_path,
        trust_remote_code=True,
        num_proc=cfg.system.data_workers,
        **dataset_config,
    )
    train_split: Dataset = dataset["train"]
    validation_split: Dataset = dataset["validation"]

    train_dataset = PianoComposerDataset(
        dataset=train_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        tasks=cfg.tasks.list,
    )
    val_dataset = PianoComposerDataset(
        dataset=validation_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        tasks=cfg.tasks.list,
    )
    return train_dataset, val_dataset
