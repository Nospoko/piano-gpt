import json

import torch
from hydra.utils import to_absolute_path
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, DictConfig
from piano_dataset.piano_tasks import ParametricTaskManager
from midi_tokenizers import MidiTokenizer, ExponentialTimeTokenizer

from data.dataset import MidiDataset
from gpt2.model import GPT, GPTConfig
from data.piano_dataset import PianoDataset
from data.next_token_dataset import NextTokenDataset
from artifacts import special_tokens_in_the_wrong_place
from data.tokenizer_utils import load_tokenizer_if_exists


def load_cfg(checkpoint: dict) -> DictConfig:
    train_config = checkpoint["config"]
    return OmegaConf.create(train_config)


def load_tokenizer(
    cfg: DictConfig,
    special_tokens: list[str],
):
    tokenizer_cfg = OmegaConf.to_container(cfg.tokenizer)
    tokenizer_parameters = tokenizer_cfg["parameters"]

    # FIXME Hardcoding in a submodule is not a good way to pass special tokens
    # it should be explicit somewhere in the training flow
    special_tokens += special_tokens_in_the_wrong_place
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

    Parameters
    ----------
    cfg : DictConfig
        The configuration object.
    dataset_config : dict
        The dataset configuration.
    checkpoint : dict
        The model checkpoint.
    device : torch.device
        The device to load the model on.

    Returns
    -------
    model : GPT
        The initialized GPT model.
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


# FIXME This is not a good abstraction
def get_dataset_for_stage(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
    piano_task_manager: ParametricTaskManager,
) -> tuple[MidiDataset, tuple]:
    if "next_token" in cfg.stage:
        # Pretraining stage
        train_dataset, all_other_datasets = prepare_next_token_datasets(
            cfg=cfg,
            tokenizer=tokenizer,
        )
        return train_dataset, all_other_datasets

    if "piano-task" in cfg.stage:
        # PIANO stage
        train_dataset, all_other_datasets = prepare_piano_dataset(
            cfg=cfg,
            tokenizer=tokenizer,
            piano_task_manager=piano_task_manager,
        )
        return train_dataset, all_other_datasets

    # TODO training stages can be both used for pretraining. Maybe there can be better names than:
    # 1. next-token-pretraining
    # 2. piano-task-finetuning

    raise ValueError(f"Unknown stage: {cfg.stage}")


def prepare_dataset_base(
    cfg: DictConfig,
    tokenizer: ExponentialTimeTokenizer,
) -> tuple[Dataset, tuple[Dataset, Dataset, Dataset, Dataset]]:
    dataset_builder_config = OmegaConf.to_container(cfg.dataset)
    dataset_path = to_absolute_path(f"./midi_datasets/{cfg.dataset.name}")

    # This is a huggingface dataset name, that aggregates and tokenizes the datasets specified in the configuration.
    if cfg.dataset.name == "MidiTokenizedDataset":
        # We must pass a hashable tokenizer representation to the tokenized dataset builder
        dataset_builder_config["tokenizer_dict"] = tokenizer.to_dict()

    dataset = load_dataset(
        dataset_path,
        trust_remote_code=True,
        num_proc=cfg.system.data_workers,
        **dataset_builder_config,
    )
    train_split = dataset["train"]
    validation_split = dataset["validation"]
    validation_split.shuffle(seed=1337)

    # TODO There's more nuance in maestro composer info (e.g. "fredetic" can be spelled different)
    validation_dataset_bach = validation_split.filter(
        lambda x: json.loads(x["source"])["composer"] == "Johann Sebastian Bach",
    )
    validation_dataset_chopin = validation_split.filter(
        lambda x: json.loads(x["source"])["composer"] == "Frédéric Chopin",
    )
    validation_dataset_mozart = validation_split.filter(
        lambda x: json.loads(x["source"])["composer"] == "Wolfgang Amadeus Mozart"
    )

    return train_split, (
        validation_split,
        validation_dataset_bach,
        validation_dataset_chopin,
        validation_dataset_mozart,
    )


def prepare_next_token_datasets(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
) -> tuple[NextTokenDataset, NextTokenDataset]:
    train_split, validation_splits = prepare_dataset_base(
        cfg=cfg,
        tokenizer=tokenizer,
    )
    train_dataset = NextTokenDataset(
        dataset=train_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        num_proc=cfg.system.data_workers,
    )
    val_dataset = NextTokenDataset(
        dataset=validation_splits[0],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        num_proc=cfg.system.data_workers,
    )
    val_dataset_bach = NextTokenDataset(
        dataset=validation_splits[1],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        num_proc=cfg.system.data_workers,
    )
    val_dataset_chopin = NextTokenDataset(
        dataset=validation_splits[2],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        num_proc=cfg.system.data_workers,
    )
    val_dataset_mozart = NextTokenDataset(
        dataset=validation_splits[3],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        num_proc=cfg.system.data_workers,
    )
    return train_dataset, (val_dataset, val_dataset_bach, val_dataset_chopin, val_dataset_mozart)


def prepare_piano_dataset(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
    piano_task_manager: ParametricTaskManager,
) -> tuple[PianoDataset, tuple[PianoDataset, PianoDataset, PianoDataset, PianoDataset]]:
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
    validation_dataset_bach = validation_split.filter(
        lambda x: json.loads(x["source"])["composer"] == "Johann Sebastian Bach",
    )
    validation_dataset_chopin = validation_split.filter(
        lambda x: json.loads(x["source"])["composer"] == "Frédéric Chopin",
    )
    validation_dataset_mozart = validation_split.filter(
        lambda x: json.loads(x["source"])["composer"] == "Wolfgang Amadeus Mozart"
    )

    train_dataset = PianoDataset(
        dataset=train_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        piano_task_manager=piano_task_manager,
        num_proc=cfg.system.data_workers,
    )
    val_dataset = PianoDataset(
        dataset=validation_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        piano_task_manager=piano_task_manager,
        num_proc=cfg.system.data_workers,
    )
    val_dataset_bach = PianoDataset(
        dataset=validation_dataset_bach,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        piano_task_manager=piano_task_manager,
        num_proc=cfg.system.data_workers,
    )
    val_dataset_chopin = PianoDataset(
        dataset=validation_dataset_chopin,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        piano_task_manager=piano_task_manager,
        num_proc=cfg.system.data_workers,
    )
    val_dataset_mozart = PianoDataset(
        dataset=validation_dataset_mozart,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        piano_task_manager=piano_task_manager,
        num_proc=cfg.system.data_workers,
    )
    return train_dataset, (val_dataset, val_dataset_bach, val_dataset_chopin, val_dataset_mozart)
