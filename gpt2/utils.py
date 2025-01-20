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


def load_raw_dataset(cfg: DictConfig, tokenizer: MidiTokenizer) -> Dataset:
    """Load raw dataset from Hugging Face."""
    dataset_config = OmegaConf.to_container(cfg.dataset)

    if cfg.stage == "next_token_pretraining":
        dataset_path = to_absolute_path("./midi_datasets/MidiTokenizedDataset")
        dataset_config["tokenizer_dict"] = tokenizer.to_dict()
    else:  # piano_task
        dataset_path = to_absolute_path("./midi_datasets/AugmentedDataset")

    return load_dataset(
        dataset_path,
        trust_remote_code=True,
        num_proc=cfg.system.data_workers,
        **dataset_config,
    )


def create_validation_splits(validation_set: Dataset) -> dict[str, Dataset]:
    """Create composer-specific validation splits."""
    validation_set = validation_set.shuffle(seed=1337)
    return {
        "loss_batch": validation_set,
        "bach": validation_set.filter(lambda x: json.loads(x["source"])["composer"] == "Johann Sebastian Bach"),
        "chopin": validation_set.filter(lambda x: json.loads(x["source"])["composer"] == "Frédéric Chopin"),
        "mozart": validation_set.filter(lambda x: json.loads(x["source"])["composer"] == "Wolfgang Amadeus Mozart"),
    }


def create_tokenized_dataset(cfg: DictConfig, tokenizer: MidiTokenizer) -> Dataset:
    """Load raw hf dataset for next token prediction."""
    dataset_config = OmegaConf.to_container(cfg.dataset)
    dataset_config["tokenizer_dict"] = tokenizer.to_dict()

    return load_dataset(
        path=to_absolute_path("./midi_datasets/MidiTokenizedDataset"),
        trust_remote_code=True,
        num_proc=cfg.system.data_workers,
        **dataset_config,
    )


def create_augmented_dataset(cfg: DictConfig) -> Dataset:
    """Load raw hf dataset for piano tasks."""
    return load_dataset(
        path=to_absolute_path("./midi_datasets/AugmentedDataset"),
        trust_remote_code=True,
        num_proc=cfg.system.data_workers,
        **OmegaConf.to_container(cfg.dataset),
    )


def create_next_token_datasets(
    hf_dataset: Dataset, cfg: DictConfig, tokenizer: MidiTokenizer
) -> dict[str, MidiDataset | dict[str, MidiDataset]]:
    """Create next token prediction datasets."""
    train_dataset = NextTokenDataset(
        dataset=hf_dataset["train"],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        num_proc=cfg.system.data_workers,
    )

    validation_splits = create_validation_splits(hf_dataset["validation"])
    validation_datasets = {
        name: NextTokenDataset(
            dataset=split,
            tokenizer=tokenizer,
            sequence_length=cfg.data.sequence_length,
            loss_masking=cfg.loss_masking,
            num_proc=cfg.system.data_workers,
        )
        for name, split in validation_splits.items()
    }

    return {"train": train_dataset, "validation": validation_datasets}


def create_piano_datasets(
    hf_dataset: Dataset,
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
    piano_task_manager: ParametricTaskManager,
) -> dict[str, MidiDataset | dict[str, MidiDataset]]:
    """Create piano task datasets."""
    train_dataset = PianoDataset(
        dataset=hf_dataset["train"],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        piano_task_manager=piano_task_manager,
        num_proc=cfg.system.data_workers,
    )

    validation_splits = create_validation_splits(hf_dataset["validation"])
    validation_datasets = {
        name: PianoDataset(
            dataset=split,
            tokenizer=tokenizer,
            sequence_length=cfg.data.sequence_length,
            loss_masking=cfg.loss_masking,
            notes_per_record=cfg.data.notes_per_record,
            piano_task_manager=piano_task_manager,
            num_proc=cfg.system.data_workers,
        )
        for name, split in validation_splits.items()
    }

    return {"train": train_dataset, "validation": validation_datasets}


def get_dataset_for_stage(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
    piano_task_manager: ParametricTaskManager = None,
) -> dict[str, MidiDataset | dict[str, MidiDataset]]:
    """Create datasets based on training stage."""
    if cfg.stage == "next_token_pretraining":
        dataset = create_tokenized_dataset(cfg, tokenizer)
        return create_next_token_datasets(dataset, cfg, tokenizer)

    elif cfg.stage == "piano_task":
        if piano_task_manager is None:
            raise ValueError("Piano task manager required for piano task stage")
        dataset = create_augmented_dataset(cfg)
        return create_piano_datasets(dataset, cfg, tokenizer, piano_task_manager)

    raise ValueError(f"Unknown stage: {cfg.stage}")
