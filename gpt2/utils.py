import json
from functools import partial

import torch
from hydra.utils import to_absolute_path
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, DictConfig
from midi_tokenizers import MidiTokenizer, ExponentialTimeTokenizer
from piano_metrics.piano_metric import F1Metric, PianoMetric, MetricsRunner, KeyCorrelationMetric

from artifacts import special_tokens
from data.dataset import MidiDataset
from gpt2.model import GPT, GPTConfig
from data.piano_dataset import PianoDataset
from data.next_token_dataset import NextTokenDataset
from data.tokenizer_utils import load_tokenizer_if_exists
from data.piano_composer_dataset import PianoComposerDataset
from data.next_token_composer_dataset import NextTokenComposerDataset


def create_metric(name: str, metric_config: dict) -> PianoMetric:
    """Create a single metric with its configuration"""

    if name.startswith("f1"):
        use_pitch_class = name == "f1_pitch_class"
        return F1Metric(
            use_pitch_class=use_pitch_class,
            velocity_threshold=metric_config.get("velocity_threshold", 30),
            min_time_unit=metric_config.get("min_time_unit", 0.01),
        )

    elif name.startswith("key_correlation"):
        use_weighted = name == "key_correlation"
        return KeyCorrelationMetric(
            segment_duration=metric_config.get("segment_duration", 0.125),
            use_weighted=use_weighted,
        )

    else:
        raise ValueError(f"Unknown metric: {name}")


def create_metrics(cfg: DictConfig) -> list[PianoMetric]:
    """Create all metrics from config"""
    metric_configs = cfg.metrics.configs
    return [create_metric(name, metric_configs.get(name, {})) for name in cfg.metrics.names]


def create_metrics_runner(cfg: DictConfig) -> MetricsRunner:
    """Create metrics runner based on config"""
    metrics = create_metrics(cfg)
    return MetricsRunner(metrics)


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


def get_dataset_for_task(cfg: DictConfig, tokenizer: MidiTokenizer) -> tuple[MidiDataset, tuple]:
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


def prepare_dataset_base(cfg: DictConfig, dataset_name: str) -> tuple[Dataset, tuple]:
    dataset_config = OmegaConf.to_container(cfg.dataset)
    dataset_path = to_absolute_path(f"./midi_datasets/{dataset_name}")
    if dataset_name == "MidiTokenizedDataset":
        dataset_config["tokenizer_cfg"] = OmegaConf.to_container(cfg.tokenizer)

    dataset = load_dataset(
        dataset_path,
        trust_remote_code=True,
        num_proc=cfg.system.data_workers,
        **dataset_config,
    )
    train_split: Dataset = dataset["train"]
    validation_split: Dataset = dataset["validation"]
    validation_split.shuffle(seed=1337)
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


def prepare_next_token_composer_datasets(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
) -> tuple[NextTokenDataset, tuple]:
    train_split, validation_splits = prepare_dataset_base(cfg, "MidiTokenizedDataset")
    train_dataset = NextTokenComposerDataset(
        dataset=train_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    val_dataset = NextTokenComposerDataset(
        dataset=validation_splits[0],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    val_dataset_bach = NextTokenComposerDataset(
        dataset=validation_splits[1],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    val_dataset_chopin = NextTokenComposerDataset(
        dataset=validation_splits[2],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    val_dataset_mozart = NextTokenComposerDataset(
        dataset=validation_splits[3],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    return train_dataset, (val_dataset, val_dataset_bach, val_dataset_chopin, val_dataset_mozart)


def prepare_next_token_datasets(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
) -> tuple[NextTokenDataset, NextTokenDataset]:
    train_split, validation_splits = prepare_dataset_base(cfg, "MidiTokenizedDataset")
    train_dataset = NextTokenDataset(
        dataset=train_split,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    val_dataset = NextTokenDataset(
        dataset=validation_splits[0],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    val_dataset_bach = NextTokenDataset(
        dataset=validation_splits[1],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    val_dataset_chopin = NextTokenDataset(
        dataset=validation_splits[2],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    val_dataset_mozart = NextTokenDataset(
        dataset=validation_splits[3],
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
    )
    return train_dataset, (val_dataset, val_dataset_bach, val_dataset_chopin, val_dataset_mozart)


def prepare_piano_dataset(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
) -> tuple[PianoDataset, tuple]:
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
    val_dataset_bach = PianoDataset(
        dataset=validation_dataset_bach,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        tasks=cfg.tasks.list,
    )
    val_dataset_chopin = PianoDataset(
        dataset=validation_dataset_chopin,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        tasks=cfg.tasks.list,
    )
    val_dataset_mozart = PianoDataset(
        dataset=validation_dataset_mozart,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        tasks=cfg.tasks.list,
    )
    return train_dataset, (val_dataset, val_dataset_bach, val_dataset_chopin, val_dataset_mozart)


def prepare_piano_composer_dataset(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
) -> tuple[PianoComposerDataset, tuple]:
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
    val_dataset_bach = PianoComposerDataset(
        dataset=validation_dataset_bach,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        tasks=cfg.tasks.list,
    )
    val_dataset_chopin = PianoComposerDataset(
        dataset=validation_dataset_chopin,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        tasks=cfg.tasks.list,
    )
    val_dataset_mozart = PianoComposerDataset(
        dataset=validation_dataset_mozart,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        loss_masking=cfg.loss_masking,
        notes_per_record=cfg.data.notes_per_record,
        tasks=cfg.tasks.list,
    )
    return train_dataset, (val_dataset, val_dataset_bach, val_dataset_chopin, val_dataset_mozart)
