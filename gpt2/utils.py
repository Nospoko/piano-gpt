import json

from hydra.utils import to_absolute_path
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, DictConfig
from piano_dataset.piano_tasks import ParametricTaskManager
from midi_tokenizers import MidiTokenizer, ExponentialTimeTokenizer

from data.dataset import MidiDataset
from data.piano_dataset import PianoDataset
from data.next_token_dataset import NextTokenDataset


def load_cfg(checkpoint: dict) -> DictConfig:
    train_config = checkpoint["config"]
    return OmegaConf.create(train_config)


def load_tokenizer(
    cfg: DictConfig,
    special_tokens: list[str],
):
    tokenizer_options = OmegaConf.to_container(cfg.tokenizer)
    tokenizer_config = tokenizer_options["config"]
    if tokenizer_options["class_name"] == "ExponentialTimeTokenizer":
        tokenizer = ExponentialTimeTokenizer.build_tokenizer(tokenizer_config=tokenizer_config)
        tokenizer.add_special_tokens(special_tokens=special_tokens)
        return tokenizer
    else:
        raise NotImplementedError(f"Unknown class name: {tokenizer_options.class_name}")


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
        "full_val": validation_set,
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
        context_size=cfg.data.context_size,
        loss_masking=cfg.loss_masking,
        num_proc=cfg.system.data_workers,
    )

    validation_splits = create_validation_splits(hf_dataset["validation"])
    validation_datasets = {
        name: NextTokenDataset(
            dataset=split,
            tokenizer=tokenizer,
            context_size=cfg.data.context_size,
            loss_masking=cfg.loss_masking,
            num_proc=cfg.system.data_workers,
        )
        for name, split in validation_splits.items()
    }

    return {"train_split": train_dataset, "validation_splits": validation_datasets}


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
        context_size=cfg.data.context_size,
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
            context_size=cfg.data.context_size,
            loss_masking=cfg.loss_masking,
            notes_per_record=cfg.data.notes_per_record,
            piano_task_manager=piano_task_manager,
            num_proc=cfg.system.data_workers,
        )
        for name, split in validation_splits.items()
    }

    return {"train_split": train_dataset, "validation_splits": validation_datasets}
