import json

from hydra.utils import to_absolute_path
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, DictConfig
from piano_dataset.piano_tasks import PianoTaskManager
from midi_tokenizers import MidiTokenizer, ExponentialTimeTokenizer

from gpt2.data.dataset import MidiDataset
from gpt2.data.musicality import MusicManager
from gpt2.data.piano_dataset import PianoDataset
from gpt2.data.next_token_dataset import NextTokenDataset


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
        dataset_path = to_absolute_path("./gpt2/midi_datasets/MidiTokenizedDataset")
        dataset_config["tokenizer_dict"] = tokenizer.to_dict()
    else:
        # for piano_tasks
        dataset_path = to_absolute_path("./gpt2/midi_datasets/AugmentedDataset")

    return load_dataset(
        dataset_path,
        trust_remote_code=True,
        num_proc=16,
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

    dataset_path = to_absolute_path("./gpt2/midi_datasets/MidiTokenizedDataset")
    return load_dataset(
        path=dataset_path,
        trust_remote_code=True,
        num_proc=cfg.system.data_workers,
        **dataset_config,
    )


def create_augmented_dataset(cfg: DictConfig) -> Dataset:
    """Load raw hf dataset for piano tasks."""
    return load_dataset(
        path=to_absolute_path("./gpt2/midi_datasets/AugmentedDataset"),
        trust_remote_code=True,
        num_proc=cfg.system.data_workers,
        **OmegaConf.to_container(cfg.dataset),
    )


def create_next_token_datasets(
    hf_dataset: Dataset,
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
    music_manager: MusicManager,
) -> dict[str, MidiDataset | dict[str, MidiDataset]]:
    """Create next token prediction datasets."""
    train_dataset = NextTokenDataset(
        dataset=hf_dataset["train"],
        tokenizer=tokenizer,
        music_manager=music_manager,
        context_size=cfg.training.context_size,
    )

    validation_splits = create_validation_splits(hf_dataset["validation"])
    validation_datasets = {
        split_name: NextTokenDataset(
            dataset=split_dataset,
            tokenizer=tokenizer,
            music_manager=music_manager,
            context_size=cfg.training.context_size,
        )
        for split_name, split_dataset in validation_splits.items()
    }

    return {"train_split": train_dataset, "validation_splits": validation_datasets}


def create_piano_datasets(
    hf_dataset: Dataset,
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
    music_manager: MusicManager,
    piano_task_manager: PianoTaskManager,
) -> dict[str, MidiDataset | dict[str, MidiDataset]]:
    """Create piano task datasets."""
    train_dataset = PianoDataset(
        dataset=hf_dataset["train"],
        tokenizer=tokenizer,
        music_manager=music_manager,
        context_size=cfg.training.context_size,
        loss_masking=cfg.training.loss_masking,
        notes_per_record=cfg.training.notes_per_record,
        piano_task_manager=piano_task_manager,
    )

    validation_splits = create_validation_splits(hf_dataset["validation"])
    validation_datasets = {
        name: PianoDataset(
            dataset=split,
            tokenizer=tokenizer,
            music_manager=music_manager,
            context_size=cfg.training.context_size,
            loss_masking=cfg.training.loss_masking,
            notes_per_record=cfg.training.notes_per_record,
            piano_task_manager=piano_task_manager,
        )
        for name, split in validation_splits.items()
    }

    return {"train_split": train_dataset, "validation_splits": validation_datasets}
