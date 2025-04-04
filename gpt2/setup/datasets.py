import json
from typing import NamedTuple

from datasets import Dataset as HFDataset
from omegaconf import OmegaConf, DictConfig
from piano_dataset.piano_tasks import PianoTaskManager
from midi_tokenizers import MidiTokenizer, ExponentialTimeTokenizer

from gpt2.data.dataset import MidiDataset
from gpt2.setup.hardware import DeviceSetup
from gpt2.data.musicality import MusicManager
from gpt2.dataloader import CyclicalDataLoader
from gpt2.data.piano_dataset import PianoDataset
from gpt2.data.random_sampler import ValidationRandomSampler, MemoryEfficientRandomSampler
from gpt2.utils import load_tokenizer, create_augmented_dataset, create_tokenized_dataset, create_next_token_datasets


class DatasetsSetup(NamedTuple):
    hf_dataset: HFDataset
    train_dataset: MidiDataset
    music_manager: MusicManager
    train_loader: CyclicalDataLoader
    tokenizer: ExponentialTimeTokenizer
    val_datasets: dict[str, MidiDataset]
    val_loaders: dict[str, CyclicalDataLoader]


def loaders_setup(
    cfg: DictConfig,
    train_dataset: MidiDataset,
    val_datasets: dict[str, MidiDataset],
    device_setup: DeviceSetup,
) -> tuple[CyclicalDataLoader, dict[str, CyclicalDataLoader]]:
    train_sampler = MemoryEfficientRandomSampler(
        data_source=train_dataset,
        seed=4 + device_setup.seed_offset,
    )
    # Create the loaders
    train_loader = CyclicalDataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=cfg.training.microbatch_size,
        pin_memory=device_setup.device_type == "cuda",
        num_workers=cfg.system.data_workers // device_setup.world_size,
        device=device_setup.device,
    )
    val_loaders = {}

    # We need validation only for the master process - this is the only place it will be performed
    if device_setup.is_master_process:
        for split_name, dataset in val_datasets.items():
            sampler = ValidationRandomSampler(
                n_records=len(dataset),
                seed=4,
                num_samples=cfg.training.microbatch_size * cfg.eval_iters,
            )
            val_loaders[split_name] = CyclicalDataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=cfg.training.microbatch_size,
                pin_memory=device_setup.device_type == "cuda",
                num_workers=cfg.system.data_workers // device_setup.world_size,
                device=device_setup.device,
            )

    return train_loader, val_loaders


def next_token_prediction_setup(
    cfg: DictConfig,
    device_setup: DeviceSetup,
    tokenizer: ExponentialTimeTokenizer = None,
) -> DatasetsSetup:
    raise NotImplementedError(
        (
            "Next Token Prediction is out of date,"
            " if you want to use it, please implement time embeddings"
            " in the NextTokenDataset, following what happens in PianoDataset"
        )
    )

    music_manager = MusicManager(
        max_n_notes=cfg.training.max_notes_per_record,
    )
    if not tokenizer:
        tokenizer = load_tokenizer(
            cfg=cfg,
            special_tokens=music_manager.tokens,
        )
    hf_dataset = create_tokenized_dataset(
        cfg=cfg,
        tokenizer=tokenizer,
    )
    datasets = create_next_token_datasets(
        cfg=cfg,
        tokenizer=tokenizer,
        hf_dataset=hf_dataset,
        music_manager=music_manager,
    )

    train_dataset = datasets["train_split"]
    print("Train dataset samples [G]:", len(train_dataset) / 1e9)
    val_datasets = datasets["validation_splits"]

    train_loader, val_loaders = loaders_setup(
        train_dataset=train_dataset,
        val_datasets=val_datasets,
        device_setup=device_setup,
        cfg=cfg,
    )

    datasets_setup = DatasetsSetup(
        music_manager=music_manager,
        train_dataset=train_dataset,
        val_datasets=val_datasets,
        train_loader=train_loader,
        val_loaders=val_loaders,
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
    )

    return datasets_setup


def piano_task_setup(
    cfg: DictConfig,
    device_setup: DeviceSetup,
    tokenizer: ExponentialTimeTokenizer = None,
) -> DatasetsSetup:
    hf_dataset = create_augmented_dataset(cfg)

    music_manager = MusicManager(
        max_n_notes=cfg.training.max_notes_per_record,
    )
    if not tokenizer:
        tokenizer = load_tokenizer(
            cfg=cfg,
            special_tokens=music_manager.tokens,
        )

    tasks_config = OmegaConf.to_container(cfg.tasks, resolve=True)
    piano_task_manager = PianoTaskManager(tasks_config=tasks_config)

    special_tokens = piano_task_manager.get_special_tokens()
    special_tokens += [PianoDataset.generation_token, PianoDataset.generation_end_token]
    tokenizer.add_special_tokens(special_tokens)

    datasets = create_piano_datasets(
        hf_dataset=hf_dataset,
        cfg=cfg,
        tokenizer=tokenizer,
        music_manager=music_manager,
        piano_task_manager=piano_task_manager,
    )

    train_dataset = datasets["train_split"]
    n_records = len(train_dataset)
    print("Train dataset samples [G]:", n_records / 1e9, f"{n_records:.0e}")
    val_datasets = datasets["validation_splits"]

    train_loader, val_loaders = loaders_setup(
        train_dataset=train_dataset,
        val_datasets=val_datasets,
        device_setup=device_setup,
        cfg=cfg,
    )

    datasets_setup = DatasetsSetup(
        music_manager=music_manager,
        train_dataset=train_dataset,
        val_datasets=val_datasets,
        train_loader=train_loader,
        val_loaders=val_loaders,
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
    )

    return datasets_setup


def create_piano_datasets(
    hf_dataset: HFDataset,
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
        prompt_masking=cfg.training.prompt_masking,
        min_n_task_notes=cfg.training.min_n_task_notes,
        max_notes_per_record=cfg.training.max_notes_per_record,
        min_notes_per_record=cfg.training.min_notes_per_record,
        piano_task_manager=piano_task_manager,
    )

    validation_splits = create_validation_splits(hf_dataset["validation"])
    validation_datasets = {
        name: PianoDataset(
            dataset=split,
            tokenizer=tokenizer,
            music_manager=music_manager,
            piano_task_manager=piano_task_manager,
            context_size=cfg.training.context_size,
            prompt_masking=cfg.training.prompt_masking,
            min_n_task_notes=cfg.training.min_n_task_notes,
            max_notes_per_record=cfg.training.max_notes_per_record,
            min_notes_per_record=cfg.training.min_notes_per_record,
        )
        for name, split in validation_splits.items()
    }

    return {"train_split": train_dataset, "validation_splits": validation_datasets}


def create_validation_splits(validation_set: HFDataset) -> dict[str, HFDataset]:
    """Create composer-specific validation splits."""
    validation_set = validation_set.shuffle(seed=1337)
    # TODO This should be controlled in config
    # But improve the composer token logic first
    return {
        "full_val": validation_set,
        # "bach": validation_set.filter(lambda x: json.loads(x["source"])["composer"] == "Johann Sebastian Bach"),
        # "chopin": validation_set.filter(lambda x: json.loads(x["source"])["composer"] == "Frédéric Chopin"),
        # "mozart": validation_set.filter(lambda x: json.loads(x["source"])["composer"] == "Wolfgang Amadeus Mozart"),
    }
