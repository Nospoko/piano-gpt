from typing import NamedTuple

from datasets import Dataset as HFDataset
from omegaconf import OmegaConf, DictConfig
from midi_tokenizers import ExponentialTimeTokenizer
from piano_dataset.piano_tasks import PianoTaskManager

from data.dataset import MidiDataset
from data.musicality import MusicManager
from data.piano_dataset import PianoDataset
from gpt2.setup.hardware import DeviceSetup
from gpt2.dataloader import CyclicalDataLoader
from data.random_sampler import ValidationRandomSampler, MemoryEfficientRandomSampler
from gpt2.utils import (
    load_tokenizer,
    create_piano_datasets,
    create_augmented_dataset,
    create_tokenized_dataset,
    create_next_token_datasets,
)


class DatasetsSetup(NamedTuple):
    hf_dataset: HFDataset
    train_loader: CyclicalDataLoader
    tokenizer: ExponentialTimeTokenizer
    val_loaders: dict[str, CyclicalDataLoader]

    special_tokens: list[str] = []


def loaders_setup(
    cfg: DictConfig,
    train_dataset: MidiDataset,
    val_datasets: dict[str:MidiDataset],
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
        batch_size=cfg.data.batch_size,
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
                num_samples=cfg.data.batch_size * cfg.eval_iters,
            )
            val_loaders[split_name] = CyclicalDataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=cfg.data.batch_size,
                pin_memory=device_setup.device_type == "cuda",
                num_workers=cfg.system.data_workers // device_setup.world_size,
                device=device_setup.device,
            )

    return train_loader, val_loaders


def next_token_prediction_setup(
    cfg: DictConfig,
    device_setup: DeviceSetup,
    music_manager: MusicManager,
) -> DatasetsSetup:
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
    val_datasets = datasets["validation_splits"]

    train_loader, val_loaders = loaders_setup(
        train_dataset=train_dataset,
        val_datasets=val_datasets,
        device_setup=device_setup,
        cfg=cfg,
    )

    datasets_setup = DatasetsSetup(
        train_loader=train_loader,
        val_loaders=val_loaders,
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
    )

    return datasets_setup


def piano_task_setup(
    cfg: DictConfig,
    device_setup: DeviceSetup,
    music_manager: MusicManager,
) -> DatasetsSetup:
    hf_dataset = create_augmented_dataset(cfg)

    tokenizer = load_tokenizer(
        cfg=cfg,
        special_tokens=music_manager.tokens,
    )

    tasks_config = OmegaConf.to_container(cfg.tasks, resolve=True)
    piano_task_manager = PianoTaskManager(tasks_config=tasks_config)

    special_tokens = piano_task_manager.get_special_tokens()
    special_tokens += [PianoDataset.generation_token]
    tokenizer.add_special_tokens(special_tokens)

    datasets = create_piano_datasets(
        hf_dataset=hf_dataset,
        cfg=cfg,
        tokenizer=tokenizer,
        music_manager=music_manager,
        piano_task_manager=piano_task_manager,
    )

    train_dataset = datasets["train_split"]
    val_datasets = datasets["validation_splits"]

    train_loader, val_loaders = loaders_setup(
        train_dataset=train_dataset,
        val_datasets=val_datasets,
        device_setup=device_setup,
        cfg=cfg,
    )

    datasets_setup = DatasetsSetup(
        special_tokens=special_tokens,
        train_loader=train_loader,
        val_loaders=val_loaders,
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
    )

    return datasets_setup
