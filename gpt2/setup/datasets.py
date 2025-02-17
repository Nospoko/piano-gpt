from omegaconf import DictConfig

from data.dataset import MidiDataset
from gpt2.setup.hardware import DeviceSetup
from gpt2.dataloader import CyclicalDataLoader
from data.random_sampler import ValidationRandomSampler, MemoryEfficientRandomSampler


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
