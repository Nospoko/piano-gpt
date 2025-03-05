from typing import NamedTuple

import torch
from torch.utils.data import Sampler, DataLoader

from gpt2.data.dataset import MidiDataset


class PianoBatch(NamedTuple):
    x: torch.tensor
    y: torch.tensor
    mask: torch.tensor
    x_time_steps: torch.tensor
    y_time_steps: torch.tensor


class CyclicalDataLoader:
    def __init__(
        self,
        dataset: MidiDataset,
        sampler: Sampler,
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
        self.dataloader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=num_workers,
        )
        self.iterator = iter(self.dataloader)
        self.epoch = 0
        self.batch_counter = 0

    def get_batch(self) -> PianoBatch:
        try:
            batch = next(self.iterator)
        except StopIteration:
            # Reset the iterator when it's exhausted
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

            print("Dataset epoch over!", self.epoch)
            self.epoch += 1

        self.batch_counter += 1

        x = batch["source_token_ids"].to(self.device, non_blocking=True)
        y = batch["target_token_ids"].to(self.device, non_blocking=True)

        x_time_steps = batch["source_time_steps"].to(self.device, non_blocking=True)
        y_time_steps = batch["target_time_steps"].to(self.device, non_blocking=True)

        mask = batch["target_mask"].to(self.device, non_blocking=True)

        piano_batch = PianoBatch(
            x=x,
            y=y,
            mask=mask,
            x_time_steps=x_time_steps,
            y_time_steps=y_time_steps,
        )
        return piano_batch
