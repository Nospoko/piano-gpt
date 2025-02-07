from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, Sampler


class MemoryEfficientRandomSampler(Sampler):
    def __init__(self, data_source: Dataset, seed: Optional[int] = None):
        """
        For DPP make sure to use different seed for each process:
        otherwise it will use the same data throughout the node.
        """
        self.data_source = data_source
        self.num_samples = len(data_source)

        self.seed = seed
        self.generator = torch.Generator()
        if self.seed is not None:
            self.generator.manual_seed(self.seed)

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        return torch.randint(0, self.num_samples, (1,), generator=self.generator).item()

    def __len__(self) -> int:
        return self.num_samples


class ValidationRandomSampler(Sampler):
    def __init__(self, n_records: int, num_samples: int, seed: Optional[int] = None):
        """
        For DPP make sure to use different seed for each process:
        otherwise it will use the same data throughout the node.
        """
        self.n_records = n_records
        self.num_samples = num_samples
        self.seed = seed
        self.generator = torch.Generator()

        if self.seed is not None:
            self.generator.manual_seed(self.seed)

        self.indieces = torch.randint(
            low=0,
            high=self.n_records - 1,
            size=(num_samples,),
            generator=self.generator,
        )

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        idx = torch.randint(
            low=0,
            high=self.num_samples,
            size=(1,),
            generator=self.generator,
        ).item()
        return self.indieces[idx].item()

    def __len__(self) -> int:
        return self.num_samples
