import json
from typing import Optional
from abc import abstractmethod

from hydra.utils import to_absolute_path
from midi_tokenizers import MidiTokenizer
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, DictConfig
from piano_dataset.piano_tasks import ParametricTaskManager

from data.dataset import MidiDataset
from data.piano_dataset import PianoDataset
from data.next_token_dataset import NextTokenDataset


class DatasetFactory:
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: MidiTokenizer,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def prepare_base_splits(self) -> tuple[Dataset, tuple[Dataset, ...]]:
        """Common dataset loading and splitting logic"""
        dataset_builder_config = OmegaConf.to_container(self.cfg.dataset)
        # We do not need to pass the dataset class to the dataset builder
        # TODO: We can well seperate dataset class from the config, to avoid poping,
        # we know which dataset class should be use for which stage, perhaps we could work with that.
        dataset_builder_config.pop("dataset_class")
        dataset_path = to_absolute_path(f"./midi_datasets/{self.cfg.dataset.dataset_class}")

        if self.cfg.dataset.dataset_class == "MidiTokenizedDataset":
            dataset_builder_config["tokenizer_dict"] = self.tokenizer.to_dict()

        dataset = load_dataset(
            dataset_path,
            trust_remote_code=True,
            num_proc=self.cfg.system.data_workers,
            **dataset_builder_config,
        )

        train_split = dataset["train"]
        validation_splits = self._create_validation_splits(dataset["validation"])
        return train_split, validation_splits

    def _create_validation_splits(
        self,
        validation_split: Dataset,
    ) -> tuple[Dataset, ...]:
        """Create composer-specific validation splits"""
        validation_split.shuffle(seed=1337)

        composers = {"Johann Sebastian Bach": [], "Frédéric Chopin": [], "Wolfgang Amadeus Mozart": []}

        filtered_splits = [validation_split]  # Start with full validation set
        for composer in composers:
            composer_split = validation_split.filter(lambda x: json.loads(x["source"])["composer"] == composer)
            filtered_splits.append(composer_split)

        return tuple(filtered_splits)

    @abstractmethod
    def create_datasets(self) -> tuple[MidiDataset, tuple[MidiDataset, ...]]:
        pass


class NextTokenPretrainingDatasetFactory(DatasetFactory):
    def create_datasets(self) -> tuple[NextTokenDataset, tuple[NextTokenDataset, ...]]:
        train_split, validation_splits = self.prepare_base_splits()

        datasets = []
        for split in [train_split] + list(validation_splits):
            dataset = NextTokenDataset(
                dataset=split,
                tokenizer=self.tokenizer,
                sequence_length=self.cfg.data.sequence_length,
                loss_masking=self.cfg.loss_masking,
                num_proc=self.cfg.system.data_workers,
            )
            datasets.append(dataset)

        return datasets[0], tuple(datasets[1:])


class PianoTaskDatasetFactory(DatasetFactory):
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: MidiTokenizer,
        piano_task_manager: ParametricTaskManager,
    ):
        super().__init__(cfg, tokenizer)
        self.piano_task_manager = piano_task_manager

    def create_datasets(self) -> tuple[PianoDataset, tuple[PianoDataset, ...]]:
        train_split, validation_splits = self.prepare_base_splits()

        datasets = []
        for split in [train_split] + list(validation_splits):
            dataset = PianoDataset(
                dataset=split,
                tokenizer=self.tokenizer,
                sequence_length=self.cfg.data.sequence_length,
                loss_masking=self.cfg.loss_masking,
                notes_per_record=self.cfg.data.notes_per_record,
                piano_task_manager=self.piano_task_manager,
                num_proc=self.cfg.system.data_workers,
            )
            datasets.append(dataset)

        return datasets[0], tuple(datasets[1:])


class DatasetFactoryManager:
    def __init__(self):
        self._factories = {
            "next_token_pretraining": NextTokenPretrainingDatasetFactory,
            "piano_task": PianoTaskDatasetFactory,
        }

    def get_datasets(
        self,
        cfg: DictConfig,
        tokenizer: MidiTokenizer,
        piano_task_manager: Optional[ParametricTaskManager] = None,
    ) -> tuple[MidiDataset, tuple]:
        factory_class = self._factories[cfg.stage]
        if cfg.stage == "piano_task":
            if piano_task_manager is None:
                raise ValueError("Piano task manager is required for piano task datasets")
            factory: DatasetFactory = factory_class(cfg, tokenizer, piano_task_manager)
        else:
            factory: DatasetFactory = factory_class(cfg, tokenizer)
        return factory.create_datasets()
