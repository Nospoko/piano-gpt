import json

import datasets
import numpy as np
import fortepyan as ff
from datasets import Split, Dataset, DatasetInfo, GeneratorBasedBuilder
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from artifacts import special_tokens
from data.augmentation import augment_dataset
from data.tokenizer_utils import load_tokenizer_if_exists
from midi_datasets.MidiTokenizedDataset.MidiTokenizedDatasetConfig import BUILDER_CONFIGS, MidiTokenizedDatasetConfig

# NOTE: If you make some changes here, you might want to delete your huggingface cache
# at ~/.cache/huggingface/ to rebuild the datasets

_DESC = """
Dataset with MIDI files, divided into source_notes and target_notes with equal sum of notes.
"""


class MidiTokenizedDataset(GeneratorBasedBuilder):
    """
    Dataset builder for sub-sequence-prediction MIDI datasets.

    This class is responsible for downloading, processing, and splitting MIDI datasets into train, test,
    and validation sets, applying augmentations, seperating input and target sequences and generating examples.
    """

    def _info(self) -> DatasetInfo:
        """
        Returns dataset metadata.

        Returns:
            DatasetInfo: Metadata about the dataset.
        """
        return DatasetInfo(description=_DESC)

    # Define the configuration class and available configurations
    BUILDER_CONFIG_CLASS = MidiTokenizedDatasetConfig
    BUILDER_CONFIGS = BUILDER_CONFIGS
    DEFAULT_CONFIG_NAME = "basic-no-overlap"

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:
        # Load the base dataset and additional datasets
        base = datasets.load_dataset(self.config.base_dataset_name)
        other_datasets = [datasets.load_dataset(path, split="train") for path in self.config.extra_datasets]

        # Concatenate all datasets and apply augmentation
        dataset = datasets.concatenate_datasets(other_datasets)
        dataset = augment_dataset(
            dataset=dataset,
            max_pitch_shift=self.config.augmentation["max_pitch_shift"],
            speed_change_factors=self.config.augmentation["speed_change_factors"],
        )

        # Enable multiprocessing by splitting the dataset into shards
        n_shards = 32
        train_shards = [dataset.shard(n_shards, it) for it in range(n_shards)]
        validation_shards = [base["validation"].shard(n_shards, it) for it in range(n_shards)]
        test_shards = [base["test"].shard(n_shards, it) for it in range(n_shards)]

        self.tokenizer = self.get_tokenzier()
        return [
            datasets.SplitGenerator(name=Split.TRAIN, gen_kwargs={"dataset_shards": train_shards}),
            datasets.SplitGenerator(name=Split.TEST, gen_kwargs={"dataset_shards": test_shards}),
            datasets.SplitGenerator(name=Split.VALIDATION, gen_kwargs={"dataset_shards": validation_shards}),
        ]

    def filter_pauses(self, piece: ff.MidiPiece) -> list[ff.MidiPiece]:
        """
        Splits a MIDI piece into smaller pieces based on pauses (silent periods).

        Parameters:
            piece (ff.MidiPiece): MIDI piece to split based on pauses.

        Returns:
            list[ff.MidiPiece]: List of MIDI pieces without long pauses.
        """
        next_start = piece.df.start.shift(-1)
        silent_distance = next_start - piece.df.end
        ids = silent_distance > self.config.pause_detection_threshold
        break_idxs = np.where(ids)[0]
        if len(break_idxs) == 0:
            return [piece]

        pieces = []
        start = 0
        for break_idx in break_idxs:
            finish = break_idx.item() + 1
            piece_part = piece[start:finish]
            pieces.append(piece_part)
            start = finish

        return pieces

    def _generate_examples(self, dataset_shards: list[Dataset]):
        """
        Generates examples from the dataset shards.

        Parameters:
            dataset_shards (list[Dataset]): List of dataset shards to generate examples from.

        Yields:
            dict: Key-value pairs representing each example.
        """
        for shard_id, dataset in enumerate(dataset_shards):
            for it, record in enumerate(dataset):
                piece = ff.MidiPiece.from_huggingface(dict(record))
                pieces = self.filter_pauses(piece)
                all_records = [self.create_record(piece) for piece in pieces]
                all_records = [record for record in all_records if record["note_token_ids"] is not None]
                for jt, sequence in enumerate(all_records):
                    key = f"{it}_{jt}_{shard_id}"
                    yield key, sequence

    def create_record(self, piece: ff.MidiPiece) -> tuple[dict, bool]:
        """
        Method that defines a record in the dataset.
        """
        notes = piece.df
        try:
            encoding = self.tokenizer.encode(notes=notes)
        except KeyError:
            encoding = None
        record = {
            "note_token_ids": encoding,
            "source": json.dumps(piece.source),
        }

        return record

    def get_tokenzier(self) -> ExponentialTimeTokenizer | AwesomeMidiTokenizer:
        tokenizer_parameters = self.config.tokenizer_cfg["parameters"]
        # FIXME ...
        tokenizer_parameters |= {"special_tokens": special_tokens}
        if self.config.tokenizer_cfg["name"] == "ExponentialTimeTokenizer":
            return ExponentialTimeTokenizer(**tokenizer_parameters)
        else:
            return load_tokenizer_if_exists(tokenizer_cfg=self.config.tokenizer_cfg)
