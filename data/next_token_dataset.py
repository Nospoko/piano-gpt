import json
from typing import Literal
from functools import partial
from multiprocessing import Manager

import torch
from datasets import Dataset as HuggingFaceDataset
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from data.dataset import MidiDataset
from artifacts import get_dataset_token, get_composer_token


class NextTokenDataset(MidiDataset):
    """
    A PyTorch Dataset class for generating next token predictions from tokenized MIDI datasets.
    Uses indexing similar to MedianDataset for consistent data access.
    """

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        sequence_length: int,
        loss_masking: Literal["finetuning", "pretraining"] = "pretraining",
        num_proc: int = 16,
    ):
        """
        Initialize the NextTokenDataset.

        Parameters:
            dataset (HuggingFaceDataset): The HuggingFace dataset containing tokenized MIDI data.
            tokenizer (MidiTokenizer): The MidiTokenizer used for tokenizing the MIDI data.
            sequence_length (int): The length of the input sequence.
            loss_masking (str): The type of loss masking to use.
        """
        super().__init__(dataset=dataset, tokenizer=tokenizer, loss_masking=loss_masking)
        self.sequence_length = sequence_length
        self.length = 0
        self.record_lengths = {}
        self.num_proc = num_proc
        self._build_record_lengths()

    def __rich_repr__(self):
        yield "NextTokenDataset"
        yield "size", len(self)
        yield "sequence_length", self.sequence_length
        yield "n_midi_records", len(self.record_lengths)

    def _build_record_lengths(self):
        """
        Calculate the length of each record in the dataset.
        This method uses multiprocessing for efficiency.
        """

        def get_length(record, idx, sequence_length, shared_dict):
            n_tokens = len(record["note_token_ids"])
            length = max(n_tokens - sequence_length, 0)
            shared_dict[idx] = length

        # Use HuggingFace's .map method for simplicity and multiprocessing.Manager for shared recources
        with Manager() as manager:
            shared_dict = manager.dict()
            get_length_partial = partial(
                get_length,
                sequence_length=self.sequence_length,
                shared_dict=shared_dict,
            )

            self.dataset.map(
                get_length_partial,
                num_proc=self.num_proc,
                desc="Building record lengths",
                with_indices=True,
            )
            self.record_lengths = dict(shared_dict)

        # Calculate total dataset length
        self.length = sum(self.record_lengths.values())

    def __len__(self) -> int:
        """Return the total length of the dataset."""
        return self.length

    def _index_to_record_and_start(self, idx: int) -> tuple[int, int]:
        """
        Convert a global index to a record ID and start position within that record.

        Parameters:
            idx (int): Global index in the dataset.

        Returns:
            tuple: (record_id, start_position)
        """
        for record_id, length in self.record_lengths.items():
            if idx < length:
                return record_id, idx
            idx -= length
        raise IndexError("Index out of range")

    def __getitem__(self, idx: int) -> dict:
        """
        Get an item from the dataset at the specified index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the source and target token ids for next token prediction.
        """
        # Get the record ID and start point for the given index
        record_id, start_point = self._index_to_record_and_start(idx)
        record = self.dataset[record_id]

        # Prepare tokens with music metadata
        piece_source = json.loads(record["source"])
        composer_token = get_composer_token(
            composer=piece_source.get("composer", ""),
        )
        dataset_token = get_dataset_token(
            piece_source=piece_source,
        )
        prefix_tokens = [dataset_token, composer_token]
        prefix_token_ids = self.tokenizer.encode_tokens(prefix_tokens)

        # Get the full encoding for the record
        full_encoding = record["note_token_ids"]

        # Extract the relevant sequence
        # FIXME "sequence_length + 1" situation is very confusing
        # TODO What's up with the +1?
        encoding = full_encoding[start_point : start_point + self.sequence_length]

        # Join with special tokens
        encoding = prefix_token_ids + encoding

        # Add padding if necessary
        if len(encoding) <= self.sequence_length:
            full_encoding = self.tokenizer.pad_to_size(
                token_ids=full_encoding,
                target_size=self.sequence_length + 1,
            )
            # TODO Explain +1, should it be included in the call above? How can I find it out?
            # padding = [self.tokenizer.pad_token_id] * (self.sequence_length + 1 - n_tokens)
            # full_encoding = full_encoding + padding

            # TODO Why was is this under if??
            # n_tokens = self.sequence_length + 1

        # Create source and target encodings
        source_encoding = encoding[:-1]
        target_encoding = encoding[1:]

        # Convert to tensors
        source_token_ids = torch.tensor(source_encoding[: self.sequence_length], dtype=torch.int64)
        target_token_ids = torch.tensor(target_encoding[: self.sequence_length], dtype=torch.int64)

        # Create target mask
        target_mask = target_token_ids != self.tokenizer.pad_token_id

        # Prepare the output dictionary
        out = {
            "source_token_ids": source_token_ids,
            "target_token_ids": target_token_ids,
            "target_mask": target_mask,
            "start_point": start_point,
            "source": record["source"],
            # In PIANO dataset this is the length of the prompt part of the sequence
            # Here we consider half of the sequence to be a prompt part for validation purpouses
            "prompt_length": self.sequence_length // 2,
            "prefix_tokens": prefix_tokens,
        }
        return out
