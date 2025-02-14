import json

import torch
import numpy as np
from datasets import Dataset as HuggingFaceDataset
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from data.dataset import MidiDataset
from data.musicality import MusicManager


class NextTokenDataset(MidiDataset):
    """
    A PyTorch Dataset class for generating next token predictions from tokenized MIDI datasets.
    Uses indexing similar to MedianDataset for consistent data access.
    """

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: ExponentialTimeTokenizer | AwesomeMidiTokenizer,
        music_manager: MusicManager,
        context_size: int,
    ):
        """
        Initialize the NextTokenDataset.

        Parameters:
            dataset (HuggingFaceDataset): The HuggingFace dataset containing tokenized MIDI data.
            tokenizer (MidiTokenizer): The MidiTokenizer used for tokenizing the MIDI data.
            context_size (int): The length of the input sequence.
        """
        super().__init__(dataset=dataset, tokenizer=tokenizer)
        self.context_size = context_size
        self.music_manager = music_manager
        self.length = 0
        self._build_record_lengths()

    def __rich_repr__(self):
        yield "NextTokenDataset"
        yield "size", len(self)
        yield "context_size", self.context_size
        yield "n_midi_records", len(self.record_lengths)

    def _build_record_lengths(self):
        """
        Calculate the length of each record in the dataset.
        This method uses multiprocessing for efficiency.
        """
        # Each record in the input dataset offers a *record_length* number
        # of possible starting points to get a subsequence with *context_size*
        record_lengths = np.array(self.dataset["n_tokens"]) - self.context_size

        # Records shorted than context are effectively discarded
        self.record_lengths = record_lengths.clip(min=0)

        # Calculate total dataset length
        self.length = self.record_lengths.sum()

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
        start_point = idx

        for record_id, length in enumerate(self.record_lengths):
            if start_point < length:
                return record_id, start_point
            start_point -= length

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
        composer_token = self.music_manager.get_composer_token(
            composer=piece_source.get("composer", ""),
        )
        dataset_token = self.music_manager.get_dataset_token(
            piece_source=piece_source,
        )
        prefix_tokens = [dataset_token, composer_token]
        prefix_token_ids = self.tokenizer.encode_tokens(prefix_tokens)

        # Get the full encoding for the record
        full_encoding = record["note_token_ids"]

        # Extract the relevant sequence
        # encoding should be context_size + 1, because we are using [:-1] and [1:] when defining source and target
        # which will make source and target token sequences one less than encoding length
        encoding = full_encoding[start_point : start_point + self.context_size + 1]

        # Join with special tokens
        encoding = prefix_token_ids + encoding

        # Add padding if necessary
        if len(encoding) <= self.context_size:
            full_encoding = self.tokenizer.pad_to_size(
                token_ids=full_encoding,
                target_size=self.context_size + 1,
            )

        # Create source and target encodings
        source_encoding = encoding[:-1]
        target_encoding = encoding[1:]

        # Convert to tensors
        source_token_ids = torch.tensor(source_encoding[: self.context_size], dtype=torch.int64)
        target_token_ids = torch.tensor(target_encoding[: self.context_size], dtype=torch.int64)

        # Create target mask
        target_mask = target_token_ids != self.tokenizer.pad_token_id

        # Prepare the output dictionary
        out = {
            "source_token_ids": source_token_ids,
            "target_token_ids": target_token_ids,
            "target_mask": target_mask,
            "start_point": start_point,
            "record_id": record_id,
            "source": record["source"],
            # In PIANO dataset this is the length of the prompt part of the sequence
            # Here we consider half of the sequence to be a prompt part for validation purpouses
            "prompt_length": self.context_size // 2,
            "prefix_tokens": prefix_tokens,
        }
        return out
