import json
from typing import Literal

import torch
import numpy as np
import pandas as pd
from datasets import Dataset as HuggingFaceDataset
from piano_dataset.piano_tasks import PianoTaskManager
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from data.dataset import MidiDataset
from artifacts import get_dataset_token, get_composer_token


class PianoDataset(MidiDataset):
    generation_token = "<GENAI>"

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: AwesomeMidiTokenizer | ExponentialTimeTokenizer,
        # TODO How can I find out what's the relation between *context_size* and *notes_per_record*?
        context_size: int,
        notes_per_record: int,
        piano_task_manager: PianoTaskManager,
        loss_masking: Literal["finetuning", "pretraining"] = "pretraining",
        num_proc: int = 16,
    ):
        # Initialize the parent class and set instance variables
        super().__init__(dataset=dataset, tokenizer=tokenizer, loss_masking=loss_masking)
        self.context_size = context_size
        self.notes_per_record = notes_per_record
        self.length = 0

        self.piano_task_manager = piano_task_manager
        # TODO Maybe a .get_task(task_id) would be better for task management
        self.piano_task_names = piano_task_manager.list_task_names()
        self.num_tasks = len(self.piano_task_manager.tasks)

        self.num_proc = num_proc
        self._build_records()

    def __rich_repr__(self):
        yield "size", len(self)
        yield "n_midi_records", len(self.record_lengths)
        yield "notes_per_record", self.notes_per_record
        yield "context_size", self.context_size
        yield "piano_tasks", self.piano_task_names

    def _build_records(self):
        """
        Calculate the length of each record in the dataset.
        This method uses multiprocessing for efficiency.
        """
        # Each record in the input dataset offers a *record_length* number
        # of possible starting points to get a note subsequence with *notes_per_record*
        record_lengths = np.array(self.dataset["n_notes"]) - self.notes_per_record + 1

        # Records shorted than context are effectively discarded
        self.record_lengths = record_lengths.clip(min=0)

        # Calculate total dataset length
        self.length = self.record_lengths.sum() * self.num_tasks

    def __len__(self):
        # Return the total length of the dataset
        return self.length

    def _index_to_record(self, idx: int) -> tuple[int, int, str]:
        # Convert global index to record ID and start point within that record
        # First get the task number
        task_number = idx % self.num_tasks
        task_name = self.piano_task_names[task_number]

        start_point = idx // self.num_tasks
        for record_id, record_length in enumerate(self.record_lengths):
            if start_point < record_length:
                return record_id, start_point, task_name
            start_point -= record_length
        raise IndexError("Index out of range")

    def __getitem__(self, idx: int) -> dict:
        # Get the record ID and start point for the given index
        record_id, start_point, task_name = self._index_to_record(idx)
        record = self.dataset[record_id]
        piece_source = json.loads(record["source"])

        # Convert notes to a DataFrame and select the desired range
        notes_df = pd.DataFrame(record["notes"])
        notes_df = notes_df.iloc[start_point : start_point + self.notes_per_record]

        # Normalize start and end times
        offset = notes_df.start.min()
        notes_df.start = notes_df.start - offset
        notes_df.end = notes_df.end - offset

        # Break the music into prompt and target parts
        piano_task = self.piano_task_manager.get_task(task_name=task_name)
        piece_split = piano_task.prompt_target_split(notes_df=notes_df)

        # Encode prompt part ...
        source_token_ids = self.tokenizer.encode_notes_df(
            notes_df=piece_split.source_df,
        )
        # ... add special tokens ...
        composer_token = get_composer_token(
            composer=piece_source.get("composer", ""),
        )
        dataset_token = get_dataset_token(
            piece_source=piece_source,
        )
        source_prefix_tokens = [dataset_token, composer_token] + piano_task.prefix_tokens
        prefix_token_ids = self.tokenizer.encode_tokens(source_prefix_tokens)

        # ... and join into a single promp sequence of token ids
        prompt_token_ids = prefix_token_ids + source_token_ids

        # Same for the target sequence
        target_token_ids = self.tokenizer.encode_notes_df(
            notes_df=piece_split.target_df,
        )
        # TODO I think this should be a tokenizer-level special token, similar to <PAD>?
        target_prefix_tokens = [self.generation_token]
        target_prefix_token_ids = self.tokenizer.encode_tokens(target_prefix_tokens)
        answer_token_ids = target_prefix_token_ids + target_token_ids

        # Join both input and output into a single sequence
        encoding = prompt_token_ids + answer_token_ids

        # Add safeguard ensuring the encoding is at most context_size + 1
        encoding = encoding[: self.context_size + 1]

        # encoding should be context_size + 1,
        # because we are using [:-1] and [1:] when defining source and target
        encoding_padded = self.tokenizer.pad_to_size(
            token_ids=encoding,
            target_size=self.context_size + 1,
        )

        # Convert into next-token-prediction task
        source_encoding = encoding_padded[:-1]
        target_encoding = encoding_padded[1:]

        # Convert encodings to tensors
        source_token_ids = torch.tensor(source_encoding, dtype=torch.int64)
        target_token_ids = torch.tensor(target_encoding, dtype=torch.int64)

        # Create target mask
        target_mask = target_token_ids != self.tokenizer.pad_token_id
        if self.loss_masking == "finetuning":
            target_mask[: len(prompt_token_ids)] = False

        # Prepare the output dictionary
        out = {
            "task": task_name,
            "target_mask": target_mask,
            "start_point": start_point,
            "piece_source": json.dumps(piece_source),
            "source_token_ids": source_token_ids,
            "target_token_ids": target_token_ids,
            "prompt_length": len(prompt_token_ids),
            "source_prefix_tokens": source_prefix_tokens,
        }
        return out
