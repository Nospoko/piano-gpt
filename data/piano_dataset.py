from typing import Literal
from functools import partial
from multiprocessing import Manager

import torch
import pandas as pd
from datasets import Dataset as HuggingFaceDataset
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from data.tasks import Task
from data.dataset import MidiDataset


class PianoDataset(MidiDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: AwesomeMidiTokenizer | ExponentialTimeTokenizer,
        sequence_length: int,
        notes_per_record: int,
        tasks: list,
        loss_masking: Literal["finetuning", "pretraining"] = "pretraining",
        num_proc: int = 16,
    ):
        # Initialize the parent class and set instance variables
        super().__init__(dataset=dataset, tokenizer=tokenizer, loss_masking=loss_masking)
        self.sequence_length = sequence_length
        self.notes_per_record = notes_per_record
        self.length = 0
        self.record_lengths = {}
        self.tasks = tasks
        self.num_tasks = len(self.tasks)
        self.num_proc = num_proc
        self._build_records()

    def __rich_repr__(self):
        yield "size", len(self)
        yield "n_midi_records", len(self.record_lengths)
        yield "notes_per_record", self.notes_per_record
        yield "sequence_length", self.sequence_length
        yield "piano_tasks", self.tasks

    def _build_records(self):
        # Helper function to calculate the length of each record
        def get_record_definition(record, idx, notes_per_record, shared_dict):
            length = len(record["notes"]["pitch"]) - notes_per_record + 1
            shared_dict[idx] = max(length, 0)

        # Use HuggingFace's .map method for simplicity and multiprocessing.Manager for shared recources
        with Manager() as manager:
            shared_dict = manager.dict()
            get_length_partial = partial(
                get_record_definition,
                notes_per_record=self.notes_per_record,
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
        self.length = sum([record_length for record_length in self.record_lengths.values()]) * self.num_tasks

    def __len__(self):
        # Return the total length of the dataset
        return self.length

    def _index_to_record(self, idx):
        # Convert global index to record ID and start point within that record
        # First get the task number
        task_number = idx % self.num_tasks
        task = self.tasks[task_number]
        idx = idx // self.num_tasks

        for record_id, record_length in self.record_lengths.items():
            if idx < record_length:
                return record_id, idx, task
            idx -= record_length
        raise IndexError("Index out of range")

    def prepare_encodings(
        self,
        record: dict,
        start_point: int,
        task: str,
    ):
        # Convert notes to a DataFrame and select the desired range
        notes = pd.DataFrame(record["notes"])
        notes = notes.iloc[start_point : start_point + self.notes_per_record]

        # Normalize start and end times
        offset = notes.start.min()
        notes.start = notes.start - offset
        notes.end = notes.end - offset

        task_generator = Task.get_task(task_name=task)
        source_notes, target_notes = task_generator.generate(notes=notes)
        source_prefix = task_generator.source_token
        target_prefix = task_generator.target_token

        # Encode source and target notes
        prompt_token_ids = self.tokenizer.encode(
            notes=source_notes,
            prefix_tokens=[source_prefix],
        )
        target_token_ids = self.tokenizer.encode(
            notes=target_notes,
            prefix_tokens=[target_prefix],
        )

        return prompt_token_ids, target_token_ids

    def __getitem__(self, idx: int) -> dict:
        # Get the record ID and start point for the given index
        record_id, start_point, task = self._index_to_record(idx)
        record = self.dataset[record_id]

        prompt_token_ids, target_token_ids = self.prepare_encodings(
            record=record,
            start_point=start_point,
            task=task,
        )
        prompt_length = len(prompt_token_ids)
        encoding = prompt_token_ids + target_token_ids

        # Add padding to reach the desired sequence length
        padding_size = self.sequence_length - len(encoding) + 1
        padding = [self.tokenizer.pad_token_id] * padding_size
        encoding = encoding + padding

        # Create source and target encodings
        source_encoding = encoding[:-1]
        target_encoding = encoding[1:]

        # Convert encodings to tensors
        source_token_ids = torch.tensor(source_encoding[: self.sequence_length], dtype=torch.int64)
        target_token_ids = torch.tensor(target_encoding[: self.sequence_length], dtype=torch.int64)

        # Create target mask
        target_mask = target_token_ids != self.tokenizer.pad_token_id
        if self.loss_masking == "finetuning":
            target_mask[: len(prompt_token_ids)] = False

        # Prepare the output dictionary
        out = {
            "source_token_ids": source_token_ids,
            "target_token_ids": target_token_ids,
            "target_mask": target_mask,
            "start_point": start_point,
            "task": task,
            "prediction_task": "high_median_prediction",
            "source": record["source"],
            # The length of the prompt part of the sequence
            "prompt_length": prompt_length,
        }
        return out
