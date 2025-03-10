import json
from typing import NamedTuple

import torch
import numpy as np
import pandas as pd
from datasets import Dataset as HuggingFaceDataset
from piano_dataset.piano_tasks import PianoTaskManager
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from gpt2.data.dataset import MidiDataset
from gpt2.data.musicality import MusicManager


class PianoIndex(NamedTuple):
    task_name: str
    start_point: int
    record_idx: int
    n_notes: int


class PianoDataset(MidiDataset):
    generation_token = "<GENAI>"
    generation_end_token = "<EOGENAI>"

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: AwesomeMidiTokenizer | ExponentialTimeTokenizer,
        # TODO How can I find out what's the relation between *context_size* and *notes_per_record*?
        context_size: int,
        max_notes_per_record: int,
        min_notes_per_record: int,
        music_manager: MusicManager,
        piano_task_manager: PianoTaskManager,
        prompt_masking: bool = False,
    ):
        # Initialize the parent class and set instance variables
        super().__init__(dataset=dataset, tokenizer=tokenizer)

        self.length = 0
        self.prompt_masking = prompt_masking
        self.context_size = context_size
        self.music_manager = music_manager
        self.max_notes_per_record = max_notes_per_record
        self.min_notes_per_record = min_notes_per_record

        self.piano_task_manager = piano_task_manager
        # TODO Maybe a .get_task(task_id) would be better for task management
        self.piano_task_names = piano_task_manager.list_task_names()
        self.num_tasks = len(self.piano_task_manager.tasks)

        self._build_records()

    def __rich_repr__(self):
        yield "size", len(self)
        yield "n_midi_records", len(self.record_lengths)
        yield "max_notes_per_record", self.max_notes_per_record
        yield "min_notes_per_record", self.min_notes_per_record
        yield "context_size", self.context_size
        yield "n_piano_tasks", len(self.piano_task_names)
        yield "prompt_masking", self.prompt_masking

    def _build_records(self):
        """
        Calculate the length of each record in the dataset.
        This method uses multiprocessing for efficiency.
        """
        # Each record in the input dataset offers a *record_length* number
        # of possible starting points to get a note subsequence with *notes_per_record*
        record_lengths = np.array(self.dataset["n_notes"]) - self.max_notes_per_record + 1

        # For every record we can have that many different subsequences with different lengths
        self.n_duration_options = self.max_notes_per_record - self.min_notes_per_record

        # Records shorter than context are effectively discarded
        self.record_lengths = record_lengths.clip(min=0)

        # Calculate total dataset length
        self.length = self.record_lengths.sum() * self.num_tasks * self.n_duration_options

    def __len__(self):
        # Return the total length of the dataset
        return self.length

    def _decode_piano_index(self, idx: int) -> PianoIndex:
        # Convert global index to record ID and start point within that record
        # First get the task number
        task_number = idx % self.num_tasks
        task_name = self.piano_task_names[task_number]

        # Go to the second level index ...
        idx_bis = idx // self.num_tasks

        # ... and decode the number of notes for this record
        n_notes = self.min_notes_per_record + (idx_bis % self.n_duration_options)

        # ... and then decode the starting note idx
        start_point = idx_bis // self.n_duration_options

        for record_idx, record_length in enumerate(self.record_lengths):
            if start_point < record_length:
                piano_index = PianoIndex(
                    task_name=task_name,
                    start_point=start_point,
                    record_idx=record_idx,
                    n_notes=n_notes,
                )
                return piano_index

            start_point -= record_length

        raise IndexError("Index out of range")

    def __getitem__(self, idx: int) -> dict:
        # Get the record ID and start point for the given index
        # record_id, start_point, task_name = self._index_to_record(idx)
        piano_index = self._decode_piano_index(idx)
        record = self.dataset[piano_index.record_idx]
        piece_source = json.loads(record["source"])

        # Convert notes to a DataFrame and select the desired range
        notes_df = pd.DataFrame(record["notes"])
        notes_df = notes_df.iloc[piano_index.start_point : piano_index.start_point + piano_index.n_notes]

        # Normalize start and end times
        offset = notes_df.start.min()
        notes_df.start = notes_df.start - offset
        notes_df.end = notes_df.end - offset

        # Break the music into prompt and target parts
        piano_task = self.piano_task_manager.get_task(task_name=piano_index.task_name)
        piece_split = piano_task.prompt_target_split(notes_df=notes_df)

        # Encode prompt part ...
        source_token_ids = self.tokenizer.encode_notes_df(
            notes_df=piece_split.source_df,
        )

        # ... add special tokens ...
        composer_token = self.music_manager.get_composer_token(
            composer=piece_source.get("composer", ""),
        )
        dataset_token = self.music_manager.get_dataset_token(
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

        # This is to indicate the beginning of generation ...
        target_prefix_tokens = [self.generation_token]
        target_prefix_token_ids = self.tokenizer.encode_tokens(target_prefix_tokens)

        # ... and this is for the end
        target_finish_token = [self.generation_end_token]
        target_finish_token_ids = self.tokenizer.encode_tokens(target_finish_token)

        answer_token_ids = target_prefix_token_ids + target_token_ids + target_finish_token_ids

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
        time_steps_padded = self.tokenizer.token_ids_to_time_steps(
            token_ids=encoding_padded,
            restart_tokens=[self.generation_token],
        )

        # Convert into next-token-prediction task
        source_encoding = encoding_padded[:-1]
        source_time_steps = time_steps_padded[:-1]

        target_encoding = encoding_padded[1:]

        # Convert encodings to tensors
        source_token_ids = torch.tensor(source_encoding, dtype=torch.int64)
        source_time_steps = torch.tensor(source_time_steps, dtype=torch.int64)

        target_token_ids = torch.tensor(target_encoding, dtype=torch.int64)

        # Create target mask, False means ignore
        target_mask = target_token_ids != self.tokenizer.pad_token_id

        # Ignore the prompt for loss calculation
        # only worry about the response
        if self.prompt_masking:
            target_mask[: len(prompt_token_ids)] = False

        # Prepare the output dictionary
        out = {
            "task": piano_index.task_name,
            "target_mask": target_mask,
            "n_notes": piano_index.n_notes,
            "start_point": piano_index.start_point,
            "piece_source": json.dumps(piece_source),
            "source_token_ids": source_token_ids,
            "target_token_ids": target_token_ids,
            "source_time_steps": source_time_steps,
            "prompt_length": len(prompt_token_ids),
            "source_prefix_tokens": source_prefix_tokens,
        }
        return out
