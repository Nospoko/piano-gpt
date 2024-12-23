import json

import torch

from artifacts import get_composer_token
from data.next_token_dataset import NextTokenDataset


class NextTokenComposerDataset(NextTokenDataset):
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
        source = json.loads(record["source"])
        if "composer" in source.keys():
            composer_token = get_composer_token(composer=source["composer"])
        else:
            composer_token = "<UNKNOWN_COMPOSER>"
        # Get the full encoding for the record
        full_encoding = record["note_token_ids"]

        composer_token_id = self.tokenizer.token_to_id[composer_token]
        full_encoding = [composer_token_id] + full_encoding

        n_tokens = len(full_encoding)

        # Add padding if necessary
        if n_tokens <= self.sequence_length:
            padding = [self.tokenizer.pad_token_id] * (self.sequence_length + 1 - n_tokens)
            full_encoding = full_encoding + padding
            n_tokens = self.sequence_length + 1

        # Extract the relevant sequence
        encoding = full_encoding[start_point : start_point + self.sequence_length + 1]

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
            "composer_token": composer_token,
            "start_point": start_point,
            "source": record["source"],
        }
        return out
