import json

import pandas as pd

from data.tasks import Task
from artifacts import get_composer_token
from data.piano_dataset import PianoDataset


class PianoComposerDataset(PianoDataset):
    def prepare_encodings(
        self,
        record: dict,
        start_point: int,
        task: str,
    ):
        # Convert notes to a DataFrame and select the desired range
        notes = pd.DataFrame(record["notes"])
        source: dict = json.loads(record["source"])
        composer = source.get("composer", "")
        composer_token = get_composer_token(composer=composer)
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
            prefix_tokens=[target_prefix, composer_token],
        )

        return prompt_token_ids, target_token_ids
