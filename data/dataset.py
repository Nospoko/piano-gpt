from typing import Literal
from abc import abstractmethod

from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset as TorchDataset

from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer


class MidiDataset(TorchDataset):
    """
    A PyTorch Dataset class for handling tokenized MIDI datasets.

    Attributes:
        dataset (HuggingFaceDataset): The HuggingFace dataset containing tokenized MIDI data.
        tokenizer (MidiTokenizer): The MidiTokenizer used for tokenizing the MIDI data.
    """

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: ExponentialTokenizer | AwesomeTokenizer,
        loss_masking: Literal["finetuning", "pretraining"] = "pretraining",
    ):
        """
        Initialize the MidiDataset.

        Parameters:
            dataset (HuggingFaceDataset): The HuggingFace dataset containing tokenized MIDI data.
            tokenizer (MidiTokenizer): The MidiTokenizer used for tokenizing the MIDI data.
        """
        super().__init__()

        # MidiTokenizer which was used during creation of the dataset
        self.tokenizer = tokenizer
        self.loss_masking = loss_masking

        # Dataset with tokenized MIDI data
        self.dataset = dataset

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The number of records in the dataset.
        """
        return len(self.dataset)

    def get_complete_record(self, idx: int) -> dict:
        """
        Retrieves the complete record at the specified index, including token ids and other stored information.

        Parameters:
            idx (int): The index of the record to retrieve.

        Returns:
            dict: The complete record containing token ids and additional information.
        """
        # The usual token ids + everything we store
        out = self[idx] | self.dataset[idx]
        return out

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """
        Abstract method to retrieve a record at the specified index. Must be implemented by subclasses.

        Parameters:
            idx (int): The index of the record to retrieve.

        Returns:
            dict: The record at the specified index.
        """
        pass
