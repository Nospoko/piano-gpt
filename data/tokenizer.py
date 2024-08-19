from pandas import DataFrame
from tokenizers import Tokenizer
from midi_trainable_tokenizers import AwesomeMidiTokenizer
from midi_tokenizers import MidiTokenizer, ExponentialTimeTokenizer


class ExponentialTokenizer(ExponentialTimeTokenizer):
    def __init__(
        self,
        min_time_unit: float = 0.01,
        n_velocity_bins: int = 128,
        special_tokens: list[str] = None,
    ):
        super().__init__(min_time_unit, n_velocity_bins, special_tokens)
        self.pad_token_id = self.token_to_id["<PAD>"]

    def encode(
        self,
        notes: DataFrame,
        pad_to_size: int = 0,
        prefix_tokens: list[str] = [],
    ) -> list[int]:
        encoding = super().encode(notes)

        padding_size = pad_to_size - len(encoding) - len(prefix_tokens)

        suffix_ids = [self.token_to_id[token] for token in prefix_tokens]
        padding = [self.pad_token_id] * padding_size

        return suffix_ids + encoding + padding


class AwesomeTokenizer(AwesomeMidiTokenizer):
    def __init__(
        self,
        base_tokenizer: MidiTokenizer,
        bpe_tokenizer: Tokenizer = None,
        max_vocab_size: int = 30000,
        max_token_length: int = 128,
        special_tokens: list[str] = None,
    ):
        super().__init__(base_tokenizer, bpe_tokenizer, max_vocab_size, max_token_length, special_tokens)
        self.pad_token_id = self.token_to_id["<PAD>"]

    def encode(
        self,
        notes: DataFrame,
        pad_to_size: int = 0,
        prefix_tokens: list[str] = [],
    ) -> list[int]:
        encoding = super().encode(notes)
        padding_size = pad_to_size - len(encoding) - len(prefix_tokens)
        suffix_ids = [self.token_to_id[token] for token in prefix_tokens]
        padding = [self.pad_token_id] * padding_size

        return suffix_ids + encoding + padding
