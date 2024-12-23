import os
import json
import hashlib

from datasets import load_dataset
from hydra.utils import to_absolute_path
from midi_tokenizers import AwesomeMidiTokenizer

from data.augmentation import augment_dataset


def train_awesome_tokenizer(tokenizer: AwesomeMidiTokenizer, dataset_dict) -> AwesomeMidiTokenizer:
    dataset = load_dataset("roszcz/maestro-sustain-v2", split="train")
    dataset = augment_dataset(dataset=dataset, **dataset_dict["augmentation"])
    tokenizer.train(train_dataset=dataset)
    return tokenizer


def hash_tokenizer_desc(tokenizer_cfg: dict) -> str:
    # Special tokens are always the same hence we do not use them for hashing
    if "special_tokens" in tokenizer_cfg["parameters"].keys():
        tokenizer_cfg["parameters"].pop("special_tokens")
    tokenizer_json = json.dumps(tokenizer_cfg)
    hasher = hashlib.sha256()
    hasher.update(tokenizer_json.encode("utf-8"))
    return hasher.hexdigest()


def get_tokenizer_path(tokenizer_hash: str) -> str:
    return to_absolute_path(os.path.join("tmp", "tokenizers", f"{tokenizer_hash}.json"))


def check_cache_for_tokenizer(tokenizer_hash: str):
    tokenizer_path = to_absolute_path(get_tokenizer_path(tokenizer_hash))
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "r") as f:
            return json.load(f)
    return None


def load_tokenizer_if_exists(tokenizer_cfg: dict) -> AwesomeMidiTokenizer:
    tokenizer_hash = hash_tokenizer_desc(tokenizer_cfg)

    # Check if a tokenizer with this hash exists in the cache
    cached_tokenizer_desc = check_cache_for_tokenizer(tokenizer_hash)

    if cached_tokenizer_desc:
        return AwesomeMidiTokenizer.from_dict(cached_tokenizer_desc)
    else:
        raise FileNotFoundError("Tokenizer not found. Run tokenizer scripts.")


def get_time_passage(tokens: list[str]) -> list[int]:
    """
    Extract time steps from a list of tokens, maintaining the sequence order and accumulating
    time values. For non-time tokens, repeats the previous time value.

    Example:
        tokens = ['1T', 'NOTE', '2T', 'VELOCITY', '4T', 'NOTE']
        get_time_steps(tokens) -> [1, 1, 3, 3, 7, 7]
    """
    time_steps = []
    current_time = 0

    for token in tokens:
        if token.endswith("T"):
            try:
                time_value = int(token[:-1])
                current_time += time_value
            except ValueError:
                pass
        time_steps.append(current_time)

    return time_steps


def get_time_steps(tokens: list[str]) -> list[int]:
    """
    Extract time steps from a list of tokens. For time tokens extracts the value,
    for non-time tokens outputs 0.

    Example:
        tokens = ['1T', 'NOTE', '2T', 'VELOCITY', '4T', 'NOTE']
        get_time_steps(tokens) -> [1, 0, 2, 0, 4, 0]
    """
    time_steps = []

    for token in tokens:
        if token.endswith("T"):
            try:
                time_value = int(token[:-1])
                time_steps.append(time_value)
            except ValueError:
                time_steps.append(0)
        else:
            time_steps.append(0)

    return time_steps
