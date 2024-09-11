import os
import json
import hashlib

from datasets import load_dataset
from hydra.utils import to_absolute_path
from midi_trainable_tokenizers import AwesomeMidiTokenizer

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
