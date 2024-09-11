import os
import json
import hashlib
from multiprocessing import Event, Process

from datasets import load_dataset
from midi_trainable_tokenizers import AwesomeMidiTokenizer
from midi_tokenizers_generation.base_tokenizer_generator import generate_tokenizer

from artifacts import special_tokens
from data.augmentation import augment_dataset


def train_awesome_tokenizer(tokenizer: AwesomeMidiTokenizer, dataset_dict) -> AwesomeMidiTokenizer:
    dataset = load_dataset("roszcz/maestro-sustain-v2", split="train")
    dataset = augment_dataset(dataset=dataset, **dataset_dict["augmentation"])
    tokenizer.train(train_dataset=dataset)
    return tokenizer


def hash_tokenizer_desc(tokenizer_desc: dict) -> str:
    tokenizer_json = json.dumps(tokenizer_desc)
    hasher = hashlib.sha256()
    hasher.update(tokenizer_json.encode("utf-8"))
    return hasher.hexdigest()


def get_tokenizer_path(tokenizer_hash: str) -> str:
    return os.path.join("tmp", "tokenizers", f"{tokenizer_hash}.json")


def check_cache_for_tokenizer(tokenizer_hash: str):
    tokenizer_path = get_tokenizer_path(tokenizer_hash)
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "r") as f:
            return json.load(f)
    return None


def load_and_cache_tokenizer(tokenizer_cfg: dict, tokenizer_hash: str, done_event):
    new_tokenizer = load_tokenizer_from_cfg(tokenizer_cfg)
    cache_tokenizer(tokenizer_hash, new_tokenizer)
    done_event.set()


def cache_tokenizer(tokenizer_hash: str, tokenizer: AwesomeMidiTokenizer):
    tokenizer_path = get_tokenizer_path(tokenizer_hash)
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    tokenizer.save_tokenizer(path=tokenizer_path)


def load_tokenizer_from_cfg(tokenizer_cfg: dict) -> AwesomeMidiTokenizer:
    awesome_tokenizer_parameters = tokenizer_cfg["parameters"] | {"special_tokens": special_tokens}
    base_tokenizer_name = tokenizer_cfg["base_tokenizer"]
    base_tokenizer_parameters = tokenizer_cfg["base_tokenizer_parameters"] | {"special_tokens": special_tokens}
    base_tokenizer = generate_tokenizer(
        name=base_tokenizer_name,
        parameters=base_tokenizer_parameters,
    )
    tokenizer = AwesomeMidiTokenizer(
        base_tokenizer=base_tokenizer,
        **awesome_tokenizer_parameters,
    )
    tokenizer = train_awesome_tokenizer(
        tokenizer=tokenizer,
        dataset_dict={
            "augmentation": {
                "max_pitch_shift": 5,
                "speed_change_factors": [0.975, 0.95, 1.025, 1.05],
            },
        },
    )
    return tokenizer.from_dict(tokenizer.to_dict())


def load_tokenizer_if_exists(tokenizer_cfg: dict) -> AwesomeMidiTokenizer:
    tokenizer_hash = hash_tokenizer_desc(tokenizer_cfg)

    # Check if a tokenizer with this hash exists in the cache
    cached_tokenizer_desc = check_cache_for_tokenizer(tokenizer_hash)

    if cached_tokenizer_desc:
        return AwesomeMidiTokenizer.from_dict(cached_tokenizer_desc)
    else:
        # If not in cache, create a new tokenizer and cache it
        done_event = Event()
        # Running this is seperate process, because otherwise
        # huggingface leaves garbage in RAM that cannot be then collected after training the tokenizer
        p = Process(target=load_and_cache_tokenizer, args=(tokenizer_cfg, tokenizer_hash, done_event))
        p.start()

        # Wait for the process to finish
        done_event.wait()

        # Ensure the process is properly terminated
        p.join()

        # Now that the tokenizer has been cached, we can load it
        cached_tokenizer_desc = check_cache_for_tokenizer(tokenizer_hash)
        if cached_tokenizer_desc is None:
            raise RuntimeError("Failed to cache tokenizer")

        return AwesomeMidiTokenizer.from_dict(cached_tokenizer_desc)
