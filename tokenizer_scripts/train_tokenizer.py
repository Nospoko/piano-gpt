import os
import json
import hashlib

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig
from midi_trainable_tokenizers import AwesomeMidiTokenizer
from midi_tokenizers_generation.base_tokenizer_generator import generate_tokenizer

from artifacts import special_tokens


def load_dataset(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read().split("\n")


def hash_config(cfg: DictConfig) -> str:
    """Generate a hash from the configuration."""
    config_str = json.dumps(OmegaConf.to_container(cfg))
    hasher = hashlib.sha256()
    hasher.update(config_str.encode("utf-8"))
    return hasher.hexdigest()


def get_dataset_path(cfg: DictConfig) -> str:
    """Generate the dataset file path based on the configuration hash."""
    config_hash = hash_config(cfg)
    output_dir = os.path.join("tmp", "tokenizer_datasets")
    return to_absolute_path(os.path.join(output_dir, f"{config_hash}.txt"))


def get_tokenizer_path(cfg: DictConfig) -> str:
    """Generate the tokenizer file path based on the configuration hash."""
    config_hash = hash_config(cfg)
    output_dir = os.path.join("tmp", "trained_tokenizers")
    return os.path.join(output_dir, f"{config_hash}.json")


def train_tokenizer(cfg: DictConfig):
    tokenizer_path = get_tokenizer_path(cfg.tokenizer)
    if os.path.exists(tokenizer_path):
        print(f"Tokenizer already trained: {tokenizer_path}")
        return

    # Load the prepared dataset
    dataset_path = get_dataset_path(cfg)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = load_dataset(dataset_path)

    # Initialize the base tokenizer
    base_tokenizer_name = cfg.tokenizer.base_tokenizer
    base_tokenizer_parameters = OmegaConf.to_container(cfg.tokenizer.base_tokenizer_parameters) | {
        "special_tokens": special_tokens
    }
    base_tokenizer = generate_tokenizer(
        name=base_tokenizer_name,
        parameters=base_tokenizer_parameters,
    )

    # Initialize the AwesomeMidiTokenizer
    awesome_tokenizer_parameters = OmegaConf.to_container(cfg.tokenizer.parameters) | {"special_tokens": special_tokens}
    tokenizer = AwesomeMidiTokenizer(
        base_tokenizer=base_tokenizer,
        **awesome_tokenizer_parameters,
    )
    tokenizer.train_from_text_dataset(dataset)
    print(tokenizer.token_to_id)
    # Save the trained tokenizer
    output_dir = to_absolute_path(os.path.join("tmp", "tokenizers"))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{hash_config(cfg.tokenizer)}.json")
    tokenizer.save_tokenizer(output_path)
    print(f"Trained tokenizer saved to: {output_path}")

    # Save the configuration alongside the tokenizer
    config_file = output_path.replace(".json", "_config.yaml")
    config_file = to_absolute_path(config_file)
    OmegaConf.save(cfg, config_file)
    print(f"Configuration saved to {config_file}")


@hydra.main(config_path="configs", config_name="tokenizer", version_base=None)
def main(cfg: DictConfig):
    train_tokenizer(cfg)


if __name__ == "__main__":
    main()
