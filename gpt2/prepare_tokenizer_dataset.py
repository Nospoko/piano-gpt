import os
import json
import hashlib
from typing import Any, Dict

import hydra
import pandas as pd
from datasets import load_dataset
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig
from midi_tokenizers import AwesomeMidiTokenizer
from midi_tokenizers.midi_tokenizers_generation.base_tokenizer_generator import generate_tokenizer

from artifacts import special_tokens
from data.augmentation import augment_dataset


def hash_config(cfg: DictConfig) -> str:
    """Generate a hash from the configuration."""
    config_str = json.dumps(OmegaConf.to_container(cfg))
    hasher = hashlib.sha256()
    hasher.update(config_str.encode("utf-8"))
    return hasher.hexdigest()


def get_output_path(cfg: DictConfig) -> str:
    """Generate the output file path based on the configuration hash."""
    config_hash = hash_config(cfg)
    output_dir = to_absolute_path(os.path.join("tmp", "tokenizer_datasets"))

    os.makedirs(output_dir, exist_ok=True)
    return to_absolute_path(os.path.join(output_dir, f"{config_hash}.txt"))


def prepare_dataset_for_tokenizer_training(
    cfg: DictConfig,
):
    """
    Prepare and save a dataset for training an AwesomeMidiTokenizer.

    Parameters:
        cfg (DictConfig): The complete configuration object.
    """
    output_file = get_output_path(cfg)
    if os.path.exists(output_file):
        print(f"Dataset already created: {output_file}")
        return
    # Load and augment the dataset
    dataset = load_dataset(cfg.dataset.base_dataset_name, split="train")
    dataset = augment_dataset(dataset=dataset, **cfg.dataset.augmentation)

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

    def process_record(record: Dict[str, Any]) -> str:
        notes = pd.DataFrame(record["notes"])
        tokens = base_tokenizer.encode(notes=notes)
        awesome_tokens = tokenizer.base_ids_to_awesome_tokens(tokens)

        # Split tokens into chunks of less than max_token_length characters
        chunked_tokens = []
        for i in range(0, len(tokens), cfg.tokenizer.parameters.max_token_length):
            chunk = "".join(str(token) for token in awesome_tokens[i : i + cfg.tokenizer.parameters.max_token_length])
            chunked_tokens.append(chunk)

        # Join chunks with whitespace
        return " ".join(chunked_tokens) + "\n"

    # Process and write the dataset to the output file
    with open(file=output_file, mode="w", encoding="utf-8") as file:
        for record in dataset:
            result = process_record(record=record)
            file.write(result)

    print(f"Dataset prepared and saved to {output_file}")

    # Save the configuration alongside the dataset
    config_file = output_file.replace(".txt", "_config.yaml")
    OmegaConf.save(cfg, config_file)
    print(f"Configuration saved to {config_file}")


@hydra.main(config_path="configs", config_name="tokenizer_training", version_base=None)
def main(cfg: DictConfig):
    prepare_dataset_for_tokenizer_training(cfg)


if __name__ == "__main__":
    main()
