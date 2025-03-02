import os

from dotenv import load_dotenv
from datasets import Dataset, load_dataset

load_dotenv()
HF_WRITE_TOKEN = os.environ["HF_WRITE_TOKEN"]


def prepare_augmented_dataset(
    augmentation_config: dict,
    target_dataset: str,
) -> Dataset:
    # *base_dataset* is NOT used at all within this script
    # - we probably should think about a dataset management design
    # that wouldn't require us to provide it here anyway
    extra_datasets = [target_dataset]
    dataset_config = {
        "base_dataset_name": "epr-labs/maestro-sustain-v2",
        "extra_datasets": extra_datasets,
        "pause_detection_threshold": 4,
        "augmentation": augmentation_config,
    }
    dataset_path = "./midi_datasets/AugmentedDataset"

    dataset = load_dataset(
        dataset_path,
        trust_remote_code=True,
        num_proc=64,
        **dataset_config,
    )
    return dataset


def augmentation_storage_process(
    dataset_name: str,
    augmentation_config: dict,
):
    target_dataset = f"epr-labs/{dataset_name}"
    dataset = prepare_augmented_dataset(
        target_dataset=target_dataset,
        augmentation_config=augmentation_config,
    )
    dataset = dataset["train"]

    new_dataset_name = dataset_name + "-augmented"
    new_dataset_path = f"epr-labs/{new_dataset_name}"

    print("Uploading:", new_dataset_path)
    dataset.push_to_hub(new_dataset_path, token=HF_WRITE_TOKEN, private=True)


if __name__ == "__main__":
    # Augmentation adds new samples, so speed=1, and shift=0 are not needed
    augmentation_config = {
        "speed_change_factors": [0.95, 0.975, 1.025, 1.05],
        "max_pitch_shift": 5,
    }
    # This implementation of the dataset builder uses the *base_dataset* only to create
    # validation splits, so in order to get the augmented train split, we have to pass i
    # as an "extra_dataset"
    dataset = prepare_augmented_dataset(
        target_dataset="epr-labs/maestro-sustain-v2",
        augmentation_config=augmentation_config,
    )
    dataset = dataset["train"]
    dataset.push_to_hub("epr-labs/maestro-augmented", token=HF_WRITE_TOKEN, private=True)

    dataset = prepare_augmented_dataset(
        target_dataset="epr-labs/giant-midi-sustain-v2",
        augmentation_config=augmentation_config,
    )
    dataset = dataset["train"]
    dataset.push_to_hub("epr-labs/giant-midi-augmented", token=HF_WRITE_TOKEN, private=True)

    dataset = prepare_augmented_dataset(
        target_dataset="epr-labs/pianofor-ai-sustain-v2",
        augmentation_config=augmentation_config,
    )
    dataset = dataset["train"]
    dataset.push_to_hub("epr-labs/pianofor-ai-augmented", token=HF_WRITE_TOKEN, private=True)

    dataset = prepare_augmented_dataset(
        target_dataset="epr-labs/imslp-midi-v2",
        augmentation_config=augmentation_config,
    )
    dataset = dataset["train"]
    dataset.push_to_hub("epr-labs/imslp-augmented", token=HF_WRITE_TOKEN, private=True)

    dataset = prepare_augmented_dataset(
        target_dataset="epr-labs/pijamia-midi-v1",
        augmentation_config=augmentation_config,
    )
    dataset = dataset["train"]
    dataset.push_to_hub("epr-labs/pijamia-midi-v1-augmented", token=HF_WRITE_TOKEN, private=True)

    dataset_names = [
        "vgmidi",
        "jsb-chorales",
        "piano-midi-de",
        "atepp-1.1-sustain-v2",
        "music-net",
        "piast-midi",
        "pianofor-ai-base-v3",
    ]
    for dataset_name in dataset_names:
        print("Augmenting:", dataset_name)
        augmentation_storage_process(
            dataset_name=dataset_name,
            augmentation_config=augmentation_config,
        )
