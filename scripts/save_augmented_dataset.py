import os

from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()
HF_WRITE_TOKEN = os.environ["HF_WRITE_TOKEN"]


def prepare_piano_dataset(extra_datasets: list[str]):
    # *base_dataset* is NOT used at all within this script
    # - we probably should think about a dataset management design
    # that wouldn't require us to provide it here anyway
    dataset_config = {
        "base_dataset_name": "roszcz/maestro-sustain-v2",
        "extra_datasets": extra_datasets,
        "pause_detection_threshold": 4,
        "augmentation": {
            "speed_change_factors": [0.95, 0.975, 1.025, 1.05],
            "max_pitch_shift": 5,
        },
    }
    dataset_path = "./midi_datasets/AugmentedDataset"

    dataset = load_dataset(
        dataset_path,
        trust_remote_code=True,
        num_proc=64,
        **dataset_config,
    )
    return dataset


if __name__ == "__main__":
    # This implementation of the dataset builder uses the *base_dataset* only to create
    # validation splits, so in order to get the augmented train split, we have to pass i
    # as an "extra_dataset"
    dataset = prepare_piano_dataset(["roszcz/maestro-sustain-v2"])
    dataset = dataset["train"]
    dataset.push_to_hub("epr-labs/maestro-augmented", token=HF_WRITE_TOKEN, private=True)

    dataset = prepare_piano_dataset(["roszcz/giant-midi-sustain-v2"])
    dataset = dataset["train"]
    dataset.push_to_hub("epr-labs/giant-midi-augmented", token=HF_WRITE_TOKEN, private=True)

    dataset = prepare_piano_dataset(["roszcz/pianofor-ai-sustain-v2"])
    dataset = dataset["train"]
    dataset.push_to_hub("epr-labs/pianofor-ai-augmented", token=HF_WRITE_TOKEN, private=True)

    dataset = prepare_piano_dataset(["roszcz/imslp-midi-v1"])
    dataset = dataset["train"]
    dataset.push_to_hub("epr-labs/imslp-augmented", token=HF_WRITE_TOKEN, private=True)
