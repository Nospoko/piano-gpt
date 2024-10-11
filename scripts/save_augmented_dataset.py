import os

from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()
HF_WRITE_TOKEN = os.environ["HF_WRITE_TOKEN"]


def prepare_piano_dataset(extra_datasets: list[str]):
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
    dataset = prepare_piano_dataset([])
    dataset = dataset["train"]
    dataset.push_to_hub("wmatejuk/maestro-augmented", token=HF_WRITE_TOKEN, private=True)
    dataset = prepare_piano_dataset(["roszcz/giant-midi-sustain-v2"])
    dataset = dataset["train"]
    dataset.push_to_hub("wmatejuk/giant-midi-augmented", token=HF_WRITE_TOKEN, private=True)
    dataset = prepare_piano_dataset(["roszcz/pianofor-ai-sustain-v2"])
    dataset = dataset["train"]
    dataset.push_to_hub("wmatejuk/pianofor-ai-augmented", token=HF_WRITE_TOKEN, private=True)
