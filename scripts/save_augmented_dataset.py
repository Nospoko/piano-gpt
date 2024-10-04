from datasets import load_dataset


def prepare_piano_dataset():
    dataset_config = {
        "base_dataset_name": "roszcz/maestro-sustain-v2",
        "extra_datasets": [
            "roszcz/giant-midi-sustain-v2",
            "roszcz/pianofor-ai-sustain-v2",
            "roszcz/imslp-midi-v1",
            "roszcz/pijamia-midi-v1",
            "roszcz/lakh-lmd-full",
        ],
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
    dataset = prepare_piano_dataset()
    dataset = dataset["train"]
    dataset.push_to_hub("wmatejuk/colossal-augmented-dataset")
