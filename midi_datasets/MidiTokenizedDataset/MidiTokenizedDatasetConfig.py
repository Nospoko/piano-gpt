import datasets
from datasets import BuilderConfig


class MidiTokenizedDatasetConfig(BuilderConfig):
    """
    Configuration class for creating a sub-sequence MIDI dataset.

    Attributes:
        base_dataset_name (str): Name of the dataset used for test and validation – right now we only
            use maestro for this, as it already has well defined splits. Train split is ignored for this dataset.
        extra_datasets (list[str]): List of datasets used for training, splits other than "train" are ignored.
        notes_per_record (int): Length of the sequences.
        step (int): Step size between sequences.
        pause_detection_threshold (int): Threshold for detecting pauses.
        augmentation (dict): Parameters for augmentation
    """

    def __init__(
        self,
        tokenizer_dict: dict,
        base_dataset_name: str = "roszcz/maestro-sustain-v2",
        extra_datasets: list[str] = [],
        pause_detection_threshold: int = 4,
        augmentation: dict = {
            "speed_change_factors": None,
            "max_pitch_shift": 0,
        },
        **kwargs,
    ):
        """
        Initialize the SubSequenceDatasetConfig.

        Parameters:
            base_dataset_name (str): Name of the base dataset.
            extra_datasets (list[str]): List of additional datasets.
            notes_per_record (int): Length of the sequences.
            step (int): Step size between sequences.
            pause_detection_threshold (int): Threshold for detecting pauses.
            augmentation (dict): Parameters for augmentation (max_pitch_shift, speed_change_factors))
            **kwargs: Additional keyword arguments.
        """
        # Initialize the version and other parameters
        super().__init__(version=datasets.Version("0.0.1"), **kwargs)

        # Assign the provided arguments to the class attributes
        self.base_dataset_name = base_dataset_name
        self.extra_datasets = extra_datasets
        self.tokenizer_dict = tokenizer_dict
        self.pause_detection_threshold = pause_detection_threshold
        self.augmentation = augmentation

    @property
    def builder_parameters(self):
        """
        Returns the builder parameters as a dictionary.

        Returns:
            dict: Builder parameters.
        """
        return {
            "base_dataset_name": self.base_dataset_name,
            "extra_datasets": self.extra_datasets,
            "tokenizer_cfg": self.tokenizer_cfg,
            "pause_detection_threshold": self.pause_detection_threshold,
            "augmentation": self.augmentation,
        }


augmentation_parameters = {
    "speed_change_factors": [0.95, 0.975, 1.05, 1.025],
    "max_pitch_shift": 5,
}
# List of configurations for different datasets for debugging
BUILDER_CONFIGS = [
    MidiTokenizedDatasetConfig(
        base_dataset_name="roszcz/maestro-sustain-v2",
        extra_datasets=[],
        tokenizer_dict={},
        pause_detection_threshold=4,
        name="basic-no-overlap",
    ),
]
