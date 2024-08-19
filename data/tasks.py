from abc import ABC, abstractmethod
from typing import Dict, Type, Tuple

import numpy as np
import pandas as pd


class Task(ABC):
    def __init__(self, source_token: str, target_token: str):
        self.source_token = source_token
        self.target_token = target_token

    @abstractmethod
    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @classmethod
    def get_task(cls, task_name: str) -> "Task":
        return task_map[task_name]()


class AboveMedianPrediction(Task):
    def __init__(self):
        super().__init__("<HIGH_FROM_MEDIAN>", "<LOW_FROM_MEDIAN>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        median = notes.pitch.median()
        source_notes = notes[notes.pitch < median]
        target_notes = notes[notes.pitch >= median]
        return source_notes, target_notes


class AboveLowQuartilePrediction(Task):
    def __init__(self):
        super().__init__("<ABOVE_LOW_QUARTILE>", "<BELOW_LOW_QUARTILE>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q1 = notes.pitch.quantile(0.25)
        source_notes = notes[notes.pitch < q1]
        target_notes = notes[notes.pitch >= q1]
        return source_notes, target_notes


class AboveHighQuartilePrediction(Task):
    def __init__(self):
        super().__init__("<ABOVE_HIGH_QUARTILE>", "<BELOW_HIGH_QUARTILE>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q3 = notes.pitch.quantile(0.75)
        source_notes = notes[notes.pitch < q3]
        target_notes = notes[notes.pitch >= q3]
        return source_notes, target_notes


class BelowLowQuartilePrediction(Task):
    def __init__(self):
        super().__init__("<BELOW_LOW_QUARTILE>", "<ABOVE_LOW_QUARTILE>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q1 = notes.pitch.quantile(0.25)
        source_notes = notes[notes.pitch >= q1]
        target_notes = notes[notes.pitch < q1]
        return source_notes, target_notes


class BelowHighQuartilePrediction(Task):
    def __init__(self):
        super().__init__("<BELOW_HIGH_QUARTILE>", "<ABOVE_HIGH_QUARTILE>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q3 = notes.pitch.quantile(0.75)
        source_notes = notes[notes.pitch >= q3]
        target_notes = notes[notes.pitch < q3]
        return source_notes, target_notes


class BelowMedianPrediction(Task):
    def __init__(self):
        super().__init__("<LOW_FROM_MEDIAN>", "<HIGH_FROM_MEDIAN>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        median = notes.pitch.median()
        source_notes = notes[notes.pitch >= median]
        target_notes = notes[notes.pitch < median]
        return source_notes, target_notes


class MiddleQuartilesPrediction(Task):
    def __init__(self):
        super().__init__("<MIDDLE_QUARTILES>", "<EXTREME_QUARTILES>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q1 = notes.pitch.quantile(0.25)
        q3 = notes.pitch.quantile(0.75)
        source_notes = notes[(notes.pitch < q1) | (notes.pitch >= q3)]
        target_notes = notes[(notes.pitch >= q1) & (notes.pitch < q3)]
        return source_notes, target_notes


class ExtremeQuartilesPrediction(Task):
    def __init__(self):
        super().__init__("<EXTREME_QUARTILES>", "<MIDDLE_QUARTILES>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q1 = notes.pitch.quantile(0.25)
        q3 = notes.pitch.quantile(0.75)
        target_notes = notes[(notes.pitch < q1) | (notes.pitch >= q3)]
        source_notes = notes[(notes.pitch >= q1) & (notes.pitch < q3)]
        return source_notes, target_notes


class LoudPrediction(Task):
    def __init__(self):
        super().__init__("<LOUD>", "<SOFT>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        median = notes.velocity.median()
        soft_notes = notes[notes.velocity < median]
        loud_notes = notes[notes.velocity >= median]
        return soft_notes, loud_notes


class VerySoftPrediction(Task):
    def __init__(self):
        super().__init__("<ABOVE_VERY_SOFT>", "<VERY_SOFT>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q1 = notes.velocity.quantile(0.25)
        source_notes = notes[notes.velocity >= q1]
        target_notes = notes[notes.velocity < q1]
        return source_notes, target_notes


class VeryLoudPrediction(Task):
    def __init__(self):
        super().__init__("<VERY_LOUD>", "<BELOW_VERY_LOUD>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q3 = notes.velocity.quantile(0.75)
        source_notes = notes[notes.velocity < q3]
        target_notes = notes[notes.velocity >= q3]
        return source_notes, target_notes


class SoftPrediction(Task):
    def __init__(self):
        super().__init__("<SOFT>", "<LOUD>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        median = notes.velocity.median()
        source_notes = notes[notes.velocity < median]
        target_notes = notes[notes.velocity >= median]
        return source_notes, target_notes


class ModerateVelocityPrediction(Task):
    def __init__(self):
        super().__init__("<MODERATE_VOLUME>", "<EXTREME_VOLUME>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q1 = notes.velocity.quantile(0.25)
        q3 = notes.velocity.quantile(0.75)
        source_notes = notes[(notes.velocity < q1) | (notes.velocity >= q3)]
        target_notes = notes[(notes.velocity >= q1) & (notes.velocity < q3)]
        return source_notes, target_notes


class ExtremeVelocityPrediction(Task):
    def __init__(self):
        super().__init__("<EXTREME_VOLUME>", "<MODERATE_VOLUME>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        q1 = notes.velocity.quantile(0.25)
        q3 = notes.velocity.quantile(0.75)
        target_notes = notes[(notes.velocity < q1) | (notes.velocity >= q3)]
        source_notes = notes[(notes.velocity >= q1) & (notes.velocity < q3)]
        return source_notes, target_notes


class VelocityDenoising(Task):
    def __init__(self):
        super().__init__("<CLEAN>", "<NOISY_VOLUME>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        noisy_notes = self._add_noise_to_notes(
            notes,
        )
        return noisy_notes, notes

    @staticmethod
    def _add_noise_to_notes(
        notes: pd.DataFrame,
        max_velocity_change: int = 30,
    ) -> pd.DataFrame:
        noisy_notes = notes.copy()
        noise = np.random.normal(0, max_velocity_change, len(noisy_notes))
        noisy_notes["velocity"] += noise.astype(int)
        noisy_notes["velocity"] = noisy_notes["velocity"].clip(0, 127)
        return noisy_notes


class PitchDenoising(Task):
    def __init__(self):
        super().__init__("<CLEAN>", "<NOISY_PITCH>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        noisy_notes = self._add_noise_to_notes(
            notes,
            max_pitch_shift=5,
        )
        return noisy_notes, notes

    @staticmethod
    def _add_noise_to_notes(
        notes: pd.DataFrame,
        max_pitch_shift: int,
    ) -> pd.DataFrame:
        noisy_notes = notes.copy()
        noise = np.random.normal(0, max_pitch_shift, len(noisy_notes))
        noisy_notes["pitch"] += noise.astype(int)
        noisy_notes["pitch"] = noisy_notes["pitch"].clip(21, 108)
        return noisy_notes


class StartTimeDenoising(Task):
    def __init__(self):
        super().__init__("<CLEAN>", "<NOISY_START_TIME>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        noisy_notes = self._add_noise_to_notes(
            notes,
            max_time_change=0.1,
        )
        return noisy_notes, notes

    @staticmethod
    def _add_noise_to_notes(notes: pd.DataFrame, max_time_change: float) -> pd.DataFrame:
        noisy_notes = notes.copy()
        noise = np.random.normal(0, max_time_change, len(noisy_notes))
        noisy_notes["start"] += noise
        noisy_notes["start"] = noisy_notes["start"].clip(0)
        noisy_notes["end"] = noisy_notes["start"] + noisy_notes["duration"]
        return noisy_notes


class TimeDenoising(Task):
    def __init__(self):
        super().__init__("<CLEAN>", "<NOISY_TIME>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        noisy_notes = self._add_noise_to_notes(
            notes,
            max_time_change=0.1,
        )
        return noisy_notes, notes

    @staticmethod
    def _add_noise_to_notes(notes: pd.DataFrame, max_time_change: float) -> pd.DataFrame:
        noisy_notes = notes.copy()

        start_noise = np.random.normal(0, max_time_change, len(noisy_notes))
        duration_noise = np.random.normal(0, max_time_change, len(noisy_notes))

        noisy_notes["start"] += start_noise
        noisy_notes["start"] = noisy_notes["start"].clip(0)
        noisy_notes["duration"] += duration_noise
        noisy_notes["duration"] = noisy_notes["duration"].clip(0)
        noisy_notes["end"] = noisy_notes["start"] + noisy_notes["duration"]
        return noisy_notes


class ComprehensiveDenoising(Task):
    def __init__(self):
        super().__init__("<CLEAN>", "<NOISY>")

    def generate(self, notes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        noisy_notes = self._add_comprehensive_noise(
            notes,
            max_pitch_shift=5,
            max_time_change=0.1,
            max_velocity_change=30,
        )
        return noisy_notes, notes

    @staticmethod
    def _add_comprehensive_noise(
        notes: pd.DataFrame,
        max_pitch_shift: int,
        max_time_change: float,
        max_velocity_change: int,
    ) -> pd.DataFrame:
        noisy_notes = notes.copy()

        # Add noise to velocity
        velocity_noise = np.random.normal(0, max_velocity_change, len(noisy_notes))
        noisy_notes["velocity"] += velocity_noise.astype(int)
        noisy_notes["velocity"] = noisy_notes["velocity"].clip(0, 127)

        # Add noise to pitch
        pitch_noise = np.random.normal(0, max_pitch_shift, len(noisy_notes))
        noisy_notes["pitch"] += pitch_noise.astype(int)
        noisy_notes["pitch"] = noisy_notes["pitch"].clip(21, 108)

        # Add noise to time (start and duration)
        start_noise = np.random.normal(0, max_time_change, len(noisy_notes))
        duration_noise = np.random.normal(0, max_time_change, len(noisy_notes))

        noisy_notes["start"] += start_noise
        noisy_notes["start"] = noisy_notes["start"].clip(0)
        noisy_notes["duration"] += duration_noise
        noisy_notes["duration"] = noisy_notes["duration"].clip(0)
        noisy_notes["end"] = noisy_notes["start"] + noisy_notes["duration"]

        return noisy_notes


task_map: Dict[str, Type[Task]] = {
    "above_median_prediction": AboveMedianPrediction,
    "above_low_quartile_prediction": AboveLowQuartilePrediction,
    "above_high_quartile_prediction": AboveHighQuartilePrediction,
    "below_low_quartile_prediction": BelowLowQuartilePrediction,
    "below_high_quartile_prediction": BelowHighQuartilePrediction,
    "below_median_prediction": BelowMedianPrediction,
    "middle_quartiles_prediction": MiddleQuartilesPrediction,
    "extreme_quartiles_prediction": ExtremeQuartilesPrediction,
    "loud_prediction": LoudPrediction,
    "very_soft_prediction": VerySoftPrediction,
    "very_loud_prediction": VeryLoudPrediction,
    "soft_prediction": SoftPrediction,
    "moderate_velocity_prediction": ModerateVelocityPrediction,
    "extreme_velocity_prediction": ExtremeVelocityPrediction,
    "velocity_denoising": VelocityDenoising,
    "pitch_denoising": PitchDenoising,
    "start_time_denoising": StartTimeDenoising,
    "time_denoising": TimeDenoising,
    "comprehensive_denoising": ComprehensiveDenoising,
}


all_tasks = [
    # Dynamically calculated pitch tasks
    "high_median_prediction",
    "above_median_prediction",
    "low_median_prediction",
    "above_low_quartile_prediction",
    "above_high_quartile_prediction",
    "below_low_quartile_prediction",
    "below_high_quartile_prediction",
    "middle_quartiles_prediction",
    "extreme_quartiles_prediction",
    # Velocity tasks
    "loud_prediction",
    "very_soft_prediction",
    "very_loud_prediction",
    "soft_prediction",
    "moderate_velocity_prediction",
    "extreme_velocity_precition",
    # Denoising tasks
    "velocity_denoising",
    "pitch_denoising",
    "start_time_denoising",
    "time_denoising",
    "comprehensive_denoising",
]
