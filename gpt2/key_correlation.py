import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class SpiralPoint:
    x: float
    y: float
    z: float

    def __add__(
        self,
        other: "SpiralPoint",
    ) -> "SpiralPoint":
        return SpiralPoint(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __mul__(
        self,
        scalar: float,
    ) -> "SpiralPoint":
        return SpiralPoint(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
        )

    def distance(
        self,
        other: "SpiralPoint",
    ) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2,
        )


class SpiralArray:
    """
    Implementation of Chew's Spiral Array model for key detection.
    Reference: Chew, E. (2000). Towards a Mathematical Model of Tonality.
    """

    def __init__(self):
        # Parameters for the spiral array
        self.h = 1.0  # height of one rotation
        self.r = 1.0  # radius
        self.p = 4.0  # pitch constant (perfect fifth = 7 semitones)

        # Initialize pitch class representations on the spiral
        self.pitch_classes = self._generate_pitch_spiral()

        # Generate major and minor key representations
        self.major_keys = self._generate_major_keys()
        self.minor_keys = self._generate_minor_keys()

        # Map of key indices to key names
        self.key_names = self._generate_key_names()

    def _generate_pitch_spiral(self) -> Dict[int, SpiralPoint]:
        """Generate pitch class positions on the spiral"""
        pitch_points = {}
        for pc in range(12):
            # Convert pitch class to position on spiral
            angle = 2 * math.pi * pc / 12
            x = self.r * math.cos(angle)
            y = self.r * math.sin(angle)
            z = self.h * pc / 12
            pitch_points[pc] = SpiralPoint(x, y, z)
        return pitch_points

    def _generate_major_keys(self) -> List[SpiralPoint]:
        """Generate major key representations"""
        major_keys = []
        # Major key template: root(4), major third(3), perfect fifth(3)
        for root in range(12):
            third = (root + 4) % 12
            fifth = (root + 7) % 12

            # Weighted combination of pitch classes
            key_point = self.pitch_classes[root] * 0.4 + self.pitch_classes[third] * 0.3 + self.pitch_classes[fifth] * 0.3
            major_keys.append(key_point)
        return major_keys

    def _generate_minor_keys(self) -> List[SpiralPoint]:
        """Generate minor key representations"""
        minor_keys = []
        # Minor key template: root(4), minor third(3), perfect fifth(3)
        for root in range(12):
            third = (root + 3) % 12  # Minor third
            fifth = (root + 7) % 12

            key_point = self.pitch_classes[root] * 0.4 + self.pitch_classes[third] * 0.3 + self.pitch_classes[fifth] * 0.3
            minor_keys.append(key_point)
        return minor_keys

    def _generate_key_names(self) -> Dict[int, str]:
        """Generate mapping of key indices to key names"""
        pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_names = {}

        # Major keys (0-11)
        for i in range(12):
            key_names[i] = f"{pitch_names[i]} major"

        # Minor keys (12-23)
        for i in range(12):
            key_names[i + 12] = f"{pitch_names[i]} minor"

        return key_names

    def get_center_of_effect(
        self,
        pitches: List[int],
        durations: List[float],
        velocities: List[float],
    ) -> SpiralPoint:
        """
        Calculate the center of effect for a set of pitches with durations and velocities.
        """
        if not pitches:
            return SpiralPoint(0, 0, 0)

        # Normalize weights (duration * velocity)
        weights = [d * v for d, v in zip(durations, velocities)]
        total_weight = sum(weights)
        if total_weight == 0:
            return SpiralPoint(0, 0, 0)

        normalized_weights = [w / total_weight for w in weights]

        # Calculate weighted average position
        center = SpiralPoint(0, 0, 0)
        for pitch, weight in zip(pitches, normalized_weights):
            pitch_class = pitch % 12
            center = center + (self.pitch_classes[pitch_class] * weight)

        return center

    def get_key(
        self,
        center: SpiralPoint,
        return_distribution: bool = False,
    ) -> Union[int, Tuple[int, np.ndarray]]:
        """
        Determine key from center of effect.
        Returns key index (0-23) where 0-11 are major keys and 12-23 are minor keys.
        If return_distribution=True, also returns probability distribution over all keys.
        """
        # Calculate distances to all key representations
        major_distances = [center.distance(k) for k in self.major_keys]
        minor_distances = [center.distance(k) for k in self.minor_keys]
        all_distances = major_distances + minor_distances

        # Convert distances to probabilities using softmax
        max_dist = max(all_distances)
        exp_distances = [math.exp(-(d - max_dist)) for d in all_distances]
        total_exp = sum(exp_distances)
        probabilities = np.array([d / total_exp for d in exp_distances])

        # Find most likely key
        key_index = np.argmin(all_distances)

        if return_distribution:
            return key_index, probabilities
        return key_index


def calculate_key_correlation(
    target_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    segment_duration: float = 0.125,  # 125ms segments as suggested in paper
    use_weighted: bool = True,
) -> Tuple[float, Dict]:
    """
    Calculate correlation coefficient between target and generated MIDI sequences
    using Spiral Array key detection algorithm.

    Parameters:
        target_df (pd.DataFrame): DataFrame with columns: pitch, velocity, start, end
        generated_df (pd.DataFrame): DataFrame with columns: pitch, velocity, start, end
        segment_duration (float): Duration of each segment in seconds for key analysis
        use_weighted (bool): If True, weight pitch contributions by note duration and velocity

    Returns:
        float: Correlation coefficient (-1 to 1)
        dict: Additional metrics including key distributions
    """

    def get_piece_duration(df: pd.DataFrame) -> float:
        return max(df["end"].max(), 0)

    def segment_piece(df: pd.DataFrame, duration: float) -> List[pd.DataFrame]:
        """Split piece into equal duration segments"""
        segments = []
        current_time = 0

        while current_time < duration:
            segment_end = current_time + segment_duration
            # Get notes that overlap with this segment
            mask = (df["start"] < segment_end) & (df["end"] > current_time)
            segments.append(df[mask].copy())
            current_time = segment_end

        return segments

    def get_key_distribution(
        segments: List[pd.DataFrame],
        spiral: SpiralArray,
    ) -> np.ndarray:
        """Calculate key distribution using Spiral Array algorithm"""
        # Initialize distribution for all major/minor keys
        key_distribution = np.zeros(24)

        for segment_df in segments:
            if len(segment_df) == 0:
                continue

            # Prepare pitch data for the segment
            pitches = segment_df["pitch"].tolist()

            if use_weighted:
                # Calculate actual durations within segment
                durations = [min(row["end"], segment_duration) - max(row["start"], 0) for _, row in segment_df.iterrows()]
                velocities = [v / 127.0 for v in segment_df["velocity"]]
            else:
                durations = [1.0] * len(pitches)
                velocities = [1.0] * len(pitches)

            # Get center of effect
            center = spiral.get_center_of_effect(pitches, durations, velocities)

            # Get key probabilities
            _, key_probs = spiral.get_key(center, return_distribution=True)
            key_distribution += key_probs

        # Normalize distribution
        total = np.sum(key_distribution)
        if total > 0:
            key_distribution = key_distribution / total

        return key_distribution

    # Initialize Spiral Array
    spiral = SpiralArray()

    # Get total duration (use longer of the two pieces)
    total_duration = max(get_piece_duration(target_df), get_piece_duration(generated_df))

    # Segment both pieces
    target_segments = segment_piece(target_df, total_duration)
    generated_segments = segment_piece(generated_df, total_duration)

    # Get key distributions using Spiral Array
    target_dist = get_key_distribution(target_segments, spiral)
    generated_dist = get_key_distribution(generated_segments, spiral)

    # Calculate correlation coefficient
    correlation = np.corrcoef(target_dist, generated_dist)[0, 1]

    # Prepare detailed metrics
    metrics = {
        "target_distribution": target_dist,
        "generated_distribution": generated_dist,
        "num_segments": len(target_segments),
        "segment_duration": segment_duration,
        "key_names": spiral.key_names,
        # Add top 3 keys for each piece
        "target_top_keys": [spiral.key_names[i] for i in np.argsort(-target_dist)[:3]],
        "generated_top_keys": [spiral.key_names[i] for i in np.argsort(-generated_dist)[:3]],
    }

    return correlation, metrics


def create_correlation_example():
    """Create example comparing two similar MIDI sequences"""
    # Create example with C major scale in different octaves
    target = pd.DataFrame(
        {
            "pitch": [60, 62, 64, 65, 67, 69, 71, 72],  # C4 scale
            "velocity": [80] * 8,
            "start": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "end": [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9],
        }
    )

    # Similar sequence but with some variations
    generated = pd.DataFrame(
        {
            "pitch": [48, 50, 52, 53, 55, 57, 59, 60],  # C3 scale
            "velocity": [75] * 8,
            "start": [0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6],
            "end": [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9],
        }
    )

    correlation, metrics = calculate_key_correlation(target, generated)

    return {"correlation": correlation, "metrics": metrics}


if __name__ == "__main__":
    example_results = create_correlation_example()
    print(f"Correlation coefficient: {example_results['correlation']:.3f}")
    print("\nTop keys in target:", ", ".join(example_results["metrics"]["target_top_keys"]))
    print("Top keys in generated:", ", ".join(example_results["metrics"]["generated_top_keys"]))
