from typing import Set, Tuple

import pandas as pd


def normalize_pitch_to_class(pitch: int) -> int:
    """Convert MIDI pitch (0-127) to pitch class (0-11)"""
    return pitch % 12


def calculate_f1(
    target_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    min_time_unit: float = 0.01,
    velocity_threshold: float = 30,
    use_pitch_class: bool = True,
) -> Tuple[float, dict]:
    """
    Calculate F1 score between target and generated MIDI-like note sequences.
    Only calculates at note boundary events and weights by duration.

    Parameters:
        target_df (pd.DataFrame) : DataFrame with columns: pitch, velocity, start, end
        generated_df (pd.DataFrame) : DataFrame with columns: pitch, velocity, start, end
        min_time_step (float) : Minimum time unit for duration calculations (in seconds)
        velocity_threshold (float) : Maximum allowed velocity difference for notes to be considered matching
        use_pitch_class (bool) : If True, normalize pitches to pitch classes (0-11), treating octaves as equivalent

    Returns:
        float : Duration-weighted average F1 score
        dict : Detailed metrics including precision, recall per event
    """
    # Get all unique time points where notes change (starts or ends)
    time_points = sorted(
        set(list(target_df["start"]) + list(target_df["end"]) + list(generated_df["start"]) + list(generated_df["end"]))
    )

    if not time_points:
        return 0.0, {"time_points": [], "precision": [], "recall": [], "f1": [], "durations": []}

    metrics = {
        "time_points": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "durations": [],  # store duration of each segment in minimum time units
    }

    def get_active_notes(df: pd.DataFrame, time_point: float) -> Set[tuple]:
        """
        Return set of (pitch, velocity) tuples active at given time point.
        If use_pitch_class is True, normalize pitches to 0-11.
        """
        mask = (df["start"] <= time_point) & (df["end"] >= time_point)
        return set(zip(df[mask]["pitch"], df[mask]["velocity"]))

    def find_matching_notes(target_notes: Set[tuple], generated_notes: Set[tuple], velocity_threshold: float) -> int:
        """
        Count matching notes considering velocity threshold.
        When using pitch classes, notes that are octaves apart can match.
        """
        matches = 0
        if use_pitch_class:
            target_notes = {(normalize_pitch_to_class(pitch), vel) for pitch, vel in target_notes}
            generated_notes = {(normalize_pitch_to_class(pitch), vel) for pitch, vel in generated_notes}

        for t_pitch, t_vel in target_notes:
            for g_pitch, g_vel in generated_notes:
                if t_pitch == g_pitch and abs(t_vel - g_vel) <= velocity_threshold:
                    matches += 1
                    break
        return matches

    # Calculate metrics for each segment between events
    for i in range(len(time_points) - 1):
        current_time = time_points[i]
        next_time = time_points[i + 1]
        duration = next_time - current_time

        # Convert duration to number of minimum time units
        duration_units = round(duration / min_time_unit)

        target_notes = get_active_notes(target_df, current_time + min_time_unit / 2)
        generated_notes = get_active_notes(generated_df, current_time + min_time_unit / 2)

        if len(target_notes) == 0 and len(generated_notes) == 0:
            continue

        true_positives = find_matching_notes(target_notes, generated_notes, velocity_threshold)

        precision = true_positives / len(generated_notes) if len(generated_notes) > 0 else 0
        recall = true_positives / len(target_notes) if len(target_notes) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics["time_points"].append(current_time)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)
        metrics["durations"].append(duration_units)

    # Calculate duration-weighted average F1 score
    total_duration = sum(metrics["durations"])
    weighted_f1 = (
        sum(f1 * dur for f1, dur in zip(metrics["f1"], metrics["durations"])) / total_duration if total_duration > 0 else 0.0
    )

    return weighted_f1, metrics


def create_example():
    # Create example with notes in different octaves
    target = pd.DataFrame({"pitch": [60, 72, 67], "velocity": [80, 80, 80], "start": [0.0, 0.5, 1.0], "end": [0.4, 0.9, 1.4]})

    generated = pd.DataFrame(
        {"pitch": [48, 72, 79], "velocity": [82, 78, 80], "start": [0.02, 0.48, 1.05], "end": [0.38, 0.95, 1.35]}
    )

    # Compare with and without pitch class normalization
    f1_with_pitch = calculate_f1(target, generated, use_pitch_class=False)
    f1_with_pitch_class = calculate_f1(target, generated, use_pitch_class=True)

    return {
        "with_pitch": f1_with_pitch,
        "with_pitch_class": f1_with_pitch_class,
    }


if __name__ == "__main__":
    create_example()
