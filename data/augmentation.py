import json
from os import cpu_count

import numpy as np
import pandas as pd
from datasets import Dataset


def change_speed(df: pd.DataFrame, factor: float = None) -> tuple[pd.DataFrame, float]:
    """
    Change the speed of the MIDI notes in the DataFrame by a given factor.
    If no factor is provided, a random factor within a specified range is used.

    Parameters:
        df (pd.DataFrame): DataFrame containing MIDI notes.
        factor (float, optional): Factor by which to change the speed. Defaults to None.

    Returns:
        tuple[pd.DataFrame, float]: The modified DataFrame and the factor used.
    """
    if not factor:
        slow = 0.8
        change_range = 0.4
        factor = slow + np.random.random() * change_range

    df.start /= factor
    df.end /= factor
    df.duration = df.end - df.start
    return df, factor


def check_pitch_shift(df: pd.DataFrame, pitch_shift: int) -> bool:
    PITCH_LOW = 21
    PITCH_HIGH = 108
    min_pitch = df.pitch.min()
    max_pitch = df.pitch.max()

    ok_low = min_pitch + pitch_shift >= PITCH_LOW
    ok_high = max_pitch + pitch_shift <= PITCH_HIGH

    is_ok = ok_low & ok_high

    return is_ok & pitch_shift != 0


def pitch_shift(df: pd.DataFrame, shift: int = 5) -> tuple[pd.DataFrame, int]:
    """
    Shift the pitch of the MIDI notes in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing MIDI notes.
        shift(int): Number of semitones to shift.

    Returns:
        tuple[pd.DataFrame, int]: The modified DataFrame and the shift amount used.
    """
    if not check_pitch_shift(df, shift):
        return None, None
    df.pitch += shift
    return df, shift


def apply_pitch_shift(
    batch: dict,
    max_pitch_shift: int = 5,
) -> dict:
    """
    Apply pitch shift augmentation to a batch of MIDI notes.

    Parameters:
        batch (dict): Batch of MIDI notes.
        max_pitch_shift: Maximal pitch shift in both directions.

    Returns:
        dict: Augmented batch of MIDI notes.
    """
    assert len(batch["notes"]) == 1
    source = json.loads(batch["source"][0])
    notes = batch["notes"][0]
    df = pd.DataFrame(notes)
    for shift in range(-max_pitch_shift, max_pitch_shift + 1):
        augmented, shift = pitch_shift(df=df.copy(), shift=shift)
        if augmented is not None:
            batch["notes"].append(augmented.to_dict(orient="series"))
            batch["source"].append(json.dumps(source | {"pitch_shift": shift}))

    return batch


def apply_speed_change(batch: dict, speed_change_factors: list[float]) -> dict:
    """
    Apply speed change augmentation to a batch of MIDI notes.

    Parameters:
        batch (dict): Batch of MIDI notes.
        speed_change_factors (list[float]): Speed change factors.

    Returns:
        dict: Augmented batch of MIDI notes.
    """
    assert len(batch["notes"]) == 1
    source = json.loads(batch["source"][0])
    notes = batch["notes"][0]
    df = pd.DataFrame(notes)
    for factor in speed_change_factors:
        augmented, factor = change_speed(df=df.copy(), factor=factor)
        batch["notes"].append(augmented.to_dict(orient="series"))
        batch["source"].append(json.dumps(source | {"change_speed_factor": factor}))

    return batch


def augment_dataset(
    dataset: Dataset,
    speed_change_factors: list[float] = None,
    max_pitch_shift: int = 5,
) -> Dataset:
    """
    Augment the dataset by applying pitch shift and speed change augmentations using all available CPUs.

    Parameters:
        dataset (Dataset): Dataset to augment.
        speed_change_factors (list[float]): Change speed factors.
        max_pitch_shift (int): Maximal pitch shift in both directions.

    Returns:
        Dataset: Augmented dataset.
    """
    if max_pitch_shift == 0 and (speed_change_factors is None or len(speed_change_factors) == 0):
        return dataset

    num_cpus = cpu_count()
    if num_cpus > 8:
        num_cpus -= 4  # Use all CPUs except 4
    else:
        num_cpus -= 1

    pitch_shift_args = {
        "max_pitch_shift": max_pitch_shift,
    }
    change_speed_args = {
        "speed_change_factors": speed_change_factors,
    }
    dataset = dataset.map(
        apply_pitch_shift,
        fn_kwargs=pitch_shift_args,
        batched=True,
        batch_size=1,
        num_proc=num_cpus,
    )
    dataset = dataset.map(
        apply_speed_change,
        fn_kwargs=change_speed_args,
        batched=True,
        batch_size=1,
        num_proc=num_cpus,
    )
    return dataset
