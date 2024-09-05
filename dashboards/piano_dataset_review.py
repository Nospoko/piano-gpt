import json
from functools import partial

import torch
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from midi_tokenizers.no_loss_tokenizer import ExponentialTimeTokenizer

from artifacts import special_tokens
from data.tasks import Task, task_map
from data.piano_composer_dataset import PianoComposerDataset


@st.cache_data()
def load_hf_dataset(
    config: dict,
    dataset_name: str,
    dataset_split: str,
):
    dataset = load_dataset(
        f"midi_datasets/{dataset_name}",
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
        **config,
    )
    return dataset


def plot_target_mask(target_mask):
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.imshow(target_mask.unsqueeze(0), cmap="binary", aspect="auto")
    ax.set_yticks([])
    ax.set_xlabel("Token Position")
    ax.set_title("Target Mask")
    return fig


@st.cache_data()
def load_piano_dataset(
    tokenizer_parameters,
    config,
    dataset_name,
    dataset_split,
    sequence_length,
    notes_per_record,
    tasks,
    loss_masking,
    selected_composers,
    selected_title,
):
    tokenizer = ExponentialTimeTokenizer(**tokenizer_parameters)

    dataset = load_hf_dataset(
        config=config,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )

    # Filter dataset
    def filter_dataset(record):
        source_data = json.loads(record["source"])
        composer_match = source_data.get("composer") in selected_composers
        title_match = selected_title == "All" or source_data.get("title") == selected_title
        return composer_match and title_match

    filtered_dataset = dataset.filter(filter_dataset)

    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    dataset = Dataset.from_generator(partial(gen_from_iterable_dataset, filtered_dataset), features=filtered_dataset.features)
    piano_dataset = PianoComposerDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        notes_per_record=notes_per_record,
        tasks=tasks,
        loss_masking=loss_masking,
    )
    return piano_dataset, dataset


def main():
    st.title("Piano MIDI Dataset Review Dashboard")

    dataset_split = st.selectbox(label="Split", options=["train", "test", "validation"])

    with st.form(key="config_form"):
        col1, col2 = st.columns(2)
        with col1:
            base_dataset_name = st.text_input(
                label="Base Dataset Name",
                value="roszcz/maestro-sustain-v2",
            )
            notes_per_record = st.number_input(
                label="Notes per Record",
                min_value=1,
                value=256,
            )
        with col2:
            sequence_length = st.number_input(
                label="Sequence Length",
                min_value=1,
                value=2048,
                step=1024,
            )
            loss_masking = st.selectbox(
                label="Loss Calculation Style",
                options=["pretraining", "finetuning"],
            )

        tasks = st.multiselect(
            label="Prediction Tasks",
            options=task_map.keys(),
            default=["above_median_prediction"],
        )

        st.form_submit_button(label="Update Config")

    with st.form(key="tokenizer_form"):
        col1, col2 = st.columns(2)
        with col1:
            min_time_unit = st.number_input(
                label="Min Time Unit",
                value=0.01,
                step=0.01,
                format="%.3f",
            )
        with col2:
            n_velocity_bins = st.number_input(
                label="Velocity Bins",
                min_value=1,
                value=32,
                step=1,
            )

        st.form_submit_button(label="Update Tokenizer")

    config = {
        "base_dataset_name": base_dataset_name,
    }

    tokenizer_parameters = {
        "min_time_unit": min_time_unit,
        "n_velocity_bins": n_velocity_bins,
        "special_tokens": special_tokens,
    }
    dataset_name = "AugmentedDataset"
    dataset = load_hf_dataset(
        config=config,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )

    composers = set()
    titles = set()
    for record in dataset:
        source_data = json.loads(record["source"])
        if "composer" in source_data:
            composers.add(source_data["composer"])
        if "title" in source_data:
            titles.add(source_data["title"])
    composers = list(composers)
    titles = list(titles)

    selected_composers = st.multiselect("Filter by Composer", options=composers, default=["Johann Sebastian Bach"])
    selected_title = st.selectbox("Filter by Title", options=["All"] + titles, index=0)

    piano_dataset, dataset = load_piano_dataset(
        tokenizer_parameters=tokenizer_parameters,
        config=config,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        sequence_length=sequence_length,
        notes_per_record=notes_per_record,
        tasks=tasks,
        loss_masking=loss_masking,
        selected_composers=selected_composers,
        selected_title=selected_title,
    )

    st.write(f"Total Samples: {len(piano_dataset)}")

    with st.expander("Configuration"):
        st.json(config)

    with st.expander("Tokenizer Parameters"):
        st.json(tokenizer_parameters)

    idx = st.number_input(
        label="Sample ID",
        value=0,
        min_value=0,
        max_value=len(piano_dataset) - 1,
    )
    sample = piano_dataset[idx]
    record_id, start_point, task = piano_dataset._index_to_record(idx=idx)
    st.write(f"Record ID: {record_id}, Start Point: {start_point}, Task: {task}")

    with st.expander(label="Source Data"):
        st.json(sample["source"])

    st.write(f"Prediction Task: {sample['prediction_task']}")

    src_token_ids = sample["source_token_ids"]
    tgt_token_ids = sample["target_token_ids"]

    st.write("### Token Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Source Tokens:")
        st.write(src_token_ids)
    with col2:
        st.write("Target Tokens:")
        st.write(tgt_token_ids)

    st.write("### Target Mask Visualization")
    target_mask_fig = plot_target_mask(sample["target_mask"])
    st.pyplot(target_mask_fig)

    # Display some statistics about the target mask
    true_count = torch.sum(sample["target_mask"]).item()
    total_count = len(sample["target_mask"])
    true_percentage = (true_count / total_count) * 100

    st.write("Target Mask Statistics:")
    st.write(f"- Total tokens: {total_count}")
    st.write(f"- Tokens used for loss calculation: {true_count}")
    st.write(f"- Percentage of tokens used: {true_percentage:.2f}%")

    src_tokens = [piano_dataset.tokenizer.vocab[token_id] for token_id in src_token_ids]

    task_generator = Task.get_task(task_name=task)
    source_token = task_generator.source_token
    target_token = task_generator.target_token
    source_position = src_tokens.index(source_token)
    target_position = src_tokens.index(target_token)

    source_tokens = src_tokens[source_position:target_position]
    target_tokens = src_tokens[target_position:]

    source_notes = piano_dataset.tokenizer.untokenize(source_tokens)
    target_notes = piano_dataset.tokenizer.untokenize(target_tokens)
    record_id, start_point, _ = piano_dataset._index_to_record(idx)
    all_notes = ff.MidiPiece.from_huggingface(dataset[record_id])
    all_notes.df = all_notes.df.iloc[start_point : start_point + piano_dataset.notes_per_record]
    streamlit_pianoroll.from_fortepyan(piece=all_notes)
    st.write(pd.concat([source_notes, target_notes]).sort_values(by="start").reset_index(drop=True))
    st.write(piano_dataset.tokenizer.untokenize(piano_dataset.tokenizer.tokenize(all_notes.df)))

    source_piece = ff.MidiPiece(source_notes)
    target_piece = ff.MidiPiece(target_notes)

    st.write("### Visualizations")
    st.write("#### Combined View:")
    streamlit_pianoroll.from_fortepyan(piece=source_piece, secondary_piece=target_piece)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"#### {source_token}:")
        streamlit_pianoroll.from_fortepyan(piece=source_piece)
    with col2:
        st.write(f"#### {target_token}:")
        streamlit_pianoroll.from_fortepyan(piece=target_piece)


if __name__ == "__main__":
    main()
