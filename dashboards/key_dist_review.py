import json
from functools import partial
from typing import List, Dict, Optional

import torch
import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

from midi_tokenizers import ExponentialTimeTokenizer



def select_part_dataset(midi_dataset: Dataset, key: str = "0") -> Dataset:
    """
    Allows the user to select a part of the dataset based on composer and title.

    Parameters:
        midi_dataset (Dataset): The MIDI dataset to select from.

    Returns:
        Dataset: The selected part of the dataset.
    """
    source_df = midi_dataset.to_pandas()
    source_df["source"] = source_df["source"].map(lambda source: yaml.safe_load(source))
    source_df["composer"] = [source["composer"] for source in source_df.source]
    source_df["title"] = [source["title"] for source in source_df.source]

    composers = source_df.composer.unique()
    selected_composer = st.selectbox(
        "Select composer",
        options=composers,
        index=3,
        key=f"composer_{key}",
    )

    ids = source_df.composer == selected_composer
    piece_titles = source_df[ids].title.unique()
    selected_title = st.selectbox("Select title", options=piece_titles, key=f"title_{key}",)

    ids = (source_df.composer == selected_composer) & (source_df.title == selected_title)
    part_df = source_df[ids]
    part_dataset = midi_dataset.select(part_df.index.values)

    return part_dataset


def dataset_configuration(key: str = "0"):
    st.header("Dataset Configuration")
    col1, col2 = st.columns(2)
    with col1:
        dataset_path = st.text_input(
            "Dataset Path",
            value="roszcz/maestro-sustain-v2",
            help="Enter the path to the dataset",
            key=f"d_path_{key}",
        )
    with col2:
        dataset_split = st.selectbox(
            "Dataset Split",
            options=["validation", "train", "test"],
            help="Choose the dataset split to use",
            key=f"split_{key}",
        )

    dataset = load_hf_dataset(
        dataset_path=dataset_path,
        dataset_split=dataset_split,
    )
    dataset = select_part_dataset(midi_dataset=dataset, key=key)

    st.success(f"Dataset loaded! Total records: {len(dataset)}")
    return dataset


@st.cache_data
def load_hf_dataset(dataset_path: str, dataset_split: str):
    dataset = load_dataset(
        dataset_path,
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
    )
    return dataset

def plot_key_distribution(distribution: np.ndarray, key_names: List[str], title: str = "") -> go.Figure:
    """Create a bar plot of key distribution."""
    fig = go.Figure()
    
    # Split into major and minor keys
    major_keys = key_names[:12]
    minor_keys = key_names[12:]
    major_dist = distribution[:12]
    minor_dist = distribution[12:]
    
    # Add major keys
    fig.add_trace(go.Bar(
        x=major_keys,
        y=major_dist,
        name='Major Keys',
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Add minor keys
    fig.add_trace(go.Bar(
        x=minor_keys,
        y=minor_dist,
        name='Minor Keys',
        marker_color='rgb(26, 118, 255)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_tickangle=-45,
        yaxis_title='Probability',
        barmode='group',
        showlegend=True
    )
    
    return fig

def plot_key_correlation_heatmap(
    target_dist: np.ndarray,
    generated_dist: np.ndarray,
    key_names: List[str]
) -> go.Figure:
    """Create a heatmap comparing two key distributions."""
    correlation_matrix = np.outer(target_dist, generated_dist)
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=key_names,
        y=key_names,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Key Distribution Correlation Heatmap',
        xaxis_tickangle=-45,
        yaxis_tickangle=0
    )
    
    return fig

def analyze_pieces(
    piece1_notes: pd.DataFrame,
    piece2_notes: pd.DataFrame,
    segment_duration: float = 0.125,
    use_weighted: bool = True
) -> Dict:
    """Analyze two pieces and return key distribution metrics"""
    from gpt2.key_correlation import calculate_key_correlation, SpiralArray
    
    correlation, metrics = calculate_key_correlation(
        target_df=piece1_notes,
        generated_df=piece2_notes,
        segment_duration=segment_duration,
        use_weighted=use_weighted
    )
    
    return {
        "correlation": correlation,
        "key_distributions": {
            "piece1": metrics["target_distribution"],
            "piece2": metrics["generated_distribution"]
        },
        "top_keys": {
            "piece1": metrics["target_top_keys"],
            "piece2": metrics["generated_top_keys"]
        },
        "key_names": metrics["key_names"]
    }

def main():
    st.title("Key Distribution Analysis Dashboard")    
    
    col1, col2 = st.columns(2)
    with col1:
        dataset_1 = dataset_configuration(key="0")
        
    with col2:
        dataset_2 = dataset_configuration(key="1")
    
    # Find selected pieces in dataset
    piece1_record = None
    piece2_record = None
    
    piece1_record = dataset_1[0]
    piece2_record = dataset_2[0]
    
    if piece1_record and piece2_record:
        # Convert to fortepyan pieces
        piece1 = ff.MidiPiece.from_huggingface(piece1_record)
        piece2 = ff.MidiPiece.from_huggingface(piece2_record)
        
        # Analysis parameters
        st.header("Analysis Parameters")
        segment_duration = st.slider("Segment Duration (seconds)", 0.05, 0.5, 0.125, 0.025)
        use_weighted = st.checkbox("Use weighted key detection", value=True)
        
        # Visualization of pieces
        st.header("Piece Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            streamlit_pianoroll.from_fortepyan(piece=piece1)
            
        with col2:
            streamlit_pianoroll.from_fortepyan(piece=piece2)
        
        # Analyze pieces
        with st.spinner("Analyzing key distributions..."):
            analysis = analyze_pieces(
                piece1_notes=piece1.df,
                piece2_notes=piece2.df,
                segment_duration=segment_duration,
                use_weighted=use_weighted
            )
        
        # Display results
        st.header("Analysis Results")
        st.metric("Correlation Coefficient", f"{analysis['correlation']:.3f}")
        
        # Key distributions
        st.subheader("Key Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Top keys in first piece:", ", ".join(analysis["top_keys"]["piece1"]))
            fig1 = plot_key_distribution(
                analysis["key_distributions"]["piece1"],
                list(analysis["key_names"].values()),
                f"Key Distribution 1"
            )
            st.plotly_chart(fig1)
            
        with col2:
            st.write("Top keys in second piece:", ", ".join(analysis["top_keys"]["piece2"]))
            fig2 = plot_key_distribution(
                analysis["key_distributions"]["piece2"],
                list(analysis["key_names"].values()),
                f"Key Distribution 2"
            )
            st.plotly_chart(fig2)
        
        # Correlation heatmap
        st.subheader("Key Distribution Correlation")
        fig3 = plot_key_correlation_heatmap(
            analysis["key_distributions"]["piece1"],
            analysis["key_distributions"]["piece2"],
            list(analysis["key_names"].values())
        )
        st.plotly_chart(fig3)
        
    else:
        st.error("Could not find selected pieces in dataset")

if __name__ == "__main__":
    main()