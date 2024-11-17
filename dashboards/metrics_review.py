from typing import Dict, List

import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

from gpt2.metrics import calculate_f1


def create_example_notes() -> List[Dict]:
    """Create example MIDI notes with pattern 60 80 64 80 68 in target"""
    return [
        {"pitch": 60, "velocity": 80, "start": 0.0, "end": 0.4},  # C4
        {"pitch": 64, "velocity": 80, "start": 0.5, "end": 0.9},  # E4
        {"pitch": 68, "velocity": 80, "start": 1.0, "end": 1.4},  # G#4
    ]


def create_shifted_notes() -> List[Dict]:
    """Create example MIDI notes with pattern 56 78 60 168 31 in generated"""
    return [
        {"pitch": 60, "velocity": 78, "start": 0.0, "end": 0.4},  # G#3
        {"pitch": 76, "velocity": 68, "start": 0.5, "end": 0.9},  # C4
        {"pitch": 56, "velocity": 82, "start": 1.0, "end": 1.4},  # G1
    ]


def main():
    st.title("MIDI Note Sequence F1 Score Analysis")
    st.write(
        """
    This dashboard demonstrates the calculation of F1 scores between two MIDI note sequences.
    It shows both exact pitch matching and pitch class matching (normalizing octaves).
    """
    )

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        min_time_unit = st.number_input("Minimum Time Unit (s)", value=0.01, step=0.001, format="%.3f")
        velocity_threshold = st.number_input("Velocity Threshold", value=30, step=1)

    # Create initial example data
    if "target_notes" not in st.session_state:
        st.session_state.target_notes = create_example_notes()
    if "generated_notes" not in st.session_state:
        st.session_state.generated_notes = create_shifted_notes()

    # Note editors
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Notes")
        target_df = pd.DataFrame(
            st.data_editor(
                st.session_state.target_notes,
                num_rows="dynamic",
                column_config={
                    "pitch": st.column_config.NumberColumn("Pitch", min_value=0, max_value=127),
                    "velocity": st.column_config.NumberColumn("Velocity", min_value=0, max_value=127),
                    "start": st.column_config.NumberColumn("Start Time", min_value=0.0, format="%.2f"),
                    "end": st.column_config.NumberColumn("End Time", min_value=0.0, format="%.2f"),
                },
            )
        )

    with col2:
        st.subheader("Generated Notes")
        generated_df = pd.DataFrame(
            st.data_editor(
                st.session_state.generated_notes,
                num_rows="dynamic",
                column_config={
                    "pitch": st.column_config.NumberColumn("Pitch", min_value=0, max_value=127),
                    "velocity": st.column_config.NumberColumn("Velocity", min_value=0, max_value=127),
                    "start": st.column_config.NumberColumn("Start Time", min_value=0.0, format="%.2f"),
                    "end": st.column_config.NumberColumn("End Time", min_value=0.0, format="%.2f"),
                },
            )
        )

    # Calculate F1 scores
    weighted_f1_exact, metrics_exact = calculate_f1(
        target_df=target_df,
        generated_df=generated_df,
        min_time_unit=min_time_unit,
        velocity_threshold=velocity_threshold,
        use_pitch_class=False,
    )

    weighted_f1_norm, metrics_norm = calculate_f1(
        target_df=target_df,
        generated_df=generated_df,
        min_time_unit=min_time_unit,
        velocity_threshold=velocity_threshold,
        use_pitch_class=True,
    )

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("F1 Score (Exact Pitch)", f"{weighted_f1_exact:.3f}")
    with col2:
        st.metric("F1 Score (Pitch Class)", f"{weighted_f1_norm:.3f}")

    # Visualizations
    st.subheader("Piano Roll Visualization")
    target_piece = ff.MidiPiece(target_df)
    generated_piece = ff.MidiPiece(generated_df)
    streamlit_pianoroll.from_fortepyan(piece=target_piece, secondary_piece=generated_piece)

    # Segment Analysis
    st.subheader("Detailed Analysis")
    tab1, tab2 = st.tabs(["Exact Pitch Matching", "Pitch Class Matching"])

    with tab1:
        df_exact = pd.DataFrame(
            {
                "Time": metrics_exact["time_points"],
                "Duration (ms)": [d * min_time_unit * 1000 for d in metrics_exact["durations"]],
                "F1": metrics_exact["f1"],
                "Precision": metrics_exact["precision"],
                "Recall": metrics_exact["recall"],
            }
        ).round(3)
        st.dataframe(df_exact)

    with tab2:
        df_norm = pd.DataFrame(
            {
                "Time": metrics_norm["time_points"],
                "Duration (ms)": [d * min_time_unit * 1000 for d in metrics_norm["durations"]],
                "F1": metrics_norm["f1"],
                "Precision": metrics_norm["precision"],
                "Recall": metrics_norm["recall"],
            }
        ).round(3)
        st.dataframe(df_norm)

    # Pitch mapping
    st.subheader("Pitch Classes")
    mapping_data = []
    for i, (t, g) in enumerate(zip(target_df.itertuples(), generated_df.itertuples())):
        mapping_data.append(
            {
                "Target": f"{t.pitch}",
                "Target Class": t.pitch % 12,
                "Generated": f"{g.pitch}",
                "Generated Class": g.pitch % 12,
                "Same Class": t.pitch % 12 == g.pitch % 12,
                "Velocity Diff": abs(t.velocity - g.velocity),
            }
        )
    st.dataframe(pd.DataFrame(mapping_data))


if __name__ == "__main__":
    main()
