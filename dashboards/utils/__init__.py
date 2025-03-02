from glob import glob

import torch
import streamlit as st


def device_model_selection():
    with st.sidebar:
        st.header("Model Configuration")
        devices = [f"cuda:{it}" for it in range(torch.cuda.device_count())] + ["cpu", "mps"]
        device = st.selectbox("Select Device", options=devices, help="Choose the device to run the model on")
        checkpoint_path = st.selectbox(
            "Select Checkpoint",
            options=glob("tmp/checkpoints/*.pt"),
            help="Choose the model checkpoint to use",
        )

    return device, checkpoint_path
