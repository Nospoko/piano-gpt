import io
import os
import json
import secrets
import zipfile
from glob import glob
from pathlib import Path

import torch
import requests
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from omegaconf import OmegaConf
from midi_tokenizers import ExponentialTimeTokenizer
from piano_dataset.piano_tasks import PianoTaskManager

from gpt2.model import GPT
from gpt2.data.musicality import MusicManager
from dashboards.utils.components import download_button


def main():
    devices = [f"cuda:{it}" for it in range(torch.cuda.device_count())]
    devices += ["cpu", "mps"]
    device = st.selectbox(
        label="Select Device",
        options=devices,
        help="Choose the device to run the model on",
    )

    # Get checkpoints starting with the latest
    checkpoint_paths = [f for f in Path("tmp/checkpoints/").iterdir() if f.is_file()]
    checkpoint_paths = sorted(
        checkpoint_paths,
        key=lambda f: f.stat().st_ctime,
        reverse=True,
    )

    checkpoint_path = st.selectbox(
        label="Select Checkpoint",
        options=checkpoint_paths,
        help="Choose the model checkpoint to use",
    )
    st.write("Checkpoint:", checkpoint_path)
    model_setup = load_cache_checkpoint(checkpoint_path, device=device)
    run_config = model_setup["run_config"]

    if run_config.model_task != "piano_task":
        st.write(
            (
                "This dashboard is designed for the PIANO Task generations,"
                "but your model was trained for something else"
            )
        )
        return

    with st.expander("model info"):
        st.json(OmegaConf.to_container(run_config), expanded=False)

        st.write("Training settings:")
        st.write(OmegaConf.to_container(run_config.training))

        st.write("Training stats:")
        st.write(model_setup["run_stats"])

    st.write("# Prompt")
    st.write(
        ("Upload a new prompt midi file, or select one from the list.\n" "Uploading will add a file to that list.")
    )

    uploaded_file = st.file_uploader(
        label="Choose a file",
        type=["mid", "midi"],
    )
    if uploaded_file is not None:
        savepath = f"tmp/prompts/{uploaded_file.name}"
        with open(savepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.write("File saved", savepath)

    with st.form("prompt selection"):
        # TODO: What would be a convenient way to manage prompts
        # for a user?
        prompt_options = glob("tmp/prompts/*.mid") + [None]
        if uploaded_file is not None:
            idx = prompt_options.index(savepath)
        else:
            idx = None
        prompt_path = st.selectbox(
            label="select prompt file",
            options=prompt_options,
            index=idx,
        )
        st.form_submit_button()

    if not prompt_path:
        st.write("Submit your prompt selection pls")
        return

    prompt_piece = ff.MidiPiece.from_file(prompt_path)

    # TODO Remove inplace operations from fortepyan
    prompt_piece.time_shift(-prompt_piece.df.start.min())

    st.write("### Prompt modification setup")
    with st.form("prompt setup"):
        speedup_factor = st.number_input(
            label="speedup factor",
            value=1.0,
            min_value=0.3,
            max_value=2.5,
        )

        start_note_idx = st.number_input(
            label="start note idx",
            value=0,
            min_value=0,
            max_value=prompt_piece.size - 5,
        )
        finish_note_idx = st.number_input(
            label="finish note idx",
            value=prompt_piece.size,
            min_value=2,
            max_value=prompt_piece.size,
        )
        pitch_shift = st.number_input(
            label="pitch shift",
            value=0,
            min_value=-12,
            max_value=12,
        )
        _ = st.form_submit_button()

    # Get just the file name, without extension
    prompt_name = os.path.splitext(os.path.basename(prompt_path))[0]
    prompt_setup = {
        "prompt_name": prompt_name,
        "pitch_shift": pitch_shift,
        "speedup_factor": speedup_factor,
        "start_note_idx": start_note_idx,
        "finish_note_idx": finish_note_idx,
    }

    prompt_piece = prompt_piece[start_note_idx:finish_note_idx]
    prompt_piece.df.pitch += pitch_shift

    streamlit_pianoroll.from_fortepyan(prompt_piece)
    st.write("Prompt notes:", prompt_piece.size)

    st.write("### Generation settings")
    with st.form("generation setup"):
        random_seed = st.number_input(
            label="random seed",
            value=137,
            max_value=100_000,
            min_value=0,
        )
        max_new_tokens = st.number_input(
            label="max new tokens",
            min_value=64,
            max_value=4096,
            value=128,
        )
        temperature = st.number_input(
            label="temperature",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.05,
        )

        cols = st.columns(2)
        top_k = cols[0].number_input(
            label="top k",
            value=5,
            min_value=1,
        )
        use_top_k = cols[1].checkbox(
            label="use top k",
            value=False,
        )
        if not use_top_k:
            top_k = None

        music_manager = MusicManager(
            max_n_notes=run_config.training.max_notes_per_record,
        )
        dataset_tokens = music_manager.dataset_tokens
        dataset_token = st.selectbox(
            label="Select a dataset token:",
            options=dataset_tokens,
            help="Choose from available special tokens to add to your prompt",
        )

        n_target_notes = st.number_input(
            label="N target notes",
            min_value=0,
            max_value=music_manager.max_n_notes,
            value=music_manager.max_n_notes // 3,
        )
        n_notes_token = music_manager.get_n_notes_token(n_target_notes)

        piano_task_manager: PianoTaskManager = model_setup["piano_task_manager"]

        # Using prompt path as a key to force st to restart this form whenever
        # I change the checkpoint of the prompt
        piano_task_name = st.selectbox(
            label="Select PIANO task:",
            options=piano_task_manager.list_task_names(),
            help="Choose from tasks used during training",
            key=prompt_path + str(checkpoint_path),
            index=None,
        )

        pianoroll_apikey = st.text_input(
            label="pianoroll apikey",
            type="password",
        )
        _ = st.form_submit_button()

    if not piano_task_name:
        st.write("Select a PIANO task and submit generation arguments")
        return

    piano_task = piano_task_manager.get_task(piano_task_name)

    prompt_notes_df: pd.DataFrame = prompt_piece.df
    prompt_notes_df.start /= speedup_factor
    prompt_notes_df.end /= speedup_factor

    model = model_setup["model"]
    tokenizer = model_setup["tokenizer"]

    st.write("# Generations")

    composer_tokens = ["<BACH>", "<MOZART>", "<CHOPIN>", "<UNKNOWN_COMPOSER>"]
    for composer_token in composer_tokens:
        pre_input_tokens = [dataset_token, composer_token, n_notes_token] + piano_task.prefix_tokens

        st.write("Pre-input tokens:", pre_input_tokens)

        # Generator randomness comes from torch.multinomial, so we can make it
        # fully deterministic by setting global torch random seed
        for it in range(2):
            # TODO Try to de-indent this loop
            local_seed = random_seed + it * 1000
            generation_setup = {
                "seed": local_seed,
                "temperature": temperature,
                "pre_input_tokens": pre_input_tokens,
                "piano_task": piano_task.name,
                "top_k": top_k,
                "model_id": os.path.basename(checkpoint_path),
            }
            st.write(generation_setup)
            st.write("".join(pre_input_tokens))

            # This acts as a caching key
            generation_info = {
                "generation_setup": generation_setup,
                "prompt_setup": prompt_setup,
            }

            generated_notes_df = cache_generation(
                prompt_notes_df=prompt_notes_df,
                seed=local_seed,
                _model=model,
                _tokenizer=tokenizer,
                device=device,
                top_k=top_k,
                pre_input_tokens=pre_input_tokens,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                generation_info=generation_info,
            )

            prompt_piece = ff.MidiPiece(prompt_notes_df)
            generated_piece = ff.MidiPiece(generated_notes_df)

            streamlit_pianoroll.from_fortepyan(prompt_piece, generated_piece)
            st.write("Generated notes:", generated_piece.size)

            unique_id = secrets.token_hex(10)
            if pianoroll_apikey:
                # TODO: Add title and description control
                make_proll_post = st.button(
                    label="post to pianoroll.io",
                    key=f"{it}-{composer_token}",
                )
                if make_proll_post:
                    post_to_pianoroll(
                        model_piece=generated_piece,
                        prompt_piece=prompt_piece,
                        pianoroll_apikey=pianoroll_apikey,
                        generation_info=generation_info,
                        unique_id=unique_id,
                    )
                    st.write("POSTED!")

            midi_package_buffer = make_midi_package(
                prompt_piece=prompt_piece,
                generated_piece=generated_piece,
            )
            download_filename = f"{prompt_name}-{unique_id}.zip"
            st.markdown(
                download_button(
                    object_to_download=midi_package_buffer.getvalue(),
                    download_filename=download_filename,
                    button_text="Download midi with context",
                ),
                unsafe_allow_html=True,
            )
            st.write("---")


def make_midi_package(
    prompt_piece: ff.MidiPiece,
    generated_piece: ff.MidiPiece,
) -> io.BytesIO:
    prompt_bytes = io.BytesIO()
    prompt_piece.to_midi().write(prompt_bytes)
    prompt_bytes.seek(0)

    generated_bytes = io.BytesIO()
    generated_piece.to_midi().write(generated_bytes)
    generated_bytes.seek(0)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("prompt.mid", prompt_bytes.getvalue())
        zip_file.writestr("generation.mid", generated_bytes.getvalue())

        zip_buffer.seek(0)

    return zip_buffer


def post_to_pianoroll(
    model_piece: ff.MidiPiece,
    prompt_piece: ff.MidiPiece,
    pianoroll_apikey: str,
    generation_info: dict,
    unique_id: str,
):
    model_notes = model_piece.df.to_dict(orient="records")
    prompt_notes = prompt_piece.df.to_dict(orient="records")

    description = json.dumps(generation_info, indent=4)
    post_title = "ai riff " + unique_id
    payload = {
        "model_notes": model_notes,
        "prompt_notes": prompt_notes,
        "post_title": post_title,
        "post_description": description,
    }

    headers = {
        "UserApiToken": pianoroll_apikey,
    }
    api_endpoint = "https://pianoroll.io/api/v1/genai_posts"
    r = requests.post(api_endpoint, headers=headers, json=payload)

    st.write(r)
    st.write(r.json())


@st.cache_data
def cache_generation(
    prompt_notes_df: pd.DataFrame,
    seed: int,
    _model,
    _tokenizer,
    pre_input_tokens: list[str],
    generation_info: dict,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    temperature: int = 1,
    top_k: int = None,
) -> pd.DataFrame:
    torch.random.manual_seed(seed)
    generation_token = "<GENAI>"

    with st.spinner("gpu goes brrrrrrrrrr"):
        input_tokens = pre_input_tokens + _tokenizer.tokenize(prompt_notes_df)
        input_tokens.append(generation_token)

        input_token_ids = _tokenizer.encode_tokens(input_tokens)

        # Add a batch size 1 dim, and move to target device
        input_token_ids = torch.tensor(input_token_ids).unsqueeze(0).to(device)

        generated_token_ids = []
        for it in range(max_new_tokens):
            # Check if the input sequence is within context size
            too_long = input_token_ids.size(1) > _model.config.context_size
            if too_long:
                input_token_ids = input_token_ids[:, -_model.config.context_size :]

            time_steps = _tokenizer.token_ids_to_time_steps(
                token_ids=input_token_ids[0],
                restart_tokens=[generation_token],
            )
            time_steps = torch.tensor(time_steps).unsqueeze(0).to(device)

            next_token_id = _model.generate_new_token(
                input_token_ids=input_token_ids,
                time_steps=time_steps,
                temperature=temperature,
                top_k=top_k,
            )

            if next_token_id.item() == _tokenizer.token_to_id["<EOGENAI>"]:
                st.write("FINISH")
                break

            input_token_ids = torch.cat(
                tensors=[input_token_ids, next_token_id],
                dim=1,
            )
            generated_token_ids.append(next_token_id.item())

        generated_notes_df = _tokenizer.decode(generated_token_ids)

        with st.expander("tokens"):
            generated_tokens = [_tokenizer.vocab[token_id] for token_id in generated_token_ids]
            st.write(generated_tokens)

    return generated_notes_df


@st.cache_data
def load_cache_checkpoint(checkpoint_path: str, device) -> dict:
    # Load a pre-trained model
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=False,
    )

    run_config = checkpoint["run_config"]
    model_cfg = checkpoint["model_cfg"]

    tokenizer = ExponentialTimeTokenizer.from_dict(checkpoint["tokenizer_desc"])

    model = GPT(
        config=model_cfg,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    state_dict = checkpoint["model"]
    model.load_state(state_dict=state_dict)
    model.to(device)
    model.eval()

    tasks_config = run_config.tasks
    piano_task_manager = PianoTaskManager(tasks_config=tasks_config)

    # TODO Function is missnamed, this is not a checkpoint
    return {
        "model": model,
        "tokenizer": tokenizer,
        "run_config": run_config,
        "run_stats": checkpoint["run_stats"],
        "piano_task_manager": piano_task_manager,
    }


if __name__ == "__main__":
    main()
