import os
from glob import glob

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
from dashboards.utils import device_model_selection
from dashboards.utils.components import download_button


def main():
    # See the readme to figure out how you can get this checkpoint
    # checkpoint_path = "checkpoints/midi-gpt2-302M-subsequence-4096-ctx-2024-09-08-19-42last.pt"
    device, checkpoint_path = device_model_selection()
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

    st.write("Training settings:")
    st.write(OmegaConf.to_container(run_config.training))

    st.write("Training stats:")
    st.write(model_setup["run_stats"])

    with st.form("prompt selection"):
        # TODO: What would be a convenient way to manage prompts
        # for a user? Definitely needs an upload
        prompt_options = glob("tmp/prompts/*.mid") + [None]
        prompt_path = st.selectbox(
            label="select prompt file",
            options=prompt_options,
            index=None,
        )
        st.form_submit_button()

    if not prompt_path:
        return

    prompt_piece = ff.MidiPiece.from_file(prompt_path)

    # TODO Remove inplace operations from fortepyan
    prompt_piece.time_shift(-prompt_piece.df.start.min())

    streamlit_pianoroll.from_fortepyan(prompt_piece)
    st.write("Prompt notes:", prompt_piece.size)

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

        speedup_factor = st.number_input(
            label="speedup factor",
            value=1.0,
            min_value=0.3,
            max_value=2.5,
        )

        dataset_tokens = MusicManager().dataset_tokens
        dataset_token = st.selectbox(
            label="Select a dataset token:",
            options=dataset_tokens,
            help="Choose from available special tokens to add to your prompt",
        )

        piano_task_manager: PianoTaskManager = model_setup["piano_task_manager"]

        piano_task_name = st.selectbox(
            label="Select PIANO task:",
            options=piano_task_manager.list_task_names(),
            help="Choose from tasks used during training",
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

    # TODO this dashboards tries hard to work both for next token prediction
    # and for the piano task generations – we should probably have separate dashboards
    composer_tokens = ["<BACH>", "<MOZART>", "<CHOPIN>", "<UNKNOWN_COMPOSER>"]
    for composer_token in composer_tokens:
        pre_input_tokens = [dataset_token, composer_token] + piano_task.prefix_tokens

        st.write("Pre-input tokens:", pre_input_tokens)

        # Generator randomness comes from torch.multinomial, so we can make it
        # fully deterministic by setting global torch random seed
        for it in range(2):
            local_seed = random_seed + it * 1000
            generation_setup = {
                "seed": local_seed,
                "temperature": temperature,
                "dataset_token": dataset_token,
                "composer_token": composer_token,
                "piano_task": piano_task.name,
            }
            st.write(generation_setup)
            st.write("".join(pre_input_tokens))

            # This acts as a caching key
            generation_properties = {
                "speedup_factor": speedup_factor,
                "prompt_path": prompt_path,
                "local_seed": local_seed,
            }

            prompt_notes_df, generated_notes = cache_generation(
                prompt_notes_df=prompt_notes_df,
                seed=local_seed,
                _model=model,
                _tokenizer=tokenizer,
                device=device,
                pre_input_tokens=pre_input_tokens,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                **generation_properties,
            )

            prompt_piece = ff.MidiPiece(prompt_notes_df)
            generated_piece = ff.MidiPiece(generated_notes)

            streamlit_pianoroll.from_fortepyan(prompt_piece, generated_piece)

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
                    )
                    st.write("POSTED!")

            out_piece = ff.MidiPiece(pd.concat([prompt_notes_df, generated_notes]))
            # Allow download of the full MIDI with context
            full_midi_path = f"tmp/tmp_{composer_token}_{local_seed}.mid"
            out_piece.to_midi().write(full_midi_path)
            with open(full_midi_path, "rb") as file:
                st.markdown(
                    download_button(
                        object_to_download=file.read(),
                        download_filename=full_midi_path.split("/")[-1],
                        button_text="Download midi with context",
                    ),
                    unsafe_allow_html=True,
                )
            st.write("---")
    os.unlink(full_midi_path)


def post_to_pianoroll(
    model_piece: ff.MidiPiece,
    prompt_piece: ff.MidiPiece,
    pianoroll_apikey: str,
):
    model_notes = model_piece.df.to_dict(orient="records")
    prompt_notes = prompt_piece.df.to_dict(orient="records")

    payload = {
        "model_notes": model_notes,
        "prompt_notes": prompt_notes,
        "post_title": "GENAI",
        "post_description": "My model did this!",
    }

    headers = {
        "UserApiToken": pianoroll_apikey,
    }
    api_endpoint = "https://pianoroll.io/api/v1/genai_posts"
    r = requests.post(api_endpoint, headers=headers, json=payload)

    st.write(r)


@st.cache_data
def cache_generation(
    prompt_notes_df: pd.DataFrame,
    seed: int,
    _model,
    _tokenizer,
    pre_input_tokens: list[str],
    device: str = "cuda",
    max_new_tokens: int = 2048,
    temperature: int = 1,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    torch.random.manual_seed(seed)
    with st.spinner("gpu goes brrrrrrrrrr"):
        input_tokens = pre_input_tokens + _tokenizer.tokenize(prompt_notes_df)
        input_tokens.append("<GENAI>")

        input_token_ids = _tokenizer.encode_tokens(input_tokens)
        input_token_ids = torch.tensor(input_token_ids).unsqueeze(0).to(device)

        generated_token_ids = _model.generate_new_tokens(
            idx=input_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None,
        )
        generated_token_ids = generated_token_ids.squeeze(0).cpu().numpy()
        generated_notes = _tokenizer.decode(generated_token_ids)

        with st.expander("tokens"):
            generated_tokens = [_tokenizer.vocab[token_id] for token_id in generated_token_ids]
            st.write(generated_tokens)

    return prompt_notes_df, generated_notes


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

    if run_config.model_task == "piano_task":
        tasks_config = run_config.tasks
        piano_task_manager = PianoTaskManager(tasks_config=tasks_config)
    else:
        piano_task_manager = None

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
