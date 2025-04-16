from enum import Enum
from glob import glob
from dataclasses import dataclass

import fortepyan as ff
import streamlit as st
import streamlit_pianoroll


class ComposerTokenSelection(Enum):
    CHOPIN = "chopin"
    BACH = "bach"


@dataclass
class GenerationInput:
    start_note_idx: int
    finish_note_idx: int
    prompt_name: str
    composer_token: ComposerTokenSelection
    random_seed: int = 137


def main():
    st.write("# Prompt")
    st.write(
        "Upload a new prompt midi file, or select one from the list.",
        "Uploading will add a file to that list.",
    )

    if "generation_history" not in st.session_state:
        st.session_state["generation_history"] = []
        st.write("initialized generation history!")

    prompt_path = prompt_selection()

    if not prompt_path:
        st.write("Submit your prompt selection pls")
        return

    prompt_piece = ff.MidiPiece.from_file(prompt_path)

    # TODO Remove inplace operations from fortepyan TODO
    prompt_piece.time_shift(-prompt_piece.df.start.min())

    st.write("## Input piece")
    streamlit_pianoroll.from_fortepyan(prompt_piece)

    show_generation_history()

    with st.form("FOO"):
        generations_parameters = get_generations_parameters(n_generations=5)
        st.write(generations_parameters)
        st.write("---")

        yo = st.form_submit_button()

    if not yo:
        st.write("yo!")
        return

    for it, generation_parameters in enumerate(generations_parameters):

        def add_to_history(p):
            st.session_state.generation_history.append(p)

        st.write(generation_parameters)
        click_me = st.button(
            label="ADD",
            key=f"add_{it}",
            on_click=add_to_history,
            args=[generation_parameters],
        )
        st.write(click_me)
        st.write("---")

    return

    # 1. Select a sub-prompt from the input piece
    #   - allow time selection and note idx selection
    # 2. st.form with all the generation parameters
    # 3. Generation! (5 seeds?, all token control by user)
    # 4. Select one or go back to 2.
    # 5. Select a new sub-prompt including what was generated in 4.
    #   - basically go back to 1. with the new input piece

    # NOTES:
    # - at every step display a pianoroll where all generated notes
    #   are shown with the secondary color
    # - same for midi download
    # - pianoroll posts only for the final _banger_

    # Questions:
    # - how to make the number of iterations dynamic? user should
    #   be allowed to just click "add one more gen step" :thinking:

    st.write("# Generation Setup")
    if "n_iterations" not in st.session_state:
        st.session_state["n_iterations"] = 1

    n_iterations = st.session_state["n_iterations"]

    st.write("---")
    st.write("---")

    foo = {}
    for it in range(n_iterations - 1):
        label = f"Iteration {it}"
        with st.expander(label=label, expanded=False):
            bar = get_generations_parameters(it=it)
            st.write(bar)

        foo[it] = bar
        st.write("---")

    with st.form("FOO"):
        it = n_iterations - 1
        label = f"Iteration {it}"
        with st.expander(label=label, expanded=True):
            bar = get_generations_parameters(it=it)
        foo[it] = bar
        st.write("---")

        yo = st.form_submit_button()

    if not yo:
        st.write("yo!")
        return

    st.write(foo)

    def bump_iteration():
        st.session_state["n_iterations"] += 1

    st.button(
        label="add iteration!",
        on_click=bump_iteration,
    )

    st.write(st.session_state)


def show_generation_history():
    if not st.session_state.get("generation_history"):
        st.info("No generations yet. Start by selecting a time range and adding an iteration.")
        return

    st.write(st.session_state.generation_history)

    for generation_parameters in st.session_state.generation_history:
        # generation_parameters = GenerationInput(**generation_parameters)
        st.write(generation_parameters.start_note_idx)


def get_generations_parameters(n_generations: int) -> list[GenerationInput]:
    start_note_idx = st.number_input(
        label="start note idx {it}",
        key="start_note_idx",
        min_value=0,
        max_value=200,
        value=0,
    )
    finish_note_idx = st.number_input(
        label="finish note idx {it}",
        key="finish_note_idx",
        min_value=0,
        max_value=200,
        value=20,
    )
    prompt_name = st.text_input(
        label="name {it}",
        value="my promp",
        key="prompt_name",
    )
    composer_token = st.selectbox(
        label="composer token {it}",
        key="composer_token",
        options=[token.name for token in ComposerTokenSelection],
    )
    random_seed = st.number_input(
        label="random seed {it}",
        key="random_seed",
        value=137,
    )

    generation_inputs = []
    for it in range(n_generations):
        local_seed = random_seed + 1000 * it
        generation_input = GenerationInput(
            start_note_idx=start_note_idx,
            finish_note_idx=finish_note_idx,
            prompt_name=prompt_name,
            composer_token=composer_token,
            random_seed=local_seed,
        )
        generation_inputs.append(generation_input)

    return generation_inputs


def prompt_selection() -> str:
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

    return prompt_path


def get_current_gen_params() -> dict:
    # I have to manually snatch everything by key
    # from the state in order to get the values that
    # are currently actually set in the form :(
    keys = [
        "start_note_idx",
        "finish_note_idx",
        "prompt_name",
        "random_seed",
        "composer_token",
    ]
    data = {key: st.session_state[key] for key in keys}

    return data


if __name__ == "__main__":
    main()
