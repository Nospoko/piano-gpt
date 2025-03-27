# Piano-GPT: MIDI Piano Music Generation

## Quickstart

Train a 10M model:

```sh
# This will create checkpoints in ./tmp/checkpoints and logs in wandb
python -m gpt2.main dataset=small model=gpt2_10M

# No wandb, small memory footprint
python -m gpt2.main dataset=small model=gpt2_10M training.microbatch_size=2 logging.wandb_log=false
```

Resume training:

```sh
python -m gpt2.main --config-name=resume_training checkpoint_path=</path/to/checkpoint.pt>

# torchrun version
PYTHONPATH=. torchrun --nproc-per-node=4 -m gpt2.main --config-name=resume_training checkpoint_path=</path/to/checkpoint.pt>
```

Tune model:

```sh
python -m gpt2.main --config-name=model_tuning checkpoint_path=</path/to/checkpoint.pt>
```

Calculate PIANO metrics:

```sh
python -m gpt2.piano_eval init_from=<checkpoint path>
```

Launch the generation dashboard:

```sh
PYTHONPATH=. streamlit run dashboards/prompt_practice.py
```

## Overview

Piano-GPT is a project utilizing the GPT-2 architecture for generating and processing MIDI piano music. It introduces the PIANO (Performance Inference And Note Orchestration) dataset, a multi-task benchmark for voice and dynamic reconstruction in MIDI piano rolls.

### Tokenization

Tokenization is described to full extent in [midi-tokenizers repo](https://github.com/Nospoko/midi-tokenizers)

## PIANO Dataset

The PIANO dataset is designed to standardize approaches and provide a benchmark for the music modeling community. It focuses on specific subsets of music generation tasks:

1. **Voice Inference**: Inferring missing notes within specific voice parts of a musical composition, based on the surrounding musical context of incomplete sequences.

2. **Dynamic Reconstruction**: Recovering notes from different volume ranges (velocity in MIDI terminology) to challenge models in reconstructing the dynamics of a piece.

3. **Noise Reduction**: Reconstructing original note information from noisy inputs, including pitch, velocity, and timing. This task simulates scenarios where MIDI data might be imperfectly recorded or played.

### Task Categories

- **Pitch-based Tasks**: Divide notes into groups based on relative pitch height (e.g., above/below median pitch, highest/lowest quartiles).
- **Volume-based Tasks**: Categorize notes based on loudness (velocity), such as loud/soft or very loud/very soft.
- **Denoising Tasks**: Add controlled random variations to pitch, volume, or timing of notes.
- **Comprehensive Denoising**: Combine variations in pitch, volume, and timing simultaneously.
- **Performance Task**: Simplify a piece by standardizing note length, volume, and timing.

## Project Structure

- `tmp/checkpoints/`: Saved model checkpoints
- `dashboards/`: Streamlit dashboards for data visualization
- `gpt2/`: Core GPT-2 model implementation and training scripts
- `commands/`: Utility scripts for model management and evaluation

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/piano-gpt.git
   cd piano-gpt
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Data Visualization
Browse through piano dataset by running:

```
PYTHONPATH=. streamlit run dashboards/piano_dataset_review.py
```

## Configuration

The project uses Hydra for configuration management. Main configuration files are located in `gpt2/configs/`.

## Acknowledgments

- This project uses the GPT-2 architecture developed by OpenAI.
- The PIANO dataset is based on the MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization) dataset, provided by the International Piano-e-Competition and the Tensorflow Magenta team.

## Important Links
- **Maestro Dataset**: [Link to dataset](https://magenta.tensorflow.org/datasets/maestro)
- **GitHub Repository**: [piano-gpt](https://github.com/Nospoko/piano-gpt)
- **Midi Tokenizers Repository**: [midi-tokenizers](https://github.com/Nospoko/midi-tokenizers)
- **Platform for pianists and algorithmic music enthusiasts**: [pianoroll.io](https://pianoroll.io)

## References

1. Oore, S., et al. (2018). This Time with Feeling: Learning Expressive Musical Performance. Neural Information Processing Systems (NeurIPS).

## Development

### Code Style

This repository uses pre-commit hooks with forced python formatting ([black](https://github.com/psf/black),
[flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected.
`black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running `git commit`:

```sh
pre-commit run --all-files
```
