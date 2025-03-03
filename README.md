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

Piano-GPT is a project leveraging the GPT-2 architecture for generating and processing MIDI piano music. It introduces the PIANO (Performance Inference And Note Orchestration) dataset, a multi-task benchmark for voice and dynamic reconstruction in MIDI piano rolls.

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
- `data/`: Dataset handling and preprocessing modules
- `database/`: Database connection and management utilities
- `gpt2/`: Core GPT-2 model implementation and training scripts
- `midi_datasets/`: Custom dataset classes for MIDI data
- `scripts/`: Utility scripts for model management and evaluation

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

## Usage

### Training

```sh
python -m gpt2.train.py
```

You can run the script in DDP mode and with custom configuration. You can change the configuration in
`gpt2/configs/*.yaml`, or specify the training hyperparameters from command line, for example
```sh
PYTHONPATH=. torchrun --nproc-per-node=8 \
    gpt2/train.py --config-name=gpt2_pretraining \
    data.batch_size=32 \
    optimizer.gradient_accumulation_steps=8 \
    optimizer.max_iters=30000 \
    data.context_size=4096 \
    dataset.extra_datasets="['epr-labs/maestro-sustain-v2', 'epr-labs/giant-midi-sustain-v2', 'epr-labs/pianofor-ai-sustain-v2']" \
    dataset.augmentation.max_pitch_shift=5 \
    "dataset.augmentation.speed_change_factors=[0.975, 0.95, 1.025, 1.05]" \
    lr.warmup_iters=1000 \
    lr.learning_rate=1e-5 \
    lr.min_lr=1e-6 \
    model=gpt2_large \
    system.data_workers=64 \
    system.compile=true \
    init_from=scratch
```

or, for downstream tasks:
```sh
PYTHONPATH=. torchrun --nproc-per-node=4 \
    gpt2/train.py --config-name=gpt2_piano \
    tasks = subsequence \
    data.batch_size=64 \
    optimizer.gradient_accumulation_steps=4 \
    optimizer.max_iters=30000 \
    data.context_size=1024 \
    data.notes_per_record=128 \
    dataset.extra_datasets="['epr-labs/maestro-sustain-v2', 'epr-labs/giant-midi-sustain-v2', 'epr-labs/pianofor-ai-sustain-v2']" \
    dataset.augmentation.max_pitch_shift=5 \
    dataset.augmentation.speed_change_factors="[0.95, 1.05]" \
    lr.learning_rate=8e-5 \
    system.data_workers=128 \
    system.compile=true \
    prompt_masking=true \
    checkpoint_path=midi-gpt2-my-awesome-model.pt

```

### Awesome Tokenizer training
```sh
python3.10 -m gpt2.prepare_tokenizer_dataset; \
python3.10 -m gpt2.train_tokenizer; \
PYTHONPATH=. torchrun --nproc-per-node=4 \
gpt2/train.py --config-name=gpt2_pretraining \
model=gpt2 \
lr.learning_rate=8e-5 \
lr.min_lr=8e-6 \
lr.warmup_iters=1000 \
system.data_workers=124 \
optimizer.gradient_accumulation_steps=4 \
task=next_token_prediction_with_composer \
eval_iters=200 eval_interval=1000 \
"dataset.extra_datasets=['epr-labs/maestro-sustain-v2', 'epr-labs/giant-midi-sustain-v2', 'epr-labs/pianofor-ai-sustain-v2']" \
data.batch_size=20 \
data.context_size=4096 \
logging.wandb_run_name_suffix=huge-pretraining-4096-ctx \
tokenizer=awesome \
logging.wandb_project=piano-awesome-gpt
```

`prepare_tokenizer_dataset` will create a text file in `tmp/tokenizer_datasets`, with a dump of tokenized and augmented MAESTRO dataset.

The text will be in a format in which tokenizer will be able to train on.
`train_tokenizer` script will then train an AwesomeMidiTokenizer on this data and dump json format of the tokenizer to `tmp/tokenizers`

Both of these scripts use `gpt2/configs/tokenizer_training` as a default hydra config. It is equivalent to `dataset` + `tokenizer` training config.

During model training initialization the program will look for a tokenizer saved with the same `dataset` and `tokenizer` configuration as training config.

### Evaluation

```
python -m gpt2.eval.py init_from=path_to_checkpoint.pt
```

### Generation
To generate with your model refer to:
https://github.com/Nospoko/piano-generation
a repository fully commited to generation methods.

### Data Visualization
Browse through piano dataset by running:

```
PYTHONPATH=. streamlit run dashboards/piano_dataset_review.py
```

### Model Management

- Download models: `python scripts/download_model.py <model_filename>`
- Upload models: `python scripts/upload_models.py`
- Run multi-task evaluation: `python scripts/run_evaluation.py <model_paths> <device> [--tasks task1 task2 ...]`

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
