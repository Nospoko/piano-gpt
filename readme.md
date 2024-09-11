
# Piano-GPT: MIDI Piano Music Generation

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

- `artifacts.py`: Utility functions and constants
- `checkpoints/`: Saved model checkpoints
- `dashboards/`: Streamlit dashboards for data visualization
- `data/`: Dataset handling and preprocessing modules
- `database/`: Database connection and management utilities
- `gpt2/`: Core GPT-2 model implementation and training scripts
- `midi_datasets/`: Custom dataset classes for MIDI data
- `scripts/`: Utility scripts for model management and evaluation

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/piano-gpt.git
   cd piano-gpt
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training

```
python -m gpt2.train.py
```

You can run the script in DDP mode and with custom configuration, for example
```
PYTHONPATH=. torchrun --nproc-per-node=2 \
gpt2/train.py --config-name=gpt2_pretraining \
data.batch_size=48 \
optimizer.gradient_accumulation_steps=8 \
optimizer.max_iters=30000 \
data.sequence_length=1024 \
data.notes_per_record=128 \
dataset.extra_datasets="['roszcz/giant-midi-sustain-v2', 'roszcz/pianofor-ai-sustain-v2']" \
dataset.augmentation.max_pitch_shift=0 \
dataset.augmentation.speed_change_factors=[] \
lr.warmup_iters=4000 \
lr.learning_rate=6e-5 \
lr.min_lr=6e-6 \
model=gpt2 \
system.data_workers=44 \
system.compile=false \
loss_masking=pretrianing \
init_from=scratch
```

or, for downstream tasks:
```
PYTHONPATH=. torchrun --nproc-per-node=4 \
gpt2/train.py --config-name=gpt2_piano \
tasks = subsequence \
data.batch_size=64 \
optimizer.gradient_accumulation_steps=4 \
optimizer.max_iters=30000 \
data.sequence_length=1024 \
data.notes_per_record=128 \
dataset.extra_datasets="['roszcz/giant-midi-sustain-v2', 'roszcz/pianofor-ai-sustain-v2']" \
dataset.augmentation.max_pitch_shift=5 \
dataset.augmentation.speed_change_factors=[0.95, 1.05] \
lr.learning_rate=8e-5 \
system.data_workers=128 \
system.compile=true \
loss_masking=finetuning \
init_from=midi-gpt2-my-awesome-model.pt  # has to be located in checkpoints and the name needs to start with midi-gpt2

```

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
