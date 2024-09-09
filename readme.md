
# Piano-GPT: MIDI Piano Music Generation

## Overview

Piano-GPT is a project leveraging the GPT-2 architecture for generating and processing MIDI piano music. It introduces the PIANO (Performance Inference And Note Orchestration) dataset, a multi-task benchmark for voice and dynamic reconstruction in MIDI piano rolls.

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
python gpt2/train.py
```

Customize training by modifying configuration files in `gpt2/configs/`.

### Evaluation

```
python gpt2/eval.py init_from=path_to_checkpoint.pt
```

### Data Visualization

```
streamlit run dashboards/piano_dataset_review.py
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

## References

1. Oore, S., et al. (2018). This Time with Feeling: Learning Expressive Musical Performance. Neural Information Processing Systems (NeurIPS).
