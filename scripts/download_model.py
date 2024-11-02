import os
import sys

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
HF_READ_TOKEN = os.environ.get("HF_READ_TOKEN")

MODELS_DIR = "checkpoints"
REPO_ID = "wmatejuk/piano-gpt"


def download_model(repo_id, filename):
    """Download a model from Hugging Face Hub."""
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=HF_READ_TOKEN, local_dir=MODELS_DIR)
        print(f"Downloaded {filename} from {repo_id}")
        return local_path
    except Exception as e:
        print(f"Error downloading {filename} from {repo_id}: {str(e)}")
        return None


def load_model(model_path):
    """Load a PyTorch model from a file."""
    try:
        model = torch.load(model_path)
        print(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.download_one_model.py <model_filename>")
        sys.exit(1)

    model_filename = sys.argv[1]

    os.makedirs(MODELS_DIR, exist_ok=True)

    local_path = download_model(REPO_ID, model_filename)
    if local_path:
        print(f"Downloaded model: {model_filename}")
    else:
        print(f"Failed to download model: {model_filename}")
