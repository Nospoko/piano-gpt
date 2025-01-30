import os

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv()
HF_READ_TOKEN = os.environ.get("HF_READ_TOKEN")

MODELS_DIR = "tmp/checkpoints"

REPO_ID = "wmatejuk/piano-gpt"

api = HfApi()


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
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        repo_files = api.list_repo_files(repo_id=REPO_ID, token=HF_READ_TOKEN)
    except Exception as e:
        print(f"Error listing repository files: {str(e)}")
        exit(1)

    model_files = [f for f in repo_files if f.endswith(".pt")]

    # Download and load models
    loaded_models = []
    for filename in model_files:
        local_path = download_model(REPO_ID, filename)
        if local_path:
            model = load_model(local_path)
            if model:
                loaded_models.append(model)

    print(f"Successfully loaded {len(loaded_models)} models out of {len(model_files)} found in the repository.")
