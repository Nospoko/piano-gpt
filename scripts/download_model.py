import os
import argparse

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
HF_READ_TOKEN = os.environ.get("HF_READ_TOKEN")


def download_model(
    repo_id: str,
    checkpoint_filename: str,
    local_dir: str,
):
    """Download a model from Hugging Face Hub."""
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            subfolder="checkpoints",
            filename=checkpoint_filename,
            token=HF_READ_TOKEN,
            local_dir=local_dir,
        )
        print(f"Downloaded {checkpoint_filename} from {repo_id}")
        return local_path
    except Exception as e:
        print(f"Error downloading {checkpoint_filename} from {repo_id}: {str(e)}")
        return None


if __name__ == "__main__":
    # python -m scripts.download_model -r epr-labs/piano-gpt -l tmp/ -c 303M-pretraining-2025...pt
    parser = argparse.ArgumentParser(description="Download a model checkpoint from HF repository")
    parser.add_argument("-c", "--checkpoint_filename", type=str, help="Path to the file")
    parser.add_argument("-r", "--repo_id", type=str, help="Huggingface repo id")
    parser.add_argument("-l", "--local_dir", type=str, help="Local folder to store the checkpoint")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    local_path = download_model(
        repo_id=args.repo_id,
        checkpoint_filename=args.checkpoint_filename,
        local_dir=args.local_dir,
    )
    if local_path:
        print("Downloaded model!")
    else:
        print("Failed to download model!")
