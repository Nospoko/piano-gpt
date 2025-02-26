import os
import argparse

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
HF_WRITE_TOKEN = os.environ.get("HF_WRITE_TOKEN")  # Make sure to keep this secret!

# Initialize Hugging Face API
api = HfApi()


def main():
    parser = argparse.ArgumentParser(description="Upload a model checkpoint to HF repository")
    parser.add_argument("-p", "--path", type=str, help="Path to the file")
    parser.add_argument("-r", "--repo_id", type=str, help="Huggingface repo ID")
    args = parser.parse_args()

    path_in_repo = os.path.join("/checkpoints", os.path.basename(args.path))

    print(path_in_repo)

    api.upload_file(
        path_or_fileobj=args.path,
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="model",
        token=HF_WRITE_TOKEN,
    )

    print(f"File {args.path} uploaded successfully!")


if __name__ == "__main__":
    # python -m scripts.upload_model_file -p tmp/checkpoints/10M-pretraining-2025-02-09-07-21.pt -r epr-labs/piano-gpt
    main()
