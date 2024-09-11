import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
HF_WRITE_TOKEN = os.environ.get("HF_WRITE_TOKEN")  # Make sure to keep this secret!

# Set the local directory containing your checkpoints
CHECKPOINTS_DIR = "checkpoints"

# Initialize Hugging Face API
api = HfApi()

# Main script
if __name__ == "__main__":
    # Create a repository name based on the subdirectory name
    repo_name = "wmatejuk/piano-gpt2"

    try:
        api.upload_folder(folder_path=CHECKPOINTS_DIR, repo_id=repo_name, repo_type="model", token=HF_WRITE_TOKEN)
    except Exception as e:
        print(f"Error uploading models: {str(e)}")

    print("All models uploaded successfully!")
