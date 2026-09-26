"""
Data Registration script.

Registers the raw "tourism.csv" dataset with the Hugging Face Hub as a
versioned Dataset repository, so that every stage of the pipeline
(data prep, training, CI/CD) always works off a single source of truth.

This script is safe to run without credentials: if no HF_TOKEN is found
in the environment, it simply reports that the upload step was skipped
instead of failing (or fabricating a fake success).
"""
import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = os.getenv("HF_USERNAME", "ASNaik")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-package-dataset"
LOCAL_DATA_FILE = "tourism_project/data/tourism.csv"


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print(
            "HF_TOKEN not set - skipping upload to the Hugging Face Hub.\n"
            f"(This script would upload '{LOCAL_DATA_FILE}' to the "
            f"dataset repo '{DATASET_REPO_ID}'.)"
        )
        return

    api = HfApi(token=token)
    create_repo(
        repo_id=DATASET_REPO_ID, repo_type="dataset",
        token=token, exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=LOCAL_DATA_FILE,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded '{LOCAL_DATA_FILE}' to https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
