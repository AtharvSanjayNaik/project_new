"""
Hosting script: pushes the deployment artifacts (Streamlit app, Dockerfile,
requirements.txt and the trained model) to a Hugging Face Space so the
prediction app is publicly reachable.

Safe to run without credentials: if HF_TOKEN is not set, it reports what
it would have done instead of failing or faking a successful deployment.
"""
import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = os.getenv("HF_USERNAME", "ASNaik")
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-wellness-package-app"
DEPLOYMENT_DIR = "tourism_project/deployment"


def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        print(
            "HF_TOKEN not set - skipping push to Hugging Face Spaces.\n"
            f"(This script would upload the contents of '{DEPLOYMENT_DIR}' "
            f"to the Space '{SPACE_REPO_ID}', running on the Docker SDK.)"
        )
        return

    api = HfApi(token=token)
    create_repo(
        repo_id=SPACE_REPO_ID, repo_type="space", space_sdk="static",
        token=token, exist_ok=True,
    )
    api.upload_folder(
        folder_path=DEPLOYMENT_DIR,
        repo_id=SPACE_REPO_ID,
        repo_type="space",
    )
    print(f"Deployed app to https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()

