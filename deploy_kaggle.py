# deploy_kaggle.py
# Automates the entire Kaggle CLI deployment workflow in Python (BOM-free and robust)

import os
import re
import json
import subprocess

def main():
    # 1. Read access_token and get username
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    username = None

    # Try reading from config first using CLI
    try:
        result = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if line.strip().startswith("- username:"):
                username = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    if not username and os.path.exists(kaggle_json_path):
        try:
            with open(kaggle_json_path, "r") as f:
                data = json.load(f)
                username = data.get("username")
        except Exception:
            pass

    if not username:
        raise ValueError("Error: Could not retrieve Kaggle username. Please ensure Kaggle API is authenticated.")

    print(f"Detected Kaggle Username: {username}")

    # 2. Configure kernel-metadata.json
    metadata_path = "kaggle_job/kernel-metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["id"] = f"{username}/whisper-scaling-laws"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Updated: {metadata_path}")

    # 3. Read tokens
    pat_path = "kaggle_job/github_pat.txt"
    hf_token_path = "kaggle_job/hf_token.txt"
    if not os.path.exists(pat_path):
        raise FileNotFoundError(f"Error: {pat_path} not found! Please create it and write your GitHub PAT inside.")
    if not os.path.exists(hf_token_path):
        raise FileNotFoundError(f"Error: {hf_token_path} not found! Please create it and write your Hugging Face Access Token inside.")

    with open(pat_path, "r", encoding="utf-8") as f:
        pat_token = f.read().strip()
    with open(hf_token_path, "r", encoding="utf-8") as f:
        hf_token = f.read().strip()

    # 4. Embed tokens temporarily and push
    run_kaggle_path = "kaggle_job/run_kaggle.py"
    with open(run_kaggle_path, "r", encoding="utf-8") as f:
        run_kaggle_content = f.read()

    embedded_content = run_kaggle_content.replace("GH_PAT_PLACEHOLDER", pat_token).replace("HF_TOKEN_PLACEHOLDER", hf_token)

    # Write embedded script
    with open(run_kaggle_path, "w", encoding="utf-8") as f:
        f.write(embedded_content)

    try:
        print("Pushing training kernel to Kaggle...")
        subprocess.run(["kaggle", "kernels", "push", "-p", "kaggle_job", "--accelerator", "NvidiaTeslaT4"], check=True)
    finally:
        # Always restore original file
        print("Restoring run_kaggle.py...")
        with open(run_kaggle_path, "w", encoding="utf-8") as f:
            f.write(run_kaggle_content)

    print("Deployment completed successfully!")
    print(f"Check status using: kaggle kernels status {username}/whisper-scaling-laws")

if __name__ == "__main__":
    main()
