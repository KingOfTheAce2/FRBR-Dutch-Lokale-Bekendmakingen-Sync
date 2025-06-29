# step4_upload_shards.py

import os
import sys
import json
import time
import tempfile
import argparse
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
REPO_ID = "vGassen/Dutch-Officiele-Publicaties-Lokale-Bekendmakingen"
INPUT_JSONL_FILE = "cleaned_data.jsonl"
PROGRESS_FILE = "upload_progress.json"
SHARD_SIZE = 200
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0

# --- Helper Functions ---
def load_local_progress() -> int:
    """Load the last successfully uploaded index from the progress file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f).get("last_index", 0)
        except (json.JSONDecodeError, IOError):
            pass
    return 0

def save_local_progress(index: int):
    """Save the last successfully uploaded index."""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"last_index": index}, f)

def flush_repo(api: HfApi, repo_id: str, token: str) -> None:
    """Delete all files from the repository."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        for path in files:
            api.delete_file(path_in_repo=path, repo_id=repo_id, repo_type="dataset", token=token)
        print(f"Flushed repository '{repo_id}'.")
    except Exception as e:
        print(f"Could not flush repository '{repo_id}': {e}", file=sys.stderr)

def get_remote_index(api: HfApi, repo_id: str, token: str) -> int:
    """Find the max index from shard filenames on the remote dataset."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        shards = [f for f in files if f.startswith("data/shard_")]
        if not shards:
            return 0
        
        # Filename format: data/shard_000000_000200.jsonl
        ends = [int(f.split('_')[-1].split('.')[0]) for f in shards]
        return max(ends) if ends else 0
    except Exception as e:
        print(f"Could not retrieve remote index for {repo_id}. Assuming 0. Error: {e}", file=sys.stderr)
        return 0

def upload_shard_with_retry(api: HfApi, local_path: str, remote_path: str, repo_id: str, token: str) -> bool:
    """Upload a single shard with a retry mechanism."""
    for attempt in range(MAX_RETRIES):
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
            return True
        except Exception as e:
            print(f"Failed to upload {remote_path} (attempt {attempt + 1}/{MAX_RETRIES}): {e}", file=sys.stderr)
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_FACTOR * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait:.2f} seconds...", file=sys.stderr)
                time.sleep(wait)
    return False

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Upload JSONL data in shards to a Hugging Face dataset.")
    parser.add_argument("--repo_id", type=str, default=REPO_ID, help="Hugging Face repository ID.")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face API token.")
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE, help="Number of records per shard.")
    parser.add_argument("--flush", action="store_true", help="Delete all remote files before uploading.")
    args = parser.parse_args()

    if not args.token:
        sys.exit("Hugging Face token not found. Please set HF_TOKEN environment variable or use --token.")

    if not os.path.exists(INPUT_JSONL_FILE):
        sys.exit(f"Error: Cleaned data file '{INPUT_JSONL_FILE}' not found. Please run Step 3 first.")

    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f:
        all_records = f.readlines()
    
    total_records = len(all_records)
    if total_records == 0:
        print("No records to upload.")
        return

    print(f"--- Step 4: Upload {total_records} Records in Shards ---")
    
    api = HfApi()
    print(f"Ensuring repository '{args.repo_id}' exists...")
    create_repo(args.repo_id, repo_type="dataset", token=args.token, exist_ok=True)

    if args.flush:
        flush_repo(api, args.repo_id, args.token)
        save_local_progress(0)
        start_index = 0
    else:
        # Determine starting point
        local_start_index = load_local_progress()
        remote_start_index = get_remote_index(api, args.repo_id, args.token)
        start_index = max(local_start_index, remote_start_index)
    
    if start_index >= total_records:
        print("Dataset is already up-to-date. All local records have been uploaded.")
        return

    print(f"Resuming upload from record {start_index} / {total_records}.")

    for i in range(start_index, total_records, args.shard_size):
        batch = all_records[i : i + args.shard_size]
        end_index = i + len(batch)
        
        # Create a temporary file for the shard
        shard_name = f"data/shard_{i:06d}_{end_index:06d}.jsonl"
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', suffix=".jsonl") as tmp:
            tmp.writelines(batch)
            tmp_path = tmp.name

        print(f"Uploading {shard_name} ({len(batch)} records)...")
        
        success = upload_shard_with_retry(api, tmp_path, shard_name, args.repo_id, args.token)
        os.remove(tmp_path)

        if success:
            save_local_progress(end_index)
            print(f"Successfully uploaded. Progress: {end_index}/{total_records}")
        else:
            print("Upload failed after multiple retries. Stopping.", file=sys.stderr)
            sys.exit(1)

    print("\n--- Upload Complete ---")

if __name__ == "__main__":
    main()
