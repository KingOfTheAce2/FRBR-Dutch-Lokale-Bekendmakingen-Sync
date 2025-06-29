# step2_download_items.py

import os
import sys
import json
import time
import random
import requests
from urllib.parse import urlparse

# --- Constants ---
URL_LIST_FILE = "url_list.json"
DATA_DIR = "data"
MAX_RETRIES = 5
BACKOFF_FACTOR = 2
REQUEST_TIMEOUT = 60 # seconds for potentially larger files

# --- Helper Functions ---
def get_session() -> requests.Session:
    """Creates and configures a requests session."""
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    session.headers.update(headers)
    return session

def fetch_url_content(session: requests.Session, url: str) -> bytes | None:
    """Fetches raw content from a URL with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.content
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            print(f"Error downloading {url}: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_FACTOR * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait:.2f} seconds...", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"Failed to download {url} after {MAX_RETRIES} attempts.", file=sys.stderr)
    return None

def get_local_path(item: dict) -> str:
    """Generates a unique local file path for a given URL item."""
    source = item['source']
    url = item['url']
    
    # Extract a unique identifier from the URL path
    # e.g., /frbr/officielepublicaties/gmb-2025-12345/1/xml/gmb-2025-12345.xml
    path_parts = [part for part in urlparse(url).path.split('/') if part]
    filename = path_parts[-1]
    
    # Create a safe directory structure
    output_dir = os.path.join(DATA_DIR, source)
    os.makedirs(output_dir, exist_ok=True)
    
    return os.path.join(output_dir, filename)

# --- Main Logic ---
def main():
    """Main function to download content from URLs."""
    print("--- Step 2: Download Items ---")
    if not os.path.exists(URL_LIST_FILE):
        sys.exit(f"Error: {URL_LIST_FILE} not found. Please run Step 1 first.")

    with open(URL_LIST_FILE, 'r', encoding='utf-8') as f:
        urls_to_download = json.load(f)

    if not urls_to_download:
        sys.exit("URL list is empty. Nothing to download.")
        
    print(f"Found {len(urls_to_download)} URLs to process.")
    
    session = get_session()
    downloaded_count = 0
    skipped_count = 0

    for i, item in enumerate(urls_to_download):
        local_path = get_local_path(item)
        
        if os.path.exists(local_path):
            skipped_count += 1
            continue

        print(f"Downloading [{i+1}/{len(urls_to_download)}]: {item['url']}")
        content = fetch_url_content(session, item['url'])

        if content:
            try:
                with open(local_path, 'wb') as f:
                    f.write(content)
                downloaded_count += 1
            except IOError as e:
                print(f"Error writing file {local_path}: {e}", file=sys.stderr)
        
        # Politeness delay
        time.sleep(0.2)

    print("\n--- Download Summary ---")
    print(f"Successfully downloaded: {downloaded_count} files.")
    print(f"Skipped (already exist): {skipped_count} files.")
    print(f"Total files in data directory.")

if __name__ == "__main__":
    main()
