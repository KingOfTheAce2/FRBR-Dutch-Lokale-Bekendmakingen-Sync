# step3_clean_text.py

import os
import sys
import json
import glob
from bs4 import BeautifulSoup

# --- Constants ---
DATA_DIR = "data"
URL_LIST_FILE = "url_list.json"
OUTPUT_JSONL_FILE = "cleaned_data.jsonl"

# --- Main Logic ---
def main():
    """Converts downloaded XML/HTML files to a clean JSONL file."""
    print("--- Step 3: Clean Text and Create JSONL ---")

    if not os.path.exists(DATA_DIR):
        sys.exit(f"Error: Data directory '{DATA_DIR}' not found. Please run Step 2 first.")

    if not os.path.exists(URL_LIST_FILE):
        sys.exit(f"Error: {URL_LIST_FILE} not found. Cannot map files back to URLs.")
    
    # Create a mapping from filename back to the original URL and source
    with open(URL_LIST_FILE, 'r', encoding='utf-8') as f:
        url_items = json.load(f)
    
    filename_to_item = {
        os.path.basename(item['url']): item for item in url_items
    }

    # Find all downloaded files
    search_pattern = os.path.join(DATA_DIR, "**", "*.*")
    downloaded_files = glob.glob(search_pattern, recursive=True)

    if not downloaded_files:
        sys.exit("No downloaded files found in the data directory.")

    print(f"Found {len(downloaded_files)} files to process.")

    records_written = 0
    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as out_f:
        for i, file_path in enumerate(downloaded_files):
            basename = os.path.basename(file_path)
            item_info = filename_to_item.get(basename)
            
            if not item_info:
                print(f"Warning: Could not find original URL for file {basename}. Skipping.", file=sys.stderr)
                continue

            print(f"Processing [{i+1}/{len(downloaded_files)}]: {file_path}")
            
            try:
                with open(file_path, 'rb') as f:
                    content_bytes = f.read()
                
                # Use 'lxml' for XML and 'html.parser' as a fallback.
                # BeautifulSoup handles encoding detection reasonably well.
                soup = BeautifulSoup(content_bytes, 'lxml-xml')
                
                # If parsing as XML yields no text, try as HTML
                if not soup.get_text(strip=True):
                    soup = BeautifulSoup(content_bytes, 'html.parser')

                clean_text = soup.get_text(separator=' ', strip=True)

                if clean_text:
                    record = {
                        "url": item_info['url'],
                        "content": clean_text,
                        "source": item_info['source']
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_written += 1
                else:
                    print(f"Warning: No text extracted from {file_path}", file=sys.stderr)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}", file=sys.stderr)

    print("\n--- Cleaning Summary ---")
    print(f"Successfully processed and wrote {records_written} records.")
    print(f"Output saved to {OUTPUT_JSONL_FILE}.")

if __name__ == "__main__":
    main()
