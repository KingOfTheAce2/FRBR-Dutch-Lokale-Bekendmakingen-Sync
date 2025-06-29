# step1_collect_urls.py

import os
import sys
import json
import time
import random
import requests
from bs4 import BeautifulSoup
from typing import List, Set, Dict, Any
from urllib.parse import urlparse

# --- Constants ---
BASE_REPOS = [
    "https://repository.overheid.nl/frbr/officielepublicaties",
    "https://repository.overheid.nl/frbr/lokalebekendmakingen"
]
TARGET_URL_COUNT = 1000
URL_LIST_FILE = "url_list.json"
MAX_RETRIES = 5
BACKOFF_FACTOR = 2
REQUEST_TIMEOUT = 30 # seconds

# --- Helper Functions ---

def get_session() -> requests.Session:
    """Creates and configures a requests session."""
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    session.headers.update(headers)
    return session

def fetch_url(session: requests.Session, url: str) -> requests.Response | None:
    """Fetches a URL with retries and backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            print(f"Error fetching {url}: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_FACTOR * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait:.2f} seconds...", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"Failed to fetch {url} after {MAX_RETRIES} attempts.", file=sys.stderr)
    return None

def load_progress() -> List[Dict[str, str]]:
    """Loads the list of already collected URLs."""
    if os.path.exists(URL_LIST_FILE):
        try:
            with open(URL_LIST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not read {URL_LIST_FILE}. Starting fresh.", file=sys.stderr)
    return []

def save_progress(urls: List[Dict[str, str]]):
    """Saves the list of collected URLs."""
    with open(URL_LIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(urls, f, indent=2, ensure_ascii=False)

ALLOWED_DOMAINS = [
    "repository.overheid.nl",
    "zoek.officielebekendmakingen.nl",
    "lokalebekendmakingen.nl",
]

def find_links(soup: BeautifulSoup) -> List[str]:
    """Finds all absolute links on a page within the allowed domains."""
    base_url = "https://repository.overheid.nl"
    links: Set[str] = set()

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']

        # Skip query-only or parent references
        if href.startswith('?') or href.startswith('../'):
            continue

        # Convert to absolute if needed
        if not href.startswith('http'):
            href = f"{base_url}{href}" if href.startswith('/') else f"{base_url}/{href}"

        # Only keep links that stay within the allowed domains
        domain = urlparse(href).netloc
        if any(domain == dom or domain.endswith(f".{dom}") for dom in ALLOWED_DOMAINS):
            links.add(href)

    return sorted(links, reverse=True)


# --- Main Logic ---

def main():
    """Main function to crawl and collect URLs."""
    print("--- Step 1: Collect URLs ---")
    session = get_session()
    collected_urls = load_progress()
    collected_urls_set = {item['url'] for item in collected_urls}

    if len(collected_urls) >= TARGET_URL_COUNT:
        print(f"Already have {len(collected_urls)} URLs. Exiting.")
        return

    print(f"Starting with {len(collected_urls)} URLs. Need to reach {TARGET_URL_COUNT}.")

    for base_repo in BASE_REPOS:
        source_type = "officielepublicaties" if "officielepublicaties" in base_repo else "lokalebekendmakingen"
        print(f"\nCrawling repository: {base_repo}")
        
        # 1. Get Years
        resp_year = fetch_url(session, base_repo)
        if not resp_year: continue
        soup_year = BeautifulSoup(resp_year.text, 'html.parser')
        
        for year_link in find_links(soup_year):
            if len(collected_urls) >= TARGET_URL_COUNT: break
            
            # 2. Get Expressions (Work/Publication level)
            resp_expr = fetch_url(session, year_link)
            if not resp_expr: continue
            soup_expr = BeautifulSoup(resp_expr.text, 'html.parser')
            
            for expr_link in find_links(soup_expr):
                if len(collected_urls) >= TARGET_URL_COUNT: break
                
                # 3. Get Manifestations (File level)
                resp_manifest = fetch_url(session, expr_link)
                if not resp_manifest: continue
                soup_manifest = BeautifulSoup(resp_manifest.text, 'html.parser')

                for manifest_link in find_links(soup_manifest):
                    # We are looking for the final downloadable item
                    is_item_link = manifest_link.endswith(('.xml', '.html'))
                    is_metadata = 'metadata' in manifest_link
                    
                    if is_item_link and not is_metadata:
                        if manifest_link not in collected_urls_set:
                            print(f"Found item: {manifest_link}")
                            url_data = {"url": manifest_link, "source": source_type}
                            collected_urls.append(url_data)
                            collected_urls_set.add(manifest_link)
                            
                            if len(collected_urls) % 20 == 0:
                                print(f"Progress: {len(collected_urls)} / {TARGET_URL_COUNT}")
                                save_progress(collected_urls)

                        if len(collected_urls) >= TARGET_URL_COUNT:
                            break
            time.sleep(0.5) # Politeness

    save_progress(collected_urls)
    print(f"\nFinished collecting URLs. Total found: {len(collected_urls)}.")
    print(f"URL list saved to {URL_LIST_FILE}.")


if __name__ == "__main__":
    main()
