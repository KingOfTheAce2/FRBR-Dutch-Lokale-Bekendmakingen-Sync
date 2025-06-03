#!/usr/bin/env python3
"""
Scrapes the latest 500 'lokale bekendmakingen', extracts text,
removes duplicates using a checkpoint, and pushes to Hugging Face.
"""
import os
import json
import time
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
from datasets import Dataset, load_dataset

SRU_URL = "https://repository.overheid.nl/sru"
QUERY = "c.product-area==lokalebekendmakingen"
BATCH = 50
HF_REPO = "vGassen/Dutch-Lokale-Bekendmakingen"
CHECKPOINT_FILE = "lb_checkpoint.json"

NS = {
    "sru": "http://docs.oasis-open.org/ns/search-ws/sruResponse",
    "gzd": "http://standaarden.overheid.nl/sru",
    "gzd2": "http://standaarden.overheid.nl/gzd"
}

def load_checkpoint():
    # TEMPORARILY RESETTING CHECKPOINT — ignore file
    print("[DEBUG] Temporarily ignoring checkpoint file")
    return {"seen_urls": []}

def save_checkpoint(state):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def fetch_xml(start: int = 1, size: int = BATCH) -> ET.Element:
    params = {
        "version": "2.0",
        "operation": "searchRetrieve",
        "query": QUERY,
        "startRecord": start,
        "maximumRecords": size
    }
    r = requests.get(SRU_URL, params=params, headers={"Accept": "application/xml"}, timeout=20)
    r.raise_for_status()
    return ET.fromstring(r.content)

def iter_records(root: ET.Element):
    for rec in root.findall(".//sru:record", NS):
        block = rec.find(".//gzd:gzd", {"gzd": NS["gzd2"]})
        if block is not None:
            yield block

def url_from(block: ET.Element) -> str | None:
    # First try bronIdentifier (preferred)
    node = block.find(".//overheidwetgeving:bronIdentifier", {
        "overheidwetgeving": "http://standaarden.overheid.nl/wetgeving/"
    })
    if node is not None and node.text:
        return node.text.strip()

    # Fallback to older method
    for tag in ("preferredUrl", "url", "itemUrl"):
        node = block.find(f".//gzd2:{tag}", {"gzd2": NS["gzd2"]})
        if node is not None and node.text:
            return node.text.strip()

    return None

def scrape_text(page_url: str) -> str:
    try:
        html = requests.get(page_url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    except Exception as exc:
        print(f"[WARN] Could not scrape {page_url}: {exc}")
        return ""

def collect_new_rows(seen_urls: list[str], max_records=500):
    all_rows = []
    print(f"[DEBUG] Starting record scan, initial checkpoint has {len(seen_urls)} URLs")
    for start in range(1, max_records + 1, BATCH):
        print(f"[DEBUG] Fetching records {start} to {start + BATCH - 1}")
        root = fetch_xml(start=start, size=BATCH)
        new_rows = []
        for block in iter_records(root):
            url = url_from(block)
            if not url:
                print("[DEBUG] Skipping block without URL")
                continue
            if url in seen_urls:
                print(f"[DEBUG] Already seen: {url}")
                continue
            text = scrape_text(url)
            if text:
                print(f"[DEBUG] New valid item: {url}")
                new_rows.append({"url": url, "content": text, "source": "Lokale Bekendmakingen"})
                seen_urls.append(url)
            time.sleep(0.2)
        if not new_rows:
            print("[DEBUG] No new valid items in this batch, stopping early.")
            break
        all_rows.extend(new_rows)
    print(f"[INFO] Collected {len(all_rows)} new items")
    return all_rows, seen_urls

def push_to_hub(rows):
    if not rows:
        print("[INFO] No new data to push.")
        return
    try:
        existing = Dataset.from_list(load_dataset(HF_REPO, split="train").to_list())
        combined = Dataset.from_list(existing.to_list() + rows)
    except Exception as e:
        print(f"[WARN] Could not load existing dataset: {e}")
        combined = Dataset.from_list(rows)

    print(f"[INFO] Uploading {len(combined)} entries to Hugging Face…")
    combined.push_to_hub(HF_REPO)

def main():
    print("[INFO] Starting scrape...")
    checkpoint = load_checkpoint()
    seen_urls = checkpoint.get("seen_urls", [])
    rows, updated_urls = collect_new_rows(seen_urls)
    push_to_hub(rows)
    checkpoint["seen_urls"] = updated_urls
    save_checkpoint(checkpoint)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
