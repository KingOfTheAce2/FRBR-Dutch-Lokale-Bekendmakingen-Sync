#!/usr/bin/env python3
"""
Scrapes the latest 250 'lokale bekendmakingen', extracts text,
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
BATCH = 100
HF_REPO = "vGassen/Dutch-Lokale-Bekendmakingen"
CHECKPOINT_FILE = "lb_checkpoint.json"

NS = {
    "sru": "http://docs.oasis-open.org/ns/search-ws/sruResponse",
    "gzd": "http://standaarden.overheid.nl/sru",
    "gzd2": "http://standaarden.overheid.nl/gzd"
}

def load_checkpoint():
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
    node = block.find(".//overheidwetgeving:bronIdentifier", {
        "overheidwetgeving": "http://standaarden.overheid.nl/wetgeving/"
    })
    if node is not None and node.text:
        url = node.text.strip()
        print(f"[DEBUG] bronIdentifier raw URL: {url}")  # ← required
        if url.endswith(".xml") or "repository.overheid.nl" in url:
            print(f"[DEBUG] Skipping metadata-only URL: {url}")
            return None
        print("[DEBUG] Using bronIdentifier for URL")
        return url

    for tag in ("preferredUrl", "url", "itemUrl"):
        node = block.find(f".//gzd2:{tag}", {"gzd2": NS["gzd2"]})
        if node is not None and node.text:
            url = node.text.strip()
            print(f"[DEBUG] fallback raw URL <{tag}>: {url}")
            if url.endswith(".xml") or "repository.overheid.nl" in url:
                print(f"[DEBUG] Skipping fallback XML URL: {url}")
                continue
            print(f"[DEBUG] Using fallback tag <{tag}> for URL")
            return url

    return None

def scrape_text(page_url: str) -> str:
    try:
        print(f"[DEBUG] Requesting page: {page_url}")
        html = requests.get(page_url, timeout=10).text

        if html.strip().startswith("<?xml"):
            print(f"[DEBUG] ❌ Got XML instead of HTML from: {page_url}")
            return ""

        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        if not paragraphs:
            print(f"[DEBUG] ❌ No <p> tags found at: {page_url}")
        return " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    except Exception as exc:
        print(f"[WARN] Could not scrape {page_url}: {exc}")
        return ""

def collect_new_rows(seen_urls: list[str], max_records=250):
    all_rows = []
    print(f"[DEBUG] Starting record scan, initial checkpoint has {len(seen_urls)} URLs")
    for start in range(1, max_records + 1, BATCH):
        print(f"[DEBUG] Fetching records {start} to {min(start + BATCH - 1, max_records)}")
        root = fetch_xml(start=start, size=BATCH)
        new_rows = []
        for block in iter_records(root):
            url = url_from(block)
            if not url:
                print("[DEBUG] Skipping block without valid URL")
                continue
            if url in seen_urls:
                print(f"[DEBUG] Already seen: {url}")
                continue

            print(f"[DEBUG] → Evaluating URL: {url}")
            text = scrape_text(url)
            if text:
                print(f"[DEBUG] ✅ Scraped OK: {url}")
                new_rows.append({"url": url, "content": text, "source": "Lokale Bekendmakingen"})
                seen_urls.append(url)
            else:
                print(f"[DEBUG] ❌ No usable content at: {url}")
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
