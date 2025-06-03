#!/usr/bin/env python3
"""
Download the latest 50 ‘lokale bekendmakingen’, scrape the HTML that
belongs to each record, and append the results (URL + raw text) to a
Hugging Face dataset file.
"""
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from huggingface_hub import HfApi, RepositoryNotFoundError, HfHubHTTPError


SRU_URL   = "https://repository.overheid.nl/sru"
QUERY     = "c.product-area==lokalebekendmakingen"
BATCH     = 50
HF_REPO   = "vGassen/Dutch-Lokale-Bekendmakingen"
TARGET    = "lokale_bekendmakingen.jsonl"
HEADERS   = {"Accept": "application/xml"}

NS = {
    # XML namespaces used in the SRU response
    "sru": "http://docs.oasis-open.org/ns/search-ws/sruResponse",
    "gzd": "http://standaarden.overheid.nl/sru",          # record wrapper
    "gzd2": "http://standaarden.overheid.nl/gzd"          # inside gzd:gzd
}


def fetch_xml(start: int = 1, size: int = BATCH) -> ET.Element:
    params = {
        "version": "2.0",
        "operation": "searchRetrieve",
        "query": QUERY,
        "startRecord": start,
        "maximumRecords": size
    }
    r = requests.get(SRU_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return ET.fromstring(r.content)


def iter_records(root: ET.Element):
    """Yield the <gzd:gzd> blocks inside each <sru:record>."""
    for rec in root.findall(".//sru:record", NS):
        block = rec.find(".//gzd:gzd", {"gzd": NS["gzd2"]})
        if block is not None:
            yield block


def url_from(block: ET.Element) -> str | None:
    """Pick the first URL field that is present."""
    for tag in ("preferredUrl", "url", "itemUrl"):
        node = block.find(f".//gzd2:{tag}", {"gzd2": NS["gzd2"]})
        if node is not None and node.text:
            return node.text.strip()
    return None


def scrape_text(page_url: str) -> str:
    try:
        html = requests.get(page_url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        # grab paragraphs only; skip scripts etc.
        return " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    except Exception as exc:           # network errors, parser errors etc.
        print(f" ⚠ could not scrape {page_url}: {exc}")
        return ""


def collect():
    root = fetch_xml()
    rows = []

    for block in iter_records(root):
        page = url_from(block)
        if not page:
            continue

        text = scrape_text(page)
        if text:                       # keep only non-empty scrapes
            rows.append({"URL": page,
                         "content": text,
                         "source": "Lokale Bekendmakingen"})

    print(f"✓ extracted {len(rows)} usable records")
    return rows


def make_repo_file(api: HfApi):
    """Ensure the jsonl file exists in the repo so we can append to it."""
    try:
        if TARGET not in api.list_repo_files(HF_REPO, repo_type="dataset"):
            print("Creating empty dataset file in the repo…")
            api.upload_file(
                path_or_fileobj=Path(__file__).with_name("empty.jsonl"),
                path_in_repo=TARGET,
                repo_id=HF_REPO,
                repo_type="dataset",
                token=os.environ["HF_TOKEN"],
                commit_message="initialise empty dataset file"
            )
    except (RepositoryNotFoundError, HfHubHTTPError) as exc:
        raise SystemExit(f"Repository problem: {exc}")


def push(rows):
    if not rows:
        print("Nothing to push – aborting.")
        return

    temp = Path("append.jsonl")
    temp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8")

    api = HfApi()
    make_repo_file(api)

    print(f"Pushing {temp} to {HF_REPO}/{TARGET} …")
    api.upload_file(
        path_or_fileobj=temp,
        path_in_repo=TARGET,
        repo_id=HF_REPO,
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
        commit_message=f"append {len(rows)} nieuwe bekendmakingen"
    )
    temp.unlink()


if __name__ == "__main__":
    push(collect())
