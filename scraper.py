#!/usr/bin/env python3
"""
Scrape Officiële Publicaties (incl. former Lokale Bekendmakingen) and
append URL, plain‑text content and constant source field to a JSONL file.
Pushing to Hugging Face is optional but supported.
"""
from __future__ import annotations
import os, sys, json, time, datetime as dt
from pathlib import Path
from typing import Iterator, List
import requests
from bs4 import BeautifulSoup
from lxml import etree           # robust XML parsing
from huggingface_hub import HfApi, HfFolder

HF_REPO      = "vGassen/Dutch-Officiele-Publicaties-Lokale-Bekendmakingen"
OUT_PATH     = Path("data/officielepublicaties.jsonl")
EVENT_BASE   = "https://repository.officiele-overheidspublicaties.nl/officielepublicaties/_events/"

HEADERS = {
    "User-Agent": "OP-scraper/1.0 (https://github.com/vGassen/Dutch-Officiele-Publicaties-Lokale-Bekendmakingen)"
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION.timeout = (10, 30)  # connect, read


def daily_event_url(day: dt.date) -> str:
    return f"{EVENT_BASE}{day.isoformat()}.xml"


def iter_identifiers(event_xml: bytes) -> Iterator[str]:
    """Yield dt.identifier values from an event file."""
    root = etree.fromstring(event_xml)
    for elem in root.iterfind(".//{*}identifier"):
        if elem.text:
            yield elem.text.strip()


def fetch_sru_record(identifier: str) -> etree._Element:
    """Download one SRU record (recordSchema=gzd)."""
    params = {
        "operation": "searchRetrieve",
        "version": "2.0",
        "maximumRecords": "1",
        "recordSchema": "gzd",
        "query": f'dt.identifier="{identifier}"'
    }
    r = SESSION.get("https://repository.overheid.nl/sru", params=params)
    r.raise_for_status()
    return etree.fromstring(r.content)


def extract_manifest_xml_url(sru_root: etree._Element) -> str:
    """Return the gzd:itemUrl with manifestation='xml' (fallback to pdf/html)."""
    ns = {"gzd": "http://standaarden.overheid.nl/gzd/1.0"}
    for item in sru_root.iterfind(".//gzd:itemUrl", namespaces=ns):
        if item.get("manifestation") == "xml":
            return item.text
    # fallback – rare but safe
    first = sru_root.find(".//gzd:itemUrl", namespaces=ns)
    if first is None or first.text is None:
        raise ValueError("itemUrl not found in SRU response")
    return first.text


def plain_text_from_xml(xml_bytes: bytes) -> str:
    """Very lightweight text extraction that keeps paragraphs/line breaks."""
    soup = BeautifulSoup(xml_bytes, "lxml-xml")
    # remove metadata / boilerplate you do not need
    for tag in soup.select("meta, head, style, script"):
        tag.decompose()
    # join all text with newlines to keep some structure
    return "\n".join(t.strip() for t in soup.stripped_strings)


def main() -> None:
    target_day = dt.date.fromisoformat(os.getenv("SCRAPE_DATE", dt.date.today().isoformat()))

    try:
        print(f"Downloading event file for {target_day}")
        event_xml = SESSION.get(daily_event_url(target_day)).content
    except requests.HTTPError as e:
        print(f"No event file for {target_day} ({e}). Exiting.")
        return

    n_new = 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("a", encoding="utf-8") as fh:
        for identifier in iter_identifiers(event_xml):
            try:
                sru_root = fetch_sru_record(identifier)
                manifest_xml_url = extract_manifest_xml_url(sru_root)
                manifest_xml = SESSION.get(manifest_xml_url).content
                text = plain_text_from_xml(manifest_xml)

                # The public‑facing URL preferred by KOOP/OB <gzd:preferredUrl>
                preferred_el = sru_root.find(".//{http://standaarden.overheid.nl/gzd/1.0}prefferedUrl")
                public_url = preferred_el.text if preferred_el is not None else manifest_xml_url

                record = {
                    "url": public_url,
                    "content": text,
                    "source": "Officiële Publicaties"
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_new += 1
            except Exception as ex:
                print(f"[warn] could not process {identifier}: {ex}", file=sys.stderr)

    print(f"Scraping finished – {n_new} new/updated documents appended.")

    # Optional: push raw file to Hugging Face
    if os.getenv("HF_TOKEN"):
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(OUT_PATH),
            path_in_repo=OUT_PATH.name,
            repo_id=HF_REPO,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"),
            commit_message=f"Update for {target_day} ({n_new} records)"
        )
        print("Uploaded to HF.")


if __name__ == "__main__":
    main()
