#!/usr/bin/env python3
import os, json, datetime as dt
from pathlib import Path
import requests
from lxml import etree
from bs4 import BeautifulSoup
from huggingface_hub import HfApi

HF_REPO = "vGassen/Dutch-Officiele-Publicaties-Lokale-Bekendmakingen"
OUT_PATH = Path("data/officielepublicaties.jsonl")
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "dual-sru-scraper"
SESSION.timeout = (10, 30)

COLLECTIONS = ["officielepublicaties", "lokalebekendmakingen"]
SRU_URL = "https://repository.overheid.nl/sru"

def search_records(product_area: str, date: str) -> list[str]:
    """Returns a list of identifiers modified on a given date in a collection."""
    query = f'c.product-area=={product_area} AND dt.modified=="{date}"'
    params = {
        "operation": "searchRetrieve",
        "version": "2.0",
        "maximumRecords": "1000",
        "recordSchema": "gzd",
        "query": query
    }
    r = SESSION.get(SRU_URL, params=params)
    r.raise_for_status()
    root = etree.fromstring(r.content)
    return [el.text for el in root.findall(".//{*}identifier") if el.text]

def fetch_sru_record(identifier: str) -> etree._Element:
    params = {
        "operation": "searchRetrieve",
        "version": "2.0",
        "maximumRecords": "1",
        "recordSchema": "gzd",
        "query": f'dt.identifier="{identifier}"'
    }
    r = SESSION.get(SRU_URL, params=params)
    r.raise_for_status()
    return etree.fromstring(r.content)

def extract_manifest_xml_url(root: etree._Element) -> str:
    for el in root.iterfind(".//{*}itemUrl"):
        if el.get("manifestation") == "xml":
            return el.text
    raise ValueError("No XML manifestation found")

def extract_preferred_url(root: etree._Element) -> str | None:
    el = root.find(".//{*}prefferedUrl")
    return el.text if el is not None else None

def plain_text_from_xml(xml_bytes: bytes) -> str:
    soup = BeautifulSoup(xml_bytes, "lxml-xml")
    for tag in soup.select("meta, head, style, script"):
        tag.decompose()
    return "\n".join(s.strip() for s in soup.stripped_strings)

def main():
    scrape_date = os.getenv("SCRAPE_DATE", dt.date.today().isoformat())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with OUT_PATH.open("a", encoding="utf-8") as fh:
        for collection in COLLECTIONS:
            try:
                identifiers = search_records(collection, scrape_date)
            except Exception as e:
                print(f"Failed fetching from {collection}: {e}")
                continue

            for identifier in identifiers:
                try:
                    root = fetch_sru_record(identifier)
                    manifest_url = extract_manifest_xml_url(root)
                    manifest_xml = SESSION.get(manifest_url).content
                    text = plain_text_from_xml(manifest_xml)
                    url = extract_preferred_url(root) or manifest_url
                    record = {
                        "url": url,
                        "content": text,
                        "source": "Officiële Publicaties"
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                except Exception as e:
                    print(f"Error on {identifier}: {e}")
    
    print(f"Finished: {written} documents written for {scrape_date}")

    if os.getenv("HF_TOKEN"):
        HfApi().upload_file(
            path_or_fileobj=str(OUT_PATH),
            path_in_repo=OUT_PATH.name,
            repo_id=HF_REPO,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"),
            commit_message=f"Update {scrape_date} ({written} items)"
        )
        print("Uploaded to Hugging Face.")

if __name__ == "__main__":
    main()
