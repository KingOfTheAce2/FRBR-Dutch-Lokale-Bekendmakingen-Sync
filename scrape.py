import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import json
from huggingface_hub import HfApi, upload_file
import os

SRU_URL = "https://repository.overheid.nl/sru"
QUERY = "c.product-area==lokalebekendmakingen"
HEADERS = {"Accept": "application/xml"}
BATCH_SIZE = 50
DATASET_FILE = "lokale_bekendmakingen.jsonl"

def fetch_records(start=1, max_records=50):
    params = {
        "version": "2.0",
        "operation": "searchRetrieve",
        "query": QUERY,
        "startRecord": start,
        "maximumRecords": max_records
    }
    response = requests.get(SRU_URL, params=params, headers=HEADERS)
    response.raise_for_status()
    return ET.fromstring(response.content)

def extract_entries(xml_root):
    ns = {'gzd': 'http://standaarden.overheid.nl/gzd'}
    records = []
    for record in xml_root.findall(".//gzd:record", ns):
        url_elem = record.find(".//gzd:preferredUrl", ns)
        url = url_elem.text if url_elem is not None else None
        content = ""
        if url:
            try:
                html = requests.get(url, timeout=10).text
                soup = BeautifulSoup(html, 'html.parser')
                paragraphs = soup.find_all('p')
                content = " ".join(p.get_text(strip=True) for p in paragraphs)
            except Exception:
                content = ""
        if url and content:
            records.append({
                "URL": url,
                "content": content,
                "source": "Lokale Bekendmakingen"
            })
    return records

def save_records(records):
    with open(DATASET_FILE, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def upload_to_huggingface():
    api = HfApi()
    repo_id = "vGassen/Dutch-Lokale-Bekendmakingen"
    api.upload_file(
        path_or_fileobj=DATASET_FILE,
        path_in_repo="lokale_bekendmakingen.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
        commit_message="Append 50 new records"
    )

if __name__ == "__main__":
    xml_root = fetch_records()
    records = extract_entries(xml_root)
    if records:
        save_records(records)
        upload_to_huggingface()
