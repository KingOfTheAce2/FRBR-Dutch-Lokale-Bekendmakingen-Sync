import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import json
import os
from huggingface_hub import HfApi

SRU_URL = "https://repository.overheid.nl/sru"
QUERY = "c.product-area==lokalebekendmakingen"
HEADERS = {"Accept": "application/xml"}
BATCH_SIZE = 50
HF_REPO = "vGassen/Dutch-Lokale-Bekendmakingen"
TARGET_FILE = "lokale_bekendmakingen.jsonl"

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

def upload_to_huggingface(jsonl_text):
    temp_file = "temp_append.jsonl"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(jsonl_text)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=temp_file,
        path_in_repo=TARGET_FILE,
        repo_id=HF_REPO,
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
        commit_message="Append 50 new Lokale Bekendmakingen records"
    )
    os.remove(temp_file)

if __name__ == "__main__":
    xml_root = fetch_records()
    records = extract_entries(xml_root)
    if records:
        jsonl_data = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records)
        upload_to_huggingface(jsonl_data)
