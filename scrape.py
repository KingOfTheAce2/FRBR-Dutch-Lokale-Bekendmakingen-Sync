import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import json
import os
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

SRU_URL = "https://repository.overheid.nl/sru"
QUERY = "c.product-area==lokalebekendmakingen"
HEADERS = {"Accept": "application/xml"}
BATCH_SIZE = 50
HF_REPO = "vGassen/Dutch-Lokale-Bekendmakingen"
TARGET_FILE = "lokale_bekendmakingen.jsonl"


def fetch_records(start=1, max_records=50):
    print(f"Fetching {max_records} records starting from {start}...")
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
            except Exception as e:
                print(f"Failed to scrape content from {url}: {e}")
                content = ""
        if url and content:
            records.append({
                "URL": url,
                "content": content,
                "source": "Lokale Bekendmakingen"
            })
    print(f"Extracted {len(records)} valid records.")
    if records:
        print("Sample record:", json.dumps(records[0], indent=2, ensure_ascii=False))
    return records


def ensure_file_exists(api, repo_id, file_path):
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if file_path not in files:
            print(f"File {file_path} not found in repo. Creating empty file...")
            with open("empty.jsonl", "w", encoding="utf-8") as f:
                f.write("")
            api.upload_file(
                path_or_fileobj="empty.jsonl",
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=os.environ["HF_TOKEN"],
                commit_message="Initialize empty dataset file"
            )
            os.remove("empty.jsonl")
        else:
            print(f"File {file_path} exists in repo.")
    except (RepositoryNotFoundError, HfHubHTTPError) as e:
        print(f"Error accessing repo or checking file: {e}")
        raise


def upload_to_huggingface(jsonl_text):
    temp_file = "temp_append.jsonl"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(jsonl_text)

    api = HfApi()
    ensure_file_exists(api, HF_REPO, TARGET_FILE)

    print(f"Uploading {temp_file} to {HF_REPO}/{TARGET_FILE}...")
    api.upload_file(
        path_or_fileobj=temp_file,
        path_in_repo=TARGET_FILE,
        repo_id=HF_REPO,
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
        commit_message="Append 50 new Lokale Bekendmakingen records"
    )
    print("Upload complete.")
    os.remove(temp_file)


if __name__ == "__main__":
    xml_root = fetch_records()
    records = extract_entries(xml_root)
    if records:
        jsonl_data = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records)
        upload_to_huggingface(jsonl_data)
    else:
        print("No valid records found. Skipping upload.")
