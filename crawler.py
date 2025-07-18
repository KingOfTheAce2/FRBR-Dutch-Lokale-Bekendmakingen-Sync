import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List
import requests
from lxml import etree
from huggingface_hub import HfApi, create_repo

# ---------------- CONFIGURATION ----------------
SRU_URL = "https://repository.overheid.nl/sru"
CQL_QUERY = "c.product-area==lokalebekendmakingen"   # Lokale Bekendmakingen
SRU_VERSION = "2.0"
BATCH_SIZE = 1000              # Max per request
STATE_PATH = "crawler_state.json"
OUTPUT_JSONL = "output.jsonl"
HF_REPO_ID = "vGassen/Dutch-Lokale-Bekendmakingen"
SOURCE = "Lokale Bekendmakingen"
SHARD_SIZE = 300               # ≤300 records per HF push
LOGLEVEL = logging.INFO
# ------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=LOGLEVEL
)

def load_state() -> int:
    if Path(STATE_PATH).exists():
        with open(STATE_PATH, "r") as f:
            return json.load(f).get("start_record", 1)
    return 1

def save_state(start_record: int):
    with open(STATE_PATH, "w") as f:
        json.dump({"start_record": start_record}, f)

def strip_html(text: str) -> str:
    try:
        parser = etree.HTMLParser(recover=True)
        root = etree.fromstring(f"<div>{text}</div>", parser=parser)
        cleaned = " ".join(root.itertext())
        return " ".join(cleaned.split())
    except Exception:
        return text

def parse_record(record_xml: bytes) -> Dict[str, str] | None:
    try:
        root = etree.fromstring(record_xml)
        # Find the URL in enrichedData/locationURI if present
        url = ""
        url_el = root.find(".//locationURI")
        if url_el is not None and url_el.text:
            url = url_el.text.strip()
        # Gather content from meta, body, fallback to all text
        parts = []
        meta = root.find(".//meta")
        if meta is not None:
            for e in meta.iter():
                if e.text and e.tag not in ["identifier", "locationURI"]:
                    parts.append(e.text)
        body = root.find(".//body")
        if body is not None:
            parts.append(strip_html(etree.tostring(body, encoding="unicode")))
        if not parts:
            parts.append(strip_html(etree.tostring(root, encoding="unicode")))
        content = "\n".join(parts)
        content = strip_html(content)
        if not url:
            # fallback: use identifier if locationURI missing
            id_el = root.find(".//identifier")
            url = id_el.text.strip() if id_el is not None and id_el.text else ""
        return {
            "URL": url,
            "Content": content,
            "Source": SOURCE
        }
    except Exception as ex:
        logging.warning(f"Failed to parse record: {ex}")
        return None

def fetch_batch(start_record: int) -> List[Dict[str, str]]:
    params = {
        "version": SRU_VERSION,
        "operation": "searchRetrieve",
        "query": CQL_QUERY,
        "startRecord": start_record,
        "maximumRecords": BATCH_SIZE
    }
    for retry in range(5):
        try:
            resp = requests.get(SRU_URL, params=params, timeout=60)
            resp.raise_for_status()
            root = etree.fromstring(resp.content)
            records = root.findall(".//{*}recordData")
            batch = []
            for r in records:
                # Some SRU XML wraps <recordData><gzd>...</gzd></recordData>
                gzd = r[0] if len(r) > 0 else r
                rec_xml = etree.tostring(gzd, encoding="utf-8")
                doc = parse_record(rec_xml)
                if doc and doc["Content"].strip():
                    batch.append(doc)
            logging.info(f"Fetched batch: {len(batch)} records (startRecord={start_record})")
            return batch
        except Exception as ex:
            logging.warning(f"Fetch batch failed (attempt {retry+1}): {ex}")
            time.sleep(2 ** retry)
    raise RuntimeError("Failed to fetch batch after retries")

def append_jsonl(records: List[Dict[str, str]], path: str):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def shard_jsonl(path: str, shard_size: int) -> List[str]:
    shards = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i in range(0, len(lines), shard_size):
        shard_path = f"{path}_shard_{i}_{i+shard_size}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as s:
            s.writelines(lines[i:i+shard_size])
        shards.append(shard_path)
    return shards

def push_to_hf(shard_paths: List[str]):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN env variable for Hugging Face access.")
    api = HfApi()
    create_repo(HF_REPO_ID, repo_type="dataset", exist_ok=True, token=token)
    for shard_path in shard_paths:
        name = Path(shard_path).name
        logging.info(f"Pushing {name} to HuggingFace ({HF_REPO_ID}) ...")
        api.upload_file(
            path_or_fileobj=shard_path,
            path_in_repo=f"data/{name}",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=token
        )
        logging.info(f"Uploaded: {name}")

def main():
    start_record = load_state()
    all_count = 0
    while True:
        batch = fetch_batch(start_record)
        if not batch:
            logging.info("No more records. Finished.")
            break
        append_jsonl(batch, OUTPUT_JSONL)
        all_count += len(batch)
        start_record += len(batch)
        save_state(start_record)
        logging.info(f"Saved {all_count} total records so far. Next startRecord={start_record}")
        # Shard and push after each batch or at the end
        if all_count % SHARD_SIZE == 0:
            shards = shard_jsonl(OUTPUT_JSONL, SHARD_SIZE)
            push_to_hf(shards)
            logging.info("Pushed all shards to Hugging Face.")
            for s in shards:
                os.remove(s)
    # Final push for leftovers
    shards = shard_jsonl(OUTPUT_JSONL, SHARD_SIZE)
    push_to_hf(shards)
    logging.info("Done: All data uploaded.")

if __name__ == "__main__":
    main()
