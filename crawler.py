import os
import json
import time
import logging
import requests
from lxml import etree, html
from huggingface_hub import HfApi, HfFolder, upload_file

# ---------------- CONFIGURATION ----------------

API_URL = "https://repository.overheid.nl/sru"  # Change as needed
API_PARAMS = {
    "operation": "searchRetrieve",
    "query": "collection=officielepublicaties",  # SRU query
    "maximumRecords": 100,  # Batch size (SRU max is 1000)
    "startRecord": 1
}
STATE_FILE = "crawler_state.json"
OUTPUT_JSONL = "output.jsonl"
SOURCE_LABEL = "Officiële Publicaties"  # Change as needed
HF_REPO_ID = "vGassen/Dutch-Officiele-Publicaties"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
BATCH_SIZE = 100  # SRU allows up to 1000
SHARD_SIZE = 300  # number of records per JSONL shard file
MAX_ENTRIES = 10000  # Max entries per run
TIMEOUT = 20
RETRIES = 5

# ---------------- LOGGING SETUP ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------- UTILITIES ----------------

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_record": 1, "processed": set()}

def save_state(state):
    # Sets not serializable: convert
    state['processed'] = list(state['processed'])
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
    state['processed'] = set(state['processed'])

def append_jsonl(records, fname):
    with open(fname, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def strip_html(text):
    tree = html.fromstring(text)
    return " ".join(tree.xpath('//text()[normalize-space()]')).strip()

def extract_main_content(xml_bytes):
    """Extract main content from Rechtspraak/OFFP SRU XML record"""
    try:
        tree = etree.fromstring(xml_bytes)
        # Most official publications have <dc:identifier> for the URL and main text in <dcterms:abstract> or similar
        namespaces = tree.nsmap
        url = tree.findtext(".//{*}identifier")
        text = ""
        # Try multiple options for text content
        for tag in [
            ".//{*}description",
            ".//{*}abstract",
            ".//{*}body",
            ".//{*}text",
        ]:
            node = tree.find(tag)
            if node is not None and node.text:
                text = node.text
                break
        # fallback: all text nodes concatenated
        if not text:
            text = " ".join(tree.xpath('.//text()'))
        return url, strip_html(text)
    except Exception as ex:
        logging.error(f"Failed to extract content: {ex}")
        return None, None

def download_url(url):
    tries = 0
    while tries < RETRIES:
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp.text
            else:
                logging.warning(f"Failed to fetch {url}: {resp.status_code}")
        except Exception as ex:
            logging.warning(f"Error downloading {url}: {ex}")
        tries += 1
        time.sleep(2 * tries)
    return None

def push_to_hf(local_path, repo_id, hf_token, filename):
    api = HfApi(token=hf_token)
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    except Exception:
        pass  # repo may already exist
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
        commit_message="Add new data shard"
    )
    logging.info(f"Pushed {filename} to {repo_id}")

# ---------------- MAIN CRAWLER LOOP ----------------

def crawl():
    state = load_state()
    state['processed'] = set(state.get('processed', []))
    total_processed = 0
    shard_counter = 1
    jsonl_records = []

    start_record = state["last_record"]
    logging.info(f"Starting from record {start_record}")

    while total_processed < MAX_ENTRIES:
        params = API_PARAMS.copy()
        params['startRecord'] = start_record
        params['maximumRecords'] = BATCH_SIZE

        tries = 0
        success = False
        while tries < RETRIES:
            try:
                resp = requests.get(API_URL, params=params, timeout=TIMEOUT)
                if resp.status_code == 200:
                    success = True
                    break
                else:
                    logging.warning(f"SRU API error: {resp.status_code}")
            except Exception as ex:
                logging.warning(f"SRU connection error: {ex}")
            tries += 1
            time.sleep(2 * tries)
        if not success:
            logging.error("Max retries exceeded for SRU, exiting.")
            break

        root = etree.fromstring(resp.content)
        records = root.findall(".//{*}record")
        if not records:
            logging.info("No more records.")
            break

        for record in records:
            raw_xml = etree.tostring(record, encoding="utf-8")
            url, main_text = extract_main_content(raw_xml)
            if not url or url in state['processed']:
                continue
            # Fetch main URL for full content
            full_content = download_url(url)
            clean_text = strip_html(full_content) if full_content else main_text
            # Write to memory
            rec = {
                "URL": url,
                "Content": clean_text,
                "Source": SOURCE_LABEL
            }
            jsonl_records.append(rec)
            state['processed'].add(url)
            total_processed += 1
            if total_processed % 50 == 0:
                logging.info(f"Processed {total_processed} records.")

            # Write shard and upload if needed
            if len(jsonl_records) >= SHARD_SIZE:
                shard_fname = f"data_shard_{shard_counter}.jsonl"
                append_jsonl(jsonl_records, shard_fname)
                push_to_hf(shard_fname, HF_REPO_ID, HF_TOKEN, shard_fname)
                jsonl_records = []
                shard_counter += 1

            if total_processed >= MAX_ENTRIES:
                break

        start_record += len(records)
        state['last_record'] = start_record
        save_state(state)

    # Write remaining
    if jsonl_records:
        shard_fname = f"data_shard_{shard_counter}.jsonl"
        append_jsonl(jsonl_records, shard_fname)
        push_to_hf(shard_fname, HF_REPO_ID, HF_TOKEN, shard_fname)

    save_state(state)
    logging.info(f"Finished, total processed: {total_processed}")

if __name__ == "__main__":
    crawl()
