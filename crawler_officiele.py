#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crawler for Dutch Official Publications and Local Announcements
Crawls XML records from the FRBR endpoints for officielepublicaties and lokalebekendmakingen,
strips XML/HTML tags, and uploads to a Hugging Face dataset.
Designed to run inside GitHub Actions.

Environment variables:
- HF_TOKEN           : Hugging Face token with write access
- HF_DATASET_REPO    : "vGassen/Dutch-Officiele-Publicaties-Lokale-Bekendmakingen"
- HF_PRIVATE         : "true" or "false" (optional; default false)

Usage:
    python crawler_officiele.py --max-items 500 [--delay 0.2] [--resume]
"""

import os
import time
import logging
from pathlib import Path
from typing import Iterator, Tuple, Dict, Set, List
import argparse

import requests
from bs4 import BeautifulSoup
from lxml import etree
from tqdm import tqdm
from datasets import Dataset, Features, Value
from requests.adapters import HTTPAdapter, Retry

# Constants
BASE_URL = "https://repository.overheid.nl"
ROOT_PATHS = ["/frbr/officielepublicaties", "/frbr/lokalebekendmakingen"]
HEADERS = {"User-Agent": "Dutch-FRBR-Crawler"}
DEFAULT_DELAY = 0.2
PAGE_RETRIES = Retry(total=2, backoff_factor=1.0, status_forcelist=[500,502,503,504])
REQUEST_TIMEOUT = 15
VISITED_FILE = "visited_urls.txt"
CHUNK_SIZE = 1000
SOURCE_LABEL = "Officiele publicaties"

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

class BaseCrawler:
    def __init__(self, delay: float = DEFAULT_DELAY):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.session.mount("https://", HTTPAdapter(max_retries=PAGE_RETRIES))

    def fetch_soup(self, path: str) -> BeautifulSoup:
        url = f"{BASE_URL}{path}"
        for attempt in range(3):
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.ok:
                return BeautifulSoup(resp.text, "lxml")
            logging.warning("Retry %s for %s (%s)", attempt+1, url, resp.status_code)
            time.sleep(self.delay * 2)
        resp.raise_for_status()

    @staticmethod
    def strip_xml(xml_bytes: bytes) -> str:
        parser = etree.XMLParser(recover=True, encoding="utf-8")
        root = etree.fromstring(xml_bytes, parser=parser)
        return " ".join(chunk.strip() for chunk in root.itertext() if chunk.strip())

    @staticmethod
    def push_chunk(
        data: List[Dict[str,str]],
        features: Features,
        repo: str,
        token: str,
        private: bool
    ) -> None:
        ds = Dataset.from_list(data, features=features)
        ds.push_to_hub(
            repo_id=repo,
            token=token,
            split="train",
            private=private,
            max_shard_size="500MB",
        )
        logging.info("Pushed %d records to %s", len(data), repo)

class DutchFRBRCrawler(BaseCrawler):
    def __init__(self, root_path: str, delay: float = DEFAULT_DELAY):
        super().__init__(delay)
        self.root_path = root_path

    def iter_document_paths(self) -> Iterator[str]:
        seen: Set[str] = set()
        offset = 0
        while True:
            page_path = f"{self.root_path}?start={offset}" if offset else self.root_path
            soup = self.fetch_soup(page_path)
            links = soup.select(f"a[href^='{self.root_path}']")
            new_found = 0
            for a in links:
                href = a['href'].split('?')[0].rstrip('/')
                if href.count('/') >= 4 and href not in seen:
                    seen.add(href)
                    yield href
                    new_found += 1
            if new_found == 0:
                break
            offset += 50
            time.sleep(self.delay)

    def iter_xml_records(self, doc_path: str) -> Iterator[Tuple[str, bytes]]:
        # Attempt direct XML endpoint
        xml_url = f"{BASE_URL}{doc_path}/1?format=xml"
        try:
            resp = self.session.get(xml_url, timeout=REQUEST_TIMEOUT)
            if resp.ok and resp.headers.get('Content-Type','').startswith('application/xml'):
                yield xml_url, resp.content
                return
        except Exception:
            pass
        # Fallback: look for individual .xml links
        soup = self.fetch_soup(f"{doc_path}/1/xml/")
        links = soup.select("a[href$='.xml']")
        for a in links:
            href = a['href']
            url = href if href.startswith('http') else f"{BASE_URL}{href}"
            try:
                r = self.session.get(url, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                yield url, r.content
            except Exception as e:
                logging.warning("Failed to fetch XML %s: %s", url, e)
            time.sleep(self.delay)

    def records_stream(self, limit: int = None, resume: bool = False) -> Iterator[Dict[str,str]]:
        seen: Set[str] = set()
        fh = None
        if resume and os.path.exists(VISITED_FILE):
            with open(VISITED_FILE, 'r', encoding='utf-8') as f:
                seen.update(line.strip() for line in f)
        if resume:
            fh = open(VISITED_FILE, 'a', encoding='utf-8')

        grabbed = 0
        features = Features({
            'url': Value('string'),
            'content': Value('string'),
            'source': Value('string'),
        })

        for doc_path in tqdm(self.iter_document_paths(), desc=f"Listing {self.root_path}"):
            for url, xml_bytes in tqdm(self.iter_xml_records(doc_path), desc="Fetching XML", leave=False):
                if url in seen:
                    continue
                text = self.strip_xml(xml_bytes)
                yield {'url': url, 'content': text, 'source': SOURCE_LABEL}
                grabbed += 1
                if fh:
                    fh.write(url + '\\n')
                    fh.flush()
                if limit is not None and grabbed >= limit:
                    if fh: fh.close()
                    return
            time.sleep(self.delay)
        if fh:
            fh.close()

def push_datasets(args: argparse.Namespace):
    repo = os.environ['HF_DATASET_REPO']
    token = os.environ['HF_TOKEN']
    private = os.getenv('HF_PRIVATE','false').lower() == 'true'
    features = Features({
        'url': Value('string'),
        'content': Value('string'),
        'source': Value('string'),
    })

    chunk: List[Dict[str,str]] = []
    total = 0
    # iterate both collections
    for root in ROOT_PATHS:
        crawler = DutchFRBRCrawler(root, delay=args.delay)
        for record in crawler.records_stream(limit=args.max_items, resume=args.resume):
            chunk.append(record)
            if len(chunk) >= CHUNK_SIZE:
                BaseCrawler.push_chunk(chunk, features, repo, token, private)
                total += len(chunk)
                chunk.clear()
    if chunk:
        BaseCrawler.push_chunk(chunk, features, repo, token, private)
        total += len(chunk)
    logging.info("Upload complete: %d records", total)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crawl Dutch FRBR collections")
    parser.add_argument('--max-items', type=int, default=500, help='Max XML records to fetch')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY, help='Crawl delay (s)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    args = parser.parse_args()
    try:
        push_datasets(args)
    except KeyError as e:
        logging.critical("Missing environment variable: %s", e)
        exit(1)
