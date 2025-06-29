# Dutch Lokale Bekendmakingen Scraper

This project collects recent entries from the Dutch "Lokale Bekendmakingen" (local government announcements) and publishes them to the [Hugging Face dataset](https://huggingface.co/datasets/vGassen/Dutch-Lokale-Bekendmakingen).

### Fields
- `url` – link to the document
- `content` – cleaned text content
- `source` – either `officielepublicaties` or `lokalebekendmakingen`

### Usage

Python 3.11+ is recommended. The scraping pipeline consists of four scripts that should be run in order:

1. **Collect URLs**
   ```bash
   python step1_collect_urls.py
   ```
2. **Download the items**
   ```bash
   python step2_download_items.py
   ```
3. **Clean text and create JSONL**
   ```bash
   python step3_clean_text.py
   ```
4. **Upload to the dataset**
   ```bash
   python step4_upload_shards.py --token <HF_TOKEN>
   ```

Pass `--flush` on the first manual run to remove any existing shards from the
Hugging Face dataset before uploading new data.

You can also store the Hugging Face token in a `.env` file using `HF_TOKEN=<token>`.
