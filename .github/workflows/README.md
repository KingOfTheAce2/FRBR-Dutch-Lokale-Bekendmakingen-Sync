# Dutch Lokale Bekendmakingen Scraper

This repo collects recent entries from the Dutch "Lokale Bekendmakingen" (local government announcements) and publishes them to the Hugging Face dataset:

👉 https://huggingface.co/datasets/vGassen/Dutch-Lokale-Bekendmakingen

### Fields:
- `URL`: Link to the full document
- `content`: Scraped text content
- `source`: Always `"Lokale Bekendmakingen"`

### Usage

Run locally:
```bash
python scrape.py
