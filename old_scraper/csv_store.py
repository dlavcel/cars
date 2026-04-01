import csv
import os
import pandas as pd
from config import OUT_CSV, FIELDNAMES
from scraper.utils import record_id_from_url

def get_next_id(filename: str) -> int:
    if not os.path.isfile(filename):
        return 1
    try:
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            last_id = None
            for row in reader:
                last_id = row.get("id")
            if last_id and str(last_id).strip().isdigit():
                return int(str(last_id).strip()) + 1
    except Exception:
        pass
    return 1

def save_to_csv(filename, data):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        row = {k: ("" if data.get(k) is None else data.get(k)) for k in FIELDNAMES}
        writer.writerow(row)

def load_existing_ids_from_csv():
    if not os.path.isfile(OUT_CSV):
        return set()
    df = pd.read_csv(OUT_CSV)
    ids = set()
    for url in df["url"]:
        rid = record_id_from_url(str(url))
        if rid:
            ids.add(rid)
    return ids
