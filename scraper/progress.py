import json
import os
from config import PROGRESS_FILE, LIST_URL

def load_progress():
    if os.path.isfile(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            p = json.load(f)
            if "seen_ids" not in p:
                p["seen_ids"] = []
            return p
    return {"current_list_url": LIST_URL, "seen_ids": []}

def save_progress(current_list_url, seen_ids):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"current_list_url": current_list_url, "seen_ids": list(seen_ids)},
            f,
            ensure_ascii=False,
            indent=2,
        )
