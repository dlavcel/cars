import os
import csv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

CSV_FILE = "cars.csv"
BASE_DIR = "cars"
MAX_WORKERS = 10

def safe_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("_", "-"))

def download_one(url, filepath):
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(r.content)
            return f"Saved {filepath}"
        else:
            return f"Failed {url} ({r.status_code})"
    except Exception as e:
        return f"Error {url}: {e}"

def download_images():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    tasks = []

    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            car_id = row.get("id")
            vin = row.get("vin")
            auction = (row.get("auction_source") or "COPART").upper()
            images_field = row.get("images")

            if not car_id or not vin or not images_field:
                continue

            folder_name = safe_filename(f"{car_id}_{vin}")
            car_folder = os.path.join(BASE_DIR, folder_name)
            os.makedirs(car_folder, exist_ok=True)

            image_urls = [u.strip() for u in images_field.split("|") if u.strip()]

            if auction == "IAAI":
                image_urls = image_urls[:5]
            else:
                image_urls = image_urls[:7]

            for idx, url in enumerate(image_urls, start=1):
                ext = ".jpg"
                if ".png" in url.lower():
                    ext = ".png"

                filepath = os.path.join(car_folder, f"{idx}{ext}")
                tasks.append((url, filepath))

    print(f"Total images to download: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_one, url, path) for url, path in tasks]

        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    download_images()
