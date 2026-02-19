import random
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from scraper.config import OUT_CSV, LIST_URL
from scraper.csv_store import load_existing_ids_from_csv, get_next_id, save_to_csv
from scraper.progress import load_progress, save_progress
from scraper.scraping import get_car_links_from_list, scrape_one_car_with_retries
from scraper.selenium_safety import is_rate_limited, rate_limit_cooldown, is_blocked_or_challenge, wait_out_cloudflare, \
    sleep_jitter
from scraper.utils import get_page_from_progress_url, record_id_from_url


def main():
    progress = load_progress()
    seen_ids = load_existing_ids_from_csv()

    # start page iš progress.json
    start_url = progress.get("current_list_url") or LIST_URL
    page = get_page_from_progress_url(start_url)

    next_id = get_next_id(OUT_CSV)

    opts = webdriver.ChromeOptions()
    opts.debugger_address = "127.0.0.1:9222"
    driver = webdriver.Chrome(options=opts)
    wait = WebDriverWait(driver, 30)

    cars_visited = 0
    new_written = 0
    total = 0

    while True:
        current_list_url = f"https://autohelperbot.com/en/sales?vehicle=AUTOMOBILE&page={page}"
        print(f"\n=== PAGE {page} ===")
        driver.get(current_list_url)

        if is_rate_limited(driver):
            rate_limit_cooldown(1)
            driver.get(current_list_url)

        if is_blocked_or_challenge(driver):
            print("CF/challenge. Laukiam, kol praeis automatiškai...")
            ok = wait_out_cloudflare(driver, timeout=120)
            if not ok:
                print("CF nepraėjo per timeout. Darom ilgesnį backoff ir bandom vėliau.")
                time.sleep(random.uniform(60, 120))
                driver.get(current_list_url)
                ok = wait_out_cloudflare(driver, timeout=120)
            if not ok:
                print("Reikia rankinio sprendimo (captcha). Praeik naršyklėje ir tęsiam automatiškai.")
                while is_blocked_or_challenge(driver):
                    time.sleep(5)
                driver.get(current_list_url)

        try:
            wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, ".lot_list a.ajax_link")) > 0)
        except Exception:
            print("Nebėra įrašų (arba puslapis tuščias). Baigiam.")
            break

        car_urls = get_car_links_from_list(driver, wait)
        print(f"Cars found: {len(car_urls)}")

        if not car_urls:
            print("Nebėra automobilių šiame puslapyje. Stop.")
            break

        for url in car_urls:
            rec_id = record_id_from_url(url)

            if rec_id in seen_ids:
                print(f"↩️ already seen record_id={rec_id}")
                continue

            data = scrape_one_car_with_retries(driver, wait, url, max_retries=2)
            cars_visited += 1

            if data:
                data["id"] = next_id
                next_id += 1

                save_to_csv(OUT_CSV, data)
                seen_ids.add(rec_id)
                total += 1
                new_written += 1

                # kad kitą kartą startuotų nuo šito page
                save_progress(current_list_url, seen_ids)

            # agresyvumas sumažintas
            sleep_jitter(8, 16)

            # kas 15 aplankytų auto – trumpas cooldown
            if cars_visited % 15 == 0:
                cool = random.uniform(20, 45)
                print(f"🧊 Cooldown after {cars_visited} cars: {cool:.0f}s")
                time.sleep(cool)

            # kas 60 naujų įrašų – ilgesnis cooldown
            if new_written > 0 and new_written % 60 == 0:
                cool = random.uniform(60, 181)
                print(f"😴 Long cooldown after {new_written} new rows: {cool:.0f}s")
                time.sleep(cool)

        save_progress(current_list_url, seen_ids)
        page += 1

        # tarp puslapių pauzė
        sleep_jitter(16, 25.2)

    print("DONE. Total new rows:", total)

if __name__ == "__main__":
    main()
