import time
import random

from selenium.webdriver.common.by import By
from scraper.parsers import get_all_image_urls, parse_vehicle_from_html
from scraper.selenium_safety import dismiss_any_alert, is_rate_limited, rate_limit_cooldown, is_blocked_or_challenge
from scraper.utils import vin_from_car_url, norm_vin

def scrape_one_car_with_retries(driver, wait, url, max_retries=2):

    expected_vin = vin_from_car_url(url)

    for attempt in range(1, max_retries):
        driver.get(url)
        dismiss_any_alert(driver)

        if is_rate_limited(driver):
            rate_limit_cooldown(attempt)
            continue

        if is_blocked_or_challenge(driver):
            backoff = 45 * attempt + random.uniform(10, 30)
            print(f"Panašu į apsaugą. Palaukiam {backoff:.1f}s ir bandom dar kartą ({attempt}/{max_retries})")
            time.sleep(backoff)
            continue

        try:
            wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "h1")) > 0)
            wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, ".lot_information")) > 0)
        except Exception:
            backoff = 10 * attempt + random.uniform(0, 10)
            print(f"⚠️ Neužsikrovė H1/lot info. Palaukiam {backoff:.1f}s ir bandom dar kartą ({attempt}/{max_retries})")
            time.sleep(backoff)
            continue

        images = get_all_image_urls(driver)
        images_str = "|".join(images)

        data = parse_vehicle_from_html(driver, url, images_str=images_str)
        got = norm_vin(data.get("vin") or "")

        # jei tikrinimui matom VIN neatitikimą, dar kartą bandome (dažnai CF/overlay)
        if expected_vin and got and expected_vin != got:
            print(f"⚠️ VIN mismatch (quality check). url_vin={expected_vin} parsed_vin={got}. Retrying page...")
            time.sleep(random.uniform(6, 14))
            continue

        if got:
            data["vin"] = got
            print(f"✅ {got} photos={len(images)}")
        else:
            print(f"✅ (no vin?) photos={len(images)}")

        return data

    return None

def get_car_links_from_list(driver, wait):
    wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, ".lot_list a.ajax_link")) > 0)
    links = driver.find_elements(By.CSS_SELECTOR, ".lot_list a.ajax_link")

    urls = []
    for a in links:
        href = a.get_attribute("href")
        if href and "/en/car/" in href:
            urls.append(href)

    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq