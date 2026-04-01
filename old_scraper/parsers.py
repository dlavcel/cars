import json
import re
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By

from scraper.selenium_safety import safe_page_source
from scraper.utils import norm_vin, parse_mileage, vin_from_car_url, parse_int_like, normalize_drive_text


def extract_currency(soup: BeautifulSoup) -> str | None:
    # 1) JSON-LD
    for s in soup.select("script[type='application/ld+json']"):
        txt = (s.string or s.get_text() or "").strip()
        if not txt:
            continue
        try:
            obj = json.loads(txt)
        except Exception:
            continue

        # kartais JSON-LD būna list
        items = obj if isinstance(obj, list) else [obj]
        for it in items:
            if isinstance(it, dict) and it.get("@type") == "Vehicle":
                offers = it.get("offers") or {}
                cur = offers.get("priceCurrency")
                if cur:
                    return str(cur).strip().upper()

    # 2) HTML fallback: <span class="currency">USD</span>
    cur_el = soup.select_one(".lot_sold .currency, .lot_price_range .currency")
    if cur_el:
        return cur_el.get_text(" ", strip=True).upper()

    return None

def extract_auction_source(soup: BeautifulSoup):
    """
    Auction source iš 'Auction:' eilutės.
    Veikia tiek kai yra <a href="copart...">, tiek kai lieka tik <img title="Copart">.
    """
    info = soup.select_one(".lot_information")
    if not info:
        return None

    # randam eilutę su label "Auction"
    for row in info.select("div"):
        b = row.find("b")
        if not b:
            continue
        label = b.get_text(" ", strip=True).rstrip(":").strip().lower()
        if label != "auction":
            continue

        # 1) jei dar yra linkas
        a = row.find("a", href=True)
        if a:
            href = (a.get("href") or "").lower()
            if "copart" in href:
                return "COPART"
            if "iaai" in href:
                return "IAAI"
            return "OTHER"

        # 2) jei linko nėra, bet yra img
        img = row.find("img")
        if img:
            title = (img.get("title") or "").strip().lower()
            alt = (img.get("alt") or "").strip().lower()
            src = (img.get("src") or "").strip().lower()

            blob = " ".join([title, alt, src])

            if "copart" in blob:
                return "COPART"
            if "iaai" in blob:
                return "IAAI"
            return "OTHER"

        return None

    return None

def extract_raw_year_vin_from_h1(h1_text: str):
    """
    H1: '2019 Land Rover Range Rover Sport Se vin: SALWG2RV7KA836801'
    -> raw: 'Land Rover Range Rover Sport Se'
       year: '2019'
       vin: 'SALWG2RV7KA836801'
    """
    if not h1_text:
        return None, None, None

    txt = " ".join(h1_text.split())

    vin = None
    mvin = re.search(r"\bvin\s*:\s*([A-HJ-NPR-Z0-9]{11,20})\b", txt, flags=re.IGNORECASE)
    if mvin:
        vin = mvin.group(1)

    left = re.split(r"\bvin\s*:\b", txt, flags=re.IGNORECASE)[0].strip()

    year = None
    raw = left.strip()

    my = re.match(r"^\s*(\d{4})\s+(.*)$", left)
    if my:
        year = my.group(1)
        raw = my.group(2).strip()

    raw = re.sub(r"\bvin\s*:\s*[A-HJ-NPR-Z0-9]{11,20}\b", "", raw, flags=re.IGNORECASE).strip()
    return raw or None, year, vin

def extract_kv_from_lot_information(soup: BeautifulSoup):
    """
    Iš <div class="mt-2"><b>Label:</b> value</div> sudeda į dict
    """
    data = {}
    info = soup.select_one(".lot_information")
    if not info:
        return data

    for b in info.find_all("b"):
        label = b.get_text(" ", strip=True).rstrip(":").strip().lower()
        parent = b.parent
        if not parent:
            continue
        full = parent.get_text(" ", strip=True)
        value = re.sub(r"^\s*" + re.escape(b.get_text(" ", strip=True)) + r"\s*", "", full).strip()
        if label:
            data[label] = value

    sold_box = soup.select_one(".lot_sold")
    if sold_box:
        txt = sold_box.get_text(" ", strip=True)
        m = re.search(r"Winning bet:\s*\$?\s*([\d\s]+)", txt, flags=re.IGNORECASE)
        if m:
            data["winning_bet"] = m.group(1).strip()

    for el in soup.select(".lot_price_range .text-left"):
        t = el.get_text(" ", strip=True)
        m = re.search(r"Average price:\s*\$?\s*([\d\s]+)", t, flags=re.IGNORECASE)
        if m:
            data["average_price"] = m.group(1).strip()
            break

    return data

def get_all_image_urls(driver):
    anchors = driver.find_elements(By.CSS_SELECTOR, "a[data-fancybox='gallery'][href]")
    urls = []
    for a in anchors:
        href = a.get_attribute("href")
        if href and (".jpg" in href.lower() or ".jpeg" in href.lower() or ".png" in href.lower()):
            urls.append(href)

    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def parse_vehicle_from_html(driver, url: str, images_str: str):
    soup = BeautifulSoup(safe_page_source(driver), "html.parser")

    h1 = soup.select_one("h1")
    h1_text = h1.get_text(" ", strip=True) if h1 else ""

    raw, year, vin_h1 = extract_raw_year_vin_from_h1(h1_text)

    kv = extract_kv_from_lot_information(soup)

    # VIN imame iš lot info; jei nėra – iš H1; jei vis tiek nėra – iš URL
    vin = norm_vin((kv.get("vin") or "").strip() or (vin_h1 or ""))
    if not vin:
        vin = vin_from_car_url(url)

    mileage_val, mileage_unit = parse_mileage(kv.get("mileage", ""))

    price = None
    if kv.get("winning_bet"):
        price = parse_int_like(kv.get("winning_bet"))
    elif kv.get("average_price"):
        price = parse_int_like(kv.get("average_price"))

    currency = extract_currency(soup) or "USD"

    who_sell = soup.select_one(".who_sell")
    seller = who_sell.get_text(" ", strip=True) if who_sell else None

    auction_source = extract_auction_source(soup)

    damage_primary = kv.get("primary damage")
    damage_secondary = kv.get("secondary damage")

    drive = normalize_drive_text(kv.get("drive"))

    key_val = kv.get("key")
    cost_of_repair = parse_int_like(kv.get("cost of repair"))
    market_value = parse_int_like(kv.get("market value"))

    return {
        "url": url,
        "vin": vin,
        "year": year,
        "raw": raw,

        "price": price,
        "currency": currency,
        "seller": seller,
        "auction_source": auction_source,

        "mileage": mileage_val,
        "mileage_unit": mileage_unit,
        "engine": kv.get("engine"),
        "fuel": kv.get("fuel"),
        "transmission": kv.get("transmission"),
        "color": kv.get("color"),

        "damage_primary": damage_primary,
        "damage_secondary": damage_secondary,
        "drive": drive,

        "key": key_val,
        "cost_of_repair": cost_of_repair,
        "market_value": market_value,

        "images": images_str or "",
    }