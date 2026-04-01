import json
import random
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import cloudscraper
from bs4 import BeautifulSoup

from proxy_login import create_authenticated_scraper, HEADERS, LoginError


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "tasks.db")

LIST_URL = "https://autohelperbot.com/en/sales"
LIST_PARAMS_BASE = {"vehicle": "AUTOMOBILE"}
BASE_DOMAIN = "https://autohelperbot.com"

REQUEST_TIMEOUT = 30

LIST_SLEEP = (1.5, 3.0)
DETAIL_SLEEP = (1.0, 2.5)

BATCH_SIZE_LIST = 10
BATCH_PAUSE_LIST = (30, 90)

BATCH_SIZE_DETAIL = 25
BATCH_PAUSE_DETAIL = (15, 40)

thread_local = threading.local()


class RateLimitedError(Exception):
    pass


class CloudflareChallengeError(Exception):
    pass


class AuthExpiredError(Exception):
    pass


class UnexpectedContentError(Exception):
    pass


class PaceController:
    def __init__(self, min_delay=1.5, max_delay=3.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_ts = 0.0
        self.lock = threading.Lock()

    def update(self, min_delay: float, max_delay: float):
        with self.lock:
            self.min_delay = min_delay
            self.max_delay = max_delay

    def wait(self):
        with self.lock:
            now = time.time()
            target_gap = random.uniform(self.min_delay, self.max_delay)
            elapsed = now - self.last_ts
            if elapsed < target_gap:
                time.sleep(target_gap - elapsed)
            self.last_ts = time.time()


def log(msg: str):
    print(msg, flush=True)


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_detail_url(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def load_txt_account(path: str, line_number: int = 1) -> dict:
    """
    Формат строки:
    ip:port:username:password:email:site_password
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if line_number < 1 or line_number > len(lines):
        raise ValueError(f"Invalid line_number={line_number}, available lines: {len(lines)}")

    raw = lines[line_number - 1]
    parts = raw.split(":", 5)

    if len(parts) != 6:
        raise ValueError(
            "Invalid account format. Expected: ip:port:username:password:email:site_password"
        )

    ip, port, proxy_user, proxy_pass, email, site_password = parts
    proxy = f"http://{proxy_user}:{proxy_pass}@{ip}:{port}"

    return {
        "worker_name": f"worker_{line_number}",
        "email": email,
        "password": site_password,
        "proxy": proxy,
        "proxy_ip": ip,
        "proxy_port": port,
        "proxy_user": proxy_user,
        "proxy_pass": proxy_pass,
    }

def split_known_vehicle_damages(value: str) -> tuple[str, str]:
    """
    knownVehicleDamages: "FRONT END, UNDERCARRIAGE"
    -> primary_damage="FRONT END", secondary_damage="UNDERCARRIAGE"
    """
    value = clean_text(value)
    if not value:
        return "", ""

    parts = [clean_text(x) for x in value.split(",") if clean_text(x)]
    if not parts:
        return "", ""

    primary = parts[0]
    secondary = parts[1] if len(parts) > 1 else ""
    return primary, secondary


def extract_photo_urls(soup) -> list[str]:
    urls = []
    seen = set()

    # Лучший источник — ссылки галереи
    for a in soup.select('a[data-fancybox="gallery"]'):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href not in seen:
            seen.add(href)
            urls.append(href)

    # Фолбэк — если вдруг fancybox-ссылок нет
    if not urls:
        for img in soup.select(".mySwiper2 img, .mySwiper img"):
            src = (img.get("src") or "").strip()
            if not src:
                continue
            src = src.split("?", 1)[0]
            if src not in seen:
                seen.add(src)
                urls.append(src)

    return urls

def load_config(path: str, line_number: int | None = None) -> dict:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required = ["worker_name", "email", "password"]
        missing = [k for k in required if not data.get(k)]
        if missing:
            raise ValueError(f"Missing required config keys: {', '.join(missing)}")
        return data

    if path.lower().endswith(".txt"):
        if line_number is None:
            line_number = 1
        return load_txt_account(path, line_number)

    raise ValueError("Unsupported config format. Use .json or .txt")


def db():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 60000;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def reset_scraper():
    if hasattr(thread_local, "scraper"):
        del thread_local.scraper
    if hasattr(thread_local, "public_scraper"):
        del thread_local.public_scraper


def rate_limit_cooldown_seconds(attempt: int = 1) -> float:
    base = random.uniform(180, 420)
    return base * max(1, attempt)


def challenge_cooldown_seconds(attempt: int = 1) -> float:
    base = random.uniform(300, 900)
    return base * max(1, attempt)


def short_error_pause() -> float:
    return random.uniform(10, 25)


def auth_retry_pause() -> float:
    return random.uniform(15, 40)


def batch_pause(mode: str):
    if mode == "list":
        sec = random.uniform(*BATCH_PAUSE_LIST)
    else:
        sec = random.uniform(*BATCH_PAUSE_DETAIL)

    log(f"[{mode.upper()}][BATCH_PAUSE] sleep={sec:.1f}s")
    time.sleep(sec)


def is_blocked_or_challenge_html(html: str, url: str = "") -> bool:
    low = (html or "").lower()

    markers = [
        "checking your browser before accessing",
        "just a moment...",
        "cf-browser-verification",
        "attention required!",
        "/cdn-cgi/challenge-platform/",
        "please enable javascript and cookies to continue",
    ]

    return any(m in low for m in markers)


def is_rate_limited_response(status_code: int, html: str, headers: dict) -> bool:
    if status_code == 429:
        return True

    low = (html or "").lower()
    markers = [
        "too many attempts",
        "please contact support",
        "rate limit",
        "too many requests",
    ]
    if any(m in low for m in markers):
        return True

    remaining = headers.get("x-ratelimit-remaining")
    if remaining is not None:
        try:
            if int(remaining) <= 0:
                return True
        except Exception:
            pass

    return False


def looks_like_auth_problem(status_code: int, html: str, final_url: str = "") -> bool:
    low = (html or "").lower()
    final_url = (final_url or "").lower()

    if status_code in (401, 403):
        return True

    if final_url.endswith("/login") or "/login?" in final_url:
        return True

    if 'class="btn btn-info">sign in<' in low:
        return True

    if 'form method="post" action="https://autohelperbot.com/en/login"' in low and 'name="password"' in low:
        return True

    return False


def adaptive_delay_from_headers(headers: dict, default_rng: tuple[float, float]) -> tuple[float, float]:
    remaining = headers.get("x-ratelimit-remaining")
    try:
        remaining = int(remaining)
    except Exception:
        return default_rng

    if remaining > 100:
        return default_rng
    if remaining > 50:
        return (2.5, 5.0)
    if remaining > 20:
        return (5.0, 10.0)
    return (10.0, 20.0)


def get_pace(mode: str) -> PaceController:
    pace = getattr(thread_local, f"pace_{mode}", None)
    if pace is None:
        if mode == "list":
            pace = PaceController(*LIST_SLEEP)
        else:
            pace = PaceController(*DETAIL_SLEEP)
        setattr(thread_local, f"pace_{mode}", pace)
    return pace


def get_scraper(config: dict):
    scraper = getattr(thread_local, "scraper", None)
    if scraper is None:
        log(
            f"[AUTH] init scraper worker={config['worker_name']} "
            f"email={config['email']} proxy={config.get('proxy')}"
        )
        scraper = create_authenticated_scraper(
            email=config["email"],
            password=config["password"],
            proxy=config.get("proxy"),
            timeout=REQUEST_TIMEOUT,
        )
        thread_local.scraper = scraper
    return scraper

def get_simple_bold_label_value(soup, label_text: str) -> str:
    target = label_text.lower().rstrip(":")

    for div in soup.select("div.mt-2, div.mb-2, div.mb-4"):
        b = div.find("b")
        if not b:
            continue

        label = clean_text(b.get_text(" ", strip=True)).rstrip(":").lower()
        if label != target:
            continue

        full_text = clean_text(div.get_text(" ", strip=True))
        label_text_raw = clean_text(b.get_text(" ", strip=True))
        value = full_text.replace(label_text_raw, "", 1).strip(" :")
        return clean_text(value)

    return ""


def get_public_scraper(config: dict):
    scraper = getattr(thread_local, "public_scraper", None)
    if scraper is None:
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True}
        )

        proxy = config.get("proxy")
        if proxy:
            scraper.proxies.update({
                "http": proxy,
                "https": proxy,
            })

        thread_local.public_scraper = scraper
    return scraper


def validate_sales_page(html: str, url: str):
    low = (html or "").lower()
    if "copart and iaai lot sales history" in low:
        return
    if 'class="lot_list"' in low:
        return
    raise UnexpectedContentError(f"Unexpected sales page content: {url}")


def validate_detail_page(html: str, url: str):
    low = (html or "").lower()
    good_markers = [
        'type="application/ld+json"',
        'class="params"',
        "auction date",
        "primary damage",
        "vin",
    ]
    if any(m in low for m in good_markers):
        return

    raise UnexpectedContentError(f"Unexpected detail page content: {url}")


def get_response(config: dict, url: str, params=None, mode: str = "list", auth_required: bool = True) -> str:
    scraper = get_scraper(config) if auth_required else get_public_scraper(config)
    pace = get_pace(mode)
    pace.wait()

    r = scraper.get(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)

    rate_limit = r.headers.get("x-ratelimit-limit")
    rate_remaining = r.headers.get("x-ratelimit-remaining")
    if rate_limit or rate_remaining:
        log(
            f"[RATE] worker={config['worker_name']} mode={mode} "
            f"limit={rate_limit} remaining={rate_remaining} url={r.url}"
        )

    default_rng = LIST_SLEEP if mode == "list" else DETAIL_SLEEP
    new_rng = adaptive_delay_from_headers(r.headers, default_rng)
    pace.update(*new_rng)

    text = r.text or ""

    if is_rate_limited_response(r.status_code, text, r.headers):
        raise RateLimitedError(f"Rate limited on {r.url}")

    if auth_required and looks_like_auth_problem(r.status_code, text, r.url):
        raise AuthExpiredError(f"Auth/session problem on {r.url}")

    r.raise_for_status()

    if mode == "list":
        try:
            validate_sales_page(text, r.url)
            return text
        except UnexpectedContentError:
            pass
    else:
        try:
            validate_detail_page(text, r.url)
            return text
        except UnexpectedContentError:
            pass

    if is_blocked_or_challenge_html(text, r.url):
        raise CloudflareChallengeError(f"Cloudflare/challenge detected for {r.url}")

    raise UnexpectedContentError(f"Unexpected page content for {r.url}")


def claim_list_page(worker_name: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    cur.execute("""
        SELECT page FROM list_pages
        WHERE status = 'pending'
        ORDER BY page
        LIMIT 1
    """)
    row = cur.fetchone()
    if not row:
        conn.commit()
        conn.close()
        return None

    page = row["page"]
    cur.execute("""
        UPDATE list_pages
        SET status='in_progress', worker=?, updated_at=CURRENT_TIMESTAMP
        WHERE page=?
    """, (worker_name, page))
    conn.commit()
    conn.close()
    return page


def finish_list_page(page: int):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        UPDATE list_pages
        SET status='done', updated_at=CURRENT_TIMESTAMP
        WHERE page=?
    """, (page,))
    conn.commit()
    conn.close()


def fail_list_page(page: int, error: str, reset_to_pending: bool = False):
    conn = db()
    cur = conn.cursor()
    if reset_to_pending:
        cur.execute("""
            UPDATE list_pages
            SET status='pending', error=?, updated_at=CURRENT_TIMESTAMP
            WHERE page=?
        """, (error[:1000], page))
    else:
        cur.execute("""
            UPDATE list_pages
            SET status='failed', error=?, updated_at=CURRENT_TIMESTAMP
            WHERE page=?
        """, (error[:1000], page))
    conn.commit()
    conn.close()


def claim_detail_task(worker_name: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    cur.execute("""
        SELECT detail_url, vin FROM detail_tasks
        WHERE status = 'pending'
        ORDER BY updated_at, detail_url
        LIMIT 1
    """)
    row = cur.fetchone()
    if not row:
        conn.commit()
        conn.close()
        return None

    cur.execute("""
        UPDATE detail_tasks
        SET status='in_progress', worker=?, updated_at=CURRENT_TIMESTAMP
        WHERE detail_url=?
    """, (worker_name, row["detail_url"]))
    conn.commit()
    conn.close()
    return {"detail_url": row["detail_url"], "vin": row["vin"]}


def finish_detail_task(detail_url: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        UPDATE detail_tasks
        SET status='done', updated_at=CURRENT_TIMESTAMP
        WHERE detail_url=?
    """, (detail_url,))
    conn.commit()
    conn.close()


def fail_detail_task(detail_url: str, error: str, reset_to_pending: bool = False):
    conn = db()
    cur = conn.cursor()
    if reset_to_pending:
        cur.execute("""
            UPDATE detail_tasks
            SET status='pending', error=?, updated_at=CURRENT_TIMESTAMP
            WHERE detail_url=?
        """, (error[:1000], detail_url))
    else:
        cur.execute("""
            UPDATE detail_tasks
            SET status='failed', error=?, updated_at=CURRENT_TIMESTAMP
            WHERE detail_url=?
        """, (error[:1000], detail_url))
    conn.commit()
    conn.close()


def get_value_by_label(card, label_text: str) -> str:
    for item in card.select("div.params div.mt-1"):
        b = item.find("b")
        if not b:
            continue
        label = clean_text(b.get_text(" ", strip=True)).rstrip(":")
        if label.lower() == label_text.lower().rstrip(":"):
            value_div = item.find("div", class_="values")
            if value_div:
                return clean_text(value_div.get_text(" ", strip=True))
    return ""


def parse_lot_card(card) -> dict:
    title_link = card.select_one(".title a.ajax_link")
    title = clean_text(title_link.get_text(" ", strip=True)) if title_link else ""
    detail_url = urljoin(BASE_DOMAIN, title_link["href"]) if title_link and title_link.get("href") else ""
    detail_url = normalize_detail_url(detail_url)

    image = card.select_one(".image_box img")
    image_url = image.get("src", "").strip() if image else ""
    image_alt = image.get("alt", "").strip() if image else ""

    price_node = card.select_one(".title .text-success")
    sold_price = clean_text(price_node.get_text(" ", strip=True)) if price_node else ""

    status_node = card.select_one(".title .badge")
    status = clean_text(status_node.get_text(" ", strip=True)) if status_node else ""

    seller_badge = card.select_one(".who_sell")
    seller_type = clean_text(seller_badge.get_text(" ", strip=True)) if seller_badge else ""

    external_btn = card.select_one('a.btn.btn-info.btn-sm[target="_blank"]')
    auction_url = external_btn.get("href", "").strip() if external_btn else ""

    auction_img = card.select_one('img[alt="copart"], img[alt="iaai"]')
    auction = auction_img.get("alt", "").strip().lower() if auction_img else ""

    return {
        "title": title,
        "detail_url": detail_url,
        "status": status,
        "sold_price": sold_price,
        "seller_type": seller_type,
        "vin": get_value_by_label(card, "Vin"),
        "engine": get_value_by_label(card, "Engine"),
        "mileage": get_value_by_label(card, "Mileage"),
        "sale_date": get_value_by_label(card, "Auction Date"),
        "repair_price": get_value_by_label(card, "Cost of repair"),
        "market_price": get_value_by_label(card, "Market value"),
        "primary_damage": get_value_by_label(card, "Primary damage"),
        "auction": auction,
        "auction_url": auction_url,
        "image_url": image_url,
        "image_alt": image_alt,
    }


def parse_list_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("div.lot_list > div.row.py-2")
    out = []
    for card in cards:
        item = parse_lot_card(card)
        if item["detail_url"] or item["vin"]:
            out.append(item)
    return out


def save_list_rows(rows: list[dict]):
    conn = db()
    cur = conn.cursor()

    for row in rows:
        cur.execute("""
            INSERT INTO list_results (
                detail_url, vin, title, status, sold_price, seller_type, engine,
                mileage, sale_date, repair_price, market_price, primary_damage,
                auction, auction_url, image_url, image_alt, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(detail_url) DO UPDATE SET
                vin=excluded.vin,
                title=excluded.title,
                status=excluded.status,
                sold_price=excluded.sold_price,
                seller_type=excluded.seller_type,
                engine=excluded.engine,
                mileage=excluded.mileage,
                sale_date=excluded.sale_date,
                repair_price=excluded.repair_price,
                market_price=excluded.market_price,
                primary_damage=excluded.primary_damage,
                auction=excluded.auction,
                auction_url=excluded.auction_url,
                image_url=excluded.image_url,
                image_alt=excluded.image_alt,
                raw_json=excluded.raw_json
        """, (
            row.get("detail_url"),
            row.get("vin"),
            row.get("title"),
            row.get("status"),
            row.get("sold_price"),
            row.get("seller_type"),
            row.get("engine"),
            row.get("mileage"),
            row.get("sale_date"),
            row.get("repair_price"),
            row.get("market_price"),
            row.get("primary_damage"),
            row.get("auction"),
            row.get("auction_url"),
            row.get("image_url"),
            row.get("image_alt"),
            json.dumps(row, ensure_ascii=False),
        ))

        if row.get("detail_url"):
            cur.execute("""
                INSERT OR IGNORE INTO detail_tasks(detail_url, vin, status)
                VALUES (?, ?, 'pending')
            """, (row["detail_url"], row.get("vin")))

    conn.commit()
    conn.close()


def get_detail_value_by_label(soup, label_text: str) -> str:
    target = label_text.lower().rstrip(":")

    for item in soup.select("div.params div.mt-1, div.params li, table tr"):
        b = item.find("b")
        if not b:
            continue

        label = clean_text(b.get_text(" ", strip=True)).rstrip(":").lower()
        if label != target:
            continue

        value_div = item.find("div", class_="values")
        if value_div:
            return clean_text(value_div.get_text(" ", strip=True))

        full_text = clean_text(item.get_text(" ", strip=True))
        label_text_raw = clean_text(b.get_text(" ", strip=True))
        value = full_text.replace(label_text_raw, "", 1).strip(" :")
        return clean_text(value)

    return ""


def parse_detail_structured(html: str, detail_url: str, vin: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    result = {
        "detail_url": normalize_detail_url(detail_url),
        "vin": vin,
        "transmission": "",
        "fuel_type": "",
        "drive_type": "",
        "secondary_damage": "",
        "color": "",
        "photo_urls": [],
    }

    json_ld_data = None
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("@type") == "Vehicle":
                json_ld_data = data
                break
        except Exception:
            pass

    if json_ld_data:
        engine = json_ld_data.get("vehicleEngine") or {}
        result["transmission"] = json_ld_data.get("vehicleTransmission", "") or ""
        result["fuel_type"] = engine.get("fuelType", "") or ""
        result["drive_type"] = json_ld_data.get("driveWheelConfiguration", "") or ""
        result["color"] = json_ld_data.get("color", "") or ""

        # knownVehicleDamages: primary, secondary
        known_damages = json_ld_data.get("knownVehicleDamages", "") or ""
        _, secondary_damage = split_known_vehicle_damages(known_damages)
        result["secondary_damage"] = secondary_damage or result["secondary_damage"]

    # Фолбэк по html-лейблам
    result["transmission"] = result["transmission"] or get_detail_value_by_label(soup, "Transmission")
    result["transmission"] = result["transmission"] or get_simple_bold_label_value(soup, "Transmission")

    result["fuel_type"] = result["fuel_type"] or get_detail_value_by_label(soup, "Fuel type")
    result["fuel_type"] = result["fuel_type"] or get_detail_value_by_label(soup, "Fuel")
    result["fuel_type"] = result["fuel_type"] or get_simple_bold_label_value(soup, "Fuel")

    result["drive_type"] = result["drive_type"] or get_detail_value_by_label(soup, "Drive")
    result["drive_type"] = result["drive_type"] or get_detail_value_by_label(soup, "Drive type")
    result["drive_type"] = result["drive_type"] or get_simple_bold_label_value(soup, "Drive")

    result["secondary_damage"] = result["secondary_damage"] or get_detail_value_by_label(soup, "Secondary damage")
    result["secondary_damage"] = result["secondary_damage"] or get_simple_bold_label_value(soup, "Secondary damage")

    result["color"] = result["color"] or get_detail_value_by_label(soup, "Color")
    result["color"] = result["color"] or get_simple_bold_label_value(soup, "Color")

    # Фото
    result["photo_urls"] = extract_photo_urls(soup)

    return result


def save_detail_result(row: dict):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO detail_results(
            detail_url, vin, transmission, fuel_type, drive_type,
            secondary_damage, color, photo_urls_json, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(detail_url) DO UPDATE SET
            vin=excluded.vin,
            transmission=excluded.transmission,
            fuel_type=excluded.fuel_type,
            drive_type=excluded.drive_type,
            secondary_damage=excluded.secondary_damage,
            color=excluded.color,
            photo_urls_json=excluded.photo_urls_json,
            raw_json=excluded.raw_json
    """, (
        row["detail_url"],
        row["vin"],
        row["transmission"],
        row["fuel_type"],
        row["drive_type"],
        row["secondary_damage"],
        row["color"],
        json.dumps(row.get("photo_urls", []), ensure_ascii=False),
        json.dumps(row, ensure_ascii=False),
    ))
    conn.commit()
    conn.close()


def run_list_worker(config: dict):
    worker_name = config["worker_name"]

    try:
        reset_scraper()
        get_scraper(config)
        log(f"[LIST][INIT_AUTH_OK] {worker_name}")
    except LoginError as e:
        log(f"[LIST][INIT_AUTH_ERR] {worker_name}: {e}")
        return
    except CloudflareChallengeError as e:
        log(f"[LIST][INIT_CF_ERR] {worker_name}: {e}")
        return
    except Exception as e:
        log(f"[LIST][INIT_ERR] {worker_name}: {e}")
        return

    startup_sleep = random.uniform(3, 20)
    log(f"[LIST][STARTUP_JITTER] {worker_name}: sleep={startup_sleep:.1f}s")
    time.sleep(startup_sleep)

    rl_attempt = 1
    cf_attempt = 1
    processed_in_batch = 0

    while True:
        page = claim_list_page(worker_name)
        if page is None:
            log(f"[LIST] {worker_name}: no more pages")
            return

        try:
            params = dict(LIST_PARAMS_BASE)
            params["page"] = page

            html = get_response(
                config,
                LIST_URL,
                params=params,
                mode="list",
                auth_required=True,
            )
            rows = parse_list_page(html)
            save_list_rows(rows)
            finish_list_page(page)

            rl_attempt = 1
            cf_attempt = 1
            processed_in_batch += 1

            log(f"[LIST] {worker_name}: page={page} rows={len(rows)}")

            if processed_in_batch >= BATCH_SIZE_LIST:
                batch_pause("list")
                processed_in_batch = 0

        except RateLimitedError as e:
            fail_list_page(page, str(e), reset_to_pending=True)
            cooldown = rate_limit_cooldown_seconds(rl_attempt)
            rl_attempt += 1
            log(f"[LIST][RATE] {worker_name}: cooldown={cooldown/60:.1f} min page={page}")
            time.sleep(cooldown)
            processed_in_batch = 0

        except CloudflareChallengeError as e:
            fail_list_page(page, str(e), reset_to_pending=True)
            cooldown = challenge_cooldown_seconds(cf_attempt)
            cf_attempt += 1
            log(f"[LIST][CF] {worker_name}: cooldown={cooldown/60:.1f} min page={page}")
            time.sleep(cooldown)
            processed_in_batch = 0

        except AuthExpiredError as e:
            reset_scraper()
            fail_list_page(page, str(e), reset_to_pending=True)
            pause = auth_retry_pause()
            log(f"[LIST][AUTH] {worker_name}: relogin after {pause:.1f}s page={page}")
            time.sleep(pause)
            processed_in_batch = 0

        except LoginError as e:
            reset_scraper()
            fail_list_page(page, str(e), reset_to_pending=True)
            log(f"[LIST][LOGIN_ERR] {worker_name}: stop worker, page returned to pending, err={e}")
            return

        except Exception as e:
            fail_list_page(page, str(e), reset_to_pending=False)
            log(f"[LIST][ERR] {worker_name}: page={page} err={e}")
            time.sleep(short_error_pause())
            processed_in_batch = 0


def run_detail_worker(config: dict):
    worker_name = config["worker_name"]

    try:
        reset_scraper()
        get_public_scraper(config)
        log(f"[DETAIL][INIT_PUBLIC_OK] {worker_name}")
    except Exception as e:
        log(f"[DETAIL][INIT_ERR] {worker_name}: {e}")
        return

    startup_sleep = random.uniform(3, 20)
    log(f"[DETAIL][STARTUP_JITTER] {worker_name}: sleep={startup_sleep:.1f}s")
    time.sleep(startup_sleep)

    rl_attempt = 1
    cf_attempt = 1
    processed_in_batch = 0

    while True:
        task = claim_detail_task(worker_name)
        if task is None:
            log(f"[DETAIL] {worker_name}: no more detail tasks")
            return

        try:
            html = get_response(
                config,
                task["detail_url"],
                params=None,
                mode="detail",
                auth_required=False,
            )
            row = parse_detail_structured(html, task["detail_url"], task["vin"])
            save_detail_result(row)
            finish_detail_task(task["detail_url"])

            rl_attempt = 1
            cf_attempt = 1
            processed_in_batch += 1

            log(f"[DETAIL] {worker_name}: {task['detail_url']}")

            if processed_in_batch >= BATCH_SIZE_DETAIL:
                batch_pause("detail")
                processed_in_batch = 0

        except RateLimitedError as e:
            fail_detail_task(task["detail_url"], str(e), reset_to_pending=True)
            cooldown = rate_limit_cooldown_seconds(rl_attempt)
            rl_attempt += 1
            log(f"[DETAIL][RATE] {worker_name}: cooldown={cooldown/60:.1f} min")
            time.sleep(cooldown)
            processed_in_batch = 0

        except CloudflareChallengeError as e:
            fail_detail_task(task["detail_url"], str(e), reset_to_pending=True)
            cooldown = challenge_cooldown_seconds(cf_attempt)
            cf_attempt += 1
            log(f"[DETAIL][CF] {worker_name}: cooldown={cooldown/60:.1f} min")
            time.sleep(cooldown)
            processed_in_batch = 0

        except Exception as e:
            fail_detail_task(task["detail_url"], str(e), reset_to_pending=False)
            log(f"[DETAIL][ERR] {worker_name}: {task['detail_url']} err={e}")
            time.sleep(short_error_pause())
            processed_in_batch = 0


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python worker.py accounts.txt list 1")
        print("  python worker.py accounts.txt detail 1")
        print("  python worker.py config.json list")
        print("  python worker.py config.json detail")
        sys.exit(1)

    config_path = sys.argv[1]
    mode = sys.argv[2].strip().lower()

    line_number = None
    if len(sys.argv) >= 4:
        line_number = int(sys.argv[3])

    config = load_config(config_path, line_number)

    if mode == "list":
        run_list_worker(config)
    elif mode == "detail":
        run_detail_worker(config)
    else:
        raise ValueError("mode must be 'list' or 'detail'")


if __name__ == "__main__":
    main()