import re
from urllib.parse import urlparse, parse_qs

def record_id_from_url(url: str) -> str:
    if not url:
        return ""
    m = re.search(r"/en/car/([^/?#]+)", url)
    return m.group(1) if m else url

def norm_vin(v: str) -> str:
    if not v:
        return ""
    v = v.strip().upper()
    v = re.sub(r"[^A-HJ-NPR-Z0-9]", "", v)
    return v

def vin_from_car_url(url: str) -> str:
    if not url:
        return ""
    m = re.search(r"/en/car/([A-HJ-NPR-Z0-9]{11,20})_", url, flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""

def get_page_from_progress_url(url: str) -> int:
    if not url:
        return 1
    try:
        parsed = urlparse(url)
        q = parse_qs(parsed.query)
        p = q.get("page", [None])[0]
        return int(p) if p and str(p).isdigit() else 1
    except Exception:
        return 1

def normalize_drive_text(val: str):
    if not val:
        return None
    v = val.strip().lower()
    if "all wheel" in v:
        return "AWD"
    if "four wheel" in v or "4x4" in v:
        return "4WD"
    if "front-wheel" in v or "front wheel" in v:
        return "FWD"
    if "rear-wheel" in v or "rear wheel" in v:
        return "RWD"
    return val.strip()

def parse_int_like(s: str):
    if not s:
        return None
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None

def parse_mileage(s: str):
    if not s:
        return None, None
    txt = " ".join(s.split())
    unit = None
    if " mi" in txt:
        unit = "mi"
    elif " km" in txt:
        unit = "km"
    val = parse_int_like(txt)
    return val, unit
