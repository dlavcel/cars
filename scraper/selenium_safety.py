import random
import time
from selenium.common import NoAlertPresentException, UnexpectedAlertPresentException

def dismiss_any_alert(driver) -> bool:
    try:
        alert = driver.switch_to.alert
        txt = (alert.text or "").strip()
        print(f"⚠️ JS alert detected: {txt!r}. Dismissing...")
        alert.dismiss()
        time.sleep(random.uniform(0.6, 1.4))
        return True
    except NoAlertPresentException:
        return False
    except Exception:
        return False

def safe_page_source(driver) -> str:
    for _ in range(3):
        try:
            return driver.page_source or ""
        except UnexpectedAlertPresentException:
            dismiss_any_alert(driver)
            time.sleep(random.uniform(0.4, 1.0))
        except Exception:
            time.sleep(random.uniform(0.4, 1.0))
    return ""

def sleep_jitter(a, b):
    time.sleep(random.uniform(a, b))

def is_rate_limited(driver) -> bool:
    html = safe_page_source(driver).lower()
    return ("too many attempts" in html) and ("please contact support" in html)

def rate_limit_cooldown(attempt: int = 1):
    minutes = random.uniform(12, 25) * max(1, attempt)
    print(f"🚫 RATE LIMIT. Cooldown {minutes:.1f} min...")
    time.sleep(minutes * 60)

def is_blocked_or_challenge(driver) -> bool:
    html = safe_page_source(driver).lower()
    title = ""
    try:
        title = (driver.title or "").lower()
    except UnexpectedAlertPresentException:
        dismiss_any_alert(driver)

    if "cloudflare" in title:
        return True
    if "checking your browser" in html:
        return True
    if "/cdn-cgi/" in html:
        return True
    return False

def wait_out_cloudflare(driver, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        dismiss_any_alert(driver)
        if not is_blocked_or_challenge(driver) and not is_rate_limited(driver):
            return True
        time.sleep(random.uniform(2.0, 4.0))
    return False
