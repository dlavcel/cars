import random
import string
import sys
import time
from pathlib import Path

import cloudscraper
from bs4 import BeautifulSoup

REGISTER_URL = "https://autohelperbot.com/en/register"
LOGOUT_URL = "https://autohelperbot.com/en/logout"
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


class RegisterError(Exception):
    pass


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def random_string(n: int = 10) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


def random_name() -> tuple[str, str]:
    first_names = [
        "Alex", "Maks", "Ivan", "Leo", "Mark", "Nazar", "Roman", "Artem",
        "Nikita", "Denis", "Andriy", "Dmytro", "Ilya", "Timur"
    ]
    last_names = [
        "Stone", "Miller", "Kovalenko", "Petrov", "Melnyk", "Smith", "Brown",
        "Walker", "Bondar", "Shevchenko", "Moroz", "Kravets", "Hrytsenko"
    ]
    return random.choice(first_names), random.choice(last_names)


def random_site_password(length: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def random_email(domain: str = "gmail.com") -> str:
    local = f"ahb_{random_string(12)}"
    return f"{local}@{domain}"


def build_proxy_url(ip: str, port: str, username: str, password: str) -> str:
    return f"http://{username}:{password}@{ip}:{port}"


def parse_proxy_line(line: str) -> tuple[str, str, str, str]:
    parts = line.strip().split(":")
    if len(parts) != 4:
        raise ValueError("Expected proxy line format: ip:port:username:password")
    return parts[0], parts[1], parts[2], parts[3]


def make_scraper(proxy_url: str):
    scraper = cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "windows",
            "desktop": True,
        }
    )
    scraper.proxies.update({
        "http": proxy_url,
        "https": proxy_url,
    })
    return scraper


def get_register_token(scraper) -> str:
    r = scraper.get(REGISTER_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    token_input = soup.find("input", {"name": "_token"})
    if not token_input or not token_input.get("value"):
        raise RegisterError("CSRF _token not found on register page")

    return token_input["value"]


def looks_registered(html: str, final_url: str) -> bool:
    low = (html or "").lower()
    final_url = (final_url or "").lower()

    if "/register" in final_url or final_url.endswith("/login"):
        return False

    if "logout" in low:
        return True
    if "/orders-list" in final_url:
        return True

    return False


def register_account(proxy_url: str, email: str, site_password: str, first_name: str, last_name: str):
    scraper = make_scraper(proxy_url)
    token = get_register_token(scraper)

    payload = {
        "_token": token,
        "last_name": last_name,
        "first_name": first_name,
        "email": email,
        "password": site_password,
    }

    headers = dict(HEADERS)
    headers.update({
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://autohelperbot.com",
        "Referer": REGISTER_URL,
    })

    resp = scraper.post(
        REGISTER_URL,
        headers=headers,
        data=payload,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
    )
    resp.raise_for_status()

    if not looks_registered(resp.text, resp.url):
        snippet = clean_text(resp.text[:500])
        raise RegisterError(f"Registration did not look successful. Final URL={resp.url}. Snippet={snippet}")

    try:
        scraper.get(LOGOUT_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    except Exception:
        pass

    return True


def already_registered_proxy(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    result = set()
    for raw in output_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":", 5)
        if len(parts) == 6:
            proxy_key = ":".join(parts[:4])
            result.add(proxy_key)
    return result


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python register_accounts.py proxies.txt accounts_ready.txt")
        print("Optional:")
        print("  python register_accounts.py proxies.txt accounts_ready.txt yourdomain.com")
        raise SystemExit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    email_domain = sys.argv[3] if len(sys.argv) >= 4 else "example.com"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    done = already_registered_proxy(output_path)

    lines = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded proxies: {len(lines)}")

    success_count = 0
    fail_count = 0

    with output_path.open("a", encoding="utf-8") as out_f:
        for idx, line in enumerate(lines, start=1):
            try:
                ip, port, proxy_user, proxy_pass = parse_proxy_line(line)
                proxy_key = f"{ip}:{port}:{proxy_user}:{proxy_pass}"

                if proxy_key in done:
                    print(f"[SKIP] #{idx} already registered for proxy {ip}:{port}")
                    continue

                proxy_url = build_proxy_url(ip, port, proxy_user, proxy_pass)
                first_name, last_name = random_name()
                email = random_email(email_domain)
                site_password = random_site_password()

                print(f"[REGISTER] #{idx} proxy={ip}:{port} email={email}")

                register_account(
                    proxy_url=proxy_url,
                    email=email,
                    site_password=site_password,
                    first_name=first_name,
                    last_name=last_name,
                )

                out_line = f"{ip}:{port}:{proxy_user}:{proxy_pass}:{email}:{site_password}\n"
                out_f.write(out_line)
                out_f.flush()

                success_count += 1
                print(f"[OK] #{idx} {email}")

                time.sleep(random.uniform(5, 15))

            except Exception as e:
                fail_count += 1
                print(f"[FAIL] #{idx} {line} -> {e}")

                time.sleep(random.uniform(10, 25))

    print(f"Done. success={success_count}, failed={fail_count}, output={output_path}")


if __name__ == "__main__":
    main()