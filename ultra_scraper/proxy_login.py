import cloudscraper
from bs4 import BeautifulSoup

LOGIN_URL = "https://autohelperbot.com/en/login"
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


class LoginError(Exception):
    pass


def _build_proxies(proxy: str | None) -> dict | None:
    if not proxy:
        return None
    return {
        "http": proxy,
        "https": proxy,
    }


def create_authenticated_scraper(
    email: str,
    password: str,
    proxy: str | None = None,
    timeout: int = REQUEST_TIMEOUT,
):
    scraper = cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "windows",
            "desktop": True,
        }
    )

    proxies = _build_proxies(proxy)
    if proxies:
        scraper.proxies.update(proxies)

    resp = scraper.get(LOGIN_URL, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    token_input = soup.find("input", {"name": "_token"})
    if not token_input or not token_input.get("value"):
        raise LoginError("Не найден CSRF _token на странице логина")

    csrf_token = token_input["value"]

    login_headers = dict(HEADERS)
    login_headers.update({
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://autohelperbot.com",
        "Referer": LOGIN_URL,
    })

    payload = {
        "_token": csrf_token,
        "email": email,
        "password": password,
    }

    login_resp = scraper.post(
        LOGIN_URL,
        headers=login_headers,
        data=payload,
        timeout=timeout,
        allow_redirects=True,
    )
    login_resp.raise_for_status()

    cookies = scraper.cookies.get_dict()

    if not any(k.startswith("remember_web_") for k in cookies):
        raise LoginError("Логин не удался: не появилась remember_web cookie")

    return scraper


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python proxy_login.py configs/acc1.json")
        raise SystemExit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        config = json.load(f)

    scraper = create_authenticated_scraper(
        email=config["email"],
        password=config["password"],
        proxy=config.get("proxy"),
    )

    test_url = "https://autohelperbot.com/en/sales?page=2"
    resp = scraper.get(test_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    print(resp.status_code, resp.url)
    print(resp.text[:1000])