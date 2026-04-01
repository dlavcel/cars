import sqlite3

DB_PATH = "./tasks.db"

schema = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS list_pages (
    page INTEGER PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    worker TEXT,
    error TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS detail_tasks (
    detail_url TEXT PRIMARY KEY,
    vin TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    worker TEXT,
    error TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS list_results (
    detail_url TEXT PRIMARY KEY,
    vin TEXT,
    title TEXT,
    status TEXT,
    sold_price TEXT,
    seller_type TEXT,
    engine TEXT,
    mileage TEXT,
    sale_date TEXT,
    repair_price TEXT,
    market_price TEXT,
    primary_damage TEXT,
    auction TEXT,
    auction_url TEXT,
    image_url TEXT,
    image_alt TEXT,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS detail_results (
    detail_url TEXT PRIMARY KEY,
    vin TEXT,
    transmission TEXT,
    fuel_type TEXT,
    drive_type TEXT,
    secondary_damage TEXT,
    color TEXT,
    photo_urls_json TEXT,
    raw_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_list_pages_status_page
ON list_pages(status, page);

CREATE INDEX IF NOT EXISTS idx_detail_tasks_status_updated
ON detail_tasks(status, updated_at);

CREATE INDEX IF NOT EXISTS idx_list_results_vin
ON list_results(vin);

CREATE INDEX IF NOT EXISTS idx_detail_results_vin
ON detail_results(vin);
"""

conn = sqlite3.connect(DB_PATH)
conn.executescript(schema)
conn.commit()
conn.close()

print("DB initialized:", DB_PATH)