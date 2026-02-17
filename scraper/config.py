LIST_URL = "https://autohelperbot.com/en/sales?vehicle=AUTOMOBILE"
OUT_CSV = "../cars.csv"
PROGRESS_FILE = "../progress.json"

FIELDNAMES = [
    "id", "url", "vin", "year", "raw",
    "price", "currency", "seller", "auction_source",
    "mileage", "mileage_unit", "engine", "fuel", "transmission", "color",
    "damage_primary", "damage_secondary", "drive",
    "key", "cost_of_repair", "market_value",
    "images",
]
