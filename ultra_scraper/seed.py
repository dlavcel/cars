import sqlite3

DB_PATH = "./tasks.db"
START_PAGE = 1
END_PAGE = 500000

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

for page in range(START_PAGE, END_PAGE + 1):
    cur.execute("""
        INSERT OR IGNORE INTO list_pages(page, status)
        VALUES (?, 'pending')
    """, (page,))

conn.commit()
conn.close()

print(f"Inserted pages {START_PAGE}-{END_PAGE}")