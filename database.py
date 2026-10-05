import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "survey.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS survey_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age_group TEXT, gender TEXT, location TEXT, education TEXT,
            employment TEXT, income TEXT, smartphone TEXT, computer TEXT,
            internet_access TEXT, internet_quality TEXT, internet_usage TEXT,
            digital_literacy TEXT, online_banking TEXT, online_education TEXT,
            online_services TEXT, social_media TEXT, internet_cost TEXT,
            digital_skills TEXT, barrier TEXT, digital_divide_score REAL,
            category TEXT, created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_response(data, score, category):
    conn = get_connection()
    fields = ["age_group","gender","location","education","employment","income",
              "smartphone","computer","internet_access","internet_quality",
              "internet_usage","digital_literacy","online_banking",
              "online_education","online_services","social_media","internet_cost",
              "digital_skills","barrier"]
    values = [data.get(f, "") for f in fields]
    conn.execute(f"""
        INSERT INTO survey_responses ({",".join(fields)}, digital_divide_score, category, created_at)
        VALUES ({",".join(["?"]*len(fields))}, ?, ?, ?)
    """, values + [score, category, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    conn.commit()
    conn.close()

def fetch_all():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM survey_responses ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_response(row_id):
    conn = get_connection()
    conn.execute("DELETE FROM survey_responses WHERE id=?", (row_id,))
    conn.commit()
    conn.close()

def replace_from_dataframe(df):
    conn = get_connection()
    conn.execute("DELETE FROM survey_responses")
    fields = ["age_group","gender","location","education","employment","income",
              "smartphone","computer","internet_access","internet_quality",
              "internet_usage","digital_literacy","online_banking",
              "online_education","online_services","social_media","internet_cost",
              "digital_skills","barrier","digital_divide_score","category","created_at"]
    for _, r in df.iterrows():
        vals = [r.get(f, "") for f in fields]
        conn.execute(f"INSERT INTO survey_responses ({','.join(fields)}) VALUES ({','.join(['?']*len(fields))})", vals)
    conn.commit()
    conn.close()
