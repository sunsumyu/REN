import sqlite3
import os

db_files = ["qa_datasets.db", "local_rag.db", "medical_cache.db", "prompts.db"]

for db in db_files:
    if os.path.exists(db):
        print(f"=== Database: {db} ===")
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        for table in tables:
            tname = table[0]
            print(f"  Table: {tname}")
            cursor.execute(f"PRAGMA table_info({tname})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"    Column: {col[1]} ({col[2]})")
        conn.close()
    else:
        print(f"Database {db} does not exist.")
