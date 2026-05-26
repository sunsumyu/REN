import sqlite3
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger("MedicalQA.DatasetDB")

DB_PATH = os.path.join(os.path.dirname(__file__), "qa_datasets.db")

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT NOT NULL,
                query TEXT NOT NULL,
                full_data_json TEXT NOT NULL,
                judge_metrics_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize qa_datasets.db: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def save_dataset_record(run_date: str, query: str, full_data: dict, judge_metrics: dict):
    init_db()
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO datasets (run_date, query, full_data_json, judge_metrics_json) VALUES (?, ?, ?, ?)',
            (run_date, query, json.dumps(full_data, ensure_ascii=False), json.dumps(judge_metrics, ensure_ascii=False))
        )
        conn.commit()
        logger.info(f"Successfully saved generated dataset for query '{query}' to database.")
    except Exception as e:
        logger.error(f"Failed to save dataset record to DB: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
