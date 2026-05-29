import sqlite3
import json

DB_PATH = "qa_datasets.db"

def inspect():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='datasets'")
        table_exists = cursor.fetchone()
        if not table_exists:
            print("Table 'datasets' does not exist in the database.")
            return
            
        # Count total records
        cursor.execute("SELECT COUNT(*) FROM datasets")
        total = cursor.fetchone()[0]
        print(f"Total records in datasets table: {total}")
        
        # Get the 5 most recent records
        cursor.execute("SELECT id, run_date, query, created_at FROM datasets ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        print("\nLast 5 records:")
        for row in rows:
            print(f"ID: {row[0]}, Date: {row[1]}, Query: '{row[2]}', Created At: {row[3]}")
            
    except Exception as e:
        print(f"Error inspecting DB: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    inspect()
