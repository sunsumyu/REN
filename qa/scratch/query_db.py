import sqlite3
import os

def check_db(name):
    if not os.path.exists(name):
        print(f"[{name}] does not exist.")
        return
    print(f"\n===== Inspecting {name} =====")
    conn = sqlite3.connect(name)
    cursor = conn.cursor()
    
    # Get tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", [t[0] for t in tables])
    
    for table in [t[0] for t in tables]:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            print(f"Table '{table}' has {count} rows.")
            
            # Print schema
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [c[1] for c in cursor.fetchall()]
            print(f"  Columns: {columns}")
            
            # Search for '硬脂酸镁'
            match_cols = []
            for col in columns:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} LIKE '%硬脂酸镁%';")
                    if cursor.fetchone()[0] > 0:
                        match_cols.append(col)
                except sqlite3.OperationalError:
                    pass
            if match_cols:
                print(f"  --> FOUND '硬脂酸镁' in columns: {match_cols}!")
                # Print some matching rows
                for col in match_cols:
                    cursor.execute(f"SELECT {col} FROM {table} WHERE {col} LIKE '%硬脂酸镁%' LIMIT 3;")
                    rows = cursor.fetchall()
                    for r in rows:
                        print(f"    Match in {col}: {r[0][:150]}")
        except Exception as e:
            print(f"  Error reading table {table}: {e}")
            
    conn.close()

check_db("medical_cache.db")
check_db("qa_datasets.db")
