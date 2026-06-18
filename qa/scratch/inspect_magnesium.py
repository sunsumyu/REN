# -*- coding: utf-8 -*-
import sqlite3
import os

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(workspace_dir, "local_rag.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Checking for specific sources in local_rag.db:")
    for term in ["硬脂酸镁", "过敏", "九期一", "葶苈子", "葶苈大枣泻肺汤", "方剂"]:
        cursor.execute("SELECT source, entity_name FROM local_rag_index WHERE source LIKE ? OR entity_name LIKE ?;", (f"%{term}%", f"%{term}%"))
        rows = cursor.fetchall()
        print(f"\n--- Matches for '{term}' (found {len(rows)}) ---")
        for r in rows[:5]:
            print(f"Source: {r[0]} | Entity: {r[1]}")
            
    conn.close()

if __name__ == "__main__":
    main()
