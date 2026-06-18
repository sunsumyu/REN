# -*- coding: utf-8 -*-
import sqlite3
import os

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(workspace_dir, "local_rag.db")
    if not os.path.exists(db_path):
        print(f"DB not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 查询记录总数
    cursor.execute("SELECT count(*) FROM local_rag_index;")
    total = cursor.fetchone()[0]
    print(f"Total records in local_rag_index: {total}")

    # 打印前 15 条记录
    cursor.execute("SELECT id, source, entity_name, category FROM local_rag_index LIMIT 15;")
    rows = cursor.fetchall()
    print("\n--- TOP 15 RECORDS IN DATABASE ---")
    for r in rows:
        print(f"ID: {r['id']} | Source: {r['source']} | Entity: {r['entity_name']} | Category: {r['category']}")

    conn.close()

if __name__ == "__main__":
    main()
