import sqlite3

def query_db():
    conn = sqlite3.connect('local_rag.db')
    cursor = conn.cursor()
    cursor.execute("SELECT source, context FROM local_rag_index WHERE source LIKE '%复发性阿弗他溃疡临床路径%' AND source LIKE '%段3%'")
    for row in cursor.fetchall():
        print("SOURCE:", row[0])
        print("CONTEXT:", row[1])
        print("-" * 50)
        
    cursor.execute("SELECT source, context FROM local_rag_index WHERE source LIKE '%慢性肺源性心脏病临床路径%' AND source LIKE '%段4%'")
    for row in cursor.fetchall():
        print("SOURCE:", row[0])
        print("CONTEXT:", row[1])
        print("-" * 50)

    cursor.execute("SELECT source, context FROM local_rag_index WHERE source LIKE '%急性呼吸窘迫综合征%临床路径%' AND source LIKE '%段3%'")
    for row in cursor.fetchall():
        print("SOURCE:", row[0])
        print("CONTEXT:", row[1])
        print("-" * 50)

    cursor.execute("SELECT source, context FROM local_rag_index WHERE source LIKE '%垂体催乳素瘤临床路径%' AND source LIKE '%段3%'")
    for row in cursor.fetchall():
        print("SOURCE:", row[0])
        print("CONTEXT:", row[1])
        print("-" * 50)

    cursor.execute("SELECT source, context FROM local_rag_index WHERE source LIKE '%下颌前突畸形临床路径%' AND source LIKE '%段3%'")
    for row in cursor.fetchall():
        print("SOURCE:", row[0])
        print("CONTEXT:", row[1])
        print("-" * 50)
    
    conn.close()

if __name__ == '__main__':
    query_db()
