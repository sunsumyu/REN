import sqlite3
import os

db_path = "medical_cache.db"

def clear_cache():
    if not os.path.exists(db_path):
        print(f"No cache database found at {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. 打印当前的缓存项数量
    cursor.execute("SELECT COUNT(*) FROM api_cache;")
    count_before = cursor.fetchone()[0]
    print(f"Total cache items before clearing: {count_before}")
    
    # 2. 清理包含 Mock 脏数据的 UGT2B7 缓存
    cursor.execute("DELETE FROM api_cache WHERE query = ? OR response_json LIKE '%辅助治疗%';", ("ugt2b7",))
    deleted_count = conn.total_changes
    conn.commit()
    print(f"Deleted {deleted_count} dirty mock cache items.")
    
    # 3. 再次确认剩余缓存项数量
    cursor.execute("SELECT COUNT(*) FROM api_cache;")
    count_after = cursor.fetchone()[0]
    print(f"Total cache items after clearing: {count_after}")
    
    conn.close()

if __name__ == "__main__":
    clear_cache()
