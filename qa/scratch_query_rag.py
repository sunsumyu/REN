import sqlite3
import json

def query_local_rag():
    db_path = "d:/REN/qa/local_rag.db"
    
    print(f"[*] 正在从本地 RAG 向量数据库中排查...")
    print(f"[*] 数据库路径: {db_path}\n")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 查看表结构
        cursor.execute("PRAGMA table_info(local_rag_index)")
        schema = cursor.fetchall()
        print("[*] 表结构:", schema)
        
        # 查看第一条记录（因为日志里写着 物理ID 1）
        cursor.execute("SELECT rowid, * FROM local_rag_index LIMIT 3")
        results = cursor.fetchall()
        
        if not results:
            print("[-] local_rag_index 表是空的！")
        else:
            print(f"✅ 成功命中 {len(results)} 条记录：\n")
            
            # 获取列名
            col_names = [description[0] for description in cursor.description]
            
            for row in results:
                print("-" * 50)
                for col_name, value in zip(col_names, row):
                    print(f"[{col_name}]: {value}")
                print("-" * 50)
                
    except Exception as e:
        print(f"[!] 数据库查询报错: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    query_local_rag()
