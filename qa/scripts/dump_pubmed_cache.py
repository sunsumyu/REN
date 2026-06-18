# -*- coding: utf-8 -*-
"""
读取并格式化打印 medical_cache.db 中所有已缓存的 PubMed 文献数据
"""
import sqlite3
import json
import os

def main():
    db_path = "medical_cache.db"
    if not os.path.exists(db_path):
        print(f"❌ 错误: 缓存数据库文件 '{db_path}' 未在当前目录下找到。")
        return
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("==================================================")
    print("📚 从 medical_cache.db 提取的 PubMed 缓存文献摘要")
    print("==================================================\n")
    
    try:
        # 查询所有服务名为 pubmed 的持久化缓存数据
        cursor.execute("""
            SELECT query, response_json, created_at 
            FROM api_cache 
            WHERE service_name = 'pubmed'
            ORDER BY created_at DESC;
        """)
        
        rows = cursor.fetchall()
        if not rows:
            print("ℹ️ 缓存中未找到任何 PubMed 文献记录。")
            return
            
        print(f"共找到 {len(rows)} 组实体的 PubMed 检索历史记录：\n")
        
        for idx, row in enumerate(rows):
            query_term = row["query"]
            created_at = row["created_at"]
            response_json = row["response_json"]
            
            try:
                items = json.loads(response_json)
            except Exception as e:
                print(f"[{idx+1}] 实体: {query_term} (解析 JSON 失败: {e})")
                continue
                
            print(f"=== 🔍 检索实体 [{idx+1}]: {query_term} (创建时间: {created_at}) ===")
            if not items:
                print("  (该实体在 PubMed 中未检索到任何文献结果)")
            else:
                for item_idx, item in enumerate(items):
                    source = item.get("source", "N/A")
                    context = item.get("context", "N/A")
                    metadata = item.get("metadata", {})
                    title = metadata.get("title", "N/A")
                    authors = metadata.get("authors", "N/A")
                    pmid = metadata.get("pmid", "N/A")
                    
                    print(f"  文献 [{item_idx+1}] | PMID: {pmid}")
                    print(f"    文献标题: {title}")
                    print(f"    主要作者: {authors}")
                    print(f"    来源标记: {source}")
                    print(f"    注入 refs 的正文段落:\n      {context}")
                    print()
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ 读取数据库出错: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
