# -*- coding: utf-8 -*-
"""
离线计算工具：读取中文临床数据，执行 Markdown 噪声过滤与中英文分段计数，输出对齐的 SQLite 库与平面内积 FAISS 索引。
"""

import os
import sqlite3
import json
import re

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    DEPS_OK = True
except ImportError:
    DEPS_OK = False

def sanitize_markdown(text: str) -> str:
    """
    Markdown 清洗过滤器 (Sanitizer)
    """
    # 移除 Markdown 超链接与图片
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', ' ', text)
    # 合并连续的多余空格及空行
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_offline_rag_index(
    raw_data_json: str,
    output_db_path: str = "local_rag.db",
    output_index_path: str = "local_rag_vector.index",
    model_name: str = "shibing624/text2vec-base-chinese"
):
    if not DEPS_OK:
        raise ImportError("Missing required vector libraries. Run: pip install faiss-cpu sentence-transformers numpy")

    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    # 初始化 SQLite 持久化层
    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS local_rag_index;")
    cursor.execute("DROP TABLE IF EXISTS local_rag_fts_index;")
    
    cursor.execute("""
        CREATE TABLE local_rag_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT,
            source TEXT,
            context TEXT,
            category TEXT
        );
    """)
    # 尝试创建 FTS5
    fts_ok = True
    try:
        cursor.execute("""
            CREATE VIRTUAL TABLE local_rag_fts_index USING fts5(
                source,
                context,
                entity_name,
                category UNINDEXED,
                tokenize="unicode61"
            );
        """)
    except sqlite3.OperationalError:
        print("FTS5 extension not compiled in SQLite. FTS5 index bypassed.")
        fts_ok = False
        
    conn.commit()

    print(f"Reading clinical datasets: {raw_data_json}...")
    with open(raw_data_json, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    contexts = []
    for item in raw_items:
        # 对 context 事实进行 Markdown 去噪
        clean_context = sanitize_markdown(item["context"])
        
        # 写入普通表
        cursor.execute("""
            INSERT INTO local_rag_index (entity_name, source, context, category)
            VALUES (?, ?, ?, ?);
        """, (item["entity_name"], item["source"], clean_context, item["category"]))
        
        # 写入倒排检索表
        if fts_ok:
            cursor.execute("""
                INSERT INTO local_rag_fts_index (entity_name, source, context, category)
                VALUES (?, ?, ?, ?);
            """, (item["entity_name"], item["source"], clean_context, item["category"]))
        
        contexts.append(clean_context)
    
    conn.commit()
    conn.close()
    print(f"Data loading to SQLite completed. Rows processed: {len(contexts)}")

    if not contexts:
        print("Empty context list. Index building bypassed.")
        return

    print("Encoding texts to vectors...")
    embeddings = model.encode(contexts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # L2 归一化以支持内积相似度（等价于余弦相似度）
    faiss.normalize_L2(embeddings)

    print("Building FAISS IndexFlatIP (Inner Product) Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # 保存
    faiss.write_index(index, output_index_path)
    print(f"🎉 RAG Vector storage index saved successfully to '{output_index_path}'.")

if __name__ == "__main__":
    # 测试装载测试用例
    test_json = "temp_seed_test.json"
    test_data = [
        {
            "entity_name": "布洛芬",
            "source": "refs:《布洛芬缓释胶囊说明书》",
            "context": "【用法用量】口服。药效主要持续。成人一次1粒（0.3克），一日2次（早晚各1次）。",
            "category": "用药方案"
        },
        {
            "entity_name": "甲氨蝶呤",
            "source": "refs:《甲氨蝶呤片说明书》",
            "context": "【不良反应】常见不良反应有口腔炎、口唇溃疡、白细胞减少，部分患者可引起严重骨髓抑制。",
            "category": "安全禁忌"
        }
    ]
    with open(test_json, "w", encoding="utf-8") as tf:
        json.dump(test_data, tf, ensure_ascii=False)
    
    try:
        build_offline_rag_index(test_json)
    except Exception as e:
        print(f"Error during offline build: {e}")
    finally:
        if os.path.exists(test_json):
            os.remove(test_json)
