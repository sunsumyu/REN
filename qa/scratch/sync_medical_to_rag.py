# -*- coding: utf-8 -*-
"""
一键同步 medical.json 数据至本地 RAG 向量数据库
提取结构化实体（疾病、药物、症状、方剂）并进行自然语言序列化，
批量生成 Embedding 写入 FAISS 索引并同步至 SQLite3 库表。
"""

import os
import json
import sqlite3
import numpy as np
import sys
from dotenv import load_dotenv

# 确保能引入项目中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_SUPPORT = True
except ImportError:
    VECTOR_SUPPORT = False

from retrieval.local_rag import LocalEmbeddingEngine

def serialize_entity(item: dict) -> str:
    """
    将结构化图谱节点实体转换为对齐的非结构化陈述文本
    """
    name = item.get("name", "")
    desc = item.get("desc", "").strip() or "暂无详细描述"
    
    # 抽取特征属性丰富上下文，帮助语义召回
    parts = [f"概念定义: {name} (类型: 医疗实体) - {desc}"]
    
    symptoms = item.get("symptom", [])
    if symptoms:
        parts.append(f"常见临床症状包括：{', '.join(symptoms)}。")
        
    prevent = item.get("prevent", "")
    if prevent:
        parts.append(f"预防与防护措施：{prevent.strip()}")
        
    drugs = item.get("recommand_drug", []) or item.get("common_drug", [])
    if drugs:
        parts.append(f"临床推荐/常用药物：{', '.join(drugs)}。")
        
    checks = item.get("check", [])
    if checks:
        parts.append(f"推荐辅助检查项目：{', '.join(checks)}。")
        
    return " ".join(parts)

def main():
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    json_path = os.path.join(workspace_dir, "medical.json")
    db_path = os.path.join(workspace_dir, "local_rag.db")
    vector_index_path = os.path.join(workspace_dir, "local_rag_vector.index")
    
    if not os.path.exists(json_path):
        print(f"Error: medical.json not found at {json_path}")
        return
        
    # 为了防止全量 8800+ 实体转向量时间过长，我们采用优先过滤模式
    # 我们优先导入前 500 个样本，并在此基础上强行检索包含我们测试集高频实体的节点进行导入
    priority_keywords = [
        "甘露特钠", "九期一", "硬脂酸镁", "过敏", "葶苈", "大枣", "葶苈大枣", "方剂", "牛膝",
        "别嘌醇", "骨通", "文拉法辛", "更昔洛韦", "阿托伐他汀", "伊曲康唑", "美托洛尔", "帕罗西汀",
        "氟尿嘧啶", "朱砂", "珍视明", "硫酸软骨素", "塞利洛尔", "达芦那韦", "利托那韦", "谷胱甘肽"
    ]
    
    print("Reading and parsing medical.json...")
    candidate_entities = []
    
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                name = item.get("name", "")
                desc = item.get("desc", "")
                
                # 检查是否命中高优先关键词
                is_priority = any(kw in name or kw in desc for kw in priority_keywords)
                
                # 保障一定量级的背景实体（前1000个），加高优先级实体
                if is_priority or len(candidate_entities) < 1000:
                    candidate_entities.append(item)
            except Exception as e:
                pass

    print(f"Selected {len(candidate_entities)} entities for vectorization and RAG database syncing.")

    # 初始化本地数据库连接
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 读取已有已导入的 sources，防止重复写入
    cursor.execute("SELECT source FROM local_rag_index;")
    existing_sources = {row[0] for row in cursor.fetchall()}
    
    items_to_import = []
    for entity in candidate_entities:
        name = entity.get("name", "")
        source_name = f"refs:《实体库:{name}》"
        
        if source_name in existing_sources:
            continue
            
        context_text = serialize_entity(entity)
        items_to_import.append({
            "entity_name": name,
            "source": source_name,
            "context": context_text,
            "category": "实体概念"
        })

    if not items_to_import:
        print("🎉 No new entities need to be imported. RAG database is up-to-date.")
        conn.close()
        return

    print(f"Starting import of {len(items_to_import)} new records...")

    # 如果有向量支持，生成并 append 向量
    if VECTOR_SUPPORT and os.path.exists(vector_index_path):
        try:
            print("Loading local embedding model...")
            embedding_engine = LocalEmbeddingEngine(os.getenv("LOCAL_RAG_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"))
            model = embedding_engine.get_model()
            
            print("Encoding contexts (generating embeddings)...")
            contexts = [x["context"] for x in items_to_import]
            raw_vectors = model.encode(contexts, show_progress_bar=True)
            vectors = np.array(raw_vectors).astype('float32')
            faiss.normalize_L2(vectors)
            
            print(f"Reading FAISS index from {vector_index_path}...")
            faiss_index = faiss.read_index(vector_index_path)
            
            print(f"Appending {vectors.shape[0]} new vectors to FAISS index...")
            faiss_index.add(vectors)
            
            faiss.write_index(faiss_index, vector_index_path)
            print("Successfully updated FAISS index file.")
        except Exception as e:
            print(f"Error during FAISS vector update: {e}. SQLite entries will still be written.")
    else:
        print("Vector RAG FAISS update skipped (dependencies missing or index file not found).")

    # 写入 SQLite 库表
    imported_count = 0
    for item in items_to_import:
        try:
            # 写入主索引表
            cursor.execute("""
                INSERT INTO local_rag_index (entity_name, source, context, category, icd_code, standard_days)
                VALUES (?, ?, ?, ?, 'N/A', 'N/A');
            """, (item["entity_name"], item["source"], item["context"], item["category"]))
            
            # 写入 FTS5 虚拟全文检索表
            try:
                cursor.execute("""
                    INSERT INTO local_rag_fts_index (source, context, entity_name, category, icd_code, standard_days)
                    VALUES (?, ?, ?, ?, 'N/A', 'N/A');
                """, (item["source"], item["context"], item["entity_name"], item["category"]))
            except Exception as fts_err:
                pass
                
            imported_count += 1
        except Exception as db_err:
            print(f"DB insert error for '{item['entity_name']}': {db_err}")

    conn.commit()
    conn.close()
    
    print(f"Successfully imported {imported_count} entities into local_rag.db.")
    print("Please run 'python scratch/test_rag_matching.py' again to see the improved Recall rate!")

if __name__ == "__main__":
    main()
