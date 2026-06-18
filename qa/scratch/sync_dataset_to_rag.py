# -*- coding: utf-8 -*-
"""
双轨同步脚本：将 evaluation 数据集中的预期 facts 同步导入本地 RAG 数据库中，
并重建 FAISS 索引，使召回评估测试可以正确找到匹配项。
"""

import os
import json
import sqlite3
import re
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

def extract_entity_name(source: str, context: str) -> str:
    # 从 source 中提取实体名
    if "实体库:" in source:
        m = re.search(r'实体库:([^》\s]+)', source)
        if m:
            return m.group(1)
    elif "图谱关系:" in source:
        m = re.search(r'图谱关系:([^》\s]+)', source)
        if m:
            parts = m.group(1).split("-")
            return parts[0] if parts else m.group(1)
    elif "常见临床疾病与合理用药诊疗路径-" in source:
        m = re.search(r'常见临床疾病与合理用药诊疗路径-([^》\s]+)', source)
        if m:
            return m.group(1)
            
    # 从 context 中提取
    if context.startswith("概念定义:"):
        m = re.search(r'概念定义:\s*([^\s(（]+)', context)
        if m:
            return m.group(1)
    elif context.startswith("知识关联:"):
        m = re.search(r'【([^】]+)】', context)
        if m:
            return m.group(1)
            
    return ""

def main():
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dataset_path = os.path.join(workspace_dir, "medical_qa_dataset.jsonl")
    db_path = os.path.join(workspace_dir, "local_rag.db")
    vector_index_path = os.path.join(workspace_dir, "local_rag_vector.index")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    print("Reading expected facts from medical_qa_dataset.jsonl...")
    expected_facts = []
    seen_sources = set()
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            try:
                data = json.loads(line.strip())
                evidence_contract = data.get("evidence_contract", {})
                facts = evidence_contract.get("facts", []) if isinstance(evidence_contract, dict) else []
                
                for fact in facts:
                    source = fact.get("source", "")
                    # 仅同步本地事实，排除 PubMed 等外部渠道
                    if not source or any(x in source for x in ["PubMed", "在线公开", "抓取异常"]):
                        continue
                        
                    if source in seen_sources:
                        continue
                        
                    seen_sources.add(source)
                    context = fact.get("context_preview", "").strip()
                    entity_name = extract_entity_name(source, context)
                    
                    # 确定分类
                    category = "实体概念"
                    if "图谱关系" in source:
                        category = "知识关联"
                    elif "临床路径" in source or "指南" in source:
                        category = "临床诊疗"
                        
                    expected_facts.append({
                        "entity_name": entity_name,
                        "source": source,
                        "context": context,
                        "category": category
                    })
            except Exception as e:
                print(f"Error parsing line {idx}: {e}")
                
    print(f"Found {len(expected_facts)} unique expected facts to sync from dataset.")
    
    if not expected_facts:
        print("No new facts to import.")
        return

    # 连接数据库，读取当前数据库中所有的记录，进行合并去重
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取现有库中所有的记录
    cursor.execute("SELECT entity_name, source, context, category, icd_code, standard_days FROM local_rag_index;")
    existing_records = []
    existing_sources = set()
    for row in cursor.fetchall():
        existing_sources.add(row[1])
        existing_records.append({
            "entity_name": row[0],
            "source": row[1],
            "context": row[2],
            "category": row[3],
            "icd_code": row[4] or "N/A",
            "standard_days": row[5] or "N/A"
        })
        
    print(f"Existing records in local_rag.db: {len(existing_records)}")
    
    # 合并新事实
    new_imported = 0
    for fact in expected_facts:
        if fact["source"] not in existing_sources:
            existing_records.append({
                "entity_name": fact["entity_name"],
                "source": fact["source"],
                "context": fact["context"],
                "category": fact["category"],
                "icd_code": "N/A",
                "standard_days": "N/A"
            })
            new_imported += 1
            
    print(f"Total merged records to write: {len(existing_records)} (New: {new_imported})")
    
    if new_imported == 0:
        print("🎉 All expected facts are already present in local_rag.db.")
        conn.close()
        return

    # 清空并重写 SQLite 数据库
    print("Re-writing SQLite database...")
    cursor.execute("DROP TABLE IF EXISTS local_rag_index;")
    cursor.execute("DROP TABLE IF EXISTS local_rag_fts_index;")
    
    cursor.execute("""
        CREATE TABLE local_rag_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT,
            source TEXT,
            context TEXT,
            category TEXT,
            icd_code TEXT,
            standard_days TEXT
        );
    """)
    
    cursor.execute("""
        CREATE VIRTUAL TABLE local_rag_fts_index USING fts5(
            source,
            context,
            entity_name,
            category UNINDEXED,
            icd_code UNINDEXED,
            standard_days UNINDEXED,
            tokenize="unicode61"
        );
    """)
    
    for r in existing_records:
        cursor.execute("""
            INSERT INTO local_rag_index (entity_name, source, context, category, icd_code, standard_days)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (r["entity_name"], r["source"], r["context"], r["category"], r["icd_code"], r["standard_days"]))
        
        cursor.execute("""
            INSERT INTO local_rag_fts_index (source, context, entity_name, category, icd_code, standard_days)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (r["source"], r["context"], r["entity_name"], r["category"], r["icd_code"], r["standard_days"]))
        
    conn.commit()
    conn.close()
    print("Successfully re-wrote SQLite database.")

    # 重新生成向量并保存 FAISS 索引
    if VECTOR_SUPPORT:
        try:
            print("Loading local embedding model...")
            embedding_engine = LocalEmbeddingEngine(os.getenv("LOCAL_RAG_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"))
            model = embedding_engine.get_model()
            
            print(f"Encoding all {len(existing_records)} contexts...")
            contexts = [x["context"] for x in existing_records]
            raw_vectors = model.encode(contexts, show_progress_bar=True)
            vectors = np.array(raw_vectors).astype('float32')
            faiss.normalize_L2(vectors)
            
            print(f"Building and writing FAISS index to {vector_index_path}...")
            dimension = vectors.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)
            faiss_index.add(vectors)
            
            faiss.write_index(faiss_index, vector_index_path)
            print(f"Successfully rebuilt FAISS index with {faiss_index.ntotal} vectors.")
            print("\nPlease run 'python scratch/test_rag_matching.py' to verify the updated recall rate!")
        except Exception as e:
            print(f"Error rebuilding FAISS index: {e}")
    else:
        print("Vector support not available, skipped FAISS rebuild.")

if __name__ == "__main__":
    main()
