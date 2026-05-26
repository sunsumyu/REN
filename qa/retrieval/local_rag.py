# -*- coding: utf-8 -*-
"""
第一级：本地私有 RAG 模块。
采用内置的 SQLite3 FTS5 (Full-Text Search) 作为零依赖的高性能检索引擎，
将本地说明书与指南进行切片并建立倒排索引，确保 80% 的常见药物回答具备最高级的权威性。
"""

import os
import sqlite3
import re
import logging
from typing import List, Dict, Any
from retrieval.schemas import NormalizedClinicalRef

logger = logging.getLogger("MedicalQA.LocalRAG")

class LocalRAGService:
    def __init__(self, db_path: str = ":memory:"):
        """
        初始化本地 RAG 服务。默认使用内存型数据库，便于多进程安全和零文件冲突。
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._initialize_index()
        self._load_seed_data()

    def _initialize_index(self):
        """
        初始化 SQLite3 FTS5 虚表，提供专业的 BM25 分词全文检索能力
        """
        cursor = self.conn.cursor()
        # 创建 FTS5 虚拟表以支持全文检索，包含源、段落内容及关键字字段
        try:
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS local_rag_index USING fts5(
                    source,
                    context,
                    entity_name,
                    category UNINDEXED,
                    tokenize="unicode61"
                );
            """)
            self.conn.commit()
            logger.info("Successfully initialized SQLite3 FTS5 virtual table for local RAG indexing.")
        except Exception as e:
            logger.error(f"Failed to create FTS5 table, falling back to standard table: {e}")
            # Fallback to standard table with LIKE search if FTS5 is not compiled in host Python
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS local_rag_index (
                    source TEXT,
                    context TEXT,
                    entity_name TEXT,
                    category TEXT
                );
            """)
            self.conn.commit()

    def _load_seed_data(self):
        """
        从本地 guideline_db.py 动态加载权威说明书与指南种子数据，实现自动 Grounding。
        """
        try:
            from guideline_db import GUIDELINE_DATA
        except ImportError:
            logger.warning("guideline_db.py not found. Local RAG index will start empty.")
            return

        cursor = self.conn.cursor()
        # 清理旧数据，防止重复加载
        try:
            cursor.execute("DELETE FROM local_rag_index;")
        except Exception:
            pass

        count = 0
        for entity_name, ref_items in GUIDELINE_DATA.items():
            for item in ref_items:
                source = item.get("source", "refs:《未知说明书》")
                context = item.get("context", "")
                
                # 简单解析医学切面门类
                category = "通用"
                if "说明书" in source:
                    if "适应症" in source or "主治" in source:
                        category = "药理机制"
                    elif "用药剂量" in source or "用法用量" in source:
                        category = "用药方案"
                    elif "不良反应" in source or "禁忌" in source:
                        category = "安全禁忌"
                elif "指南" in source:
                    category = "指南推荐"

                # 插入虚表
                cursor.execute(
                    "INSERT INTO local_rag_index (source, context, entity_name, category) VALUES (?, ?, ?, ?);",
                    (source, context, entity_name, category)
                )
                count += 1

        self.conn.commit()
        logger.info(f"Loaded {count} seed clinical reference items into Tier 1 Local RAG index.")

    def search(self, query: str, entity_name: str, threshold: float = 0.3) -> List[NormalizedClinicalRef]:
        """
        混合式全文检索。通过匹配实体词与自然语言查询，提取高可信度段落。
        """
        cursor = self.conn.cursor()
        results = []
        
        # 1. 优先进行实体词的完全与子串检索（精准锚定）
        cursor.execute(
            "SELECT source, context, entity_name, category FROM local_rag_index WHERE entity_name LIKE ? OR ? LIKE '%' || entity_name || '%';",
            (f"%{entity_name}%", query)
        )
        rows = cursor.fetchall()
        
        # 转换为规范实体
        for row in rows:
            results.append(NormalizedClinicalRef(
                source=row["source"],
                context=row["context"],
                category=row["category"],
                metadata={"entity_name": row["entity_name"], "retrieval_method": "entity_like_match"}
            ))

        # 2. 如果实体词未命中，则降级为 FTS5 全文模糊检索（支持 BM25 全文分词）
        if not results:
            # 清洗查询词，去除符号，构建 FTS5 查询短语
            clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
            # 拼接词项进行模糊匹配
            words = [w for w in clean_query.split() if len(w) > 1]
            if words:
                search_term = " OR ".join(words)
                try:
                    cursor.execute(
                        "SELECT source, context, entity_name, category FROM local_rag_index WHERE local_rag_index MATCH ? LIMIT 5;",
                        (search_term,)
                    )
                    fts_rows = cursor.fetchall()
                    for row in fts_rows:
                        results.append(NormalizedClinicalRef(
                            source=row["source"],
                            context=row["context"],
                            category=row["category"],
                            metadata={"entity_name": row["entity_name"], "retrieval_method": "fts5_match"}
                        ))
                except Exception as e:
                    logger.debug(f"FTS5 MATCH failed or bypassed: {e}")
                    # 如果 FTS5 抛出异常，降级为常规关键字扫描
                    for word in words[:3]:
                        cursor.execute(
                            "SELECT source, context, entity_name, category FROM local_rag_index WHERE context LIKE ? LIMIT 3;",
                            (f"%{word}%",)
                        )
                        like_rows = cursor.fetchall()
                        for row in like_rows:
                            # 避免重复
                            if not any(r.context == row["context"] for r in results):
                                results.append(NormalizedClinicalRef(
                                    source=row["source"],
                                    context=row["context"],
                                    category=row["category"],
                                    metadata={"entity_name": row["entity_name"], "retrieval_method": "keyword_like_match"}
                                ))

        logger.info(f"Tier 1 (Local RAG) search completed. Query: '{query}', Found items: {len(results)}")
        return results

    def close(self):
        """
        释放资源
        """
        try:
            self.conn.close()
        except Exception:
            pass
