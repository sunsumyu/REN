# -*- coding: utf-8 -*-
"""
第一级：本地私有 RAG 模块。
集成 hsa-agent 优秀设计：惰性单例模式、余弦相似度检索（L2归一化 + IndexFlatIP）。
安全退避设计：在环境损坏/模型损坏时自适应关闭向量通道，退化至倒排或 LIKE 检索。
"""

import os
import sqlite3
import re
import logging
from typing import List, Dict, Any
from retrieval.schemas import NormalizedClinicalRef

# 1. 动态尝试导入向量计算依赖，保证插拔无害性
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    VECTOR_RAG_AVAILABLE = True
except ImportError:
    VECTOR_RAG_AVAILABLE = False
    class SentenceTransformer: pass # 虚类防报错

logger = logging.getLogger("MedicalQA.LocalRAG")

_ACTIVE_RAG_SERVICES = []

class LocalEmbeddingEngine:
    """
    单例模式大模型编码管理器，支持惰性加载。
    """
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LocalEmbeddingEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "shibing624/text2vec-base-chinese"):
        self.model_name = model_name

    def get_model(self):
        if self._model is None:
            # 允许加载失败抛出异常，以便外层调用捕捉并关闭向量通道
            logger.info(f"Loading SentenceTransformer: '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
        return self._model


class LocalRAGService:
    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = workspace_dir
        self.db_path = os.path.join(self.workspace_dir, os.getenv("LOCAL_RAG_SQLITE_DB_PATH", "local_rag.db"))
        self.vector_index_path = os.path.join(self.workspace_dir, os.getenv("LOCAL_RAG_VECTOR_INDEX_PATH", "local_rag_vector.index"))
        self.embedding_model_name = os.getenv("LOCAL_RAG_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")
        self.similarity_threshold = float(os.getenv("LOCAL_RAG_SIMILARITY_THRESHOLD", "0.45"))
        
        # 依赖与开关共同决定是否开启向量模式
        self.vector_enabled = VECTOR_RAG_AVAILABLE and os.getenv("LOCAL_RAG_VECTOR_ENABLED", "true").lower() in ("true", "1")
        self.embedding_engine = None
        self.vector_index = None
        self.metadata_store = {}

        # 建立 SQLite3 持久化连接
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_sqlite_schemas()

        # 尝试惰性唤醒向量模式
        if self.vector_enabled:
            self._init_vector_components()
        else:
            logger.info("Local RAG Mode: Standard Mode (SQLite3 FTS5 / SQL LIKE).")
            
        _ACTIVE_RAG_SERVICES.append(self)

    def _ensure_sqlite_schemas(self):
        cursor = self.conn.cursor()
        # 普通物理事实表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS local_rag_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_name TEXT,
                source TEXT,
                context TEXT,
                category TEXT
            );
        """)
        # FTS5 虚拟表（带 unicode61 字符感知分词）
        try:
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS local_rag_fts_index USING fts5(
                    source,
                    context,
                    entity_name,
                    category UNINDEXED,
                    tokenize="unicode61"
                );
            """)
        except sqlite3.OperationalError:
            logger.warning("FTS5 extension not compiled in host python. Virtual table creation bypassed.")
        self.conn.commit()

    def _init_vector_components(self):
        try:
            if not os.path.exists(self.vector_index_path):
                raise FileNotFoundError(f"FAISS index file missing at '{self.vector_index_path}'")
            
            # 使用单例加载器惰性加载
            self.embedding_engine = LocalEmbeddingEngine(self.embedding_model_name)
            
            # 预热加载模型，检测是否抛出 OOM 或模型加载错误
            self.embedding_engine.get_model()
            
            # 读取物理 FAISS 索引
            self.vector_index = faiss.read_index(self.vector_index_path)
            
            # 从 SQLite 数据表加载对齐的 Metadata
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, source, context, category FROM local_rag_index;")
            for row in cursor.fetchall():
                # SQLite3 自增自 1 开始，FAISS 数组索引自 0 开始，故做减 1 转换映射
                self.metadata_store[row["id"] - 1] = {
                    "source": row["source"],
                    "context": row["context"],
                    "category": row["category"]
                }
            logger.info(f"🎉 Local RAG Vector Mode initialized. Registered {self.vector_index.ntotal} vectors.")
        except Exception as e:
            # 当任何一环出错时，直接失效向量模式，确保后续 search 绝不执行向量分支
            if isinstance(e, FileNotFoundError):
                logger.info(f"Vector storage index file not found: {e}. Falling back to standard mode (SQLite3 FTS5 / SQL LIKE).")
            else:
                logger.error(f"Failed to load vector storage: {e}. Automatically falling back to standard mode.", exc_info=True)
            self.vector_enabled = False
            self.embedding_engine = None
            self.vector_index = None
            self.metadata_store = {}

    def _fts_search(self, query: str, limit: int = 5) -> List[NormalizedClinicalRef]:
        cursor = self.conn.cursor()
        results = []
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
        words = [w for w in clean_query.split() if len(w) > 1]
        
        if words:
            search_term = " OR ".join(words)
            try:
                cursor.execute("""
                    SELECT source, context, category 
                    FROM local_rag_fts_index 
                    WHERE local_rag_fts_index MATCH ? LIMIT ?;
                """, (search_term, limit))
                for row in cursor.fetchall():
                    results.append(NormalizedClinicalRef(
                        source=row["source"],
                        context=row["context"],
                        category=row["category"],
                        metadata={"retrieval_method": "fts5_match"}
                    ))
            except Exception as e:
                logger.debug(f"FTS5 Search execution failed: {e}")
        return results

    def _like_search(self, entity_name: str, limit: int = 3) -> List[NormalizedClinicalRef]:
        cursor = self.conn.cursor()
        results = []
        cursor.execute("""
            SELECT source, context, category 
            FROM local_rag_index 
            WHERE entity_name LIKE ? OR context LIKE ? LIMIT ?;
        """, (f"%{entity_name}%", f"%{entity_name}%", limit))
        for row in cursor.fetchall():
            results.append(NormalizedClinicalRef(
                source=row["source"],
                context=row["context"],
                category=row["category"],
                metadata={"retrieval_method": "sql_like_match"}
            ))
        return results

    async def search(self, query: str, entity_name: str) -> List[NormalizedClinicalRef]:
        """
        本地私有 RAG 统一检索入口 (三级自愈退让)
        """
        import asyncio
        # --- 通道一：FAISS 余弦相似度语义检索 ---
        if self.vector_enabled and self.embedding_engine and self.vector_index:
            try:
                # 获取并使用模型编码
                model = self.embedding_engine.get_model()
                raw_vector = await asyncio.to_thread(model.encode, [query], show_progress_bar=False)
                query_vector = np.array(raw_vector).astype('float32')
                
                # 强制归一化以支持 IndexFlatIP 内积计算余弦相似度
                faiss.normalize_L2(query_vector)
                
                # 相似度召回 (k=5)
                scores, indices = await asyncio.to_thread(self.vector_index.search, query_vector, 5)
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    # 校验余弦相似度分值是否大于硬限额阈值，且在 metadata 列表中
                    if idx in self.metadata_store and score >= self.similarity_threshold:
                        meta = self.metadata_store[idx]
                        results.append(NormalizedClinicalRef(
                            source=meta["source"],
                            context=meta["context"],
                            category=meta["category"],
                            metadata={
                                "entity_name": entity_name, 
                                "similarity_score": float(score),
                                "retrieval_method": "vector_cosine_match"
                            }
                        ))
                if results:
                    logger.info(f"Tier 1 Vector HIT for entity '{entity_name}' ({len(results)} items retrieved)")
                    return results
            except Exception as e:
                # 向量执行失败：报错并直接绕过，降级到 FTS5，防止异常向上抛出中断程序
                logger.error(f"Vector RAG runtime search failed: {e}. Transitioning to FTS5...")

        # --- 通道二：SQLite FTS5 倒排匹配 ---
        try:
            fts_results = self._fts_search(query)
            if fts_results:
                logger.info(f"Tier 1 FTS5 HIT for entity '{entity_name}'")
                return fts_results
        except Exception as e:
            logger.error(f"FTS5 Search execution crashed: {e}. Transitioning to SQL LIKE...")

        # --- 通道三：SQL LIKE 字符模糊对撞 ---
        try:
            like_results = self._like_search(entity_name)
            if like_results:
                logger.info(f"Tier 1 SQL LIKE HIT for entity '{entity_name}'")
                return like_results
        except Exception as e:
            logger.error(f"SQL LIKE search failed: {e}")

        # 均未命中，返回空触发 Tier 2 外网 PubMed 检索
        logger.info(f"Tier 1 RAG missed entirely for entity '{entity_name}'.")
        return []

    def close(self):
        global _ACTIVE_RAG_SERVICES
        if self in _ACTIVE_RAG_SERVICES:
            try:
                _ACTIVE_RAG_SERVICES.remove(self)
            except ValueError:
                pass
        try:
            self.conn.close()
        except Exception:
            pass

    @classmethod
    def clear_all_caches(cls):
        global _ACTIVE_RAG_SERVICES
        logger.info(f"🔄 Clearing all caches for {len(_ACTIVE_RAG_SERVICES)} active LocalRAGService instances.")
        for service in list(_ACTIVE_RAG_SERVICES):
            try:
                service.metadata_store.clear()
                if service.vector_enabled:
                    service._init_vector_components()
                logger.info("  - Cache cleared and components re-initialized successfully.")
            except Exception as e:
                logger.error(f"Error clearing cache for service instance: {e}")
