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

    def _normalize_entity_name(self, name: str) -> str:
        """
        对实体名称进行归一化清洗：
        1. 全部转为小写
        2. 剔除所有空白字符与常见符号/连字符 (_, -, ~, ®, ™ 等)
        3. 剔除中西药常见剂型/规格后缀 (片, 胶囊, 注射液, 口服溶液, 胶浆, 颗粒, 软膏)
        """
        if not name:
            return ""
        name = name.lower()
        # 去除所有空格和标点符号
        name = re.sub(r'[\s\-~_\(\)（）\+®™/\.\,，。]', '', name)
        # 去除常见剂型/产品后缀
        suffixes = ["片", "胶囊", "注射液", "口服溶液", "口服液", "胶浆", "颗粒", "软膏", "凝胶", "滴眼液", "泡腾片", "贴膏"]
        for suf in suffixes:
            if name.endswith(suf) and len(name) > len(suf):
                name = name[:-len(suf)]
        return name

    def _ensure_sqlite_schemas(self):
        cursor = self.conn.cursor()
        # 普通物理事实表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS local_rag_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_name TEXT,
                source TEXT,
                context TEXT,
                category TEXT,
                icd_code TEXT,
                standard_days TEXT
            );
        """)
        
        # 增量检查普通表，不包含新字段则 Alter 新增
        cursor.execute("PRAGMA table_info(local_rag_index);")
        columns = [row[1] for row in cursor.fetchall()]
        if "icd_code" not in columns:
            try:
                cursor.execute("ALTER TABLE local_rag_index ADD COLUMN icd_code TEXT;")
                cursor.execute("ALTER TABLE local_rag_index ADD COLUMN standard_days TEXT;")
                logger.info("Database Schema Migrated: Added 'icd_code' and 'standard_days' columns to local_rag_index table.")
            except Exception as e:
                logger.warning(f"Failed to alter local_rag_index schema during initialization: {e}")

        # FTS5 虚拟表（带 unicode61 字符感知分词）
        fts_ok = True
        try:
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS local_rag_fts_index USING fts5(
                    source,
                    context,
                    entity_name,
                    category UNINDEXED,
                    icd_code UNINDEXED,
                    standard_days UNINDEXED,
                    tokenize="unicode61"
                );
            """)
        except sqlite3.OperationalError:
            logger.warning("FTS5 extension not compiled in host python. Virtual table creation bypassed.")
            fts_ok = False
            
        # 如果 FTS5 已经存在但字段不匹配，则重建它
        if fts_ok:
            try:
                cursor.execute("PRAGMA table_info(local_rag_fts_index);")
                fts_columns = [row[1] for row in cursor.fetchall()]
                if fts_columns and "icd_code" not in fts_columns:
                    logger.info("FTS index schema is outdated. Rebuilding FTS virtual table...")
                    cursor.execute("DROP TABLE local_rag_fts_index;")
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
            except Exception as e:
                logger.warning(f"Failed to rebuild FTS index schema during initialization: {e}")

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
            cursor.execute("SELECT id, source, context, category, entity_name, icd_code, standard_days FROM local_rag_index;")
            for row in cursor.fetchall():
                # SQLite3 自增自 1 开始，FAISS 数组索引自 0 开始，故做减 1 转换映射
                self.metadata_store[row["id"] - 1] = {
                    "source": row["source"],
                    "context": row["context"],
                    "category": row["category"],
                    "entity_name": row["entity_name"],
                    "icd_code": row["icd_code"],
                    "standard_days": row["standard_days"]
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
        
        words = []
        for w in clean_query.split():
            if len(w) <= 1:
                continue
            if re.search(r'[\u4e00-\u9fa5]', w):
                if len(w) > 4:
                    for i in range(len(w) - 1):
                        words.append(w[i:i+2])
                    for i in range(len(w) - 2):
                        words.append(w[i:i+3])
                else:
                    words.append(w)
            else:
                words.append(w)
        words = list(set([wd.strip() for wd in words if len(wd.strip()) > 1]))
        
        if words:
            search_term = " OR ".join(words)
            try:
                cursor.execute("""
                    SELECT source, context, category, entity_name, icd_code, standard_days
                    FROM local_rag_fts_index 
                    WHERE local_rag_fts_index MATCH ? LIMIT ?;
                """, (search_term, limit))
                from utils.clinical_purifier import ClinicalPathwayPurifier
                for row in cursor.fetchall():
                    metadata_dict = {
                        "entity_name": row["entity_name"],
                        "icd_code": row["icd_code"],
                        "standard_days": row["standard_days"]
                    }
                    enriched_context = ClinicalPathwayPurifier.enrich_single_chunk(row["context"], metadata_dict)
                    results.append(NormalizedClinicalRef(
                        source=row["source"],
                        context=enriched_context,
                        category=row["category"],
                        metadata={
                            "entity_name": row["entity_name"],
                            "icd_code": row["icd_code"],
                            "standard_days": row["standard_days"],
                            "retrieval_method": "fts5_match"
                        }
                    ))
            except Exception as e:
                logger.debug(f"FTS5 Search execution failed: {e}")
        return results

    def _like_search(self, entity_name: str, limit: int = 3) -> List[NormalizedClinicalRef]:
        cursor = self.conn.cursor()
        results = []
        cursor.execute("""
            SELECT source, context, category, entity_name, icd_code, standard_days 
            FROM local_rag_index 
            WHERE entity_name LIKE ? OR context LIKE ? LIMIT ?;
        """, (f"%{entity_name}%", f"%{entity_name}%", limit))
        from utils.clinical_purifier import ClinicalPathwayPurifier
        for row in cursor.fetchall():
            metadata_dict = {
                "entity_name": row["entity_name"],
                "icd_code": row["icd_code"],
                "standard_days": row["standard_days"]
            }
            enriched_context = ClinicalPathwayPurifier.enrich_single_chunk(row["context"], metadata_dict)
            results.append(NormalizedClinicalRef(
                source=row["source"],
                context=enriched_context,
                category=row["category"],
                metadata={
                    "entity_name": row["entity_name"],
                    "icd_code": row["icd_code"],
                    "standard_days": row["standard_days"],
                    "retrieval_method": "sql_like_match"
                }
            ))
        return results

    async def search(self, query: str, entity_name: str) -> List[NormalizedClinicalRef]:
        """
        本地私有 RAG 统一检索入口：基于双轨 Hybrid (FAISS + FTS5) + RRF 重排与实体过滤
        """
        import asyncio
        from utils.clinical_purifier import ClinicalPathwayPurifier

        vector_hits = []
        # 1. 向量通道搜索 (Dense Search)
        if self.vector_enabled and self.embedding_engine and self.vector_index:
            try:
                model = self.embedding_engine.get_model()
                raw_vector = await asyncio.to_thread(model.encode, [query], show_progress_bar=False)
                query_vector = np.array(raw_vector).astype('float32')
                faiss.normalize_L2(query_vector)
                
                scores, indices = await asyncio.to_thread(self.vector_index.search, query_vector, 15)
                for score, idx in zip(scores[0], indices[0]):
                    if idx in self.metadata_store:
                        meta = self.metadata_store[idx]
                        vector_hits.append({
                            "id": idx,
                            "source": meta["source"],
                            "context": meta["context"],
                            "category": meta["category"],
                            "entity_name": meta.get("entity_name", "未知"),
                            "icd_code": meta.get("icd_code", "未知"),
                            "standard_days": meta.get("standard_days", "未知"),
                            "similarity_score": float(score)
                        })
            except Exception as e:
                logger.error(f"Vector RAG runtime search failed: {e}")

        # 2. 倒排通道搜索 (Sparse Search via FTS5)
        fts_hits = []
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
        
        words = []
        for w in clean_query.split():
            if len(w) <= 1:
                continue
            if re.search(r'[\u4e00-\u9fa5]', w):
                if len(w) > 4:
                    for i in range(len(w) - 1):
                        words.append(w[i:i+2])
                    for i in range(len(w) - 2):
                        words.append(w[i:i+3])
                else:
                    words.append(w)
            else:
                words.append(w)
                
        if entity_name:
            words.append(entity_name)
            
        words = list(set([wd.strip() for wd in words if len(wd.strip()) > 1]))
        
        if words:
            search_term = " OR ".join(words)
            try:
                cursor = self.conn.cursor()
                # rowid maps to id in local_rag_index, id - 1 corresponds to FAISS idx
                cursor.execute("""
                    SELECT rowid, source, context, category, entity_name, icd_code, standard_days 
                    FROM local_rag_fts_index 
                    WHERE local_rag_fts_index MATCH ? LIMIT 15;
                """, (search_term,))
                for row in cursor.fetchall():
                    fts_hits.append({
                        "id": row["rowid"] - 1,
                        "source": row["source"],
                        "context": row["context"],
                        "category": row["category"],
                        "entity_name": row["entity_name"],
                        "icd_code": row["icd_code"],
                        "standard_days": row["standard_days"]
                    })
            except Exception as e:
                logger.debug(f"FTS5 Search execution failed during hybrid search: {e}")

        # 3. Reciprocal Rank Fusion (RRF) 融合重排
        rrf_scores = {}
        candidates = {}

        # 向量排名分数累计
        for rank, hit in enumerate(vector_hits, start=1):
            doc_id = hit["id"]
            candidates[doc_id] = hit
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60.0 + rank)

        # FTS排名分数累计
        for rank, hit in enumerate(fts_hits, start=1):
            doc_id = hit["id"]
            if doc_id not in candidates:
                candidates[doc_id] = hit
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60.0 + rank)

        # 4. 弹性实体硬匹配与退避过滤 (Elastic Entity Filtering & Backoff)
        entity_matches = []
        other_matches = []
        
        target_norm = self._normalize_entity_name(entity_name) if entity_name else ""

        for doc_id, score in rrf_scores.items():
            cand = candidates[doc_id]
            cand_entity = cand.get("entity_name", "")
            cand_norm = self._normalize_entity_name(cand_entity)
            
            is_entity_match = False
            if target_norm:
                # 优先检查清洗后的互包含关系
                if (target_norm in cand_norm) or (cand_norm in target_norm):
                    is_entity_match = True
            
            if is_entity_match:
                entity_matches.append((doc_id, score))
            else:
                other_matches.append((doc_id, score))

        # 决定使用的候选集
        final_candidates = []
        retrieval_method_tag = "hybrid_rrf"
        
        if entity_matches:
            entity_matches.sort(key=lambda x: x[1], reverse=True)
            final_candidates = entity_matches
            retrieval_method_tag += "_entity"
        else:
            if not target_norm:
                # 只有当没有指定具体实体时，才允许使用 general 的其他匹配
                other_matches.sort(key=lambda x: x[1], reverse=True)
                final_candidates = other_matches
                retrieval_method_tag += "_general"
            else:
                # 【退避防御优化】：如果指定了实体但物理上没有硬匹配成功的文档，
                # 不应粗暴地直接返回空！这会导致极其优秀的语义匹配结果（如 CYP2C9 vs 塞来昔布）被强行抛弃。
                # 只要候选文档的相似度较高 (>= 0.70)，或者是 FTS/倒排高频命中，且 RRF 排序在 Top-3，则被召回。
                backup_candidates = []
                for doc_id, score in other_matches:
                    cand = candidates[doc_id]
                    sim = cand.get("similarity_score")
                    
                    is_high_value = False
                    if sim is not None and sim >= max(0.45, self.similarity_threshold - 0.10):
                        is_high_value = True
                    
                    if is_high_value:
                        backup_candidates.append((doc_id, score))
                
                if backup_candidates:
                    backup_candidates.sort(key=lambda x: x[1], reverse=True)
                    final_candidates = backup_candidates[:3] # 取前3个高质量备选
                    retrieval_method_tag += "_fallback_high_sim"
                    logger.info(f"Local RAG: Specified entity '{entity_name}' had 0 hard entity matches, but retrieved {len(final_candidates)} high-value fallback candidates.")
                else:
                    logger.info(f"Local RAG: Specified entity '{entity_name}' had 0 matching docs in Tier 1. Bypassing other_matches to trigger fallback.")

        # 5. 相似度门禁过滤与通道决定
        results = []
        for doc_id, rrf_score in final_candidates:
            cand = candidates[doc_id]
            in_vector = any(h["id"] == doc_id for h in vector_hits)
            in_fts = any(h["id"] == doc_id for h in fts_hits)
            sim_score = cand.get("similarity_score")

            passed = False
            channel = "unknown"
            if in_vector and in_fts:
                # 混合模式：适当放宽阈值
                if sim_score >= min(0.40, self.similarity_threshold):
                    passed = True
                    channel = "hybrid"
            elif in_vector:
                # 纯语义向量模式：严格阈值门禁
                if sim_score >= self.similarity_threshold:
                    passed = True
                    channel = "vector"
            elif in_fts:
                # 纯倒排文本模式：天然通过
                passed = True
                channel = "fts"

            if passed:
                metadata_dict = {
                    "entity_name": cand.get("entity_name", "未知"),
                    "icd_code": cand.get("icd_code", "未知"),
                    "standard_days": cand.get("standard_days", "未知")
                }
                # 动态装配带有元数据的展示上下文
                enriched_context = ClinicalPathwayPurifier.enrich_single_chunk(cand["context"], metadata_dict)
                results.append(NormalizedClinicalRef(
                    source=cand["source"],
                    context=enriched_context,
                    category=cand["category"],
                    metadata={
                        "entity_name": cand.get("entity_name", "未知"),
                        "similarity_score": sim_score if sim_score is not None else "N/A",
                        "rrf_score": float(rrf_score),
                        "retrieval_method": f"{retrieval_method_tag}_{channel}_match"
                    }
                ))

        if results:
            logger.info(f"Tier 1 Hybrid HIT for entity '{entity_name}' ({len(results)} items retrieved)")
            return results[:5]

        # 6. 兜底退避通道：SQL LIKE 模糊扫描
        try:
            like_results = self._like_search(entity_name)
            if like_results:
                logger.info(f"Tier 1 SQL LIKE Fallback HIT for entity '{entity_name}'")
                return like_results
        except Exception as e:
            logger.error(f"SQL LIKE fallback search failed: {e}")

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
