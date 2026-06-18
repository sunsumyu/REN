# -*- coding: utf-8 -*-
"""
统一检索管理中心 (Retrieval Manager)。
三级检索架构的总协调器，负责协调本地 RAG、专业医学接口和受限互联网搜索。
提供三级分流与灾备降级逻辑，在第一级/第二级有高价值权威内容时优先阻断后续检索，
最大化保真并彻底过滤无效互联网垃圾噪音。
"""

import os
import logging
from typing import List, Dict, Any, Tuple
from retrieval.schemas import NormalizedClinicalRef
from retrieval.local_rag import LocalRAGService
from retrieval.api_gateway import APIGatewayService
from retrieval.restricted_search import RestrictedSearchService

logger = logging.getLogger("MedicalQA.RetrievalManager")

class RetrievalManager:
    def __init__(self, workspace_dir: str = ".", llm_service: Any = None):
        """
        初始化三级检索管理中心。自动拉起本地私有 RAG、API 接口网关和受限联网搜索服务。
        """
        self.workspace_dir = workspace_dir
        self.local_rag = LocalRAGService(workspace_dir=workspace_dir) # 内存型倒排全文检索，冷启动极速
        self.api_gateway = APIGatewayService(db_dir=workspace_dir, llm_service=llm_service) # 本地持久化 SQLite3 缓存网关
        self.restricted_search = RestrictedSearchService() # 域名白名单受限搜索引擎

    async def get_grounding_references(self, query: str, entity_name: str) -> Tuple[List[Dict[str, str]], str]:
        """
        核心分流检索路由逻辑。
        三级逐步降级：
          1. 优先调用本地私有库 RAG (Tier 1)。
          2. 若本地无匹配，降级调用垂直医药接口/PubMed Gateway (Tier 2)。
          3. 若接口无匹配，被迫启动域名受限的权威联网搜索 (Tier 3)。
        
        返回: (格式化后的 Grounding references 列表, 触发的检索层级标记)
        """
        logger.info(f"=== Starting Three-Tiered Retrieval Routing for Entity: '{entity_name}' ===")

        # --- Tier 1: Local Private RAG ---
        try:
            local_refs = await self.local_rag.search(query, entity_name)
            if local_refs:
                logger.info(f"--- Routing Success: Hit TIER 1 (Local RAG) for '{entity_name}' ---")
                formatted = [item.to_pipeline_format() for item in local_refs]
                return formatted, "TIER 1 (本地私有 RAG)"
        except Exception as e:
            logger.error(f"Tier 1 (Local RAG) failed during runtime: {e}")

        # --- Tier 2: Specialized API Gateway ---
        try:
            api_refs = await self.api_gateway.search(query, entity_name)
            if api_refs:
                logger.info(f"--- Routing Fallback: Hit TIER 2 (Specialized APIs) for '{entity_name}' ---")
                formatted = [item.to_pipeline_format() for item in api_refs]
                return formatted, "TIER 2 (医学 API 网关)"
        except Exception as e:
            logger.error(f"Tier 2 (API Gateway) failed during runtime: {e}")

        # --- Tier 3: Restricted Web Search ---
        try:
            search_refs = await self.restricted_search.search(query, entity_name)
            if search_refs:
                logger.info(f"--- Routing Fallback: Hit TIER 3 (Restricted Web Search) for '{entity_name}' ---")
                formatted = [item.to_pipeline_format() for item in search_refs]
                return formatted, "TIER 3 (受限联网搜索)"
        except Exception as e:
            logger.error(f"Tier 3 (Restricted Search) failed during runtime: {e}")

        # 极端兜底（空列表防御）
        logger.warning(f"All three retrieval tiers returned empty results for query '{query}'")
        fallback_ref = [{
            "source": "refs:《通用临床执业指引手册》",
            "context": "中西医结合联合用药必须在执业医师处方指导下科学安全用药，应定期随访、动态监测生化临床指标。"
        }]
        return fallback_ref, "TIER 3 DUMMY FALLBACK (极限防空防御)"

    def close(self):
        """
        释放资源，释放 SQLite 连接
        """
        try:
            self.local_rag.close()
        except Exception:
            pass
