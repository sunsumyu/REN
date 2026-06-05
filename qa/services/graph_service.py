# -*- coding: utf-8 -*-
import logging
from typing import Dict, Any, Set
import config

logger = logging.getLogger("MedicalQA.GraphService")

from abc import ABC, abstractmethod

class IGraphService(ABC):
    @abstractmethod
    async def fetch_random_knowledge_graph(self, count: int = config.DEFAULT_ENTITY_COUNT, kb_id: int = config.DEFAULT_KNOWLEDGE_BASE_ID, hop_count: int = config.DEFAULT_HOP_COUNT) -> Dict[str, Any]:
        pass

def is_empty_or_placeholder(val: Any) -> bool:
    """
    判断一个值是否为无效的占位符/空值（包含 None, 0, 0.0, 以及空字符串）。
    """
    if val is None:
        return True
    if isinstance(val, (int, float)) and val == 0:
        return True
    if isinstance(val, str):
        return not val.strip()
    return False

def merge_records(r1: Dict[str, Any], r2: Dict[str, Any]) -> Dict[str, Any]:
    """
    智能合并两个属性字典，确保在去重的同时，最大化保留有效且完整的数据。
    """
    merged = {}
    all_keys = set(r1.keys()).union(r2.keys())
    for k in all_keys:
        v1 = r1.get(k)
        v2 = r2.get(k)
        
        empty1 = is_empty_or_placeholder(v1)
        empty2 = is_empty_or_placeholder(v2)
        
        if empty1 and not empty2:
            merged[k] = v2
        elif empty2 and not empty1:
            merged[k] = v1
        elif empty1 and empty2:
            # 两个都是空值/占位符，优先保留非 None 的值
            merged[k] = v1 if v1 is not None else v2
        else:
            # 两个都是有效值，进行去重及去噪合并
            s1, s2 = str(v1).strip(), str(v2).strip()
            if k in {"description", "relationship"}:
                if s1.replace(" ", "") == s2.replace(" ", ""):
                    merged[k] = v1 if len(s1) < len(s2) else v2
                else:
                    merged[k] = v1 if len(s1) > len(s2) else v2
            else:
                # 其它类型字段，优先保留字符串表示较长（信息更全）的
                merged[k] = v1 if len(s1) >= len(s2) else v2
    return merged



class GraphService(IGraphService):
    def __init__(self, llm_service):
        # We inject the LLMService (which wraps the Async HTTP client)
        self.llm_service = llm_service
        self.seen_entity_ids: Set[str] = set()

    async def fetch_random_knowledge_graph(
        self, 
        count: int = config.DEFAULT_ENTITY_COUNT,
        kb_id: int = config.DEFAULT_KNOWLEDGE_BASE_ID,
        hop_count: int = config.DEFAULT_HOP_COUNT
    ) -> Dict[str, Any]:
        """
        Fetch a random medical subgraph. Automatically passes seen entity IDs to exclude them.
        """
        exclude_ids_str = ",".join(list(self.seen_entity_ids))
        
        params = {
            "count": count,
            "knowledgeBaseId": kb_id,
            "hopCount": hop_count
        }
        if exclude_ids_str:
            params["entityIds"] = exclude_ids_str
            
        logger.info(f"Fetching random KG data (excluding {len(self.seen_entity_ids)} entities)...")
        
        response_data = await self.llm_service._request_with_retry("GET", config.GRAPH_API_URL, params=params)
        
        if not response_data.get("success", False):
            raise Exception(f"Knowledge Graph API failed: {response_data.get('msg', 'Unknown error')}")
            
        graph_data = response_data.get("data", {})
        entities = graph_data.get("entities", [])
        
        # 🌟 对同一次 API 返回的数据按 id 智能合并去重，最大化保留各字段及非空信息
        unique_entities = {}
        for entity in entities:
            entity_id = str(entity.get("id"))
            if entity_id not in unique_entities:
                unique_entities[entity_id] = entity
            else:
                unique_entities[entity_id] = merge_records(unique_entities[entity_id], entity)
        
        graph_data["entities"] = list(unique_entities.values())
        entities = graph_data["entities"]
        
        # 🌟 对同一次 API 返回的关系数据按 id 智能合并去重（若缺失 id，则按 source/target/relationship 去重）
        relationships = graph_data.get("relationships", [])
        unique_relationships = {}
        for rel in relationships:
            rel_id = rel.get("id")
            if rel_id:
                rel_key = str(rel_id)
            else:
                rel_key = f"{rel.get('source')}-{rel.get('target')}-{rel.get('relationship')}"
                
            if rel_key not in unique_relationships:
                unique_relationships[rel_key] = rel
            else:
                unique_relationships[rel_key] = merge_records(unique_relationships[rel_key], rel)
        
        graph_data["relationships"] = list(unique_relationships.values())
        
        new_entities_count = 0
        for entity in entities:
            entity_id = str(entity.get("id"))
            if entity_id not in self.seen_entity_ids:
                self.seen_entity_ids.add(entity_id)
                new_entities_count += 1
                
        logger.info(f"Successfully retrieved {len(entities)} entities (registered {new_entities_count} new entities).")
        return graph_data
